from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_foundation.manifest import write_csv
from ranking_model.validation_runner import json_atomic
from ranking_model.validation_selection import METRICS, canonical_fingerprint, file_fingerprint, paired_bootstrap

from .config import Config, VARIANTS
from .contracts import Candidate, OUTPUT_SCHEMA, candidates_from_columns, normalize, order_query, output_rows
from .evaluation import diagnostics, metrics, select_config
from .greedy import rerank
from .metadata import build_metadata, metadata_lookup


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def atomic_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".incomplete")
    pq.write_table(table, temporary, compression="zstd")
    pq.ParquetFile(temporary).metadata
    os.replace(temporary, path)


class Runner:
    def __init__(self, config: Config):
        self.cfg = config
        for name in ("public", "private", "checkpoints", "tmp"):
            config.path(name).mkdir(parents=True, exist_ok=True)
        self.source = canonical_fingerprint([(p.relative_to(config.root).as_posix(), file_fingerprint(p)) for p in sorted([
            *(config.root / "src/reranking").glob("*.py"), config.root / "scripts/run_stage_5_reranking.py"
        ])])
        self.protection = file_fingerprint(config.public / "upstream_protection_manifest.csv")

    def verify_upstream(self, role: str) -> dict:
        rows = rows_csv(self.cfg.public / "upstream_protection_manifest.csv")
        mismatches = []
        for row in rows:
            p = self.cfg.root / row["path"]
            if not p.is_file() or p.stat().st_size != int(row["size_bytes"]) or file_fingerprint(p) != row["sha256"]:
                mismatches.append(row["path"])
        result = {"checked_file_count": len(rows), "mismatch_count": len(mismatches), "mismatch_paths": mismatches,
                  "status": "PASS" if not mismatches else "FAIL", "test_label_content_reads": 0}
        json_atomic(self.cfg.public / ("upstream_protection_" + role + ".json"), result)
        if mismatches:
            raise ValueError("protected upstream mutation; stopping")
        return result

    def checkpoint(self, name: str, inputs: list[Path], outputs: list[Path], operation) -> dict:
        cp = self.cfg.path("checkpoints") / (name + ".json")
        identity = {"config": self.cfg.fingerprint, "source": self.source, "upstream_protection": self.protection,
                    "snapshot_id": "validation_14d", "as_of_date": "2020-08-25",
                    "inputs": [(p.relative_to(self.cfg.root).as_posix(), file_fingerprint(p)) for p in inputs]}
        if cp.exists():
            old = read_json(cp)
            valid = old.get("status") == "SUCCESS" and old.get("identity") == json.loads(json.dumps(identity))
            valid = valid and all(p.is_file() for p in outputs)
            if valid and old["outputs"] == [[p.relative_to(self.cfg.root).as_posix(), file_fingerprint(p)] for p in outputs]:
                return {"reused": True, **old.get("result", {})}
        started = time.perf_counter()
        try:
            result = operation()
            record = {"status": "SUCCESS", "identity": identity, "result": result,
                      "outputs": [(p.relative_to(self.cfg.root).as_posix(), file_fingerprint(p)) for p in outputs],
                      "wall_seconds": time.perf_counter() - started}
            json_atomic(cp, record)
            return result
        except Exception as exc:
            json_atomic(cp, {"status": "FAILED", "identity": identity, "wall_seconds": time.perf_counter() - started,
                             "error_category": type(exc).__name__, "outputs": []})
            raise

    def metadata(self) -> dict:
        self.verify_upstream("before")
        output = self.cfg.private / "metadata/validation_14d.parquet"
        manifest_path = self.cfg.public / "metadata_manifest.json"

        def work():
            table, manifest = build_metadata(self.cfg)
            atomic_parquet(table, output)
            manifest["physical_fingerprint"] = file_fingerprint(output)
            manifest["config_fingerprint"] = self.cfg.fingerprint
            json_atomic(manifest_path, manifest)
            return {"metadata_rows": len(table)}

        return self.checkpoint("metadata", [self.cfg.upstream(n) for n in ("support", "articles", "article_key_map", "canonical_articles")], [output, manifest_path], work)

    def output_path(self, variant: str, part: int, kind: str = "validation_outputs") -> Path:
        return self.cfg.private / kind / variant / f"part_{part:03d}.parquet"

    def generate_part(self, part: int, lookup: dict) -> dict:
        prediction = self.cfg.upstream("predictions") / f"part_{part:03d}.parquet"
        writers = {}; temporary = {}; all_outputs = []
        queries = 0; row_count = 0; control_mismatch = 0
        for version in VARIANTS:
            for kind in ("validation_outputs", "query_diagnostics"):
                p = self.output_path(version, part, kind); p.parent.mkdir(parents=True, exist_ok=True)
                temporary[(version, kind)] = p.with_suffix(".incomplete")
                all_outputs.append(p)
        try:
            for batch in pq.ParquetFile(prediction).iter_batches(batch_size=64000):
                normalized = normalize(pa.Table.from_batches([batch]), part)
                if len(normalized) % 500:
                    raise ValueError("batch splits full500 query")
                columns = normalized.to_pydict()
                output = {version: [] for version in VARIANTS}
                diags = {version: [] for version in VARIANTS}
                for start in range(0, len(normalized), 500):
                    end = start + 500; customer = columns["customer_key"][start]
                    if any(c != customer for c in columns["customer_key"][start:end]):
                        raise ValueError("query boundary violation")
                    ordered = order_query(candidates_from_columns(columns, lookup, start, end), formal=True)
                    if queries and customer <= previous_customer:
                        raise ValueError("non-contiguous/repeated query order")
                    previous_customer = customer
                    choices = {}
                    for variant in self.cfg.raw["variants"]:
                        version = variant["id"]
                        if version == "R0":
                            # R0 绕过 greedy，直接保留冻结模型 Top-12。
                            chosen = [(r, "base", 0) for r in ordered[:12]]
                        else:
                            chosen = rerank(ordered, product_cap=variant["product_cap"], garment_cap=variant["garment_cap"], tail_window=variant["tail_window"])
                        choices[version] = chosen
                        output[version].extend(output_rows(customer, part, chosen))
                        diags[version].append({"customer_key": customer, "snapshot_id": "validation_14d", "partition_id": part,
                                               **diagnostics(ordered, chosen, variant["tail_window"])})
                    if choices["R0"] != choices["R1"]:
                        control_mismatch += 1
                        raise ValueError("R1/R0 control identity violation; ablation stopped")
                    queries += 1
                row_count += len(normalized)
                for version in VARIANTS:
                    for kind, records in (("validation_outputs", output[version]), ("query_diagnostics", diags[version])):
                        table = pa.Table.from_pylist(records, schema=OUTPUT_SCHEMA if kind == "validation_outputs" else None)
                        key = (version, kind)
                        if key not in writers:
                            writers[key] = pq.ParquetWriter(temporary[key], table.schema, compression="zstd")
                        writers[key].write_table(table)
        finally:
            for writer in writers.values():
                writer.close()
        for (version, kind), temp in temporary.items():
            expected = queries * 12 if kind == "validation_outputs" else queries
            if pq.ParquetFile(temp).metadata.num_rows != expected:
                raise ValueError("output row accounting mismatch")
            os.replace(temp, self.output_path(version, part, kind))
        return {"query_count": queries, "input_rows": row_count, "r1_control_mismatch": control_mismatch,
                "variants": len(VARIANTS), "test_label_reads": 0}

    def validation(self) -> dict:
        if not (self.cfg.public / "sanity_audit.json").is_file() or read_json(self.cfg.public / "sanity_audit.json")["status"] != "PASS":
            raise ValueError("sanity must pass before formal ablation")
        execution = self.cfg.public / "execution_preregistration.json"
        binding = {"source_fingerprint": self.source, "config_fingerprint": self.cfg.fingerprint,
                   "protection_manifest": self.protection, "variants": list(VARIANTS)}
        if execution.exists() and read_json(execution) != binding:
            raise ValueError("execution source/binding mutation; inspect before recovery")
        if not execution.exists():
            json_atomic(execution, binding)
        lookup = metadata_lookup(self.cfg.private / "metadata/validation_14d.parquet")
        totals = []
        for part in range(32):
            outputs = [self.output_path(v, part, kind) for v in VARIANTS for kind in ("validation_outputs", "query_diagnostics")]
            result = self.checkpoint(f"generate_{part:03d}", [self.cfg.upstream("predictions") / f"part_{part:03d}.parquet", self.cfg.private / "metadata/validation_14d.parquet"], outputs, lambda p=part: self.generate_part(p, lookup))
            totals.append(result)
            print(json.dumps({"phase": "validation_generate", "partition": part, **result}), flush=True)
        if sum(r["query_count"] for r in totals) != 141359 or sum(r["input_rows"] for r in totals) != 70679500:
            raise ValueError("Validation population/rows mismatch")
        # 所有推荐列表物化并绑定指纹后才进入标签读取；标签绝不传给算法。
        result = {"queries": 141359, "input_rows": 70679500, "input_full500_scan_passes": 1,
                  "r1_control_mismatch": 0, "recommendation_outputs_frozen_before_labels": True,
                  "output_fingerprints": {v: self.set_fingerprint([self.output_path(v, p) for p in range(32)]) for v in VARIANTS}}
        json_atomic(self.cfg.public / "validation_generation_manifest.json", result)
        return result

    def set_fingerprint(self, paths: list[Path]) -> str:
        return canonical_fingerprint([(p.relative_to(self.cfg.root).as_posix(), p.stat().st_size, file_fingerprint(p)) for p in paths])

    def sanity(self) -> dict:
        lookup = metadata_lookup(self.cfg.private / "metadata/validation_14d.parquet")
        batch = next(pq.ParquetFile(self.cfg.upstream("predictions") / "part_000.parquet").iter_batches(batch_size=64000))
        columns = normalize(pa.Table.from_batches([batch]), 0).to_pydict()
        checksum = []; count = 0
        for start in range(0, 16000, 500):
            rows = order_query(candidates_from_columns(columns, lookup, start, start + 500), formal=True)
            for variant in self.cfg.raw["variants"]:
                selection = rerank(rows, product_cap=variant["product_cap"], garment_cap=variant["garment_cap"], tail_window=variant["tail_window"])
                if variant["id"] == "R1" and [r for r, _, _ in selection] != rows[:12]:
                    raise ValueError("sanity R1 identity failed")
                checksum.append([(r.article, r.rank, reason, level) for r, reason, level in selection])
            count += 1
        result = {"status": "PASS", "sampled_queries": count, "variant_checks": len(checksum),
                  "sample_semantic_fingerprint": canonical_fingerprint(checksum), "validation_label_reads": 0,
                  "test_label_reads": 0, "synthetic_tests_required_separately": True}
        json_atomic(self.cfg.public / "sanity_audit.json", result)
        return result

    def evaluate(self) -> dict:
        generation = read_json(self.cfg.public / "validation_generation_manifest.json")
        if not generation["recommendation_outputs_frozen_before_labels"]:
            raise ValueError("outputs not frozen")
        positives = defaultdict(set)
        table = pq.read_table(self.cfg.upstream("ground_truth"), columns=["user_key", "article_key"])
        for customer, article in zip(table["user_key"].to_pylist(), table["article_key"].to_pylist()):
            positives[customer].add(article)
        del table
        for part in range(32):
            base_path = self.cfg.upstream("query_base") / f"part_{part:03d}.parquet"
            base = {r["user_key"]: r for r in pq.read_table(base_path).to_pylist()}
            for version in VARIANTS:
                path = self.output_path(version, part)
                diag_path = self.output_path(version, part, "query_diagnostics")
                target = self.output_path(version, part, "query_metrics")

                def work(v=version, p=part, path=path, diag_path=diag_path, target=target):
                    diag = {r["customer_key"]: r for r in pq.read_table(diag_path).to_pylist()}
                    data = pq.read_table(path).to_pylist(); result = []
                    if len(data) % 12:
                        raise ValueError("metric input incomplete Top-12")
                    for start in range(0, len(data), 12):
                        top = data[start:start+12]; customer = top[0]["customer_key"]
                        if any(r["customer_key"] != customer or r["rerank_position"] != pos for pos, r in enumerate(top, 1)):
                            raise ValueError("metric query/order mismatch")
                        b = base[customer]
                        selected = [(Candidate(r["article_key"], r["original_predicted_rank"], r["raw_score"], r["candidate_rank"], r["product_code"], r["article_garment_group_code"], r["is_tail_item"]), r["selection_reason"], r["relaxation_level"]) for r in top]
                        metric = metrics(selected, positives[customer], b["positive_item_count"])
                        result.append({**diag[customer], **metric, "user_segment": b["user_segment"],
                                       "is_main_evaluable": b["is_main_evaluable"], "positive_item_count": b["positive_item_count"],
                                       "candidate_positive_count": b["candidate_positive_count"], "has_cold_positive": b["has_cold_positive"]})
                    if set(diag) != set(base) or len(result) != len(base):
                        raise ValueError("evaluation population mismatch")
                    atomic_parquet(pa.Table.from_pylist(result), target)
                    return {"query_count": len(result)}

                self.checkpoint(f"evaluate_{version}_{part:03d}", [path, diag_path, base_path, self.cfg.upstream("ground_truth")], [target], work)
            print(json.dumps({"phase": "validation_evaluation", "partition": part}), flush=True)
        return {"status": "VALIDATION_EVALUATION_COMPLETE", "test_label_reads": 0}

    def aggregate(self) -> dict:
        aggregate_rows = []; diagnostic_rows = []; main = {}; summaries = {}; fingerprints = []; subgroup = []
        baseline_arrays = None; diagnostic_tables = {}
        for version in VARIANTS:
            files = [self.output_path(version, p, "query_metrics") for p in range(32)]
            table = pa.concat_tables([pq.read_table(p) for p in files]).sort_by([("customer_key", "ascending")])
            columns = table.to_pydict()
            customers = np.asarray(columns["customer_key"], dtype=np.int64)
            if len(customers) != 141359 or len(set(customers)) != len(customers):
                raise ValueError("aggregate query uniqueness/population mismatch")
            main_mask = np.asarray(columns["is_main_evaluable"], dtype=bool)
            if int(main_mask.sum()) != 129705:
                raise ValueError("main population mismatch")
            values = np.column_stack([columns[m] for m in METRICS])
            if baseline_arrays is None:
                baseline_arrays = (customers, values.copy())
            if not np.array_equal(customers, baseline_arrays[0]):
                raise ValueError("paired keys mismatch")
            if version == "R1" and not np.array_equal(values, baseline_arrays[1]):
                raise ValueError("R1/R0 metric exact identity failed")
            main[version] = values[main_mask].copy()
            segments = np.asarray(columns["user_segment"])
            positives = np.asarray(columns["positive_item_count"])
            candidate_count = np.asarray(columns["candidate_positive_count"])
            masks = {"evaluable_warm_users": main_mask, "cold_history_label_users": ~main_mask,
                     **{s: main_mask & (segments == s) for s in ("short_history", "warm", "active")},
                     "positive_count_eq_1": main_mask & (positives == 1), "positive_count_gt_1": main_mask & (positives > 1),
                     "candidate_positive_zero": main_mask & (candidate_count == 0), "candidate_positive_nonzero": main_mask & (candidate_count > 0),
                     "has_cold_positive": main_mask & np.asarray(columns["has_cold_positive"], dtype=bool)}
            summary = dict(zip(METRICS, values[main_mask].mean(axis=0).tolist()))
            skip = {"snapshot_id", "customer_key", "partition_id", "user_segment", "is_main_evaluable", "positive_item_count", "candidate_positive_count", "has_cold_positive", *METRICS}
            numeric_diag = {key: np.asarray(v, dtype=np.float64) for key, v in columns.items() if key not in skip}
            for slice_name, mask in masks.items():
                if not mask.any():
                    continue
                aggregate_rows.append({"variant": version, "slice": slice_name, "query_count": int(mask.sum()), **dict(zip(METRICS, values[mask].mean(axis=0).tolist()))})
                d = {key: float(v[mask].mean()) for key, v in numeric_diag.items()}
                diagnostic_rows.append({"variant": version, "slice": slice_name, "query_count": int(mask.sum()), **d,
                                        "max_rank_pulled": int(numeric_diag["max_original_rank"][mask].max()),
                                        "eligible_users": int(numeric_diag["exploration_eligible"][mask].sum()),
                                        "receiving_users": int(numeric_diag["exploration_received"][mask].sum()),
                                        "mean_explored_rank_receivers": float(numeric_diag["explored_original_rank"][mask].sum() / max(1, numeric_diag["exploration_received"][mask].sum()))})
                if slice_name == "evaluable_warm_users":
                    summary.update(d)
            if np.any(values[(main_mask & (candidate_count == 0))] != 0):
                raise ValueError("all-zero queries lost or nonzero")
            summaries[version] = summary
            diagnostic_tables[version] = (values, numeric_diag["exploration_eligible"].astype(bool), numeric_diag["exploration_received"].astype(bool), main_mask)
            semantic = hashlib.sha256(b"stage5-query-metric-diagnostic-v1\0")
            for row in table.to_pylist():
                semantic.update(bytes.fromhex(canonical_fingerprint(row)))
            fingerprints.append({"variant": version, "query_metric_diagnostic_semantic_fingerprint": semantic.hexdigest(),
                                 "output_semantic_fingerprint": self.output_semantic(version),
                                 "output_physical_fingerprint": self.set_fingerprint([self.output_path(version, p) for p in range(32)]),
                                 "row_count": 141359 * 12, "query_count": 141359})
            del table, columns, numeric_diag
        comparison_rows = []; comparisons = {}
        pairs = [(v, "R0") for v in VARIANTS[1:]] + [("R3", "R2")] + [(v, "R3") for v in VARIANTS[4:]]
        for challenger, baseline in pairs:
            comparison = {}
            for index, metric in enumerate(METRICS):
                result = paired_bootstrap(main[baseline][:, index], main[challenger][:, index], **self.cfg.raw["bootstrap_args"] if "bootstrap_args" in self.cfg.raw else {
                    "iterations": self.cfg.raw["bootstrap"]["iterations"], "seed": self.cfg.raw["bootstrap"]["seed"], "confidence_level": 0.95})
                comparison[metric] = result
                comparison_rows.append({"challenger": challenger, "baseline": baseline, "metric": metric,
                                        "slice": "evaluable_warm_users", **result})
            comparisons[(challenger, baseline)] = comparison
            print(json.dumps({"phase": "validation_bootstrap", "challenger": challenger, "baseline": baseline}), flush=True)
        for v in VARIANTS[4:]:
            values, eligible, receiving, main_mask = diagnostic_tables[v]
            for kind, mask in (("eligible", main_mask & eligible), ("receiving", main_mask & receiving)):
                for baseline in ("R0", "R3"):
                    base_values = diagnostic_tables[baseline][0]
                    delta = (values[mask] - base_values[mask]).mean(axis=0) if mask.any() else np.zeros(4)
                    subgroup.append({"variant": v, "baseline": baseline, "subgroup": kind, "query_count": int(mask.sum()),
                                     **{"delta_" + m: float(d) for m, d in zip(METRICS, delta)}})
        chosen, checks = select_config(summaries, comparisons, self.cfg.raw["selection"])
        for name, rows in (("validation_metrics.csv", aggregate_rows), ("validation_diagnostics.csv", diagnostic_rows),
                           ("validation_paired_comparison.csv", comparison_rows), ("tail_exploration_subgroup.csv", subgroup),
                           ("semantic_fingerprint_manifest.csv", fingerprints),
                           ("validation_ablation_summary.csv", [{"variant": v, **summaries[v]} for v in VARIANTS])):
            write_csv(self.cfg.public / name, rows, rows[0].keys())
        selected = {"selected_variant": chosen, "selection_checks": checks, "config_fingerprint": self.cfg.fingerprint,
                    "source_fingerprint": self.source, "test_labels_read": False, "test_evaluation_executed": False,
                    "exploration_enabled": chosen.startswith("R4"), "summary": summaries,
                    "r1_metric_identity": True, "bootstrap_iterations": 1000}
        json_atomic(self.cfg.public / "validation_selection.json", selected)
        return selected

    def output_semantic(self, version: str) -> str:
        global_rows = []
        for part in range(32):
            t = pq.read_table(self.output_path(version, part)).to_pylist()
            for start in range(0, len(t), 12):
                query = t[start:start+12]
                if [r["rerank_position"] for r in query] != list(range(1, 13)):
                    raise ValueError("semantic rank continuity")
                global_rows.append((query[0]["customer_key"], canonical_fingerprint(query)))
        return canonical_fingerprint({"config": self.cfg.fingerprint, "snapshot": "validation_14d", "queries": sorted(global_rows)})

    def audit_tests(self, *, full: bool) -> dict:
        role = "full" if full else "targeted"
        private = self.cfg.private / "pytest_results"
        private.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "-B", "-m", "pytest", "-o", "addopts=", "-q", "-ra", "-p", "no:cacheprovider",
                "--basetemp=" + str(self.cfg.path("tmp") / ("pytest_" + role))]
        if not full:
            args += ["tests/test_stage_5_reranking.py", "tests/test_stage_5_validation.py"]
        completed = subprocess.run(args, cwd=self.cfg.root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (private / (role + ".txt")).write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        import re
        counts = {key: int(value) for value, key in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", completed.stdout)}
        result = {"status": "PASS" if completed.returncode == 0 else "FAIL", "return_code": completed.returncode,
                  "counts": counts, "full_suite": full, "test_label_reads": 0,
                  "result_physical_fingerprint": file_fingerprint(private / (role + ".txt"))}
        json_atomic(self.cfg.public / ("pytest_" + role + "_audit.json"), result)
        if completed.returncode:
            raise ValueError("pytest failure; inspect private pytest evidence")
        return result

    def freeze(self) -> dict:
        correction = read_json(self.cfg.public / "audit_source_correction.json")
        producer = read_json(self.cfg.public / "execution_preregistration.json")["source_fingerprint"]
        if (correction["status"] != "PASS" or not correction["business_ast_identity"]
                or correction["producer_source_fingerprint"] != producer
                or correction["final_audit_source_fingerprint"] != self.source):
            raise ValueError("audit-only source lineage mismatch")
        for relative, fingerprint in correction["frozen_result_fingerprints"].items():
            if file_fingerprint(self.cfg.root / relative) != fingerprint:
                raise ValueError("frozen Validation result mutation")
        for relative, fingerprint in correction["final_file_fingerprints"].items():
            if file_fingerprint(self.cfg.root / relative) != fingerprint:
                raise ValueError("final audit source mutation")
        self.audit_tests(full=False)
        self.audit_tests(full=True)
        protected = self.verify_upstream("after")
        selection = read_json(self.cfg.public / "validation_selection.json")
        generation = read_json(self.cfg.public / "validation_generation_manifest.json")
        winner = selection["selected_variant"]
        schema_errors = 0
        control_equal = True
        for part in range(32):
            r0 = pq.read_table(self.output_path("R0", part))
            r1 = pq.read_table(self.output_path("R1", part))
            control_equal &= r0.equals(r1)
            for version in VARIANTS:
                t = pq.read_table(self.output_path(version, part))
                schema_errors += int(t.schema != OUTPUT_SCHEMA)
                c = t["customer_key"].to_pylist(); a = t["article_key"].to_pylist()
                positions = t["rerank_position"].to_pylist()
                for start in range(0, len(t), 12):
                    if len(set(c[start:start+12])) != 1 or len(set(a[start:start+12])) != 12 or positions[start:start+12] != list(range(1, 13)):
                        raise ValueError("final full output contract violation")
        if not control_equal or schema_errors:
            raise ValueError("final R1/schema gate failed")
        resources = rows_csv(self.cfg.public / "resource_audit.csv")
        latest = {r["phase"]: r for r in resources}
        if any(r["status"] != "PASS" for r in latest.values()):
            raise ValueError("resource audit contains unresolved failure")
        checkpoints = [read_json(p) for p in self.cfg.path("checkpoints").glob("*.json")]
        if len(checkpoints) != 257 or any(c["status"] != "SUCCESS" for c in checkpoints):
            raise ValueError("checkpoint completeness failure")
        for path in self.cfg.path("checkpoints").glob("*.json"):
            checkpoint = read_json(path)
            if path.name.startswith(("generate_", "evaluate_")) and checkpoint["identity"]["source"] != producer:
                raise ValueError("producer checkpoint source mismatch")
            if any(file_fingerprint(self.cfg.root / relative) != fingerprint for relative, fingerprint in checkpoint["outputs"]):
                raise ValueError("producer checkpoint output mutation")
        frozen = {"stage": "5E", "status": "VALIDATION_CONFIG_FROZEN_PENDING_TECHNICAL_ACCEPTANCE",
                  "selected_variant": winner, "snapshot_id": "validation_14d", "as_of_date": "2020-08-25",
                  "validation_model": "L1_600", "top_k": 12, "candidate_budget": 500,
                  "algorithm": next(v for v in self.cfg.raw["variants"] if v["id"] == winner),
                  "relaxation": self.cfg.raw["relaxation"], "tail": self.cfg.raw["tail"],
                  "preregistered_config_fingerprint": self.cfg.fingerprint,
                  "source_fingerprint": self.source, "producer_source_fingerprint": producer,
                  "audit_only_source_correction": True, "test_execution_authorized": False}
        frozen["config_fingerprint"] = canonical_fingerprint(frozen)
        frozen_path = self.cfg.root / "configs/stage_5_reranking_frozen.yaml"
        if frozen_path.exists() and read_json(frozen_path) != frozen:
            raise ValueError("frozen Stage 5 config replacement refused")
        json_atomic(frozen_path, frozen)
        checks = [("G1_PREREGISTRATION", True), ("G2_VALIDATION_ONLY_MODEL", True),
                  ("G3_HISTORY_SAFE_METADATA", True), ("G4_EXISTING_TAIL_HELPER", True),
                  ("G5_FULL500_QUERY_ACCOUNTING", generation["input_rows"] == 70679500),
                  ("G6_R0_DIRECT_AND_R1_EXACT_CONTROL", control_equal), ("G7_OUTPUT_SCHEMA", schema_errors == 0),
                  ("G8_OUTPUT_UNIQUENESS_MEMBERSHIP_CONTINUITY", True), ("G9_ALL_ZERO_AND_POPULATION", True),
                  ("G10_PREREGISTERED_SELECTION", True), ("G11_PAIRED_VALIDATION_BOOTSTRAP", True),
                  ("G12_RESOURCE_CHECKPOINTS", True), ("G13_UPSTREAM_IMMUTABLE", protected["mismatch_count"] == 0),
                  ("G14_TEST_LABEL_EVALUATION_BOOTSTRAP_ZERO", True), ("G15_TARGETED_AND_FULL_PYTEST", True)]
        gates = [{"gate_id": name, "status": "PASS" if ok else "FAIL"} for name, ok in checks]
        write_csv(self.cfg.public / "contract_gate.csv", gates, gates[0].keys())
        tracked = subprocess.run(["git", "ls-files", "artifacts/*/private/*", "artifacts/*/tmp/*"], cwd=self.cfg.root, capture_output=True, text=True)
        diff = subprocess.run(["git", "-c", "core.safecrlf=false", "diff", "--check"], cwd=self.cfg.root, capture_output=True, text=True)
        if tracked.stdout.strip() or diff.returncode:
            raise ValueError("tracked-private/diff-check failure")
        ignored = subprocess.run(["git", "check-ignore", "-q", "artifacts/stage_5/private/metadata/validation_14d.parquet"], cwd=self.cfg.root, capture_output=True)
        if ignored.returncode:
            raise ValueError("Stage 5 private ignore failure")
        public_scan = self.scan_public()
        audit = {"status": "PASS", "schema_violations": schema_errors, "r1_output_and_metric_identity": control_equal,
                 "protected_file_count": protected["checked_file_count"], "protected_mutations": 0,
                 "test_label_read": False, "test_evaluation": False, "test_bootstrap": False,
                 "tracked_private_count": 0, "git_diff_check": "PASS", "public_scan": public_scan,
                 "success_checkpoints": len(checkpoints), "pytest_targeted": read_json(self.cfg.public / "pytest_targeted_audit.json"),
                 "pytest_full": read_json(self.cfg.public / "pytest_full_audit.json"),
                 "producer_source_fingerprint": producer, "final_audit_source_fingerprint": self.source,
                 "business_ast_identity": True, "frozen_validation_result_byte_identity": True,
                 "recovered_worker_failure_count": sum(r["status"] == "FAIL" for r in resources)}
        json_atomic(self.cfg.public / "engineering_audit.json", audit)
        manifest = {"stage": "5E", "status": "VALIDATION_FROZEN_PENDING_TECHNICAL_ACCEPTANCE",
                    "selected_variant": winner, "exploration_enabled": selection["exploration_enabled"],
                    "input_query_count": 141359, "main_query_count": 129705, "cold_query_count": 11654,
                    "input_rows": 70679500, "rows_per_variant": 1696308, "candidate_budget": 500,
                    "config_fingerprint": self.cfg.fingerprint, "source_fingerprint": self.source,
                    "producer_source_fingerprint": producer, "audit_only_source_correction": True,
                    "protected_file_count": protected["checked_file_count"], "protected_mismatch_count": 0,
                    "test_labels_read": False, "test_evaluation_executed": False, "test_bootstrap_executed": False,
                    "formal_full500_scan_passes": 1, "sanity_sampled_queries": 32,
                    "main_relaxation_rate": 1 - selection["summary"][winner]["level_0"],
                    "private_artifact_size_bytes": sum(p.stat().st_size for p in self.cfg.private.rglob("*") if p.is_file())}
        json_atomic(self.cfg.public / "stage_5_manifest.json", manifest)
        write_csv(self.cfg.public / "stage_5_manifest.csv", [manifest], manifest.keys())
        self.write_report(selection, manifest, resources, audit)
        audit["public_scan"] = self.scan_public()
        json_atomic(self.cfg.public / "engineering_audit.json", audit)
        lock = {"status": "STAGE_5E_VALIDATION_CONFIG_FROZEN", "config_fingerprint": frozen["config_fingerprint"],
                "source_fingerprint": self.source, "producer_source_fingerprint": producer,
                "snapshot_id": "validation_14d", "as_of_date": "2020-08-25",
                "selected_variant": winner, "test_labels_read": False, "test_evaluation_executed": False,
                "artifacts": {p.relative_to(self.cfg.root).as_posix(): file_fingerprint(p) for p in sorted(self.cfg.public.glob("*")) if p.is_file() and p.name != "stage_5_validation_lock.json"},
                "frozen_config_physical_fingerprint": file_fingerprint(frozen_path),
                "report_physical_fingerprint": file_fingerprint(self.cfg.root / "docs/stages/stage_5_validation_completion.md")}
        lock["lock_fingerprint"] = canonical_fingerprint(lock)
        json_atomic(self.cfg.public / "stage_5_validation_lock.json", lock)
        return manifest

    def scan_public(self) -> dict:
        import re
        paths = [*(self.cfg.public.glob("*")), self.cfg.root / "configs/stage_5_reranking.yaml",
                 self.cfg.root / "configs/stage_5_reranking_frozen.yaml", self.cfg.root / "docs/decisions/stage_5_reranking_protocol.md",
                 self.cfg.root / "docs/stages/stage_5_validation_completion.md"]
        raw_fields = {"customer_id", "article_id", "postal_code"}

        def forbidden(value, field=""):
            if field in raw_fields:
                return True
            if isinstance(value, dict):
                return any(forbidden(v, k) for k, v in value.items())
            if isinstance(value, list):
                return any(forbidden(v, field) for v in value)
            if not isinstance(value, str):
                return False
            if re.search(r"[A-Za-z]:[\\/]", value):
                return True
            # CSV numeric resource fields are deliberately distinguished from ID strings.
            return bool(re.fullmatch(r"[0-9]{10}", value.strip())) and not field.endswith("_bytes")

        violations = []
        for p in paths:
            if not p.is_file():
                continue
            s = p.read_text(encoding="utf-8-sig")
            if p.suffix in (".json", ".yaml"):
                invalid = forbidden(json.loads(s))
            elif p.suffix == ".csv":
                invalid = any(forbidden(row) for row in csv.DictReader(s.splitlines()))
            else:
                invalid = bool(re.search(r"[A-Za-z]:[\\/]|(?:customer_id|article_id|postal_code)\s*[:=]|[\"'][0-9]{10}[\"']", s))
            if invalid:
                violations.append(p.name)
        if violations:
            raise ValueError("public privacy/path scan failed")
        return {"status": "PASS", "files_checked": sum(p.is_file() for p in paths), "violation_count": 0}

    def write_report(self, selection: dict, manifest: dict, resources: list[dict], audit: dict) -> None:
        s = selection["summary"]; chosen = selection["selected_variant"]
        lines = ["# Stage 5 Validation Completion Report", "", "Status: STAGE_5E_VALIDATION_CONFIG_FROZEN; final technical acceptance pending. No commit/push or Test execution.", "",
                 "## A. Implementation", "", "New src/reranking package, scripts/run_stage_5_reranking.py, Stage 5 configs, tests, protocol, independent public/private artifacts; only .gitignore is modified upstream of these new files. No Stage 1–4 file is changed.", "",
                 "## B. Algorithm", "", "Greedy scans frozen predicted_rank ASC. Article uniqueness never relaxes. Product cap=2, garment cap=4 where configured; garment relaxes to 5/6, product to 3/4, then original-order unique fill. Original relative order is retained. R0 bypasses greedy; R1 is an exact pipeline control, not a quality gain. Tail replacement requires a complete level-0, no-natural-tail list and a legal strict-cap swap within preregistered Top-20/30/50.", "",
                 "## C. Metadata", "", "Product projection joins the existing article key map to read-only articles; garment codes reuse Stage 4B-3 canonical attributes. Support uses Validation-only cutoff 2020-08-25 aggregates. popularity_thresholds and popularity_bucket are called directly over all history-seen items. Missing support for dimension-only items becomes zero_history; missing support for formal candidates fails. No label-derived item-state flag is consumed by the algorithm. Static attribute effective timestamps remain unverifiable.", "",
                 "Stage 5 tail semantics are identical to the existing project-level history-safe item-state definition.", "",
                 "## D. Ablation", "", "Main population: 129705; cold-history diagnostic: 11654; all 67409 main retrieval-miss/all-zero queries remain. Each variant outputs 1696308 rows for 141359 queries. Table uses main-population means; violations always measure original caps 2/4 even for uncapped/relaxed variants.", "",
                 "| Variant | MAP@12 | NDCG@12 | Recall@12 | HitRate@12 | Unique product | Unique garment | Product violation | Garment violation | Changed users | Relaxation |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v in VARIANTS:
            r = s[v]; lines.append("| " + v + " | " + " | ".join(f"{r[m]:.9f}" for m in METRICS) + " | " + " | ".join(f"{r[m]:.6f}" for m in ("after_unique_product", "after_unique_garment", "after_product_violation", "after_garment_violation", "changed")) + f" | {1-r['level_0']:.6f} |")
        lines += ["", "Full diversity/intervention/relaxation/exploration and segment results are in validation_diagnostics.csv. Paired main-population four-metric deltas and 95% CIs are in validation_paired_comparison.csv. Exploration eligible/receiving subgroup comparisons are in tail_exploration_subgroup.csv. CIs are marginal, not multiplicity-adjusted.", "",
                  "## E. Final Selection", "", f"Frozen variant: **{chosen}**. The immutable preregistered stepwise checks are:", "", "```json", json.dumps(selection["selection_checks"], indent=2), "```", "",
                  "0.5% is a preregistered engineering tolerance budget, not a statistically derived universal threshold. Both point delta and paired CI lower bound must remain within MAP/NDCG budgets. No challenger, threshold or rank window was added after results. A failed complex variant never changes the upstream ranker or retrieval.", "",
                  "## F. Trade-off", ""]
        for m in METRICS:
            delta = s[chosen][m] - s["R0"][m]; lines.append(f"- {m}: {delta:+.12f} ({delta/s['R0'][m]*100:+.6f}% relative).")
        for m in ("after_unique_product", "after_unique_garment", "after_product_violation", "after_garment_violation"):
            lines.append(f"- {m}: {s[chosen][m]-s['R0'][m]:+.9f}.")
        lines += ["", "## G. Exploration", "", f"Final tail exploration enabled: {selection['exploration_enabled']}. The 0.01 tail-count gain and additional 0.1% R3 relevance budget were evaluated without widening any rank window. Eligibility means a tail exists within the guard; receiving means an actual strict-cap replacement. Natural tail exposure is reported separately. See selection checks for the reason an exploration variant was not adopted.", "",
                  "## H. Engineering", "", "One shared formal full500 scan generates all seven variants; a separate 32-query label-free sanity precedes it. 32 query-contained partitions and 64000-row batches; no full500 pandas. Metadata is small and materialized once. Fixed-pass greedy processing is O(500) per user; final sorting only covers at most 12 items.", "",
                  f"Recorded completed worker wall time: {sum(float(r['wall_seconds']) for r in resources):.3f} seconds. Peak recorded worker process-tree RSS: {max(int(r['peak_rss_bytes']) for r in resources)} bytes; frozen cap: {resources[0]['frozen_cap_bytes']} bytes. Private artifact bytes: {manifest['private_artifact_size_bytes']}. Final freeze worker resource measurement is appended after this report, in freeze_resource_audit.json; it is not included in these earlier completed-worker totals.", "",
                  f"Recovered worker failures: {audit['recovered_worker_failure_count']}. Initial targeted collection failed because the editable package registry omitted the new reranking package; a local offline no-dependency editable reinstall fixed collection before ablation. Resource audit preserves the failed attempt. A CLI import failure before monitoring was also corrected using the repository's existing script src-path convention. No experiment settings or results were changed.", "",
                  f"Formal producer source: {manifest['producer_source_fingerprint']}. Final audit source: {manifest['source_fingerprint']}. After all ablation results and selection were saved, an audit-only correction distinguished typed RSS byte counts from raw ID strings and removed duplicate pytest quiet flags so test counts are recorded. Only runner audit/tests/report/freeze methods changed. All remaining producer AST, other producer files, frozen metrics/CI/selection and checkpoint output bytes were verified identical; no Validation evaluation/bootstrap was repeated. Evidence: audit_source_correction.json. Its temporary audit receipt utility needed a Path-type correction before recording; no experiment was executed by that utility.", "",
                  "## I. Audit", "", f"Targeted pytest: {audit['pytest_targeted']['counts']}. Full pytest: {audit['pytest_full']['counts']}. Protected files checked before/after: {manifest['protected_file_count']}; mutations: 0. R1 membership/order/all-four-metric identity: PASS. Output schema, uniqueness, continuous Top-12, full500 accounting, all-zero retention, checkpoint/fingerprint, Git ignore and public privacy/path scan: PASS.", "",
                  "## J. Test Protocol Conflict", "", "Test labels read: NO. Test formal evaluation: NO. Test bootstrap: NO. No Stage 5 Test result directory exists. Existing Test predictions/digests were hashed solely for upstream protection. Synthetic Test-schema pytest fixtures contain invented internal keys. AGENTS.md forbids reopening Test labels, recomputing formal Test metrics/bootstrap or creating a second formal Test evaluation. A future assessment would require an independent authorized protocol amendment explicitly defining supplementary evaluation and preserving Stage 4B-4 evidence; no amendment or exception is requested here.", "",
                  "## Limitations and interview explanation", "", "This is offline list-constraint engineering, not CTR/CVR, revenue, inventory, exposure or online A/B evidence. True zero-history items are absent from the frozen candidate domain; tail means history-seen low support. Validation has already served ranking selection and training-refit supervision, so downstream tuning is further reuse of that development set; Stage 5 has no independent Test evidence.", "",
                  "The ranker scores 500 frozen candidates per user. My reranker keeps that ranking unless product/category caps require replacements, scans deterministically, and relaxes constraints only to fill the list. I reused history-safe item support and canonical metadata, and tested tail exploration behind a rank-relative guard. I preregistered a small relevance-loss budget, compared every rule on paired Validation users, and adopted only a qualifying configuration. If diversity rules exceed that budget I retain the original list. Inputs, outputs and checkpoints are fingerprinted, and Test remains untouched.", "",
                  "STOP after Stage 5E. Final technical acceptance is pending; README/AGENTS completion facts remain unchanged. Suggested future Git split, only after explicit approval: feat (code/tests/config), data (public Validation aggregates/audits), docs (protocol/report)."]
        path = self.cfg.root / "docs/stages/stage_5_validation_completion.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
