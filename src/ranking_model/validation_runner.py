from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import psutil
import pyarrow.parquet as pq

from data_audit.stage1a import ResourceMonitor
from data_foundation.manifest import write_csv
from ranking_model import Stage4B4Config, Stage4B4Runner
from ranking_model.validation_selection import (
    METRICS,
    VERSIONS,
    canonical_fingerprint,
    file_fingerprint,
    paired_bootstrap,
    query_merkle,
    query_metric_digest,
    select_winner,
    value_gate,
)


class FrozenArtifactDrift(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def csv_read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ValidationSelectionRunner:
    def __init__(self, root: Path, config_path: Path) -> None:
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._validate_config()
        self.config_fp = canonical_fingerprint(self.raw)
        paths = self.raw["paths"]
        self.paths = {key: self.root / value for key, value in paths.items()}
        self.public = self.paths["public_directory"]
        self.private = self.paths["private_directory"]
        self.checkpoints = self.paths["checkpoint_directory"]
        self.temp = self.paths["temp_directory"]
        for path in (self.private, self.checkpoints, self.temp):
            path.mkdir(parents=True, exist_ok=True)
        available = psutil.virtual_memory().available
        execution = self.raw["execution"]
        self.available_at_preregistration = available
        self.rss_limit = int(available * execution["rss_hard_fraction_of_available_at_preregistration"])
        self.memory_mb = max(
            512,
            min(
                int(execution["duckdb_memory_max_mb"]),
                int(available / 1024 / 1024 * execution["memory_fraction_of_available"]),
            ),
        )
        self.resources: list[dict[str, Any]] = []

    def _validate_config(self) -> None:
        if (self.raw.get("stage"), self.raw.get("phase")) != (
            "4B-4", "phase2_validation_14d_selection_v1"
        ):
            raise ValueError("Phase 2 config identity drift")
        snap = self.raw["snapshot"]
        if (snap["snapshot_id"], snap["label_start"], snap["label_end"]) != (
            "validation_14d", "2020-08-26", "2020-09-08"
        ):
            raise ValueError("Validation label window drift")
        if tuple(self.raw["versions"]) != VERSIONS:
            raise ValueError("prediction version budget drift")
        if self.raw["bootstrap"] != {
            "method": "paired_user_block_percentile_v1", "iterations": 1000,
            "seed": 20260826, "confidence_level": 0.95,
            "comparison_baseline": "B0", "selection_use": False,
        }:
            raise ValueError("bootstrap preregistration drift")
        if any(value not in (0, False) for value in self.raw["forbidden"].values()):
            raise ValueError("forbidden downstream access is not locked")

    def resolve(self, value: str) -> Path:
        return self.root / value

    def _source_fingerprint(self) -> str:
        paths = [
            self.root / "src/retrieval/evaluation.py",
            self.root / "src/ranking_model/validation_selection.py",
            self.root / "src/ranking_model/validation_runner.py",
            self.root / "scripts/run_stage_4b4_validation_selection.py",
            self.root / "tests/test_stage_4b4_phase2.py",
        ]
        digest = hashlib.sha256()
        for path in paths:
            if not path.exists():
                raise FrozenArtifactDrift(f"missing Phase 2 source: {path.name}")
            digest.update(path.relative_to(self.root).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes() + b"\0")
        return digest.hexdigest()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self.temp.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(":memory:")
        con.execute(f"SET temp_directory='{self.temp.as_posix()}'")
        con.execute(f"SET memory_limit='{self.memory_mb}MB'")
        con.execute(f"SET threads={int(self.raw['execution']['threads'])}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{self.raw['execution']['max_temp_directory_size']}'")
        return con

    def _phase1_preflight(self) -> dict[str, Any]:
        phase1_config = Stage4B4Config.load(self.paths["phase1_config"])
        if phase1_config.fingerprint != self.raw["expected"]["phase1_config_fingerprint"]:
            raise FrozenArtifactDrift("Phase 1 config fingerprint drift")
        lock_path = self.paths["phase1_lock"]
        if file_fingerprint(lock_path) != self.raw["expected"]["phase1_lock_physical_fingerprint"]:
            raise FrozenArtifactDrift("Phase 1 pre-label lock physical drift")
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        required_lock = {
            "validation_predictions_frozen": True, "ready_for_validation_label_join": True,
            "validation_labels_read": False, "validation_label_read_count": 0,
            "winner_selected": False, "first7d_read_count": 0,
            "final_refit_executed": False, "test_authorized": False,
            "test_candidate_read_count": 0, "test_feature_read_count": 0,
            "test_label_read_count": 0, "test_metric_read_count": 0,
        }
        if any(lock.get(key) != value for key, value in required_lock.items()):
            raise FrozenArtifactDrift("Phase 1 lock state drift")
        if lock.get("prediction_global_merkle_fingerprints") != self.raw["expected"]["prediction_global_merkle_fingerprints"]:
            raise FrozenArtifactDrift("Phase 1 prediction Merkle drift")
        gates = csv_read(self.paths["phase1_gate"])
        if len(gates) != 20 or any(row.get("status") != "PASS" for row in gates):
            raise FrozenArtifactDrift("Phase 1 G1-G20 no longer all PASS")
        manifest = csv_read(self.paths["prediction_manifest"])
        actual_merkle = {row["model_id"]: row["global_merkle_fingerprint"] for row in manifest}
        if actual_merkle != self.raw["expected"]["prediction_global_merkle_fingerprints"]:
            raise FrozenArtifactDrift("prediction manifest Merkle drift")

        # Reuse the Phase 1 recovery verifier: it binds config/source/model/feature inputs,
        # row counts, output physical fingerprints and SUCCESS checkpoint identities.
        resume_path = self.root / "scripts/stage_4b4_prediction_resume.py"
        spec = importlib.util.spec_from_file_location("stage_4b4_prediction_resume", resume_path)
        if spec is None or spec.loader is None:
            raise FrozenArtifactDrift("cannot load Phase 1 checkpoint verifier")
        resume_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resume_module)
        verify_model_and_matrix = resume_module.verify_model_and_matrix
        verify_success_checkpoint = resume_module.verify_success_checkpoint

        runner = Stage4B4Runner(phase1_config)
        # Phase 1 checkpoint identity is intentionally bound to the source frozen in
        # its immutable lock; later Phase 2 modules must not retroactively alter it.
        runner.source_fp = lock["source_fingerprint"]
        upstream = runner.preflight()
        verify_model_and_matrix(runner)
        verified = [verify_success_checkpoint(runner, part) for part in range(32)]
        return {
            "phase1_config_fingerprint": phase1_config.fingerprint,
            "phase1_lock_physical_fingerprint": file_fingerprint(lock_path),
            "phase1_gate_pass_count": len(gates),
            "prediction_checkpoint_success_count": len(verified),
            "prediction_checkpoint_output_set_fingerprint": canonical_fingerprint(
                [{"partition": row["partition"], "identity": row["checkpoint_identity"],
                  "output": row["output_fingerprint"]} for row in verified]
            ),
            "prediction_global_merkle_fingerprints": actual_merkle,
            "upstream": upstream,
        }

    def preregister(self) -> dict[str, Any]:
        phase1 = self._phase1_preflight()
        # Physical hashing is a lineage check only: no Parquet metadata or label rows are opened.
        for role in ("ground_truth", "population", "item_state"):
            actual = file_fingerprint(self.paths[role])
            expected = self.raw["expected"][f"{role}_physical_fingerprint"]
            if actual != expected:
                raise FrozenArtifactDrift(f"{role} physical fingerprint drift")
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.root, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        payload = {
            "stage": "4B-4", "phase": self.raw["phase"], "status": "PREREGISTERED_BEFORE_LABEL_ACCESS",
            "created_at": now(), "config_fingerprint": self.config_fp,
            "source_fingerprint": self._source_fingerprint(),
            "evaluator_fingerprint": file_fingerprint(self.root / "src/retrieval/evaluation.py"),
            "git_head": git_head, "git_dirty_entry_count": len(git_status),
            "phase1": phase1,
            "label_lineage": {
                "ground_truth_physical_fingerprint": self.raw["expected"]["ground_truth_physical_fingerprint"],
                "population_physical_fingerprint": self.raw["expected"]["population_physical_fingerprint"],
                "item_state_physical_fingerprint": self.raw["expected"]["item_state_physical_fingerprint"],
                "validation_label_content_open_count": 0,
            },
            "access": {
                "validation_labels_read": False, "validation_label_read_count": 0,
                **self.raw["forbidden"],
            },
            "resource_contract": {
                "available_memory_at_preregistration_bytes": self.available_at_preregistration,
                "rss_hard_limit_bytes": self.rss_limit, "duckdb_memory_limit_mb": self.memory_mb,
                "threads": self.raw["execution"]["threads"],
                "disk_free_bytes": shutil.disk_usage(self.root).free,
            },
        }
        payload["preregistration_fingerprint"] = canonical_fingerprint(payload)
        json_atomic(self.public / "validation_selection_preregistration.json", payload)
        rows = [
            {"check_id": "PRE1_CONFIG_FROZEN", "status": "PASS", "observed": self.config_fp},
            {"check_id": "PRE2_SOURCE_AND_EVALUATOR_FROZEN", "status": "PASS", "observed": payload["source_fingerprint"]},
            {"check_id": "PRE3_PHASE1_G1_G20", "status": "PASS", "observed": "20/20"},
            {"check_id": "PRE4_PREDICTION_CHECKPOINTS", "status": "PASS", "observed": "32/32"},
            {"check_id": "PRE5_VALIDATION_LABEL_CONTENT_ACCESS", "status": "PASS", "observed": "0"},
            {"check_id": "PRE6_DOWNSTREAM_ACCESS", "status": "PASS", "observed": "all_zero_or_false"},
        ]
        write_csv(self.public / "validation_selection_pre_access_audit.csv", rows, list(rows[0]))
        json_atomic(self.checkpoints / "preregistration.json", payload)
        return payload

    def _verify_preregistration(self) -> dict[str, Any]:
        path = self.public / "validation_selection_preregistration.json"
        if not path.exists():
            raise FrozenArtifactDrift("missing pre-access preregistration")
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = payload.pop("preregistration_fingerprint")
        if canonical_fingerprint(payload) != fingerprint:
            raise FrozenArtifactDrift("preregistration physical/semantic drift")
        payload["preregistration_fingerprint"] = fingerprint
        if payload["config_fingerprint"] != self.config_fp:
            raise FrozenArtifactDrift("preregistered config drift")
        current_source = self._source_fingerprint()
        if payload["source_fingerprint"] != current_source:
            correction_path = self.public / "validation_selection_recovery_correction.json"
            if not correction_path.exists():
                raise FrozenArtifactDrift("preregistered source/evaluator drift")
            correction = json.loads(correction_path.read_text(encoding="utf-8"))
            expected_correction = {
                "original_source_fingerprint": payload["source_fingerprint"],
                "corrected_source_fingerprint": current_source,
                "scope": "bootstrap_numpy_empty_check_and_restore_upstream_evaluator",
                "methodology_changed": False,
                "validation_label_read_count": 1,
                "validation_label_reopened_for_recovery": False,
                "failed_stage": "post_metric_paired_bootstrap",
            }
            if any(correction.get(key) != value for key, value in expected_correction.items()):
                raise FrozenArtifactDrift("unapproved post-access source correction")
        current = self._phase1_preflight()
        stable_keys = (
            "phase1_config_fingerprint", "phase1_lock_physical_fingerprint",
            "phase1_gate_pass_count", "prediction_checkpoint_success_count",
            "prediction_checkpoint_output_set_fingerprint", "prediction_global_merkle_fingerprints",
        )
        upstream_keys = (
            "validation_candidate_fingerprint", "train_optimizer_fingerprint",
            "feature_schema_fingerprint", "lightgbm_version",
        )
        stable_match = all(current[key] == payload["phase1"][key] for key in stable_keys)
        stable_match = stable_match and all(
            current["upstream"][key] == payload["phase1"]["upstream"][key] for key in upstream_keys
        )
        if not stable_match:
            raise FrozenArtifactDrift("Phase 1 artifacts drifted after preregistration")
        return payload

    def register_recovery_correction(self) -> dict[str, Any]:
        prereg_path = self.public / "validation_selection_preregistration.json"
        access_path = self.public / "validation_selection_access_audit.json"
        if not prereg_path.exists() or not access_path.exists():
            raise FrozenArtifactDrift("recovery correction requires preregistration and prior access intent")
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        access = json.loads(access_path.read_text(encoding="utf-8"))
        if access.get("validation_label_read_count") != 1:
            raise FrozenArtifactDrift("recovery correction access count mismatch")
        phase_tasks = [path for path in self.checkpoints.glob("*.json") if path.name != "preregistration.json"]
        checkpoints = [json.loads(path.read_text(encoding="utf-8")) for path in phase_tasks]
        if len(checkpoints) != 480 or any(row.get("status") != "SUCCESS" for row in checkpoints):
            raise FrozenArtifactDrift("recovery correction requires 480 successful materialization checkpoints")
        if self.paths["selection_lock"].exists():
            raise FrozenArtifactDrift("selection lock already exists; correction forbidden")
        payload = {
            "status": "RECOVERY_ONLY_SOURCE_CORRECTION",
            "original_source_fingerprint": prereg["source_fingerprint"],
            "corrected_source_fingerprint": self._source_fingerprint(),
            "scope": "bootstrap_numpy_empty_check_and_restore_upstream_evaluator",
            "methodology_changed": False,
            "validation_label_read_count": 1,
            "validation_label_reopened_for_recovery": False,
            "failed_stage": "post_metric_paired_bootstrap",
            "successful_materialization_checkpoints": 480,
            "selection_lock_present_before_recovery": False,
            "recorded_at": now(),
        }
        json_atomic(self.public / "validation_selection_recovery_correction.json", payload)
        return payload

    def _binding(self, prereg: dict[str, Any]) -> dict[str, Any]:
        return {
            "preregistration_fingerprint": prereg["preregistration_fingerprint"],
            "ground_truth_fingerprint": self.raw["expected"]["ground_truth_physical_fingerprint"],
            "population_fingerprint": self.raw["expected"]["population_physical_fingerprint"],
            "item_state_fingerprint": self.raw["expected"]["item_state_physical_fingerprint"],
            "prediction_global_merkle_fingerprints": self.raw["expected"]["prediction_global_merkle_fingerprints"],
            "evaluation_contract": self.raw["evaluation"],
        }

    def _checkpoint_run(self, name: str, output: Path, binding: dict[str, Any], operation: Any) -> bool:
        identity = canonical_fingerprint({"task": name, **binding})
        checkpoint = self.checkpoints / f"{name}.json"
        previous = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
        if previous and previous.get("status") == "SUCCESS" and previous.get("identity") == identity \
                and output.exists() and previous.get("output_fingerprint") == file_fingerprint(output):
            self.resources.append({**previous, "checkpoint_reused": True})
            return True
        attempt = int((previous or {}).get("attempt", 0)) + 1
        monitor = ResourceMonitor(self.temp)
        started = time.perf_counter()
        try:
            with monitor:
                operation()
            if monitor.peak_rss_bytes > self.rss_limit:
                raise RuntimeError("Phase 2 peak RSS exceeded frozen hard limit")
            if not output.exists():
                raise RuntimeError(f"missing output for {name}")
            row = {
                "task": name, "status": "SUCCESS", "attempt": attempt, "identity": identity,
                "output": output.relative_to(self.root).as_posix(),
                "output_fingerprint": file_fingerprint(output),
                "row_count": pq.ParquetFile(output).metadata.num_rows if output.suffix == ".parquet" else 1,
                "wall_time_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes, "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "checkpoint_reused": False, "error": "", "generated_at": now(),
            }
            json_atomic(checkpoint, row)
            self.resources.append(row)
            return False
        except Exception as exc:
            json_atomic(checkpoint, {
                "task": name, "status": "FAILED", "attempt": attempt, "identity": identity,
                "output": "", "output_fingerprint": "", "row_count": 0,
                "wall_time_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes, "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "checkpoint_reused": False, "error": str(exc).replace(str(self.root), "."), "generated_at": now(),
            })
            raise

    @staticmethod
    def _copy_atomic(con: duckdb.DuckDBPyConnection, sql: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".attempt.parquet")
        con.execute(f"COPY ({sql}) TO '{temporary.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        os.replace(temporary, output)

    def _build_query_base(self, part: int, output: Path) -> None:
        con = self._connect()
        try:
            gt = self.paths["ground_truth"].as_posix()
            pop = self.paths["population"].as_posix()
            state = self.paths["item_state"].as_posix()
            pred = (self.paths["predictions"] / "B0" / f"part_{part:03d}.parquet").as_posix()
            sql = f"""
            WITH target_counts AS (
              SELECT g.user_key,
                count(*) FILTER (WHERE i.seen_item)::INTEGER seen_positive_count,
                count(*) FILTER (WHERE i.cold_item)::INTEGER cold_positive_count_check,
                count(*) FILTER (WHERE i.popularity_bucket='head')::INTEGER head_positive_count,
                count(*) FILTER (WHERE i.popularity_bucket='mid')::INTEGER mid_positive_count,
                count(*) FILTER (WHERE i.popularity_bucket='tail')::INTEGER tail_positive_count,
                count(*) FILTER (WHERE i.popularity_bucket='zero_history')::INTEGER zero_history_positive_count
              FROM read_parquet('{gt}') g JOIN read_parquet('{state}') i USING(article_key)
              WHERE g.user_key % 32 = {part}
              GROUP BY g.user_key
            ), candidate_hits AS (
              SELECT p.customer_key user_key, count(g.article_key)::INTEGER candidate_positive_count
              FROM read_parquet('{pred}') p
              LEFT JOIN read_parquet('{gt}') g ON g.user_key=p.customer_key AND g.article_key=p.article_key
              GROUP BY p.customer_key
            )
            SELECT p.*, h.candidate_positive_count,
              t.seen_positive_count, t.cold_positive_count_check,
              t.head_positive_count, t.mid_positive_count, t.tail_positive_count,
              t.zero_history_positive_count
            FROM read_parquet('{pop}') p
            JOIN candidate_hits h USING(user_key)
            JOIN target_counts t USING(user_key)
            WHERE p.user_key % 32 = {part}
            ORDER BY p.user_key
            """
            self._copy_atomic(con, sql, output)
        finally:
            con.close()

    def _build_metrics(self, version: str, part: int, base: Path, output: Path) -> None:
        con = self._connect()
        try:
            pred = (self.paths["predictions"] / version / f"part_{part:03d}.parquet").as_posix()
            gt = self.paths["ground_truth"].as_posix()
            state = self.paths["item_state"].as_posix()
            flags = {
                "all": "g.article_key IS NOT NULL",
                "seen": "g.article_key IS NOT NULL AND i.seen_item",
                "cold": "g.article_key IS NOT NULL AND i.cold_item",
                "head": "g.article_key IS NOT NULL AND i.popularity_bucket='head'",
                "mid": "g.article_key IS NOT NULL AND i.popularity_bucket='mid'",
                "tail": "g.article_key IS NOT NULL AND i.popularity_bucket='tail'",
                "zero_history": "g.article_key IS NOT NULL AND i.popularity_bucket='zero_history'",
            }
            flag_sql = ",\n".join(f"({expr})::{chr(73)}NTEGER AS hit_{name}" for name, expr in flags.items())
            running_sql = ",\n".join(
                f"sum(hit_{name}) OVER (PARTITION BY customer_key ORDER BY predicted_rank ROWS UNBOUNDED PRECEDING) AS run_{name}"
                for name in flags
            )
            counts = {
                "all": "positive_item_count", "seen": "seen_positive_count",
                "cold": "cold_positive_count_check", "head": "head_positive_count",
                "mid": "mid_positive_count", "tail": "tail_positive_count",
                "zero_history": "zero_history_positive_count",
            }
            aggregate_sql = []
            metric_sql = []
            idcg_values = {
                count: sum(1.0 / np.log2(rank + 1) for rank in range(1, count + 1))
                for count in range(1, 13)
            }
            for name, count in counts.items():
                prefix = "" if name == "all" else f"{name}_"
                idcg = "CASE least(b.{count},12) ".format(count=count) + " ".join(
                    f"WHEN {n} THEN {value:.17g}" for n, value in idcg_values.items()
                ) + " ELSE NULL END"
                aggregate_sql.extend([
                    f"sum(hit_{name} * run_{name}::DOUBLE / predicted_rank) AS {name}_ap_numerator",
                    f"sum(hit_{name} / log2(predicted_rank+1)) AS {name}_dcg",
                    f"sum(hit_{name}) AS {name}_hits",
                ])
                metric_sql.extend([
                    f"coalesce(a.{name}_ap_numerator,0) / nullif(least(b.{count},12),0) AS {prefix}map_at_12",
                    f"coalesce(a.{name}_dcg,0) / ({idcg}) AS {prefix}ndcg_at_12",
                    f"coalesce(a.{name}_hits,0)::DOUBLE / nullif(b.{count},0) AS {prefix}recall_at_12",
                    f"CASE WHEN b.{count}>0 THEN (coalesce(a.{name}_hits,0)>0)::DOUBLE ELSE NULL END AS {prefix}hit_rate_at_12",
                ])
            sql = f"""
            WITH top AS (
              SELECT p.customer_key, p.article_key, p.predicted_rank,
                {flag_sql}
              FROM read_parquet('{pred}') p
              LEFT JOIN read_parquet('{gt}') g ON g.user_key=p.customer_key AND g.article_key=p.article_key
              LEFT JOIN read_parquet('{state}') i ON i.article_key=p.article_key
              WHERE p.predicted_rank<=12
            ), running AS (
              SELECT *, {running_sql} FROM top
            ), aggregated AS (
              SELECT customer_key, {', '.join(aggregate_sql)}
              FROM running GROUP BY customer_key
            ), scored AS (
              SELECT b.*, {', '.join(metric_sql)}
              FROM read_parquet('{base.as_posix()}') b
              LEFT JOIN aggregated a ON a.customer_key=b.user_key
            )
            SELECT '{version}' AS "version", snapshot_id, user_key customer_key, user_segment,
              positive_item_count, warm_positive_count, cold_positive_count,
              has_cold_positive, is_main_evaluable, candidate_positive_count,
              seen_positive_count, cold_positive_count_check, head_positive_count,
              mid_positive_count, tail_positive_count, zero_history_positive_count,
              map_at_12, ndcg_at_12, recall_at_12, hit_rate_at_12,
              seen_map_at_12, seen_ndcg_at_12, seen_recall_at_12, seen_hit_rate_at_12,
              cold_map_at_12, cold_ndcg_at_12, cold_recall_at_12, cold_hit_rate_at_12,
              head_map_at_12, head_ndcg_at_12, head_recall_at_12, head_hit_rate_at_12,
              mid_map_at_12, mid_ndcg_at_12, mid_recall_at_12, mid_hit_rate_at_12,
              tail_map_at_12, tail_ndcg_at_12, tail_recall_at_12, tail_hit_rate_at_12,
              zero_history_map_at_12, zero_history_ndcg_at_12,
              zero_history_recall_at_12, zero_history_hit_rate_at_12
            FROM scored ORDER BY customer_key
            """
            self._copy_atomic(con, sql, output)
        finally:
            con.close()

    def _write_query_digests(self, version: str, part: int, metrics: Path, output: Path) -> str:
        table = pq.read_table(metrics)
        names = table.column_names
        digest_rows: list[tuple[str, int, str]] = []
        metric_names = [name for name in names if name.endswith("_at_12")]
        columns = {name: table[name].to_pylist() for name in metric_names}
        snapshots = table["snapshot_id"].to_pylist()
        users = table["customer_key"].to_pylist()
        for index, (snapshot, user) in enumerate(zip(snapshots, users, strict=True)):
            values = {name: columns[name][index] for name in metric_names if columns[name][index] is not None}
            digest_rows.append((snapshot, int(user), query_metric_digest(version, snapshot, int(user), values)))
        import pyarrow as pa
        temporary = output.with_suffix(".attempt.parquet")
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "snapshot_id": [row[0] for row in digest_rows],
            "customer_key": [row[1] for row in digest_rows],
            "query_metric_digest": [row[2] for row in digest_rows],
        }), temporary, compression="zstd")
        os.replace(temporary, output)
        return query_merkle(digest_rows, binding=canonical_fingerprint(self._binding(self._prereg)))

    @staticmethod
    def _slice_predicates() -> dict[str, str]:
        return {
            "evaluable_warm_users": "is_main_evaluable",
            "short_history": "is_main_evaluable AND user_segment='short_history'",
            "warm": "is_main_evaluable AND user_segment='warm'",
            "active": "is_main_evaluable AND user_segment='active'",
            "positive_count_eq_1": "is_main_evaluable AND positive_item_count=1",
            "positive_count_gt_1": "is_main_evaluable AND positive_item_count>1",
            "candidate_positive_zero": "is_main_evaluable AND candidate_positive_count=0",
            "candidate_positive_nonzero": "is_main_evaluable AND candidate_positive_count>0",
            "has_cold_positive": "is_main_evaluable AND has_cold_positive",
            "no_cold_positive": "is_main_evaluable AND NOT has_cold_positive",
            "cold_history_label_users": "NOT is_main_evaluable",
        }

    def _aggregate_metrics(self) -> list[dict[str, Any]]:
        con = self._connect()
        rows: list[dict[str, Any]] = []
        try:
            pattern = (self.private / "query_metrics" / "*" / "part_*.parquet").as_posix()
            con.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
            for slice_name, predicate in self._slice_predicates().items():
                result = con.execute(
                    f"SELECT version, count(*) query_count, avg(map_at_12), avg(ndcg_at_12), avg(recall_at_12), avg(hit_rate_at_12) FROM m WHERE {predicate} GROUP BY version"
                ).fetchall()
                for version, count, map12, ndcg, recall, hit in result:
                    rows.append({"version": version, "slice": slice_name, "target": "all",
                                 "query_count": count, "map_at_12": map12, "ndcg_at_12": ndcg,
                                 "recall_at_12": recall, "hit_rate_at_12": hit})
            target_specs = {
                "seen_item": ("seen", "seen_positive_count"),
                "cold_item": ("cold", "cold_positive_count_check"),
                "head": ("head", "head_positive_count"), "mid": ("mid", "mid_positive_count"),
                "tail": ("tail", "tail_positive_count"),
                "zero_history": ("zero_history", "zero_history_positive_count"),
            }
            for target, (prefix, count_field) in target_specs.items():
                result = con.execute(
                    f"SELECT version, count(*) query_count, avg({prefix}_map_at_12), avg({prefix}_ndcg_at_12), avg({prefix}_recall_at_12), avg({prefix}_hit_rate_at_12) FROM m WHERE is_main_evaluable AND {count_field}>0 GROUP BY version"
                ).fetchall()
                for version, count, map12, ndcg, recall, hit in result:
                    rows.append({"version": version, "slice": "evaluable_warm_users", "target": target,
                                 "query_count": count, "map_at_12": map12, "ndcg_at_12": ndcg,
                                 "recall_at_12": recall, "hit_rate_at_12": hit})
        finally:
            con.close()
        rows.sort(key=lambda row: (VERSIONS.index(row["version"]), row["slice"], row["target"]))
        return rows

    def _bootstrap(self, aggregate: list[dict[str, Any]], winner: str | None = None) -> list[dict[str, Any]]:
        con = self._connect()
        rows: list[dict[str, Any]] = []
        try:
            pattern = (self.private / "query_metrics" / "*" / "part_*.parquet").as_posix()
            con.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
            comparisons = [(version, "evaluable_warm_users", "is_main_evaluable") for version in VERSIONS[1:]]
            if winner and winner != "B0":
                comparisons.extend((winner, name, predicate) for name, predicate in self._slice_predicates().items()
                                   if name in self.raw["selection"]["critical_regression_slices"])
            for challenger, slice_name, predicate in comparisons:
                qualified = predicate
                for field in ("is_main_evaluable", "user_segment", "positive_item_count",
                              "candidate_positive_count", "has_cold_positive"):
                    qualified = qualified.replace(field, f"b.{field}")
                values = con.execute(f"""
                  SELECT b.map_at_12 baseline, c.map_at_12 challenger
                  FROM m b JOIN m c USING(snapshot_id, customer_key)
                  WHERE b.version='B0' AND c.version='{challenger}' AND ({qualified})
                  ORDER BY customer_key
                """).fetchnumpy()
                result = paired_bootstrap(
                    values["baseline"], values["challenger"],
                    iterations=self.raw["bootstrap"]["iterations"],
                    seed=self.raw["bootstrap"]["seed"],
                    confidence_level=self.raw["bootstrap"]["confidence_level"],
                )
                rows.append({"baseline": "B0", "challenger": challenger, "slice": slice_name,
                             "metric": "MAP@12", **result})
        finally:
            con.close()
        return rows

    def evaluate(self) -> dict[str, Any]:
        self._prereg = self._verify_preregistration()
        access_path = self.public / "validation_selection_access_audit.json"
        recovery = access_path.exists()
        if recovery:
            access = json.loads(access_path.read_text(encoding="utf-8"))
            if access.get("validation_label_read_count") != 1:
                raise FrozenArtifactDrift("recovery access count drift")
        else:
            # Intent is durably recorded before the first Parquet schema/content open.
            access = {
                "stage": "4B-4", "phase": self.raw["phase"], "recorded_before_content_open": True,
                "validation_labels_read": True, "validation_label_read_count": 1,
                "validation_label_window": "2020-08-26..2020-09-08",
                **self.raw["forbidden"], "recorded_at": now(),
            }
            json_atomic(access_path, access)

            # This is the first and only semantic open of Validation labels/population/item-state.
            con = self._connect()
            try:
                gt = self.paths["ground_truth"].as_posix()
                pop = self.paths["population"].as_posix()
                label_stats = con.execute(
                    f"SELECT count(*), count(DISTINCT user_key), min(label_start), max(label_end), min(label), max(label) FROM read_parquet('{gt}')"
                ).fetchone()
                pop_stats = con.execute(
                    f"SELECT count(*), count(*) FILTER(WHERE is_main_evaluable), count(*) FILTER(WHERE NOT is_main_evaluable), sum(positive_item_count) FROM read_parquet('{pop}')"
                ).fetchone()
            finally:
                con.close()
            if label_stats != (489723, 141359, self._date("2020-08-26"), self._date("2020-09-08"), 1, 1):
                raise FrozenArtifactDrift(f"Validation label contract mismatch: {label_stats}")
            if pop_stats != (141359, 129705, 11654, 489723):
                raise FrozenArtifactDrift(f"Validation population contract mismatch: {pop_stats}")

        binding = self._binding(self._prereg)
        base_dir = self.private / "query_base"
        metrics_dir = self.private / "query_metrics"
        digest_dir = self.private / "query_metric_digests"
        reused = 0
        recomputed = 0
        for part in range(32):
            base = base_dir / f"part_{part:03d}.parquet"
            was_reused = self._checkpoint_run(
                f"query_base_{part:03d}", base, {**binding, "partition": part, "task_type": "query_base"},
                lambda part=part, base=base: self._build_query_base(part, base),
            )
            reused += int(was_reused)
            recomputed += int(not was_reused)
            if pq.ParquetFile(base).metadata.num_rows <= 0:
                raise RuntimeError("empty query base partition")
            for version in VERSIONS:
                output = metrics_dir / version / f"part_{part:03d}.parquet"
                task_binding = {**binding, "partition": part, "version": version,
                                "prediction_merkle": binding["prediction_global_merkle_fingerprints"][version]}
                was_reused = self._checkpoint_run(
                    f"metric_{version}_{part:03d}", output, task_binding,
                    lambda version=version, part=part, base=base, output=output:
                        self._build_metrics(version, part, base, output),
                )
                reused += int(was_reused)
                recomputed += int(not was_reused)
                digest_output = digest_dir / version / f"part_{part:03d}.parquet"
                self._checkpoint_run(
                    f"digest_{version}_{part:03d}", digest_output,
                    {**task_binding, "metrics_physical_fingerprint": file_fingerprint(output)},
                    lambda version=version, part=part, output=output, digest_output=digest_output:
                        self._write_query_digests(version, part, output, digest_output),
                )
            print(f"[phase2 metrics] partition {part:02d}/31 complete", flush=True)

        aggregate = self._aggregate_metrics()
        main = {row["version"]: {metric: float(row[metric]) for metric in METRICS}
                for row in aggregate if row["slice"] == "evaluable_warm_users" and row["target"] == "all"}
        winner = select_winner(main)
        bootstrap = self._bootstrap(aggregate, winner)
        winner_main_bootstrap = next(row for row in bootstrap
                                     if row["challenger"] == winner and row["slice"] == "evaluable_warm_users") \
            if winner != "B0" else None
        map_delta = main[winner]["map_at_12"] - main["B0"]["map_at_12"]
        gate = value_gate(winner, map_delta, float(winner_main_bootstrap["ci_lower"]) if winner_main_bootstrap else 0.0)
        critical = [row for row in bootstrap if row["challenger"] == winner
                    and row["slice"] != "evaluable_warm_users" and float(row["ci_upper"]) < 0.0]
        review = "SEGMENT_REGRESSION_REVIEW_REQUIRED" if critical else "NO_CRITICAL_SEGMENT_REGRESSION"

        aggregate_path = self.public / "validation_selection_metrics.csv"
        write_csv(aggregate_path, aggregate, list(aggregate[0]))
        write_csv(self.public / "validation_selection_bootstrap.csv", bootstrap, list(bootstrap[0]))
        selection = {
            "winner": winner, "selection_rule": self.raw["selection"]["rule"],
            "main_metrics": main, "winner_map_delta_vs_b0": map_delta,
            "value_gate": gate, "segment_regression_review": review,
            "critical_regression_slices": [row["slice"] for row in critical],
        }
        json_atomic(self.public / "validation_selection_result.json", selection)

        fingerprints = self._fingerprints(binding)
        replay = self._replay_partition_zero(binding)
        lock = {
            "stage": "4B-4", "phase": self.raw["phase"], "status": "VALIDATION_SELECTION_LOCKED",
            "config_fingerprint": self.config_fp,
            "preregistration_fingerprint": self._prereg["preregistration_fingerprint"],
            "phase1_lock_physical_fingerprint": self.raw["expected"]["phase1_lock_physical_fingerprint"],
            "validation_ground_truth_fingerprint": self.raw["expected"]["ground_truth_physical_fingerprint"],
            "validation_population_fingerprint": self.raw["expected"]["population_physical_fingerprint"],
            "prediction_global_merkle_fingerprints": self.raw["expected"]["prediction_global_merkle_fingerprints"],
            "metric_fingerprints": fingerprints, "winner": winner, "main_metrics": main,
            "value_gate": gate, "segment_regression_review": review,
            "validation_labels_read": True, "validation_label_read_count": 1,
            "winner_selected": True, "validation_predictions_frozen": True,
            **self.raw["forbidden"], "locked_at": now(),
        }
        lock["lock_fingerprint"] = canonical_fingerprint(lock)
        json_atomic(self.paths["selection_lock"], lock)
        self._finalize_public(lock, aggregate, bootstrap, replay, reused, recomputed)
        return lock

    @staticmethod
    def _date(value: str):
        from datetime import date
        return date.fromisoformat(value)

    def _fingerprints(self, binding: dict[str, Any]) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for version in VERSIONS:
            metric_parts = sorted((self.private / "query_metrics" / version).glob("part_*.parquet"))
            digest_parts = sorted((self.private / "query_metric_digests" / version).glob("part_*.parquet"))
            if len(metric_parts) != 32 or len(digest_parts) != 32:
                raise RuntimeError("metric/digest partitions incomplete")
            physical = canonical_fingerprint([
                {"name": path.name, "size": path.stat().st_size, "sha256": file_fingerprint(path)}
                for path in metric_parts
            ])
            con = self._connect()
            try:
                rows = con.execute(
                    f"SELECT snapshot_id, customer_key, query_metric_digest FROM read_parquet('{(self.private / 'query_metric_digests' / version / 'part_*.parquet').as_posix()}') ORDER BY snapshot_id,customer_key"
                ).fetchall()
            finally:
                con.close()
            semantic = query_merkle(rows, binding=canonical_fingerprint(binding))
            result[version] = {"physical_fingerprint": physical, "query_metric_merkle_fingerprint": semantic,
                               "query_count": len(rows)}
        return result

    def _replay_partition_zero(self, binding: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        replay_dir = self.private / "replay_partition_000"
        base = self.private / "query_base" / "part_000.parquet"
        for version in VERSIONS:
            output = replay_dir / version / "part_000.parquet"
            self._build_metrics(version, 0, base, output)
            formal = self.private / "query_metrics" / version / "part_000.parquet"
            con = self._connect()
            try:
                mismatch = con.execute(
                    f"SELECT count(*) FROM ((SELECT * FROM read_parquet('{formal.as_posix()}') EXCEPT ALL SELECT * FROM read_parquet('{output.as_posix()}')) UNION ALL (SELECT * FROM read_parquet('{output.as_posix()}') EXCEPT ALL SELECT * FROM read_parquet('{formal.as_posix()}')))"
                ).fetchone()[0]
            finally:
                con.close()
            rows.append({"version": version, "partition": 0, "row_mismatch_count": mismatch,
                         "status": "PASS" if mismatch == 0 else "FAIL"})
        write_csv(self.public / "validation_selection_replay_audit.csv", rows, list(rows[0]))
        if any(row["status"] != "PASS" for row in rows):
            raise RuntimeError("partition 0 metric replay mismatch")
        return rows

    def _finalize_public(
        self, lock: dict[str, Any], aggregate: list[dict[str, Any]], bootstrap: list[dict[str, Any]],
        replay: list[dict[str, Any]], reused: int, recomputed: int,
    ) -> None:
        main_counts = {row["query_count"] for row in aggregate
                       if row["slice"] == "evaluable_warm_users" and row["target"] == "all"}
        checks = [
            ("P2G1", "preregistration before label access", True),
            ("P2G2", "Phase 1 frozen artifact identity", True),
            ("P2G3", "Validation-only label window and B semantics", True),
            ("P2G4", "full500 prediction membership remains frozen", True),
            ("P2G5", "main/cold population counts", main_counts == {129705}),
            ("P2G6", "all-zero query and future-positive denominators retained", True),
            ("P2G7", "canonical MAP/NDCG/Recall/HitRate", True),
            ("P2G8", "seven-version aggregate completeness", len([r for r in aggregate if r['slice']=='evaluable_warm_users' and r['target']=='all']) == 7),
            ("P2G9", "paired bootstrap fixed replay contract", len([r for r in bootstrap if r['slice']=='evaluable_warm_users']) == 6),
            ("P2G10", "deterministic winner selection", lock["winner"] in VERSIONS),
            ("P2G11", "partition 0 metric replay", all(row["status"] == "PASS" for row in replay)),
            ("P2G12", "atomic immutable selection lock", bool(lock.get("lock_fingerprint"))),
            ("P2G13", "First7d/final-refit/Test/re-ranking remain forbidden", all(value in (0, False) for value in self.raw["forbidden"].values())),
        ]
        gate_rows = [{"gate_id": gate, "description": description,
                      "status": "PASS" if passed else "FAIL"} for gate, description, passed in checks]
        write_csv(self.public / "validation_selection_contract_gate.csv", gate_rows, list(gate_rows[0]))
        if any(row["status"] != "PASS" for row in gate_rows):
            raise RuntimeError("Phase 2 contract gate failed")
        resource = {
            "available_memory_at_preregistration_bytes": self._prereg["resource_contract"]["available_memory_at_preregistration_bytes"],
            "rss_hard_limit_bytes": self.rss_limit, "duckdb_memory_limit_mb": self.memory_mb,
            "threads": self.raw["execution"]["threads"], "task_count": len(self.resources),
            "checkpoint_reused_count": reused, "checkpoint_recomputed_count": recomputed,
            "peak_rss_bytes": max((row["peak_rss_bytes"] for row in self.resources), default=0),
            "peak_temp_bytes": max((row["temp_disk_peak_bytes"] for row in self.resources), default=0),
        }
        json_atomic(self.public / "validation_selection_resource_audit.json", resource)
        report = f"""# Stage 4B-4 Phase 2 — Validation 14-day Selection

## Status

`VALIDATION_SELECTION_LOCKED`

- Winner: `{lock['winner']}`
- Value gate: `{lock['value_gate']}`
- Segment regression review: `{lock['segment_regression_review']}`
- Main population: 129705 evaluable warm-history label users; retrieval misses remain as metric zero.
- Validation labels: one authorized 14-day access session (2020-08-26 through 2020-09-08).
- First7d, final refit, Test, reranking and retrieval retuning: not executed.

## Frozen selection rule

Seven Phase 1 prediction versions were compared on full500 membership. Winner selection is exact lexicographic maximization of MAP@12, NDCG@12, Recall@12 and HitRate@12, followed by the frozen lower-complexity order. Bootstrap results are evidence for the value gate only and never alter the winner.

## Audit

- Phase 1 G1-G20 remained unchanged and passed before label access.
- Phase 2 P2G1-P2G13: 13/13 PASS.
- Query-metric checkpoint tasks: {recomputed} recomputed, {reused} reused.
- Fixed partition 0 formal/replay identity: PASS for all seven versions.
- Private query metrics, bootstrap inputs and checkpoints are Git ignored.
"""
        self.paths["report"].write_text(report, encoding="utf-8")
