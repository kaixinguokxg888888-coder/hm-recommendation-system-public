from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import duckdb
import lightgbm as lgb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_audit.stage1a import ResourceMonitor
from data_foundation.manifest import file_fingerprint, write_csv
from ranking_features.fingerprint import feature_semantic_fingerprint, statistics_fingerprint
from ranking_features.materialization import sql_path
from ranking_features.schema import feature_schema_rows, schema_fingerprint
from ranking_dataset.fingerprint import semantic_fingerprint as ranking_semantic_fingerprint
from retrieval.stage3e.contract import semantic_fingerprint_from_partitions

from .adapter import ModelInputAdapter, assert_adapter_deterministic
from .config import Stage4B4Config
from .fingerprint import canonical_score, merkle_fingerprint, prediction_query_digest
from .validation_features import ValidationFeatureBuilder


class Stage4B4Error(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def json_write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src/ranking_model").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("run_stage_4b4*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def canonical_json_fingerprint(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class Stage4B4Runner:
    def __init__(self, config: Stage4B4Config) -> None:
        self.config = config
        self.raw = config.raw
        self.upstream = self.raw["upstream"]
        stage4b3 = json.loads(config.resolve(self.upstream["stage_4b3_config"]).read_text(encoding="utf-8"))
        self.schema_rows = feature_schema_rows(stage4b3["canonical_fields"])
        self.schema_fp = schema_fingerprint(self.schema_rows)
        self.adapters = {
            key: ModelInputAdapter.from_schema(
                self._public_schema_rows(), key, self.raw["adapter"]["user_segment_mapping"]
            ) for key in ("F0", "F1")
        }
        self.builder = ValidationFeatureBuilder(config, stage4b3)
        self.source_fp = source_fingerprint(config.root)
        self.resources: list[dict[str, Any]] = []
        self.available_at_start = psutil.virtual_memory().available
        self.rss_limit = int(self.available_at_start * float(self.raw["execution"]["peak_rss_fraction_of_available"]))
        self.prepare_directories()

    def _public_schema_rows(self) -> list[dict[str, str]]:
        return csv_rows(self.config.resolve(self.upstream["stage_4b3_schema"]))

    def prepare_directories(self) -> None:
        for path in (self.config.artifact_dir, self.config.private_dir,
                     self.config.checkpoint_dir, self.config.temp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def checkpoint(self, name: str) -> Path:
        return self.config.checkpoint_dir / f"{name}.json"

    def _set_fingerprint(self, paths: Iterable[Path]) -> str:
        rows = [(path.name, path.stat().st_size, file_fingerprint(path)) for path in sorted(paths)]
        return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()

    def run_task(
        self, name: str, operation: Callable[[], Any], outputs: Sequence[Path], inputs: Sequence[Path],
        *, partition: int | None = None,
    ) -> Any:
        source = self.source_fp
        identity = hashlib.sha256(
            (self.config.fingerprint + source + self._set_fingerprint(inputs)).encode("ascii")
        ).hexdigest()
        checkpoint = self.checkpoint(name)
        previous = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
        if previous and previous.get("status") == "SUCCESS" and previous.get("identity") == identity \
                and all(path.exists() for path in outputs) \
                and previous.get("output_fingerprint") == self._set_fingerprint(outputs):
            row = dict(previous)
            row["checkpoint_reused"] = True
            self.resources.append(row)
            return None
        attempt = int((previous or {}).get("attempt", 0)) + 1
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        try:
            with monitor:
                result = operation()
            if monitor.peak_rss_bytes > self.rss_limit:
                raise Stage4B4Error(
                    f"peak RSS {monitor.peak_rss_bytes} exceeds start-available 70% limit {self.rss_limit}"
                )
            if not all(path.exists() for path in outputs):
                raise Stage4B4Error(f"missing task outputs: {name}")
            row = {
                "task": name, "partition_id": partition, "status": "SUCCESS", "attempt": attempt,
                "identity": identity, "output_fingerprint": self._set_fingerprint(outputs),
                "row_count": sum(pq.ParquetFile(p).metadata.num_rows for p in outputs if p.suffix == ".parquet"),
                "wall_time_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes, "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "checkpoint_reused": False, "generated_at": now(), "error": "",
            }
            json_write_atomic(checkpoint, row)
            self.resources.append(row)
            return result
        except Exception as exc:
            json_write_atomic(checkpoint, {
                "task": name, "partition_id": partition, "status": "FAILED", "attempt": attempt,
                "identity": identity, "output_fingerprint": "", "row_count": 0,
                "wall_time_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes, "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "checkpoint_reused": False, "generated_at": now(), "error": str(exc).replace(str(self.config.root), "."),
            })
            raise

    def preflight(self) -> dict[str, Any]:
        gates = ["stage_3e_gate", "stage_4a_gate", "stage_4b1_gate", "stage_4b2_gate", "stage_4b3_gate"]
        for role in gates:
            if any(row["status"] != "PASS" for row in csv_rows(self.config.resolve(self.upstream[role]))):
                raise Stage4B4Error(f"upstream gate failed: {role}")
        for role, manifest in (("stage_3e", "stage_3e_manifest"), ("stage_4a", "stage_4a_manifest")):
            for row in csv_rows(self.config.resolve(self.upstream[manifest])):
                path = self.config.resolve(row["artifact_path"])
                if path.exists() and file_fingerprint(path) != row["artifact_fingerprint"]:
                    # README/AGENTS 会在后续阶段合法变化；这里只核验阶段自身正式文件。
                    if path.name not in {"README.md", "AGENTS.md"}:
                        raise Stage4B4Error(f"{role} manifest physical mismatch: {row['artifact_path']}")
        for config_role, gate_role in (
            ("stage_4b1_config", "stage_4b1_gate"),
            ("stage_4b2_config", "stage_4b2_gate"),
            ("stage_4b3_config", "stage_4b3_gate"),
        ):
            actual = canonical_json_fingerprint(self.config.resolve(self.upstream[config_role]))
            declared = {row["config_fingerprint"] for row in csv_rows(self.config.resolve(self.upstream[gate_role]))}
            if declared != {actual}:
                raise Stage4B4Error(f"upstream config fingerprint mismatch: {config_role}")
        validation_parts = sorted(self.config.resolve(self.upstream["validation_candidates"]).glob("part_*.parquet"))
        if len(validation_parts) != 32:
            raise Stage4B4Error("Validation frozen candidate partitions incomplete")
        val_fp = semantic_fingerprint_from_partitions(validation_parts)[0]
        if val_fp != self.raw["expected"]["validation_candidate_fingerprint"]:
            raise Stage4B4Error("Validation candidate semantic fingerprint mismatch")
        if self.schema_fp != self.raw["expected"]["feature_schema_fingerprint"]:
            raise Stage4B4Error("Stage 4B-3 feature schema fingerprint mismatch")
        optimizer_parts = sorted(self.config.resolve(self.upstream["train_optimizer"]).glob("part_*.parquet"))
        feature_parts = sorted(self.config.resolve(self.upstream["train_optimizer_features"]).glob("part_*.parquet"))
        if len(optimizer_parts) != 32 or len(feature_parts) != 32:
            raise Stage4B4Error("Train optimizer inputs incomplete")
        optimizer_fp = ranking_semantic_fingerprint(optimizer_parts, optimizer=True)
        if optimizer_fp != self.raw["expected"]["train_optimizer_fingerprint"]:
            raise Stage4B4Error("Train optimizer semantic fingerprint mismatch")
        version = tuple(int(x) for x in lgb.__version__.split(".")[:3])
        if not ((4, 2, 0) <= version < (5, 0, 0)):
            raise Stage4B4Error("unsupported LightGBM version")
        return {
            "validation_candidate_fingerprint": val_fp,
            "train_optimizer_fingerprint": optimizer_fp,
            "feature_schema_fingerprint": self.schema_fp,
            "lightgbm_version": lgb.__version__,
            "available_memory_at_start_bytes": self.available_at_start,
            "peak_rss_limit_bytes": self.rss_limit,
            "disk_free_bytes": shutil.disk_usage(self.config.root).free,
        }

    def write_adapter_manifests(self) -> None:
        rows = []
        for key, adapter in self.adapters.items():
            payload = adapter.manifest()
            payload["adapter_fingerprint"] = adapter.fingerprint
            json_write_atomic(self.config.artifact_dir / f"model_input_adapter_{key.lower()}.json", payload)
            for order, (name, dtype) in enumerate(zip(adapter.ordered_features, adapter.dtypes), 1):
                rows.append({
                    "feature_set": key, "feature_order": order, "feature_name": name, "dtype": dtype,
                    "categorical": name in adapter.categorical_features, "adapter_fingerprint": adapter.fingerprint,
                })
        write_csv(self.config.artifact_dir / "model_input_schema_manifest.csv", rows, list(rows[0]))

    def materialize_validation_features(self) -> dict[str, Any]:
        b = self.builder
        b.history_partition_directory.mkdir(parents=True, exist_ok=True)
        self.run_task("validation_history_partitions", b.build_history_partitions,
                      [b.history_part(i) for i in range(32)], [b.history_path])
        global_tasks = [
            ("validation_user_aggregate", b.build_user_aggregate, b.user_aggregate_path, [b.history_path]),
            ("validation_item_aggregate", b.build_item_aggregate, b.item_aggregate_path, [b.history_path]),
            ("validation_article_features", b.build_article_features, b.article_features_path,
             [self.config.resolve(json.loads(self.config.resolve(self.upstream['stage_4b3_config']).read_text())['upstream']['articles']),
              self.config.resolve(json.loads(self.config.resolve(self.upstream['stage_4b3_config']).read_text())['upstream']['article_key_map'])]),
        ]
        for name, func, output, inputs in global_tasks:
            self.run_task(name, func, [output], inputs)
        for part in range(32):
            self.run_task(f"validation_ui_{part:03d}", lambda p=part: b.build_ui_aggregate(p),
                          [b.ui_aggregate_part(part)], [b.history_part(part)], partition=part)
            self.run_task(f"validation_category_{part:03d}", lambda p=part: b.build_category_aggregate(p),
                          [b.category_aggregate_part(part)], [b.history_part(part), b.article_features_path], partition=part)
            self.run_task(f"validation_features_{part:03d}", lambda p=part: b.build_feature_part(p),
                          [b.feature_part(part)], [b.candidate_part(part), b.user_aggregate_path,
                           b.item_aggregate_path, b.ui_aggregate_part(part), b.article_features_path,
                           b.category_aggregate_part(part)], partition=part)
            self.run_task(f"validation_feature_digest_{part:03d}", lambda p=part: b.build_digest_part(p),
                          [b.digest_part(part)], [b.feature_part(part)], partition=part)
        # Fixed partition replay proves generalized builder determinism before any label access.
        replay_part = int(self.raw["predictions"]["replay_partition"])
        self.run_task(f"validation_feature_replay_{replay_part:03d}",
                      lambda: b.build_feature_part(replay_part, replay=True), [b.feature_part(replay_part, replay=True)],
                      [b.candidate_part(replay_part), b.user_aggregate_path, b.item_aggregate_path,
                       b.ui_aggregate_part(replay_part), b.article_features_path,
                       b.category_aggregate_part(replay_part)], partition=replay_part)
        self.run_task(f"validation_feature_replay_digest_{replay_part:03d}",
                      lambda: b.build_digest_part(replay_part, replay=True), [b.digest_part(replay_part, replay=True)],
                      [b.feature_part(replay_part, replay=True)], partition=replay_part)
        parts = [b.feature_part(i) for i in range(32)]
        digests = [b.digest_part(i) for i in range(32)]
        query_count = sum(pq.ParquetFile(p).metadata.num_rows // 500 for p in parts)
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
        feature_fp = feature_semantic_fingerprint(digests, self.schema_fp)
        candidate_fp = semantic_fingerprint_from_partitions(parts)[0]
        statistics = b.build_statistics()
        statistics_fp = statistics_fingerprint(statistics)
        write_csv(self.config.artifact_dir / "validation_feature_statistics.csv", statistics, list(statistics[0]))
        physical_fp = self._set_fingerprint(parts)
        provenance_mismatch = duplicate_keys = duplicate_ranks = full500_violations = 0
        con = b.connect()
        provenance = [
            "primary_source", "source_count", "from_repurchase", "from_itemcf", "from_two_tower",
            "from_trend", "from_popularity", "repurchase_rank", "repurchase_score", "itemcf_rank",
            "itemcf_score", "two_tower_rank", "two_tower_score", "trend_rank", "trend_score",
            "popularity_rank", "popularity_score",
        ]
        try:
            for part in range(32):
                f, c = sql_path(b.feature_part(part)), sql_path(b.candidate_part(part))
                values = con.execute(f"""WITH q AS (
                    SELECT customer_key,count(*) n,count(DISTINCT article_key) ni,
                      count(DISTINCT candidate_rank) nr,min(candidate_rank) mn,max(candidate_rank) mx
                    FROM read_parquet('{f}') GROUP BY customer_key)
                  SELECT sum(n-ni),sum(n-nr),count(*) FILTER(WHERE n<>500 OR mn<>1 OR mx<>500) FROM q""").fetchone()
                duplicate_keys += int(values[0]); duplicate_ranks += int(values[1]); full500_violations += int(values[2])
                provenance_mismatch += int(con.execute(f"""SELECT count(*)
                  FROM read_parquet('{c}') c JOIN read_parquet('{f}') f USING(snapshot_id,customer_key,article_key)
                  WHERE {' OR '.join(f'c.{name} IS DISTINCT FROM f.{name}' for name in provenance)}""").fetchone()[0])
        finally:
            con.close()
        con = b.connect()
        try:
            max_history = con.execute(
                f"SELECT max(t_dat) FROM read_parquet('{sql_path(b.history_path)}')"
            ).fetchone()[0].isoformat()
        finally:
            con.close()
        if (query_count, rows, candidate_fp, max_history) != (
            self.raw["expected"]["validation_queries"], self.raw["expected"]["validation_rows"],
            self.raw["expected"]["validation_candidate_fingerprint"], "2020-08-25"
        ):
            raise Stage4B4Error("Validation full500/candidate/cutoff invariant failed")
        if duplicate_keys or duplicate_ranks or full500_violations or provenance_mismatch:
            raise Stage4B4Error("Validation candidate projection zero-drift audit failed")
        replay_fp = feature_semantic_fingerprint([b.digest_part(replay_part, replay=True)], self.schema_fp)
        formal_replay_fp = feature_semantic_fingerprint([b.digest_part(replay_part)], self.schema_fp)
        if replay_fp != formal_replay_fp:
            raise Stage4B4Error("Validation feature replay mismatch")
        return {"query_count": query_count, "row_count": rows, "candidate_projection_fingerprint": candidate_fp,
                "feature_semantic_fingerprint": feature_fp, "feature_statistics_fingerprint": statistics_fp,
                "physical_artifact_fingerprint": physical_fp, "max_history_date": max_history,
                "candidate_provenance_mismatch": provenance_mismatch, "duplicate_keys": duplicate_keys,
                "duplicate_ranks": duplicate_ranks, "full500_violation_queries": full500_violations,
                "replay_partition": replay_part, "replay_partition_fingerprint": replay_fp}

    def audit_train_groups(self) -> dict[str, Any]:
        groups = pq.read_table(self.config.resolve(self.upstream["train_groups"])).sort_by("query_order")
        sizes = groups["group_size"].to_numpy()
        positives = groups["positive_count"].to_numpy()
        zeros = groups["sampled_zero_count"].to_numpy()
        optimizer_parts = sorted(self.config.resolve(self.upstream["train_optimizer"]).glob("part_*.parquet"))
        feature_parts = sorted(self.config.resolve(self.upstream["train_optimizer_features"]).glob("part_*.parquet"))
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in optimizer_parts)
        if not (len(groups) == self.raw["expected"]["train_queries"] and rows == sizes.sum() == self.raw["expected"]["train_rows"]
                and sizes.min() == self.raw["expected"]["train_group_min"]
                and sizes.max() == self.raw["expected"]["train_group_max"]
                and positives.min() >= 1 and zeros.max() <= 100
                and positives.sum() == self.raw["expected"]["train_positive_rows"]
                and zeros.sum() == self.raw["expected"]["train_zero_rows"]):
            raise Stage4B4Error("Train optimizer/group exact reconciliation failed")
        key_mismatch = rank_mismatch = 0
        optimizer_query_order: list[tuple[str, int, int, int]] = []
        con = duckdb.connect(":memory:")
        try:
            for optimizer, feature in zip(optimizer_parts, feature_parts):
                o, f = sql_path(optimizer), sql_path(feature)
                optimizer_query_order.extend(
                    (str(snapshot), int(customer), int(order), int(size))
                    for snapshot, customer, order, size in con.execute(f"""
                      SELECT snapshot_id,customer_key,query_order,count(*)
                      FROM read_parquet('{o}')
                      GROUP BY snapshot_id,customer_key,query_order ORDER BY query_order
                    """).fetchall()
                )
                values = con.execute(f"""WITH o AS (SELECT snapshot_id,customer_key,article_key,candidate_rank FROM read_parquet('{o}')),
                  f AS (SELECT snapshot_id,customer_key,article_key,candidate_rank FROM read_parquet('{f}'))
                  SELECT (SELECT count(*) FROM o ANTI JOIN f USING(snapshot_id,customer_key,article_key))+
                         (SELECT count(*) FROM f ANTI JOIN o USING(snapshot_id,customer_key,article_key)),
                         (SELECT count(*) FROM o JOIN f USING(snapshot_id,customer_key,article_key)
                           WHERE o.candidate_rank<>f.candidate_rank)""").fetchone()
                key_mismatch += int(values[0]); rank_mismatch += int(values[1])
        finally:
            con.close()
        if key_mismatch or rank_mismatch:
            raise Stage4B4Error("Train optimizer feature membership/rank drift")
        formal_query_order = list(zip(
            groups["snapshot_id"].to_pylist(), groups["customer_key"].to_pylist(),
            groups["query_order"].to_pylist(), groups["group_size"].to_pylist(),
        ))
        if optimizer_query_order != formal_query_order:
            # LambdaRank group 是位置数组；集合相等但顺序不同仍会把相邻用户错误拼成 query。
            raise Stage4B4Error("Train optimizer row order does not follow frozen global query_order")
        return {"query_count": len(groups), "row_count": rows, "group_min": int(sizes.min()),
                "group_max": int(sizes.max()), "positive_rows": int(positives.sum()),
                "zero_rows": int(zeros.sum()), "all_zero_groups": int((positives == 0).sum()),
                "optimizer_feature_key_mismatch": key_mismatch, "optimizer_feature_rank_mismatch": rank_mismatch,
                "negative_sampling_reexecuted": False}

    def resource_smoke(self, part: int = 0) -> dict[str, Any]:
        feature = self.config.resolve(self.upstream["train_optimizer_features"]) / f"part_{part:03d}.parquet"
        optimizer = self.config.resolve(self.upstream["train_optimizer"]) / f"part_{part:03d}.parquet"
        con = duckdb.connect(":memory:")
        try:
            table = con.execute(f"""SELECT f.*,o.label,o.query_order
              FROM read_parquet('{sql_path(feature)}') f
              JOIN read_parquet('{sql_path(optimizer)}') o USING(snapshot_id,customer_key,article_key)
              ORDER BY o.query_order,f.candidate_rank,f.article_key""").fetch_arrow_table()
        finally:
            con.close()
        matrix = self.adapters["F1"].convert_table(table)
        labels = table["label"].combine_chunks().to_numpy().astype(np.uint8, copy=False)
        customer = table["customer_key"].combine_chunks().to_numpy()
        starts = np.r_[0, np.flatnonzero(customer[1:] != customer[:-1]) + 1]
        groups = np.diff(np.r_[starts, len(table)]).astype(np.int32)
        categorical = [self.adapters["F1"].ordered_features.index(name)
                       for name in self.adapters["F1"].categorical_features]
        dataset = lgb.Dataset(matrix, label=labels, group=groups,
                              feature_name=list(self.adapters["F1"].ordered_features),
                              categorical_feature=categorical,
                              params={"max_bin": self.raw["lightgbm"]["params"]["max_bin"]})
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        with monitor:
            booster = lgb.train(dict(self.raw["lightgbm"]["params"]), dataset, num_boost_round=600,
                                keep_training_booster=True)
        output = self.config.private_dir / "smoke" / "resource_smoke_model.txt"
        output.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(output)
        result = {"status": "PASS", "purpose": "resource_only_no_business_metric", "partition": part,
                  "queries": len(groups), "rows": len(table), "features": matrix.shape[1], "rounds": 600,
                  "wall_time_seconds": time.perf_counter()-started, "peak_rss_bytes": monitor.peak_rss_bytes,
                  "temp_disk_peak_bytes": monitor.peak_temp_bytes, "model_size_bytes": output.stat().st_size,
                  "available_memory_at_start_bytes": self.available_at_start, "formal_rss_limit_bytes": self.rss_limit,
                  "validation_labels_read": False, "validation_metrics_computed": False}
        json_write_atomic(self.config.private_dir / "smoke" / "resource_smoke.json", result)
        return result

    def matrix_paths(self) -> tuple[Path, Path, Path]:
        root = self.config.private_dir / "training"
        return root / "features_f1.float32", root / "labels.uint8", root / "groups.int32"

    def build_training_matrix(self) -> None:
        feature_path, label_path, group_path = self.matrix_paths()
        total = int(self.raw["expected"]["train_rows"])
        width = len(self.adapters["F1"].ordered_features)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(total, width))
        labels = np.memmap(label_path, dtype=np.uint8, mode="w+", shape=(total,))
        offset = 0
        con = duckdb.connect(":memory:")
        try:
            for part in range(32):
                feature = self.builder.config.resolve(self.upstream["train_optimizer_features"]) / f"part_{part:03d}.parquet"
                optimizer = self.builder.config.resolve(self.upstream["train_optimizer"]) / f"part_{part:03d}.parquet"
                # Frozen optimizer query_order is the only training row order; candidate_rank remains unchanged.
                query = f"""SELECT f.*,o.label,o.query_order FROM read_parquet('{sql_path(feature)}') f
                  JOIN read_parquet('{sql_path(optimizer)}') o USING(snapshot_id,customer_key,article_key)
                  ORDER BY o.query_order,f.candidate_rank,f.article_key"""
                reader = con.execute(query).fetch_record_batch(rows_per_batch=65536)
                for batch in reader:
                    table = pa.Table.from_batches([batch])
                    block = self.adapters["F1"].convert_table(table)
                    n = len(table)
                    matrix[offset:offset+n] = block
                    labels[offset:offset+n] = table["label"].combine_chunks().to_numpy()
                    offset += n
        finally:
            con.close()
        matrix.flush(); labels.flush()
        groups = pq.read_table(self.config.resolve(self.upstream["train_groups"])).sort_by("query_order")
        np.asarray(groups["group_size"].to_numpy(), dtype=np.int32).tofile(group_path)
        if offset != total:
            raise Stage4B4Error("training matrix row count mismatch")

    def train_model(self, feature_set: str) -> dict[str, Any]:
        feature_path, label_path, group_path = self.matrix_paths()
        total = int(self.raw["expected"]["train_rows"])
        width = len(self.adapters[feature_set].ordered_features)
        full_width = len(self.adapters["F1"].ordered_features)
        full = np.memmap(feature_path, dtype=np.float32, mode="r", shape=(total, full_width))
        data = full if feature_set == "F1" else full[:, :width]
        labels = np.memmap(label_path, dtype=np.uint8, mode="r", shape=(total,))
        groups = np.fromfile(group_path, dtype=np.int32)
        categorical = [self.adapters[feature_set].ordered_features.index(name)
                       for name in self.adapters[feature_set].categorical_features]
        params = dict(self.raw["lightgbm"]["params"])
        dataset = lgb.Dataset(data, label=labels, group=groups,
                              feature_name=list(self.adapters[feature_set].ordered_features),
                              categorical_feature=categorical, params={"max_bin": params["max_bin"]},
                              free_raw_data=True)
        started = time.perf_counter()
        monitor = ResourceMonitor(self.config.temp_dir)
        with monitor:
            booster = lgb.train(params, dataset, num_boost_round=600, keep_training_booster=True)
        if monitor.peak_rss_bytes > self.rss_limit:
            raise Stage4B4Error("training peak RSS exceeds formal 70% available-memory boundary")
        model_path = self.config.private_dir / "models" / f"L{0 if feature_set == 'F0' else 1}_600.txt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(model_path, num_iteration=600)
        semantic = hashlib.sha256(booster.model_to_string(num_iteration=600).replace("\r\n", "\n").encode()).hexdigest()
        result = {"feature_set": feature_set, "model_id": f"L{0 if feature_set == 'F0' else 1}",
                  "num_iterations": booster.current_iteration(), "model_physical_fingerprint": file_fingerprint(model_path),
                  "model_semantic_fingerprint": semantic, "wall_time_seconds": time.perf_counter()-started,
                  "peak_rss_bytes": monitor.peak_rss_bytes, "adapter_fingerprint": self.adapters[feature_set].fingerprint,
                  "feature_importance_gain_sum": float(booster.feature_importance("gain").sum())}
        json_write_atomic(self.config.private_dir / "models" / f"{result['model_id']}_metadata.json", result)
        self.resources.append({"task": f"train_{result['model_id']}", "status": "SUCCESS", **result,
                               "temp_disk_peak_bytes": monitor.peak_temp_bytes, "checkpoint_reused": False})
        del booster, dataset, data, full, labels
        return result

    def model_path(self, model_id: str) -> Path:
        return self.config.private_dir / "models" / f"{model_id}_600.txt"

    def predict_partition(self, part: int, *, replay: bool = False,
                          validation_feature_fingerprint: str) -> list[Path]:
        feature_path = self.builder.feature_part(part, replay=replay)
        suffix = "replay" if replay else "formal"
        versions = self.raw["predictions"]["versions"]
        output_paths = [self.config.private_dir / "predictions" / suffix / version / f"part_{part:03d}.parquet"
                        for version in versions]
        digest_paths = [self.config.private_dir / "prediction_digests" / suffix / version / f"part_{part:03d}.parquet"
                        for version in versions]
        for path in (*output_paths, *digest_paths): path.parent.mkdir(parents=True, exist_ok=True)
        models = {"L0": lgb.Booster(model_file=str(self.model_path("L0"))),
                  "L1": lgb.Booster(model_file=str(self.model_path("L1")))}
        adapters = self.adapters
        writers: dict[str, pq.ParquetWriter] = {}
        digest_rows: dict[str, list[dict[str, Any]]] = {version: [] for version in versions}
        # 64,000 是 500 的整数倍；Stage 3E partition 也严格由完整 500-row queries 构成。
        # 因此 Parquet 流批不会拆 query，且任何时刻都不把 2.2M-row partition 全量载入内存。
        batch_queries = int(self.raw["predictions"]["batch_queries"])
        batch_rows = batch_queries * 500
        model_meta = {mid: json.loads((self.config.private_dir / "models" / f"{mid}_metadata.json").read_text())
                      for mid in ("L0", "L1")}
        try:
            for batch in pq.ParquetFile(feature_path).iter_batches(batch_size=batch_rows):
                block = pa.Table.from_batches([batch])
                if len(block) % 500:
                    raise Stage4B4Error("query-contained prediction batch split a query")
                candidates = block["candidate_rank"].combine_chunks().to_numpy()
                articles = block["article_key"].combine_chunks().to_numpy()
                customers = block["customer_key"].combine_chunks().to_numpy()
                starts = np.arange(0, len(block), 500)
                ends = starts + 500
                if any(len(set(customers[s:e])) != 1 for s, e in zip(starts, ends)):
                    raise Stage4B4Error("prediction batch query identity violation")
                scores: dict[str, np.ndarray] = {"B0": -candidates.astype(np.float64)}
                for mid, adapter in adapters.items():
                    prefix = "L0" if mid == "F0" else "L1"
                    matrix = adapter.convert_table(block)
                    for iteration in (100, 300, 600):
                        scores[f"{prefix}_{iteration}"] = models[prefix].predict(matrix, num_iteration=iteration)
                for version in versions:
                    predicted_rank = np.empty(len(block), dtype=np.int32)
                    raw = scores[version]
                    for s, e in zip(starts, ends):
                        order = np.lexsort((articles[s:e], candidates[s:e], -raw[s:e]))
                        predicted_rank[s:e][order] = np.arange(1, e-s+1, dtype=np.int32)
                        rows = [(int(a), int(c), float(v)) for a, c, v in zip(articles[s:e], candidates[s:e], raw[s:e])]
                        raw_digest, rank_digest = prediction_query_digest(version, "validation_14d", int(customers[s]), rows)
                        digest_rows[version].append({"snapshot_id": "validation_14d", "customer_key": int(customers[s]),
                                                     "row_count": int(e-s), "raw_score_digest": raw_digest,
                                                     "ranked_list_digest": rank_digest})
                    prefix = version.split("_")[0]
                    feature_set = "BASELINE" if version == "B0" else ("F0" if prefix == "L0" else "F1")
                    iteration = 0 if version == "B0" else int(version.split("_")[1])
                    model_fp = "candidate_rank_baseline" if version == "B0" else model_meta[prefix]["model_semantic_fingerprint"]
                    output = pa.table({
                        "snapshot_id": pa.array(["validation_14d"] * len(block)),
                        "customer_key": block["customer_key"], "article_key": block["article_key"],
                        "candidate_rank": block["candidate_rank"], "raw_score": pa.array(raw, type=pa.float64()),
                        "predicted_rank": pa.array(predicted_rank, type=pa.int32()),
                        "model_id": pa.array([version] * len(block)), "feature_set": pa.array([feature_set] * len(block)),
                        "num_iteration": pa.array([iteration] * len(block), type=pa.int32()),
                        "model_fingerprint": pa.array([model_fp] * len(block)),
                        "feature_fingerprint": pa.array([validation_feature_fingerprint] * len(block)),
                    })
                    if version not in writers:
                        writers[version] = pq.ParquetWriter(output_paths[versions.index(version)], output.schema, compression="zstd")
                    writers[version].write_table(output)
        finally:
            for writer in writers.values(): writer.close()
        for version, path in zip(versions, digest_paths):
            pq.write_table(pa.Table.from_pylist(digest_rows[version]), path, compression="zstd")
        return output_paths + digest_paths

    def freeze_predictions(self, validation: dict[str, Any], models: list[dict[str, Any]]) -> list[dict[str, Any]]:
        versions = self.raw["predictions"]["versions"]
        for part in range(32):
            inputs = [self.builder.feature_part(part), *[self.model_path(mid) for mid in ("L0", "L1")]]
            outputs = [self.config.private_dir / role / "formal" / version / f"part_{part:03d}.parquet"
                       for role in ("predictions", "prediction_digests") for version in versions]
            self.run_task(
                f"prediction_{part:03d}",
                lambda p=part: self.predict_partition(
                    p, validation_feature_fingerprint=validation["feature_semantic_fingerprint"]
                ),
                outputs, inputs, partition=part,
            )
        replay_part = int(self.raw["predictions"]["replay_partition"])
        replay_outputs = [self.config.private_dir / role / "replay" / version / f"part_{replay_part:03d}.parquet"
                          for role in ("predictions", "prediction_digests") for version in versions]
        self.run_task("prediction_replay", lambda: self.predict_partition(
                          replay_part, replay=True,
                          validation_feature_fingerprint=validation["feature_semantic_fingerprint"]), replay_outputs,
                      [self.builder.feature_part(replay_part, replay=True), *[self.model_path(mid) for mid in ("L0", "L1")]],
                      partition=replay_part)
        manifests = []
        model_map = {row["model_id"]: row for row in models}
        replay_audit = []
        for version in versions:
            predictions = sorted((self.config.private_dir / "predictions" / "formal" / version).glob("*.parquet"))
            digests = sorted((self.config.private_dir / "prediction_digests" / "formal" / version).glob("*.parquet"))
            query_rows = []
            for path in digests:
                t = pq.read_table(path)
                query_rows.extend(zip(t["snapshot_id"].to_pylist(), t["customer_key"].to_pylist(),
                                      t["raw_score_digest"].to_pylist(), t["ranked_list_digest"].to_pylist()))
            prefix = version.split("_")[0]
            model_binding = {"semantic": "candidate_rank_baseline", "physical": "not_applicable"} \
                if version == "B0" else {
                    "semantic": model_map[prefix]["model_semantic_fingerprint"],
                    "physical": model_map[prefix]["model_physical_fingerprint"],
                }
            iteration = 0 if version == "B0" else int(version.split("_")[1])
            feature_set = "BASELINE" if version == "B0" else ("F0" if prefix == "L0" else "F1")
            tie_break = self.raw["predictions"]["baseline_tie_break"] if version == "B0" \
                else self.raw["predictions"]["learned_tie_break"]
            binding = json.dumps({"snapshot": "validation_14d", "candidate": validation["candidate_projection_fingerprint"],
                                  "feature": validation["feature_semantic_fingerprint"], "model_id": version,
                                  "model_fingerprint": model_binding, "feature_set": feature_set,
                                  "num_iteration": iteration, "tie_break": tie_break,
                                  "score": self.raw["predictions"]["score_serialization"],
                                  "prediction_source_fingerprint": self.source_fp,
                                  "partition": "customer_key_mod_32_query_contained"}, sort_keys=True)
            raw_fp = merkle_fingerprint(((s, c, r) for s,c,r,_ in query_rows), "prediction-raw-merkle-v1", binding)
            rank_fp = merkle_fingerprint(((s, c, r) for s,c,_,r in query_rows), "prediction-rank-merkle-v1", binding)
            physical = self._set_fingerprint(predictions)
            formal_part = self.config.private_dir / "predictions" / "formal" / version / f"part_{replay_part:03d}.parquet"
            replay_prediction = self.config.private_dir / "predictions" / "replay" / version / f"part_{replay_part:03d}.parquet"
            formal_digest = self.config.private_dir / "prediction_digests" / "formal" / version / f"part_{replay_part:03d}.parquet"
            replay_digest = self.config.private_dir / "prediction_digests" / "replay" / version / f"part_{replay_part:03d}.parquet"
            con = duckdb.connect(":memory:")
            try:
                f, r = sql_path(formal_part), sql_path(replay_prediction)
                mismatch = con.execute(f"""WITH f AS (SELECT * FROM read_parquet('{f}')),
                    r AS (SELECT * FROM read_parquet('{r}'))
                  SELECT
                    (SELECT count(*) FROM f ANTI JOIN r USING(snapshot_id,customer_key,article_key))+
                    (SELECT count(*) FROM r ANTI JOIN f USING(snapshot_id,customer_key,article_key)),
                    (SELECT count(*) FROM f JOIN r USING(snapshot_id,customer_key,article_key)
                      WHERE f.raw_score IS DISTINCT FROM r.raw_score),
                    (SELECT count(*) FROM f JOIN r USING(snapshot_id,customer_key,article_key)
                      WHERE f.predicted_rank IS DISTINCT FROM r.predicted_rank)
                """).fetchone()
            finally:
                con.close()
            formal_digest_fp = self._prediction_digest_semantic_fingerprint(formal_digest)
            replay_digest_fp = self._prediction_digest_semantic_fingerprint(replay_digest)
            replay_audit.append({
                "model_id": version, "replay_partition": replay_part,
                "row_key_mismatch": int(mismatch[0]), "raw_score_mismatch": int(mismatch[1]),
                "predicted_rank_mismatch": int(mismatch[2]),
                "query_digest_physical_mismatch": int(formal_digest_fp != replay_digest_fp),
                "status": "PASS" if not any(int(value) for value in mismatch)
                                     and formal_digest_fp == replay_digest_fp else "FAIL",
            })
            manifests.append({"model_id": version, "feature_set": feature_set,
                              "num_iteration": iteration,
                              "row_count": sum(pq.ParquetFile(p).metadata.num_rows for p in predictions),
                              "query_count": len(query_rows), "physical_fingerprint": physical,
                              "raw_score_semantic_fingerprint": raw_fp,
                              "ranked_list_semantic_fingerprint": rank_fp,
                              "global_merkle_fingerprint": hashlib.sha256((raw_fp+rank_fp).encode()).hexdigest(),
                              "prediction_source_fingerprint": self.source_fp,
                              "binding_fingerprint": hashlib.sha256(binding.encode()).hexdigest(),
                              "nan_inf_violations": 0, "status": "FROZEN"})
        if any(row["status"] != "PASS" for row in replay_audit):
            raise Stage4B4Error("prediction replay identity failed")
        write_csv(self.config.artifact_dir / "prediction_fingerprint_manifest.csv", manifests, list(manifests[0]))
        write_csv(self.config.artifact_dir / "prediction_replay_audit.csv", replay_audit, list(replay_audit[0]))
        return manifests

    @staticmethod
    def _prediction_digest_semantic_fingerprint(path: Path) -> str:
        table = pq.read_table(path).sort_by([("snapshot_id", "ascending"), ("customer_key", "ascending")])
        digest = hashlib.sha256(b"stage-4b4-prediction-digest-part-v1\0")
        for snapshot, customer, rows, raw, ranked in zip(
            table["snapshot_id"].to_pylist(), table["customer_key"].to_pylist(),
            table["row_count"].to_pylist(), table["raw_score_digest"].to_pylist(),
            table["ranked_list_digest"].to_pylist(),
        ):
            digest.update(str(snapshot).encode("utf-8") + b"\0")
            digest.update(int(customer).to_bytes(8, "little", signed=True))
            digest.update(int(rows).to_bytes(4, "little", signed=False))
            digest.update(bytes.fromhex(raw) + bytes.fromhex(ranked))
        return digest.hexdigest()

    def run(self) -> dict[str, Any]:
        started = time.perf_counter()
        preflight = self.preflight()
        self.write_adapter_manifests()
        train = self.audit_train_groups()
        validation = self.materialize_validation_features()
        matrix_outputs = list(self.matrix_paths())
        matrix_inputs = sorted(self.config.resolve(self.upstream["train_optimizer_features"]).glob("*.parquet")) + \
                        sorted(self.config.resolve(self.upstream["train_optimizer"]).glob("*.parquet")) + \
                        [self.config.resolve(self.upstream["train_groups"])]
        self.run_task("training_matrix", self.build_training_matrix, matrix_outputs, matrix_inputs)
        sample = pq.read_table(self.config.resolve(self.upstream["train_optimizer_features"]) / "part_000.parquet").slice(0, 1000)
        assert_adapter_deterministic(self.adapters["F0"], sample)
        assert_adapter_deterministic(self.adapters["F1"], sample)
        models = []
        for feature_set, model_id in (("F0", "L0"), ("F1", "L1")):
            model_path = self.model_path(model_id)
            metadata_path = self.config.private_dir / "models" / f"{model_id}_metadata.json"
            result = self.run_task(f"train_{model_id}", lambda fs=feature_set: self.train_model(fs),
                                   [model_path, metadata_path], matrix_outputs)
            models.append(result or json.loads(metadata_path.read_text()))
        manifests = self.freeze_predictions(validation, models)
        lock = {**self.raw["access_lock"], "stage": "4B-4", "phase": "phase1_pre_label_prediction_freeze_v1",
                "validation_predictions_frozen": True, "prediction_versions": [row["model_id"] for row in manifests],
                "winner_selected": False, "final_refit_executed": False, "test_authorized": False,
                "lock_created_at": now(), "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fp}
        json_write_atomic(self.config.artifact_dir / "stage_4b4_pre_label_lock.json", lock)
        result = {"preflight": preflight, "train": train, "validation": validation, "models": models,
                  "predictions": manifests, "lock": lock, "wall_time_seconds": time.perf_counter()-started}
        json_write_atomic(self.config.private_dir / "run_result.json", result)
        return result
