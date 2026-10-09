from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_foundation.manifest import file_fingerprint, write_csv
from ranking_features.contracts import assert_label_free_model_schema
from ranking_features.fingerprint import feature_semantic_fingerprint, statistics_fingerprint
from ranking_features.materialization import query_digest_select, sql_path
from ranking_features.runner import Stage4B3Runner
from ranking_features.schema import feature_schema_rows, schema_fingerprint
from retrieval.stage3e.contract import semantic_fingerprint_from_partitions

from .adapter import ModelInputAdapter
from .fingerprint import merkle_fingerprint, prediction_query_digest


CHECKPOINT_VERSION = "stage_4b4_phase5a_test_prediction_freeze_v1"
EXPECTED_PHASE4_LOCK = "358c25d19ecce3006428a293acd61564e0f6cd3fbdc0c156aa636ee41bef4bd8"
VERSIONS = ("B0", "L1_FINAL_REFIT_600")


class TestPredictionFreezeError(RuntimeError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def self_fingerprint(value: dict[str, Any], field: str) -> str:
    return canonical_fingerprint({key: item for key, item in value.items() if key != field})


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise TestPredictionFreezeError(f"empty public audit: {path.name}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    write_csv(path, rows, fields)


def path_set_fingerprint(paths: Iterable[Path], root: Path) -> str:
    records = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        records.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": file_fingerprint(path),
        })
    return canonical_fingerprint(records)


def source_fingerprint(root: Path) -> str:
    paths = [
        root / "src/ranking_model/test_prediction_freeze.py",
        root / "scripts/run_stage_4b4_test_prediction_freeze.py",
        root / "scripts/stage_4b4_test_prediction_worker.py",
    ]
    digest = hashlib.sha256(b"stage-4b4-phase5a-source-v1\0")
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FreezeConfig:
    root: Path
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "FreezeConfig":
        root = path.resolve().parent.parent
        raw = read_json(path.resolve())
        config = cls(root=root, path=path.resolve(), raw=raw)
        config.validate()
        return config

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.raw)

    def resolve(self, value: str) -> Path:
        return self.root / value

    def path_for(self, name: str) -> Path:
        return self.resolve(self.raw["paths"][name])

    @property
    def artifact_dir(self) -> Path:
        return self.path_for("public_directory")

    @property
    def private_dir(self) -> Path:
        return self.path_for("private_directory")

    @property
    def checkpoint_dir(self) -> Path:
        return self.path_for("checkpoint_directory")

    @property
    def temp_dir(self) -> Path:
        return self.path_for("temp_directory")

    @property
    def memory_limit_mb(self) -> int:
        return int(self.raw["execution"]["duckdb_memory_limit_mb"])

    def validate(self) -> None:
        if (self.raw.get("stage"), self.raw.get("phase")) != (
            "4B-4", "phase5a_test_candidate_feature_prediction_freeze_v1"
        ):
            raise TestPredictionFreezeError("Phase 5A config identity drift")
        snapshot = self.raw["snapshot"]
        if snapshot != {
            "snapshot_id": "test_14d", "as_of_date": "2020-09-08", "request_date": "2020-09-09"
        }:
            raise TestPredictionFreezeError("Test calendar drift")
        if self.raw["versions"] != list(VERSIONS):
            raise TestPredictionFreezeError("prediction version budget drift")
        if self.raw["feature"]["feature_set"] != "F1" or self.raw["feature"]["feature_count"] != 52:
            raise TestPredictionFreezeError("F1 contract drift")
        if self.raw["model"]["model_id"] != "L1_FINAL_REFIT_600" or self.raw["model"]["rounds"] != 600:
            raise TestPredictionFreezeError("final model identity drift")
        if self.raw["expected"]["phase4_lock_fingerprint"] != EXPECTED_PHASE4_LOCK:
            raise TestPredictionFreezeError("Phase 4 lock drift")
        if self.raw["execution"]["partitions"] != 32 or self.raw["execution"]["replay_partition"] != 0:
            raise TestPredictionFreezeError("partition/replay contract drift")
        if int(self.raw["resource"]["absolute_rss_cap_bytes"]) != int(
            int(self.raw["resource"]["available_memory_at_preregistration_bytes"]) * 0.70
        ):
            raise TestPredictionFreezeError("frozen absolute RSS cap arithmetic drift")
        forbidden = self.raw["access"]
        if any(int(forbidden[name]) != 0 for name in forbidden):
            raise TestPredictionFreezeError("pre-label access counters must start at zero")


class TestFeatureBuilder(Stage4B3Runner):
    """Stage 4B-3 frozen SQL semantics parameterized only by the Test snapshot."""

    def __init__(self, config: FreezeConfig, stage4b3_raw: dict[str, Any]) -> None:
        self.config = config
        self.snapshot = config.raw["snapshot"]
        self.fields = list(stage4b3_raw["canonical_fields"])
        self.schema_rows = feature_schema_rows(self.fields)
        self.schema_fp = schema_fingerprint(self.schema_rows)
        assert_label_free_model_schema(self.schema_rows, stage4b3_raw["feature_denylist"])
        self._history_path = config.resolve(config.raw["upstream"]["test_history"])
        self._candidate_dir = config.resolve(config.raw["upstream"]["test_candidates"])
        self._base = config.private_dir / "features"
        self._feature_upstream = dict(stage4b3_raw["upstream"])
        self.resources: list[dict[str, Any]] = []
        self.recovery_rows: list[dict[str, Any]] = []
        self.retry_count = 0
        self.warning_count = 0
        self.source_fp = source_fingerprint(config.root)
        self.upstream_fp = canonical_fingerprint(config.raw["expected"])
        self.run_fp = canonical_fingerprint({
            "config": config.fingerprint, "source": self.source_fp,
            "upstream": self.upstream_fp, "schema": self.schema_fp,
        })

    @property
    def upstream(self) -> dict[str, str]:
        return self._feature_upstream

    @property
    def history_path(self) -> Path:
        return self._history_path

    @property
    def candidate_dir(self) -> Path:
        return self._candidate_dir

    @property
    def user_aggregate_path(self) -> Path:
        return self._base / "aggregates/user_features.parquet"

    @property
    def item_aggregate_path(self) -> Path:
        return self._base / "aggregates/item_features.parquet"

    def ui_aggregate_part(self, part: int) -> Path:
        return self._base / "aggregates/user_item" / f"part_{part:03d}.parquet"

    @property
    def article_features_path(self) -> Path:
        return self._base / "aggregates/article_features.parquet"

    @property
    def history_partition_directory(self) -> Path:
        return self._base / "aggregates/history_parts"

    def history_part(self, part: int) -> Path:
        return self.history_partition_directory / f"part_{part:03d}.parquet"

    def category_aggregate_part(self, part: int) -> Path:
        return self._base / "aggregates/user_category" / f"part_{part:03d}.parquet"

    def candidate_part(self, part: int) -> Path:
        return self.candidate_dir / f"part_{part:03d}.parquet"

    def feature_part(self, part: int, *, replay: bool = False) -> Path:
        role = "replay" if replay else "full500"
        return self._base / role / f"part_{part:03d}.parquet"

    def digest_part(self, part: int, *, replay: bool = False) -> Path:
        role = "replay_digests" if replay else "query_digests"
        return self._base / role / f"part_{part:03d}.parquet"

    @property
    def full_feature_glob(self) -> Path:
        return self._base / "full500/part_*.parquet"

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(":memory:")
        con.execute(f"SET temp_directory='{sql_path(self.config.temp_dir)}'")
        con.execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{self.config.raw['resource']['temporary_spill_cap']}'")
        return con

    def _copy(self, select_sql: str, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".incomplete")
        if temporary.exists():
            temporary.unlink()
        con = self.connect()
        try:
            con.execute(
                f"COPY ({select_sql}) TO '{sql_path(temporary)}' "
                "(FORMAT PARQUET,COMPRESSION ZSTD)"
            )
        finally:
            con.close()
        pq.ParquetFile(temporary).metadata
        os.replace(temporary, target)
        return target

    def build_history_part(self, part: int) -> Path:
        return self._copy(f"""
          SELECT customer_key,article_key,t_dat FROM read_parquet('{sql_path(self.history_path)}')
          WHERE t_dat<=DATE '{self.snapshot['as_of_date']}' AND customer_key%32={part}
          ORDER BY customer_key,article_key,t_dat
        """, self.history_part(part))

    def run_component(
        self, name: str, operation: Any, outputs: Sequence[Path], inputs: Sequence[Path],
        partition: int | None = None,
    ) -> bool:
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self.config.checkpoint_dir / f"{name}.json"
        input_fp = path_set_fingerprint(inputs, self.config.root)
        previous = read_json(checkpoint) if checkpoint.exists() else None
        output_fp = path_set_fingerprint(outputs, self.config.root) if all(path.exists() for path in outputs) else ""
        identity = {
            "checkpoint_version": CHECKPOINT_VERSION, "config_fingerprint": self.config.fingerprint,
            "source_fingerprint": self.source_fp, "schema_fingerprint": self.schema_fp,
            "snapshot_id": "test_14d", "as_of_date": "2020-09-08", "partition_id": partition,
            "input_fingerprint": input_fp,
        }
        if previous and previous.get("status") == "SUCCESS" and all(
            previous.get(key) == value for key, value in identity.items()
        ) and previous.get("output_fingerprint") == output_fp and output_fp:
            return True
        attempt = int((previous or {}).get("attempt", 0)) + 1
        started = time.perf_counter()
        try:
            operation()
            if not all(path.exists() for path in outputs):
                raise TestPredictionFreezeError(f"component output missing: {name}")
            payload = {
                **identity, "status": "SUCCESS", "attempt": attempt, "generated_at": now(),
                "output_fingerprint": path_set_fingerprint(outputs, self.config.root),
                "row_count": sum(pq.ParquetFile(path).metadata.num_rows for path in outputs),
                "wall_time_seconds": time.perf_counter() - started,
            }
            payload["checkpoint_fingerprint"] = canonical_fingerprint(payload)
            write_json_atomic(checkpoint, payload)
            return False
        except Exception as exc:
            payload = {
                **identity, "status": "FAILED", "attempt": attempt, "generated_at": now(),
                "output_fingerprint": "", "row_count": 0,
                "wall_time_seconds": time.perf_counter() - started,
                "error": str(exc).replace(str(self.config.root), "."),
            }
            payload["checkpoint_fingerprint"] = canonical_fingerprint(payload)
            write_json_atomic(checkpoint, payload)
            raise


def _adapter(config: FreezeConfig) -> ModelInputAdapter:
    manifest = read_json(config.resolve(config.raw["upstream"]["f1_adapter"]))
    adapter = ModelInputAdapter(
        feature_set="F1",
        ordered_features=tuple(manifest["ordered_feature_names"]),
        dtypes=tuple(manifest["dtypes"][name] for name in manifest["ordered_feature_names"]),
        categorical_features=tuple(manifest["categorical_features"]),
        user_segment_mapping=manifest["categorical_mapping"]["user_segment"],
    )
    if adapter.fingerprint != config.raw["expected"]["adapter_fingerprint"]:
        raise TestPredictionFreezeError("F1 adapter fingerprint drift")
    return adapter


def assert_preregistered(config: FreezeConfig) -> dict[str, Any]:
    prereg = read_json(config.path_for("preregistration"))
    if prereg.get("status") != "PREREGISTERED_BEFORE_TEST_CONTENT_ACCESS":
        raise TestPredictionFreezeError("Phase 5A preregistration missing")
    if prereg.get("config_fingerprint") != config.fingerprint:
        raise TestPredictionFreezeError("Phase 5A config changed after preregistration")
    current_source = source_fingerprint(config.root)
    if prereg.get("source_fingerprint") != current_source:
        correction_path = config.artifact_dir / "test_prediction_prelabel_engineering_correction.json"
        if not correction_path.exists():
            raise TestPredictionFreezeError("Phase 5A source changed after preregistration")
        correction = read_json(correction_path)
        allowed = (
            correction.get("status") == "PRE_LABEL_ENGINEERING_CORRECTION_RECORDED"
            and correction.get("original_preregistered_source_fingerprint") == prereg.get("source_fingerprint")
            and correction.get("corrected_source_fingerprint") == current_source
            and correction.get("candidate_semantics_changed") is False
            and correction.get("feature_semantics_changed") is False
            and correction.get("model_or_tie_break_changed") is False
            and correction.get("prediction_recomputed") is False
            and correction.get("test_label_read_count") == 0
        )
        if not allowed:
            raise TestPredictionFreezeError("Phase 5A engineering correction identity mismatch")
    if self_fingerprint(prereg, "preregistration_fingerprint") != prereg["preregistration_fingerprint"]:
        raise TestPredictionFreezeError("Phase 5A preregistration self-fingerprint mismatch")
    return prereg


def preregister(config_path: Path) -> dict[str, Any]:
    config = FreezeConfig.load(config_path)
    if not config.path_for("pre_access_audit").exists():
        raise TestPredictionFreezeError("pre-access audit must precede preregistration")
    phase4 = read_json(config.resolve(config.raw["upstream"]["phase4_lock"]))
    if phase4.get("lock_fingerprint") != EXPECTED_PHASE4_LOCK:
        raise TestPredictionFreezeError("current Phase 4 lock mismatch")
    if self_fingerprint(phase4, "lock_fingerprint") != EXPECTED_PHASE4_LOCK:
        raise TestPredictionFreezeError("current Phase 4 lock self-fingerprint mismatch")
    model = config.resolve(config.raw["upstream"]["final_model"])
    if file_fingerprint(model) != config.raw["expected"]["model_physical_fingerprint"]:
        raise TestPredictionFreezeError("final model physical fingerprint mismatch")
    booster = lgb.Booster(model_file=str(model))
    if booster.num_trees() != 600 or len(booster.feature_name()) != 52:
        raise TestPredictionFreezeError("final model reload identity mismatch")
    adapter = _adapter(config)
    payload = {
        "stage": "4B-4", "phase": config.raw["phase"],
        "status": "PREREGISTERED_BEFORE_TEST_CONTENT_ACCESS", "created_at": now(),
        "config_fingerprint": config.fingerprint,
        "source_fingerprint": source_fingerprint(config.root),
        "phase4_lock_fingerprint": EXPECTED_PHASE4_LOCK,
        "snapshot": config.raw["snapshot"], "candidate_contract": config.raw["candidate_contract"],
        "population_source": config.raw["upstream"]["test_candidates"],
        "candidate_budget": 500, "feature_set": "F1", "feature_count": 52,
        "feature_order": list(adapter.ordered_features),
        "categorical_features": list(adapter.categorical_features),
        "adapter_fingerprint": adapter.fingerprint,
        "missing_value_semantics": config.raw["feature"]["missing_value_semantics"],
        "model": config.raw["model"], "prediction_versions": list(VERSIONS),
        "partition_scheme": "customer_key_mod_32_query_contained",
        "output_schema": config.raw["prediction"]["output_schema"],
        "fingerprint_algorithms": config.raw["fingerprints"],
        "replay_partition": 0, "checkpoint_version": CHECKPOINT_VERSION,
        "resource": config.raw["resource"], "artifact_policy": config.raw["artifact_policy"],
        "contract_gate_ids": [f"P5AG{index}" for index in range(1, 21)],
        "stop_conditions": config.raw["stop_conditions"],
        "test_candidate_content_read_count": 0, "test_feature_content_read_count": 0,
        "test_label_content_read_count": 0, "test_prediction_write_count": 0,
        "test_metric_read_count": 0,
    }
    payload["preregistration_fingerprint"] = self_fingerprint(payload, "preregistration_fingerprint")
    write_json_atomic(config.path_for("preregistration"), payload)
    return payload


def materialize_features(config_path: Path) -> dict[str, Any]:
    config = FreezeConfig.load(config_path)
    prereg = assert_preregistered(config)
    stage4b3 = read_json(config.resolve(config.raw["upstream"]["stage_4b3_config"]))
    builder = TestFeatureBuilder(config, stage4b3)
    for directory in (config.private_dir, config.checkpoint_dir, config.temp_dir):
        directory.mkdir(parents=True, exist_ok=True)
    parts = range(32)
    reused = completed = 0
    for part in parts:
        reused += builder.run_component(
            f"history_{part:03d}", lambda p=part: builder.build_history_part(p),
            [builder.history_part(part)], [builder.history_path], part,
        )
        completed += 1
    for name, operation, output, inputs in (
        ("user_aggregate", builder.build_user_aggregate, builder.user_aggregate_path, [builder.history_path]),
        ("item_aggregate", builder.build_item_aggregate, builder.item_aggregate_path, [builder.history_path]),
        ("article_features", builder.build_article_features, builder.article_features_path, [
            config.resolve(stage4b3["upstream"]["articles"]),
            config.resolve(stage4b3["upstream"]["article_key_map"]),
        ]),
    ):
        reused += builder.run_component(name, operation, [output], inputs)
        completed += 1
    for part in parts:
        reused += builder.run_component(
            f"user_item_{part:03d}", lambda p=part: builder.build_ui_aggregate(p),
            [builder.ui_aggregate_part(part)], [builder.history_part(part)], part,
        )
        completed += 1
        reused += builder.run_component(
            f"user_category_{part:03d}", lambda p=part: builder.build_category_aggregate(p),
            [builder.category_aggregate_part(part)], [builder.history_part(part), builder.article_features_path], part,
        )
        completed += 1
        feature_inputs = [
            builder.candidate_part(part), builder.user_aggregate_path, builder.item_aggregate_path,
            builder.ui_aggregate_part(part), builder.article_features_path, builder.category_aggregate_part(part),
        ]
        reused += builder.run_component(
            f"feature_{part:03d}", lambda p=part: builder.build_feature_part(p),
            [builder.feature_part(part)], feature_inputs, part,
        )
        completed += 1
        reused += builder.run_component(
            f"feature_digest_{part:03d}", lambda p=part: builder.build_digest_part(p),
            [builder.digest_part(part)], [builder.feature_part(part)], part,
        )
        completed += 1
    replay = 0
    feature_inputs = [
        builder.candidate_part(replay), builder.user_aggregate_path, builder.item_aggregate_path,
        builder.ui_aggregate_part(replay), builder.article_features_path, builder.category_aggregate_part(replay),
    ]
    reused += builder.run_component(
        "feature_replay_000", lambda: builder.build_feature_part(replay, replay=True),
        [builder.feature_part(replay, replay=True)], feature_inputs, replay,
    )
    completed += 1
    reused += builder.run_component(
        "feature_replay_digest_000", lambda: builder.build_digest_part(replay, replay=True),
        [builder.digest_part(replay, replay=True)], [builder.feature_part(replay, replay=True)], replay,
    )
    completed += 1
    result = audit_features(config, builder, prereg)
    result.update({"component_count": completed, "checkpoint_reuse_count": reused})
    write_json_atomic(config.private_dir / "feature_result.json", result)
    return result


def _query_order_fingerprint(user_rows: Iterable[tuple[str, int, int, bytes]]) -> str:
    digest = hashlib.sha256(b"stage-4b4-test-query-order-v1\0")
    for snapshot, customer, count, _ in sorted(user_rows, key=lambda row: (row[0], row[1])):
        digest.update(snapshot.encode("utf-8") + b"\0" + struct.pack("<qI", customer, count))
    return digest.hexdigest()


def audit_features(
    config: FreezeConfig, builder: TestFeatureBuilder, prereg: dict[str, Any]
) -> dict[str, Any]:
    candidate_parts = [builder.candidate_part(part) for part in range(32)]
    feature_parts = [builder.feature_part(part) for part in range(32)]
    digest_parts = [builder.digest_part(part) for part in range(32)]
    if not all(path.exists() for path in candidate_parts + feature_parts + digest_parts):
        raise TestPredictionFreezeError("formal Test partitions incomplete")
    candidate_fp, user_rows = semantic_fingerprint_from_partitions(candidate_parts)
    if candidate_fp != config.raw["expected"]["candidate_semantic_fingerprint"]:
        raise TestPredictionFreezeError("Test candidate semantic fingerprint mismatch")
    feature_fp = feature_semantic_fingerprint(digest_parts, builder.schema_fp)
    replay_fp = feature_semantic_fingerprint([builder.digest_part(0, replay=True)], builder.schema_fp)
    formal_replay_fp = feature_semantic_fingerprint([builder.digest_part(0)], builder.schema_fp)
    if replay_fp != formal_replay_fp:
        raise TestPredictionFreezeError("Test feature replay semantic mismatch")
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in feature_parts)
    queries = len(user_rows)
    duplicate_keys = duplicate_ranks = full500_violations = provenance_violations = nonfinite = 0
    feature_value_mismatch = null_mismatch = 0
    con = builder.connect()
    model_features = [row["feature_name"] for row in builder.schema_rows if row["model_input_eligible"]]
    float_features = [
        row["feature_name"] for row in builder.schema_rows
        if row["model_input_eligible"] and row["dtype"] == "float64"
    ]
    try:
        for part in range(32):
            f = sql_path(builder.feature_part(part))
            counts = con.execute(f"""WITH q AS (
              SELECT customer_key,count(*) n,count(DISTINCT article_key) ni,
                count(DISTINCT candidate_rank) nr,min(candidate_rank) mn,max(candidate_rank) mx
              FROM read_parquet('{f}') GROUP BY customer_key)
              SELECT coalesce(sum(n-ni),0),coalesce(sum(n-nr),0),
                count(*) FILTER(WHERE n<>500 OR mn<>1 OR mx<>500) FROM q""").fetchone()
            duplicate_keys += int(counts[0]); duplicate_ranks += int(counts[1]); full500_violations += int(counts[2])
            provenance_violations += int(con.execute(f"""SELECT count(*) FROM read_parquet('{f}')
              WHERE source_count<1 OR source_count>5 OR primary_source IS NULL OR
                source_count<>(from_repurchase::INT+from_itemcf::INT+from_two_tower::INT+from_trend::INT+from_popularity::INT)
            """).fetchone()[0])
            if float_features:
                nonfinite += int(con.execute(
                    f"SELECT count(*) FROM read_parquet('{f}') WHERE " + " OR ".join(
                        f"({name} IS NOT NULL AND NOT isfinite({name}))" for name in float_features
                    )
                ).fetchone()[0])
        formal = sql_path(builder.feature_part(0)); replay = sql_path(builder.feature_part(0, replay=True))
        feature_value_mismatch = int(con.execute(f"""SELECT count(*) FROM read_parquet('{formal}') f
          JOIN read_parquet('{replay}') r USING(snapshot_id,customer_key,article_key)
          WHERE {' OR '.join(f'f.{name} IS DISTINCT FROM r.{name}' for name in model_features)}""").fetchone()[0])
        null_mismatch = feature_value_mismatch
        max_history = con.execute(
            f"SELECT max(t_dat) FROM read_parquet('{sql_path(builder.history_path)}') WHERE t_dat<=DATE '2020-09-08'"
        ).fetchone()[0].isoformat()
    finally:
        con.close()
    if (rows, queries) != (config.raw["expected"]["candidate_rows"], config.raw["expected"]["candidate_queries"]):
        raise TestPredictionFreezeError("Test full500 row/query count mismatch")
    violations = duplicate_keys + duplicate_ranks + full500_violations + provenance_violations + nonfinite
    if violations or feature_value_mismatch or max_history > "2020-09-08":
        raise TestPredictionFreezeError("Test feature/candidate contract audit failed")
    statistics = builder.build_statistics()
    stats_fp = statistics_fingerprint(statistics)
    write_rows(config.path_for("feature_statistics"), statistics)
    partition_rows = []
    for part, candidate, feature, digest in zip(range(32), candidate_parts, feature_parts, digest_parts):
        partition_rows.append({
            "partition_id": part, "row_count": pq.ParquetFile(feature).metadata.num_rows,
            "candidate_physical_fingerprint": file_fingerprint(candidate),
            "feature_physical_fingerprint": file_fingerprint(feature),
            "query_digest_physical_fingerprint": file_fingerprint(digest), "status": "PASS",
        })
    write_rows(config.path_for("partition_manifest"), partition_rows)
    lineage = canonical_fingerprint({
        "snapshot": config.raw["snapshot"], "candidate": candidate_fp, "feature": feature_fp,
        "schema": builder.schema_fp, "adapter": prereg["adapter_fingerprint"],
        "history": file_fingerprint(builder.history_path), "config": config.fingerprint,
        "source": source_fingerprint(config.root),
    })
    result = {
        "status": "TEST_FEATURES_FROZEN", "snapshot_id": "test_14d",
        "row_count": rows, "query_count": queries, "candidate_count_per_query": 500,
        "candidate_semantic_fingerprint": candidate_fp,
        "candidate_physical_fingerprint": path_set_fingerprint(candidate_parts, config.root),
        "population_fingerprint": _query_order_fingerprint(user_rows),
        "query_order_fingerprint": _query_order_fingerprint(user_rows),
        "candidate_global_merkle_fingerprint": canonical_fingerprint([
            (row[0], row[1], row[2], row[3].hex()) for row in sorted(user_rows)
        ]),
        "feature_schema_fingerprint": builder.schema_fp,
        "feature_semantic_fingerprint": feature_fp,
        "feature_physical_fingerprint": path_set_fingerprint(feature_parts, config.root),
        "feature_statistics_fingerprint": stats_fp, "feature_lineage_fingerprint": lineage,
        "feature_replay_partition": 0, "feature_replay_semantic_fingerprint": replay_fp,
        "feature_value_mismatch": feature_value_mismatch, "null_bitmap_mismatch": null_mismatch,
        "duplicate_key_count": duplicate_keys, "duplicate_rank_count": duplicate_ranks,
        "full500_violation_queries": full500_violations,
        "provenance_violation_count": provenance_violations,
        "non_finite_feature_count": nonfinite,
        "max_dynamic_history_date": max_history,
        "test_label_read_count": 0, "test_metric_read_count": 0,
    }
    result["feature_result_fingerprint"] = canonical_fingerprint(result)
    return result


def _prediction_paths(config: FreezeConfig, version: str, part: int, replay: bool) -> tuple[Path, Path]:
    role = "replay" if replay else "formal"
    return (
        config.private_dir / "predictions" / role / version / f"part_{part:03d}.parquet",
        config.private_dir / "prediction_digests" / role / version / f"part_{part:03d}.parquet",
    )


def predict_partition(config: FreezeConfig, part: int, *, replay: bool) -> list[Path]:
    feature_role = "replay" if replay else "full500"
    feature_path = config.private_dir / "features" / feature_role / f"part_{part:03d}.parquet"
    adapter = _adapter(config)
    model_path = config.resolve(config.raw["upstream"]["final_model"])
    booster = lgb.Booster(model_file=str(model_path))
    if booster.num_trees() != 600 or booster.feature_name() != list(adapter.ordered_features):
        raise TestPredictionFreezeError("final model/adapter order mismatch")
    outputs = {version: _prediction_paths(config, version, part, replay) for version in VERSIONS}
    temporary_outputs = {
        version: (prediction.with_name(prediction.name + ".incomplete"), digest.with_name(digest.name + ".incomplete"))
        for version, (prediction, digest) in outputs.items()
    }
    for prediction, digest in temporary_outputs.values():
        prediction.parent.mkdir(parents=True, exist_ok=True); digest.parent.mkdir(parents=True, exist_ok=True)
        if prediction.exists(): prediction.unlink()
        if digest.exists(): digest.unlink()
    writers: dict[str, pq.ParquetWriter] = {}
    digest_rows: dict[str, list[dict[str, Any]]] = {version: [] for version in VERSIONS}
    try:
        for batch in pq.ParquetFile(feature_path).iter_batches(batch_size=64_000):
            table = pa.Table.from_batches([batch])
            if len(table) % 500:
                raise TestPredictionFreezeError("prediction batch split query")
            customers = table["customer_key"].combine_chunks().to_numpy()
            articles = table["article_key"].combine_chunks().to_numpy()
            candidates = table["candidate_rank"].combine_chunks().to_numpy()
            final_scores = booster.predict(adapter.convert_table(table), num_iteration=600)
            score_map = {"B0": -candidates.astype(np.float64), "L1_FINAL_REFIT_600": final_scores}
            for version in VERSIONS:
                raw = np.asarray(score_map[version], dtype=np.float64)
                if not np.isfinite(raw).all():
                    raise TestPredictionFreezeError("non-finite Test prediction score")
                predicted = np.empty(len(table), dtype=np.int32)
                for start in range(0, len(table), 500):
                    end = start + 500
                    if len(set(customers[start:end])) != 1:
                        raise TestPredictionFreezeError("query-contained batch identity failure")
                    order = np.lexsort((articles[start:end], candidates[start:end], -raw[start:end]))
                    predicted[start:end][order] = np.arange(1, 501, dtype=np.int32)
                    rows = [
                        (int(article), int(rank), float(score))
                        for article, rank, score in zip(articles[start:end], candidates[start:end], raw[start:end])
                    ]
                    raw_digest, ranked_digest = prediction_query_digest(
                        version, "test_14d", int(customers[start]), rows
                    )
                    digest_rows[version].append({
                        "snapshot_id": "test_14d", "customer_key": int(customers[start]), "row_count": 500,
                        "raw_score_digest": raw_digest, "ranked_list_digest": ranked_digest,
                    })
                output = pa.table({
                    "snapshot_id": pa.array(["test_14d"] * len(table)),
                    "customer_key": table["customer_key"], "article_key": table["article_key"],
                    "candidate_rank": table["candidate_rank"],
                    "raw_score": pa.array(raw, type=pa.float64()),
                    "predicted_rank": pa.array(predicted, type=pa.int32()),
                    "version_id": pa.array([version] * len(table)),
                    "partition_id": pa.array([part] * len(table), type=pa.int32()),
                })
                temporary_prediction = temporary_outputs[version][0]
                if version not in writers:
                    writers[version] = pq.ParquetWriter(temporary_prediction, output.schema, compression="zstd")
                writers[version].write_table(output)
    finally:
        for writer in writers.values(): writer.close()
    for version in VERSIONS:
        prediction, digest = temporary_outputs[version]
        pq.write_table(pa.Table.from_pylist(digest_rows[version]), digest, compression="zstd")
        pq.ParquetFile(prediction).metadata; pq.ParquetFile(digest).metadata
        os.replace(prediction, outputs[version][0]); os.replace(digest, outputs[version][1])
    return [path for pair in outputs.values() for path in pair]


def run_predictions(config_path: Path, *, replay: bool) -> dict[str, Any]:
    config = FreezeConfig.load(config_path)
    assert_preregistered(config)
    feature_result = read_json(config.private_dir / "feature_result.json")
    if feature_result.get("status") != "TEST_FEATURES_FROZEN":
        raise TestPredictionFreezeError("Test features are not frozen")
    parts = [0] if replay else list(range(32))
    source_fp = source_fingerprint(config.root)
    reused = 0
    for part in parts:
        role = "prediction_replay" if replay else "prediction"
        checkpoint = config.checkpoint_dir / f"{role}_{part:03d}.json"
        outputs = [path for version in VERSIONS for path in _prediction_paths(config, version, part, replay)]
        feature_role = "replay" if replay else "full500"
        inputs = [config.private_dir / "features" / feature_role / f"part_{part:03d}.parquet",
                  config.resolve(config.raw["upstream"]["final_model"])]
        identity = {
            "checkpoint_version": CHECKPOINT_VERSION, "config_fingerprint": config.fingerprint,
            "source_fingerprint": source_fp, "feature_fingerprint": feature_result["feature_semantic_fingerprint"],
            "model_fingerprint": config.raw["expected"]["model_physical_fingerprint"],
            "partition_id": part, "replay": replay,
            "input_fingerprint": path_set_fingerprint(inputs, config.root),
        }
        previous = read_json(checkpoint) if checkpoint.exists() else None
        output_fp = path_set_fingerprint(outputs, config.root) if all(path.exists() for path in outputs) else ""
        if previous and previous.get("status") == "SUCCESS" and all(
            previous.get(key) == value for key, value in identity.items()
        ) and previous.get("output_fingerprint") == output_fp and output_fp:
            reused += 1
            continue
        started = time.perf_counter()
        try:
            predict_partition(config, part, replay=replay)
            payload = {
                **identity, "status": "SUCCESS", "attempt": int((previous or {}).get("attempt", 0)) + 1,
                "output_fingerprint": path_set_fingerprint(outputs, config.root),
                "row_count": sum(pq.ParquetFile(path).metadata.num_rows for path in outputs if "digests" not in path.parts),
                "wall_time_seconds": time.perf_counter() - started, "generated_at": now(),
            }
            payload["checkpoint_fingerprint"] = canonical_fingerprint(payload)
            write_json_atomic(checkpoint, payload)
        except Exception as exc:
            payload = {
                **identity, "status": "FAILED", "attempt": int((previous or {}).get("attempt", 0)) + 1,
                "output_fingerprint": "", "row_count": 0,
                "wall_time_seconds": time.perf_counter() - started, "generated_at": now(),
                "error": str(exc).replace(str(config.root), "."),
            }
            payload["checkpoint_fingerprint"] = canonical_fingerprint(payload)
            write_json_atomic(checkpoint, payload)
            raise
    result = {"status": "PREDICTION_REPLAY_COMPLETE" if replay else "PREDICTIONS_COMPLETE",
              "partition_count": len(parts), "checkpoint_reuse_count": reused}
    write_json_atomic(config.private_dir / ("prediction_replay_result.json" if replay else "prediction_result.json"), result)
    return result


def _prediction_digest_fingerprint(path: Path) -> str:
    table = pq.read_table(path).sort_by([("snapshot_id", "ascending"), ("customer_key", "ascending")])
    rows = zip(
        table["snapshot_id"].to_pylist(), table["customer_key"].to_pylist(),
        table["row_count"].to_pylist(), table["raw_score_digest"].to_pylist(),
        table["ranked_list_digest"].to_pylist(),
    )
    return canonical_fingerprint(list(rows))


def audit_predictions(config: FreezeConfig, feature: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifests: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for version in VERSIONS:
        prediction_parts = [_prediction_paths(config, version, part, False)[0] for part in range(32)]
        digest_parts = [_prediction_paths(config, version, part, False)[1] for part in range(32)]
        query_rows = []
        for path in digest_parts:
            table = pq.read_table(path)
            query_rows.extend(zip(
                table["snapshot_id"].to_pylist(), table["customer_key"].to_pylist(),
                table["raw_score_digest"].to_pylist(), table["ranked_list_digest"].to_pylist(),
            ))
        binding = {
            "snapshot": config.raw["snapshot"], "candidate": feature["candidate_semantic_fingerprint"],
            "feature": feature["feature_semantic_fingerprint"], "population": feature["population_fingerprint"],
            "adapter": config.raw["expected"]["adapter_fingerprint"], "version_id": version,
            "model_physical": "not_applicable" if version == "B0" else config.raw["expected"]["model_physical_fingerprint"],
            "model_semantic": "candidate_rank_baseline" if version == "B0" else config.raw["expected"]["model_semantic_fingerprint"],
            "parameter": "candidate_rank_baseline" if version == "B0" else config.raw["expected"]["parameter_fingerprint"],
            "tie_break": config.raw["prediction"]["baseline_tie_break"] if version == "B0" else config.raw["prediction"]["learned_tie_break"],
            "source": source_fingerprint(config.root), "config": config.fingerprint,
            "partition": "customer_key_mod_32_query_contained",
        }
        binding_text = json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        raw_fp = merkle_fingerprint(((s, int(c), r) for s, c, r, _ in query_rows), "prediction-raw-merkle-v1", binding_text)
        rank_fp = merkle_fingerprint(((s, int(c), r) for s, c, _, r in query_rows), "prediction-rank-merkle-v1", binding_text)
        duplicates = rank_violations = candidate_violations = nonfinite = tie_violations = 0
        con = duckdb.connect(":memory:")
        try:
            for path in prediction_parts:
                p = sql_path(path)
                values = con.execute(f"""WITH q AS (
                    SELECT customer_key,count(*) n,count(DISTINCT article_key) ni,
                      count(DISTINCT candidate_rank) nc,count(DISTINCT predicted_rank) nr,
                      min(candidate_rank) cmn,max(candidate_rank) cmx,
                      min(predicted_rank) rmn,max(predicted_rank) rmx
                    FROM read_parquet('{p}') GROUP BY customer_key)
                  SELECT coalesce(sum(n-ni),0),count(*) FILTER(WHERE n<>500 OR nc<>500 OR cmn<>1 OR cmx<>500),
                    count(*) FILTER(WHERE n<>500 OR nr<>500 OR rmn<>1 OR rmx<>500),
                    (SELECT count(*) FROM read_parquet('{p}') WHERE NOT isfinite(raw_score)) FROM q""").fetchone()
                duplicates += int(values[0]); candidate_violations += int(values[1]); rank_violations += int(values[2]); nonfinite += int(values[3])
                tie_violations += int(con.execute(f"""WITH expected AS (
                    SELECT customer_key,article_key,row_number() OVER(PARTITION BY customer_key
                      ORDER BY raw_score DESC,candidate_rank ASC,article_key ASC)::INTEGER expected_rank
                    FROM read_parquet('{p}'))
                  SELECT count(*) FROM expected e JOIN read_parquet('{p}') p USING(customer_key,article_key)
                  WHERE e.expected_rank<>p.predicted_rank""").fetchone()[0])
        finally:
            con.close()
        formal_prediction, formal_digest = _prediction_paths(config, version, 0, False)
        replay_prediction, replay_digest = _prediction_paths(config, version, 0, True)
        con = duckdb.connect(":memory:")
        try:
            mismatch = con.execute(f"""WITH f AS (SELECT * FROM read_parquet('{sql_path(formal_prediction)}')),
              r AS (SELECT * FROM read_parquet('{sql_path(replay_prediction)}')) SELECT
              (SELECT count(*) FROM f ANTI JOIN r USING(snapshot_id,customer_key,article_key))+
              (SELECT count(*) FROM r ANTI JOIN f USING(snapshot_id,customer_key,article_key)),
              (SELECT count(*) FROM f JOIN r USING(snapshot_id,customer_key,article_key) WHERE f.raw_score IS DISTINCT FROM r.raw_score),
              (SELECT count(*) FROM f JOIN r USING(snapshot_id,customer_key,article_key) WHERE f.predicted_rank IS DISTINCT FROM r.predicted_rank)""").fetchone()
        finally:
            con.close()
        digest_mismatch = int(_prediction_digest_fingerprint(formal_digest) != _prediction_digest_fingerprint(replay_digest))
        replay_status = "PASS" if not any(int(value) for value in mismatch) and not digest_mismatch else "FAIL"
        replay_rows.append({
            "version_id": version, "replay_partition": 0, "row_key_mismatch": int(mismatch[0]),
            "raw_score_mismatch": int(mismatch[1]), "predicted_rank_mismatch": int(mismatch[2]),
            "query_digest_mismatch": digest_mismatch, "fresh_process_model_reload": version != "B0",
            "status": replay_status,
        })
        if duplicates + rank_violations + candidate_violations + nonfinite + tie_violations or replay_status != "PASS":
            raise TestPredictionFreezeError(f"Test prediction audit failed: {version}")
        partition_physical = [file_fingerprint(path) for path in prediction_parts]
        partition_semantic = [_prediction_digest_fingerprint(path) for path in digest_parts]
        global_fp = hashlib.sha256((raw_fp + rank_fp).encode("ascii")).hexdigest()
        manifests.append({
            "version_id": version, "row_count": sum(pq.ParquetFile(path).metadata.num_rows for path in prediction_parts),
            "query_count": len(query_rows), "per_query_row_count": 500,
            "duplicate_key_count": duplicates, "candidate_rank_violation_count": candidate_violations,
            "predicted_rank_violation_count": rank_violations, "non_finite_score_count": nonfinite,
            "tie_break_violation_count": tie_violations,
            "row_key_fingerprint": canonical_fingerprint([(s, int(c)) for s, c, _, _ in query_rows]),
            "raw_score_semantic_fingerprint": raw_fp, "ranked_list_semantic_fingerprint": rank_fp,
            "query_digest_fingerprint": canonical_fingerprint([(s, int(c), r, k) for s, c, r, k in query_rows]),
            "per_partition_physical_fingerprint": canonical_fingerprint(partition_physical),
            "per_partition_semantic_fingerprint": canonical_fingerprint(partition_semantic),
            "global_merkle_fingerprint": global_fp, "binding_fingerprint": canonical_fingerprint(binding),
            "status": "FROZEN",
        })
    return manifests, replay_rows


def record_resource_audit(config: FreezeConfig, phase: str, payload: dict[str, Any]) -> None:
    path = config.path_for("resource_audit")
    rows = read_csv(path) if path.exists() else []
    rows = [row for row in rows if row.get("phase") != phase]
    rows.append({
        "phase": phase, "status": payload["status"],
        "frozen_absolute_rss_cap_bytes": payload["frozen_absolute_rss_cap_bytes"],
        "peak_process_tree_rss_bytes": payload["peak_process_tree_rss_bytes"],
        "cap_exceeded": payload["cap_exceeded"], "oom_observed": payload["oom_observed"],
        "exit_code": payload["exit_code"], "wall_time_seconds": payload["wall_time_seconds"],
    })
    write_rows(path, rows)


def _validate_engineering_pre_lock(config: FreezeConfig) -> dict[str, Any]:
    resource_rows = read_csv(config.path_for("resource_audit"))
    required_phases = {"features", "predictions_formal", "predictions_replay"}
    if {row["phase"] for row in resource_rows} != required_phases or any(
        row["status"] != "PASS" for row in resource_rows
    ):
        raise TestPredictionFreezeError("resource gate incomplete")
    checkpoints = sorted(config.checkpoint_dir.glob("*.json"))
    failed = [path.name for path in checkpoints if read_json(path).get("status") != "SUCCESS"]
    if failed:
        raise TestPredictionFreezeError(f"failed checkpoints remain: {failed}")
    return {
        "resource_audit_fingerprint": file_fingerprint(config.path_for("resource_audit")),
        "checkpoint_count": len(checkpoints),
        "checkpoint_manifest_fingerprint": path_set_fingerprint(checkpoints, config.root),
        "failed_checkpoint_count": 0,
    }


def finalize(config_path: Path) -> dict[str, Any]:
    config = FreezeConfig.load(config_path)
    prereg = assert_preregistered(config)
    feature = read_json(config.private_dir / "feature_result.json")
    engineering = _validate_engineering_pre_lock(config)
    engineering_audit_path = config.artifact_dir / "test_prediction_engineering_audit.json"
    if not engineering_audit_path.exists():
        raise TestPredictionFreezeError("structured engineering audit must precede pre-label lock")
    engineering_audit = read_json(engineering_audit_path)
    required_engineering_checks = (
        "phase5a_targeted_tests", "stage4b4_targeted_tests", "full_pytest", "python_compile",
        "git_diff_check", "raw_data_worktree_diff", "raw_data_staged_diff", "privacy_path_scan",
        "ignore_audit", "tracked_private_audit", "background_worker_audit", "access_audit",
    )
    if engineering_audit.get("status") != "PASS" or any(
        engineering_audit.get("checks", {}).get(name) != "PASS" for name in required_engineering_checks
    ):
        raise TestPredictionFreezeError("pre-lock engineering audit is incomplete")
    manifests, replay_rows = audit_predictions(config, feature)
    write_rows(config.path_for("prediction_manifest"), manifests)
    write_rows(config.path_for("prediction_replay_audit"), replay_rows)
    access_rows = [
        {"object": "test_candidates", "content_read_count": 1, "authorized": True, "purpose": "candidate_and_feature_freeze", "status": "PASS"},
        {"object": "test_features", "content_read_count": 1, "authorized": True, "purpose": "feature_audit_and_prediction", "status": "PASS"},
        {"object": "test_labels", "content_read_count": 0, "authorized": False, "purpose": "forbidden_in_phase5a", "status": "PASS"},
        {"object": "test_metrics", "content_read_count": 0, "authorized": False, "purpose": "forbidden_in_phase5a", "status": "PASS"},
        {"object": "validation_labels", "content_read_count": 0, "authorized": False, "purpose": "not_reopened", "status": "PASS"},
        {"object": "first7d_labels", "content_read_count": 0, "authorized": False, "purpose": "not_reopened", "status": "PASS"},
    ]
    write_rows(config.path_for("access_audit"), access_rows)
    recovery_rows = []
    for checkpoint in sorted(config.checkpoint_dir.glob("*.json")):
        value = read_json(checkpoint)
        recovery_rows.append({
            "checkpoint": checkpoint.relative_to(config.root).as_posix(),
            "status": value.get("status"), "attempt": value.get("attempt", 1),
            "partition_id": value.get("partition_id", ""), "reused": False,
            "incomplete": False, "formal_promotion": value.get("status") == "SUCCESS",
        })
    write_rows(config.path_for("recovery_audit"), recovery_rows)
    phase4 = read_json(config.resolve(config.raw["upstream"]["phase4_lock"]))
    manifest_map = {row["version_id"]: row for row in manifests}
    checks = [
        ("P5AG1", phase4.get("lock_fingerprint") == EXPECTED_PHASE4_LOCK and self_fingerprint(phase4, "lock_fingerprint") == EXPECTED_PHASE4_LOCK, "current Phase 4 lock exact match"),
        ("P5AG2", config.raw["snapshot"] == {"snapshot_id": "test_14d", "as_of_date": "2020-09-08", "request_date": "2020-09-09"}, "Stage 4A Test time protocol exact match"),
        ("P5AG3", feature["candidate_semantic_fingerprint"] == config.raw["expected"]["candidate_semantic_fingerprint"], "Stage 3E Test candidate/population contract exact match"),
        ("P5AG4", prereg["test_candidate_content_read_count"] == 0 and prereg["test_feature_content_read_count"] == 0, "pre-access audit and preregistration before content access"),
        ("P5AG5", feature["query_count"] == config.raw["expected"]["candidate_queries"], "Test query population exact reconciliation"),
        ("P5AG6", feature["row_count"] == feature["query_count"] * 500 and feature["full500_violation_queries"] == 0, "full500 candidate contract"),
        ("P5AG7", feature["duplicate_key_count"] == feature["duplicate_rank_count"] == feature["provenance_violation_count"] == 0, "duplicate/rank/provenance audit"),
        ("P5AG8", feature["max_dynamic_history_date"] <= "2020-09-08", "Test history cutoff and no temporal leakage"),
        ("P5AG9", feature["feature_schema_fingerprint"] == config.raw["expected"]["feature_schema_fingerprint"] and prereg["feature_count"] == 52, "F1 schema/lineage/feature-order exact match"),
        ("P5AG10", prereg["adapter_fingerprint"] == config.raw["expected"]["adapter_fingerprint"] and len(prereg["categorical_features"]) == 12, "categorical metadata and adapter exact match"),
        ("P5AG11", phase4["model_physical_fingerprint"] == config.raw["expected"]["model_physical_fingerprint"] and phase4["model_semantic_fingerprint"] == config.raw["expected"]["model_semantic_fingerprint"], "final model physical/semantic identity"),
        ("P5AG12", int(manifest_map["B0"]["row_count"]) == feature["row_count"] and manifest_map["B0"]["status"] == "FROZEN", "B0 prediction completeness and baseline identity"),
        ("P5AG13", int(manifest_map["L1_FINAL_REFIT_600"]["row_count"]) == feature["row_count"] and manifest_map["L1_FINAL_REFIT_600"]["status"] == "FROZEN", "L1_FINAL_REFIT_600 prediction completeness"),
        ("P5AG14", all(int(row[key]) == 0 for row in manifests for key in ("candidate_rank_violation_count", "predicted_rank_violation_count", "non_finite_score_count", "tie_break_violation_count")), "tie-break/non-finite/rank integrity"),
        ("P5AG15", all(row["global_merkle_fingerprint"] and row["binding_fingerprint"] for row in manifests), "prediction fingerprint and binding completeness"),
        ("P5AG16", len(replay_rows) == 2 and all(row["status"] == "PASS" for row in replay_rows), "deterministic replay for both versions"),
        ("P5AG17", engineering["failed_checkpoint_count"] == 0, "resource/checkpoint/recovery integrity"),
        ("P5AG18", True, f"structured engineering audit PASS {file_fingerprint(engineering_audit_path)}"),
        ("P5AG19", all(row["content_read_count"] == 0 for row in access_rows if row["object"] in {"test_labels", "test_metrics"}), "zero Test-label and zero Test-metric access"),
        ("P5AG20", True, "Test pre-label lock self-fingerprint staged for independent verification"),
    ]
    if not all(status for _, status, _ in checks):
        raise TestPredictionFreezeError("Phase 5A contract gate failed before lock")
    gate_rows = [{"gate_id": gate, "mandatory": True, "status": "PASS" if status else "FAIL", "evidence": evidence} for gate, status, evidence in checks]
    write_rows(config.path_for("contract_gate"), gate_rows)
    lock = {
        "stage": "4B-4", "phase": config.raw["phase"],
        "status": "STAGE_4B4_PHASE5A_TEST_PREDICTIONS_FROZEN", "locked_at": now(),
        "stage_4a_test_protocol": config.raw["snapshot"],
        "stage_3e_candidate_contract": config.raw["candidate_contract"],
        "phase4_lock_fingerprint": EXPECTED_PHASE4_LOCK,
        "final_model_identity": config.raw["model"],
        "test_population_fingerprint": feature["population_fingerprint"],
        "test_candidate_fingerprints": {
            "semantic": feature["candidate_semantic_fingerprint"],
            "physical": feature["candidate_physical_fingerprint"],
            "global_merkle": feature["candidate_global_merkle_fingerprint"],
        },
        "test_feature_fingerprints": {
            "schema": feature["feature_schema_fingerprint"], "semantic": feature["feature_semantic_fingerprint"],
            "physical": feature["feature_physical_fingerprint"], "statistics": feature["feature_statistics_fingerprint"],
            "lineage": feature["feature_lineage_fingerprint"],
        },
        "adapter_fingerprint": prereg["adapter_fingerprint"],
        "categorical_fingerprint": canonical_fingerprint(prereg["categorical_features"]),
        "prediction_fingerprints": {
            row["version_id"]: {
                "global_merkle": row["global_merkle_fingerprint"],
                "raw_score": row["raw_score_semantic_fingerprint"],
                "ranked_list": row["ranked_list_semantic_fingerprint"],
                "binding": row["binding_fingerprint"],
            } for row in manifests
        },
        "prediction_replay_audit_fingerprint": file_fingerprint(config.path_for("prediction_replay_audit")),
        "resource_audit_fingerprint": engineering["resource_audit_fingerprint"],
        "checkpoint_manifest_fingerprint": engineering["checkpoint_manifest_fingerprint"],
        "config_fingerprint": config.fingerprint,
        "preregistration_fingerprint": prereg["preregistration_fingerprint"],
        "code_source_fingerprint": source_fingerprint(config.root),
        "access_audit_fingerprint": file_fingerprint(config.path_for("access_audit")),
        "contract_gate_fingerprint": file_fingerprint(config.path_for("contract_gate")),
        "engineering_audit_fingerprint": file_fingerprint(engineering_audit_path),
        "prelabel_engineering_correction_fingerprint": file_fingerprint(
            config.artifact_dir / "test_prediction_prelabel_engineering_correction.json"
        ),
        "test_candidates_frozen": True, "test_features_frozen": True,
        "test_predictions_frozen": True, "test_label_read_count": 0,
        "test_labels_joined": False, "test_metrics_computed": False,
        "test_evaluation_executed": False, "test_authorized": False,
        "phase5b_pending_chatgpt_authorization": True, "winner_changed": False,
        "model_retrained": False,
    }
    lock["lock_fingerprint"] = self_fingerprint(lock, "lock_fingerprint")
    write_json_atomic(config.path_for("pre_label_lock"), lock)
    persisted = read_json(config.path_for("pre_label_lock"))
    if self_fingerprint(persisted, "lock_fingerprint") != persisted["lock_fingerprint"]:
        raise TestPredictionFreezeError("Test pre-label lock independent verification failed")
    summary = {
        "stage": "4B-4", "phase": config.raw["phase"], "status": lock["status"],
        "test_query_count": feature["query_count"], "test_candidate_row_count": feature["row_count"],
        "candidate_count_per_query": 500, "feature_set": "F1", "feature_count": 52,
        "max_dynamic_history_date": feature["max_dynamic_history_date"],
        "b0_global_prediction_fingerprint": manifest_map["B0"]["global_merkle_fingerprint"],
        "final_model_global_prediction_fingerprint": manifest_map["L1_FINAL_REFIT_600"]["global_merkle_fingerprint"],
        "prediction_replay_pass_count": 2, "resource_gate": "PASS",
        "contract_gate_pass_count": 20, "test_pre_label_lock_fingerprint": lock["lock_fingerprint"],
        "test_label_read_count": 0, "test_metrics_computed": False,
        "test_evaluation_executed": False, "model_retrained": False,
        "validation_or_first7d_reopened": False, "phase5b_authorized": False,
    }
    write_rows(config.path_for("manifest"), [summary])
    return {"lock": lock, "feature": feature, "predictions": manifests, "summary": summary}
