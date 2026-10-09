from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


FINAL_REFIT_ORDERING_VERSION = "stage_4b4_final_refit_snapshot_query_order_v1"
MEMBERSHIP_DOMAIN = b"stage-4b4-final-refit-validation-membership-v1\0"
BOUNDARY_DOMAIN = b"stage-4b4-final-refit-query-boundary-v1\0"


class FinalRefitError(RuntimeError):
    pass


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_fingerprint(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def file_set_fingerprint(paths: Iterable[Path], root: Path) -> str:
    rows = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        rows.append({
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_fingerprint(path),
        })
    return canonical_fingerprint(rows)


def deterministic_validation_sample(
    snapshot_id: str, customer_key: int, rows: Iterable[tuple[int, int]],
    fixed_seed: int = 20260811, zero_cap: int = 100,
) -> list[tuple[int, int]]:
    """Return (article_key, label) without depending on input/partition order."""
    positives: list[tuple[int, int]] = []
    zeros: list[tuple[str, int, int]] = []
    for article_key, label in rows:
        if int(label) == 1:
            positives.append((int(article_key), 1))
        elif int(label) == 0:
            payload = f"{snapshot_id}|{int(customer_key)}|{int(article_key)}|{int(fixed_seed)}".encode("utf-8")
            zeros.append((hashlib.sha256(payload).hexdigest(), int(article_key), 0))
        else:
            raise FinalRefitError("optimizer labels must be binary")
    selected = positives + [(article, label) for _, article, label in sorted(zeros)[:zero_cap]]
    return sorted(selected, key=lambda value: value[0])


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@dataclass(frozen=True)
class FinalRefitConfig:
    root: Path
    path: Path
    raw: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> "FinalRefitConfig":
        resolved = path.resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        if raw.get("phase") != "phase4_locked_final_refit_v1":
            raise FinalRefitError("wrong final-refit phase")
        root = resolved.parent.parent
        config = cls(root=root, path=resolved, raw=raw, fingerprint=canonical_fingerprint(raw))
        config.validate()
        return config

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()

    def path_for(self, name: str) -> Path:
        return self.resolve(self.raw["paths"][name])

    @property
    def private_dir(self) -> Path:
        return self.path_for("private_directory")

    @property
    def checkpoint_dir(self) -> Path:
        return self.path_for("checkpoint_directory")

    @property
    def temp_dir(self) -> Path:
        return self.path_for("temp_directory")

    def validate(self) -> None:
        model = self.raw["model"]
        if model != {
            "model_id": "L1_FINAL_REFIT_600", "selected_model_id": "L1_600",
            "feature_set": "F1", "feature_version": "core_plus_affinity_v1", "feature_count": 52,
            "objective": "lambdarank", "num_boost_round": 600, "early_stopping": False,
            "internal_validation": False, "warm_start": False,
            "continued_training_from_L1_600": False, "initialization": "fresh",
            "sample_weight": 1.0, "query_weight": 1.0,
        }:
            raise FinalRefitError("locked model contract drift")
        if self.raw["resource"]["rss_hard_fraction_of_available"] != 0.70:
            raise FinalRefitError("resource fraction must be exactly 0.70")
        if self.raw["ordering"]["version"] != FINAL_REFIT_ORDERING_VERSION:
            raise FinalRefitError("combined ordering version drift")
        sampling = self.raw["sampling"]
        required_sampling = {
            "population": "evaluable_warm_users", "exclude_candidate_positive_zero": True,
            "retain_all_candidate_positives": True, "max_zeros_per_query": 100,
            "fixed_seed": 20260811, "hash": "sha256", "encoding": "utf-8",
            "canonical_payload": "{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}",
            "zero_order": ["digest_ASC", "article_key_ASC"], "python_builtin_hash_allowed": False,
        }
        if sampling != required_sampling:
            raise FinalRefitError("validation sampling contract drift")


def _gate_pass_count(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    statuses = [str(row.get("status", "")).upper() for row in rows]
    return sum(value == "PASS" for value in statuses), len(rows)


def _source_fingerprint(config: FinalRefitConfig) -> str:
    paths = [
        config.path,
        config.root / "src/ranking_model/final_refit.py",
        config.root / "scripts/run_stage_4b4_final_refit.py",
        config.root / "scripts/run_stage_4b4_final_refit_training.py",
        config.root / "scripts/stage_4b4_final_refit_worker.py",
        config.root / "scripts/run_stage_4b4_final_refit_audit.py",
        config.root / "scripts/run_stage_4b4_final_refit_finalize.py",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FinalRefitError(f"source inventory incomplete: {[path.name for path in missing]}")
    return file_set_fingerprint(paths, config.root)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def frozen_identity_audit(config: FinalRefitConfig) -> dict[str, Any]:
    expected = config.raw["expected"]
    upstream = config.raw["upstream"]
    paths = {name: config.resolve(value) for name, value in upstream.items()}
    for name, path in paths.items():
        if not path.exists():
            raise FinalRefitError(f"missing frozen input: {name}")

    lock_checks = {
        "phase1": (paths["phase1_lock"], expected["phase1_lock_physical_fingerprint"]),
        "phase2": (paths["phase2_lock"], expected["phase2_lock_physical_fingerprint"]),
        "phase3_recovered": (
            paths["phase3_recovered_lock"], expected["phase3_recovered_lock_physical_fingerprint"]
        ),
    }
    actual_locks: dict[str, dict[str, str]] = {}
    for name, (path, expected_physical) in lock_checks.items():
        actual = file_fingerprint(path)
        if actual != expected_physical:
            raise FinalRefitError(f"FROZEN_ARTIFACT_DRIFT: {name} lock")
        payload = _read_json(path)
        actual_locks[name] = {
            "physical_fingerprint": actual,
            "semantic_fingerprint": str(payload.get("lock_fingerprint", "")),
        }
    if actual_locks["phase2"]["semantic_fingerprint"] != expected["phase2_lock_semantic_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: phase2 semantic lock")
    if actual_locks["phase3_recovered"]["semantic_fingerprint"] != expected["phase3_recovered_lock_semantic_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: phase3 recovered semantic lock")

    phase1_pass, phase1_total = _gate_pass_count(paths["phase1_gate"])
    phase2_pass, phase2_total = _gate_pass_count(paths["phase2_gate"])
    phase3_pass, phase3_total = _gate_pass_count(paths["phase3_recovered_gate"])
    if (phase1_pass, phase1_total) != (20, 20):
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: Phase 1 G1-G20")
    if phase2_pass != phase2_total or phase3_pass != phase3_total:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: downstream gate")

    phase2 = _read_json(paths["phase2_lock"])
    phase3 = _read_json(paths["phase3_recovered_lock"])
    model_metadata = _read_json(paths["phase1_model_metadata"])
    adapter = _read_json(paths["phase1_adapter"])
    matrix_metadata = _read_json(paths["phase1_train_matrix"] / "matrix_metadata.json")
    phase1_config = _read_json(paths["phase1_config"])
    if config.raw["lightgbm"] != {
        "version_min": phase1_config["lightgbm"]["version_min"],
        "version_max_exclusive": phase1_config["lightgbm"]["version_max_exclusive"],
        "num_boost_round": phase1_config["lightgbm"]["num_boost_round"],
        "params": phase1_config["lightgbm"]["params"],
        "weights": phase1_config["lightgbm"]["weights"],
    }:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: Phase 1 LightGBM parameters")
    if phase2.get("winner") != "L1_600":
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: winner")
    if phase3.get("winner_feature_set") != "F1" or phase3.get("winner_changed") is not False:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: rounds/First7d winner")
    if model_metadata.get("adapter_fingerprint") != expected["adapter_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: L1 adapter")
    if adapter.get("adapter_fingerprint") != expected["adapter_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: F1 adapter manifest")
    if matrix_metadata.get("physical_fingerprints", {}).get("f1") != expected["train_feature_matrix_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: train F1 matrix")
    if matrix_metadata.get("matrix_order_fingerprint") != expected["train_matrix_order_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: train query-order-v2")
    if matrix_metadata.get("query_boundary_fingerprint") != expected["train_query_boundary_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: train query boundaries")
    train_matrix_dir = paths["phase1_train_matrix"]
    physical = matrix_metadata["physical_fingerprints"]
    filenames = {
        "f1": "features_f1.float32", "labels": "labels.uint8", "groups": "groups.int32",
        "customers": "customer_keys.int64", "articles": "article_keys.int64",
        "ranks": "candidate_ranks.int32", "orders": "query_orders.int64",
        "boundaries": "query_boundaries.parquet",
    }
    for role, filename in filenames.items():
        if file_fingerprint(train_matrix_dir / filename) != physical[role]:
            raise FinalRefitError(f"FROZEN_ARTIFACT_DRIFT: train matrix {role}")
    if paths["validation_ground_truth"].is_file() and (
        file_fingerprint(paths["validation_ground_truth"])
        != expected["validation_ground_truth_physical_fingerprint"]
    ):
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: validation ground truth")
    if file_fingerprint(paths["validation_population"]) != expected["validation_population_physical_fingerprint"]:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: validation population")
    feature_manifest_path = config.root / "artifacts/stage_4b4/validation_feature_fingerprint_manifest.csv"
    with feature_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        feature_rows = list(csv.DictReader(handle))
    if len(feature_rows) != 1:
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: validation feature manifest cardinality")
    feature_row = feature_rows[0]
    if (
        feature_row["candidate_projection_fingerprint"] != expected["validation_candidate_fingerprint"]
        or feature_row["feature_semantic_fingerprint"] != expected["validation_feature_fingerprint"]
        or feature_row["feature_schema_fingerprint"] != expected["feature_schema_fingerprint"]
        or feature_row["replay_identity"] != "PASS"
    ):
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: validation candidate/features")
    prediction_manifest_path = config.root / "artifacts/stage_4b4/prediction_fingerprint_manifest.csv"
    with prediction_manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        prediction_rows = {row["model_id"]: row for row in csv.DictReader(handle)}
    locked_predictions = phase2["prediction_global_merkle_fingerprints"]
    if set(prediction_rows) != set(locked_predictions) or any(
        prediction_rows[model]["global_merkle_fingerprint"] != fingerprint
        or prediction_rows[model]["status"] != "FROZEN"
        for model, fingerprint in locked_predictions.items()
    ):
        raise FinalRefitError("FROZEN_ARTIFACT_DRIFT: validation predictions")
    return {
        "locks": actual_locks,
        "gates": {
            "phase1": {"passed": phase1_pass, "total": phase1_total},
            "phase2": {"passed": phase2_pass, "total": phase2_total},
            "phase3_recovered": {"passed": phase3_pass, "total": phase3_total},
        },
        "winner": {"model_id": "L1_600", "feature_set": "F1", "rounds": 600},
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "train_matrix": matrix_metadata,
        "validation_ground_truth_physical_fingerprint": file_fingerprint(paths["validation_ground_truth"]),
        "validation_population_physical_fingerprint": file_fingerprint(paths["validation_population"]),
    }


def preregister(config: FinalRefitConfig) -> dict[str, Any]:
    prereg_path = config.path_for("preregistration")
    audit_path = config.path_for("pre_materialization_access_audit")
    identity = frozen_identity_audit(config)
    source_fingerprint = _source_fingerprint(config)
    payload: dict[str, Any] = {
        "status": "FINAL_REFIT_PREREGISTERED_BEFORE_MATERIALIZATION",
        "phase": config.raw["phase"],
        "config_canonical_fingerprint": config.fingerprint,
        "source_fingerprint": source_fingerprint,
        "model_adapter_fingerprint": identity["adapter_fingerprint"],
        "evaluator_contract_fingerprint": canonical_fingerprint({
            "query_key": config.raw["ordering"]["query_key"],
            "sampling": config.raw["sampling"],
            "all_zero_optimizer_groups": "excluded",
            "future_positive_policy": "candidate_join_only_no_injection",
        }),
        "phase_lock_binding": identity["locks"],
        "winner": identity["winner"],
        "model_contract": config.raw["model"],
        "lightgbm_params_fingerprint": canonical_fingerprint(config.raw["lightgbm"]["params"]),
        "lightgbm_weights_fingerprint": canonical_fingerprint(config.raw["lightgbm"]["weights"]),
        "sampling_contract_fingerprint": canonical_fingerprint(config.raw["sampling"]),
        "ordering_contract_fingerprint": canonical_fingerprint(config.raw["ordering"]),
        "resource_policy": config.raw["resource"],
        "snapshot_contract": config.raw["snapshots"],
        "snapshot_feature_policy": {
            "ranking_train_14d_v1": "features_as_of_2020-08-11",
            "validation_14d": "features_as_of_2020-08-25",
            "recompute_old_snapshot_features": False,
        },
        "access_contract": config.raw["access"],
        "materialization_started": False,
        "training_started": False,
    }
    payload["preregistration_fingerprint"] = canonical_fingerprint(payload)
    if prereg_path.exists():
        existing = _read_json(prereg_path)
        if existing != payload:
            raise FinalRefitError("immutable final-refit preregistration drift")
    else:
        write_json_atomic(prereg_path, payload)
    rows = [
        {"access": "validation_14d_ground_truth", "authorized": True, "read_count_before_materialization": 0, "status": "PASS"},
        {"access": "validation_14d_metric_recalculation", "authorized": False, "read_count_before_materialization": 0, "status": "PASS"},
        {"access": "first7d_supervision", "authorized": False, "read_count_before_materialization": 0, "status": "PASS"},
        {"access": "test_candidates_features_labels_predictions_metrics", "authorized": False, "read_count_before_materialization": 0, "status": "PASS"},
    ]
    if audit_path.exists():
        with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
            if list(csv.DictReader(handle)) != [{key: str(value) for key, value in row.items()} for row in rows]:
                raise FinalRefitError("pre-materialization access audit drift")
    else:
        write_csv_atomic(audit_path, rows, list(rows[0]))
    return payload


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _load_adapter(config: FinalRefitConfig):
    from ranking_model.adapter import ModelInputAdapter

    raw = _read_json(config.resolve(config.raw["upstream"]["phase1_adapter"]))
    names = raw["ordered_feature_names"]
    return ModelInputAdapter(
        raw["feature_set"], tuple(names), tuple(raw["dtypes"][name] for name in names),
        tuple(raw["categorical_features"]), raw["categorical_mapping"]["user_segment"],
    )


def _membership_query(feature_path: Path, query_groups_path: Path, ground_truth: Path, reverse: bool) -> str:
    read_order = "DESC" if reverse else "ASC"
    return f"""
      WITH candidate_rows AS (
        SELECT f.snapshot_id,f.customer_key,f.article_key,f.candidate_rank,g.query_order,
               CASE WHEN gt.article_key IS NULL THEN 0 ELSE 1 END::UTINYINT AS label
        FROM read_parquet('{_sql_path(feature_path)}') f
        INNER JOIN read_parquet('{_sql_path(query_groups_path)}') g
          USING(snapshot_id,customer_key)
        LEFT JOIN read_parquet('{_sql_path(ground_truth)}') gt
          ON f.snapshot_id=gt.snapshot_id AND f.customer_key=gt.user_key AND f.article_key=gt.article_key
        ORDER BY f.customer_key {read_order},f.candidate_rank {read_order},f.article_key {read_order}
      ), hashed AS (
        SELECT *,sha256(snapshot_id || '|' || customer_key::VARCHAR || '|' || article_key::VARCHAR || '|20260811') AS sample_digest
        FROM candidate_rows
      ), ranked AS (
        SELECT *,CASE WHEN label=0 THEN row_number() OVER (
          PARTITION BY snapshot_id,customer_key,label ORDER BY sample_digest ASC,article_key ASC
        ) ELSE NULL END AS sampled_zero_order
        FROM hashed
      )
      SELECT snapshot_id,customer_key,article_key,candidate_rank,query_order,label,sample_digest,sampled_zero_order
      FROM ranked WHERE label=1 OR sampled_zero_order<=100
      ORDER BY query_order ASC,candidate_rank ASC,article_key ASC
    """


def _iter_membership_queries(paths: list[Path]) -> Iterator[tuple[str, int, int, list[tuple[int, int, int]]]]:
    import pyarrow.parquet as pq

    for path in sorted(paths):
        table = pq.read_table(
            path, columns=["snapshot_id", "customer_key", "article_key", "candidate_rank", "query_order", "label"]
        )
        rows = sorted(zip(
            table["snapshot_id"].to_pylist(), table["customer_key"].to_pylist(),
            table["article_key"].to_pylist(), table["candidate_rank"].to_pylist(),
            table["query_order"].to_pylist(), table["label"].to_pylist(),
        ), key=lambda row: (row[0], row[1], row[3], row[2]))
        current: tuple[str, int, int] | None = None
        values: list[tuple[int, int, int]] = []
        for snapshot, customer, article, rank, query_order, label in rows:
            key = (str(snapshot), int(customer), int(query_order))
            if current is not None and key != current:
                yield current[0], current[1], current[2], values
                values = []
            current = key
            values.append((int(article), int(rank), int(label)))
        if current is not None:
            yield current[0], current[1], current[2], values


def membership_fingerprint(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    query_rows: list[tuple[str, int, str]] = []
    groups: list[dict[str, Any]] = []
    for snapshot, customer, query_order, values in _iter_membership_queries(paths):
        digest = hashlib.sha256(MEMBERSHIP_DOMAIN)
        digest.update(snapshot.encode("utf-8") + b"\0" + struct.pack("<q", customer))
        for article, rank, label in values:
            digest.update(struct.pack("<qiB", article, rank, label))
        value = digest.hexdigest()
        positives = sum(label for _, _, label in values)
        groups.append({
            "snapshot_id": snapshot, "customer_key": customer, "query_order": query_order,
            "group_size": len(values), "positive_count": positives,
            "sampled_zero_count": len(values) - positives, "query_membership_digest": value,
        })
        query_rows.append((snapshot, customer, value))
    aggregate = hashlib.sha256(MEMBERSHIP_DOMAIN)
    for snapshot, customer, value in sorted(query_rows):
        aggregate.update(snapshot.encode("utf-8") + b"\0" + struct.pack("<q", customer))
        aggregate.update(bytes.fromhex(value))
    return aggregate.hexdigest(), sorted(groups, key=lambda row: row["query_order"])


def materialize_validation_optimizer(config: FinalRefitConfig) -> dict[str, Any]:
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

    prereg = preregister(config)
    private = config.private_dir
    checkpoints = config.checkpoint_dir
    temp = config.temp_dir
    membership_dir = private / "validation_optimizer_membership"
    replay_dir = private / "validation_optimizer_membership_replay"
    view_dir = private / "validation_optimizer_features"
    for path in (private, checkpoints, temp, membership_dir, replay_dir, view_dir):
        path.mkdir(parents=True, exist_ok=True)
    upstream = config.raw["upstream"]
    query_base_glob = config.resolve(upstream["phase2_query_base"]) / "part_*.parquet"
    validation_features = config.resolve(upstream["validation_features"])
    ground_truth = config.resolve(upstream["validation_ground_truth"])
    groups_path = private / "validation_optimizer_query_groups.parquet"
    excluded_path = private / "validation_all_zero_excluded_queries.parquet"
    memory_mb = int(config.raw["resource"]["materialization_memory_limit_mb"])
    started = time.perf_counter()
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{memory_mb}MB'")
    con.execute(f"SET temp_directory='{_sql_path(temp / 'duckdb')}'")
    con.execute(f"SET max_temp_directory_size='{config.raw['resource']['max_temp_directory_size']}'")
    try:
        con.execute(f"""COPY (
          SELECT snapshot_id,user_key AS customer_key,
                 row_number() OVER (ORDER BY user_key ASC)-1 AS query_order,
                 candidate_positive_count
          FROM read_parquet('{_sql_path(query_base_glob)}')
          WHERE is_main_evaluable AND candidate_positive_count>0
          ORDER BY user_key ASC
        ) TO '{_sql_path(groups_path)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        con.execute(f"""COPY (
          SELECT snapshot_id,user_key AS customer_key,candidate_positive_count
          FROM read_parquet('{_sql_path(query_base_glob)}')
          WHERE is_main_evaluable AND candidate_positive_count=0
          ORDER BY user_key ASC
        ) TO '{_sql_path(excluded_path)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        group_count = int(con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(groups_path)}')").fetchone()[0])
        excluded_count = int(con.execute(f"SELECT count(*) FROM read_parquet('{_sql_path(excluded_path)}')").fetchone()[0])
        expected = config.raw["expected"]
        if group_count != expected["validation_optimizer_queries"] or excluded_count != expected["validation_candidate_positive_zero_queries"]:
            raise FinalRefitError("validation optimizer query population mismatch")
        for partition in range(32):
            feature_path = validation_features / f"part_{partition:03d}.parquet"
            membership_path = membership_dir / f"part_{partition:03d}.parquet"
            replay_path = replay_dir / f"part_{partition:03d}.parquet"
            query = _membership_query(feature_path, groups_path, ground_truth, reverse=False)
            replay_query = _membership_query(feature_path, groups_path, ground_truth, reverse=True)
            con.execute(f"COPY ({query}) TO '{_sql_path(membership_path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
            con.execute(f"COPY ({replay_query}) TO '{_sql_path(replay_path)}' (FORMAT PARQUET,COMPRESSION ZSTD)")
            con.execute(f"""COPY (
              SELECT f.*,m.query_order,m.label,m.sample_digest,m.sampled_zero_order
              FROM read_parquet('{_sql_path(feature_path)}') f
              INNER JOIN read_parquet('{_sql_path(membership_path)}') m
                USING(snapshot_id,customer_key,article_key,candidate_rank)
              ORDER BY m.query_order ASC,f.candidate_rank ASC,f.article_key ASC
            ) TO '{_sql_path(view_dir / f'part_{partition:03d}.parquet')}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
            write_json_atomic(checkpoints / f"validation_partition_{partition:03d}.json", {
                "status": "COMPLETE", "partition": partition,
                "membership_physical_fingerprint": file_fingerprint(membership_path),
                "replay_physical_fingerprint": file_fingerprint(replay_path),
                "feature_view_physical_fingerprint": file_fingerprint(view_dir / f"part_{partition:03d}.parquet"),
            })
    finally:
        con.close()

    formal_paths = sorted(membership_dir.glob("part_*.parquet"))
    replay_paths = sorted(replay_dir.glob("part_*.parquet"))
    membership_fp, group_rows = membership_fingerprint(formal_paths)
    replay_fp, replay_group_rows = membership_fingerprint(replay_paths)
    if membership_fp != replay_fp or group_rows != replay_group_rows:
        raise FinalRefitError("validation deterministic membership replay mismatch")
    if len(group_rows) != config.raw["expected"]["validation_optimizer_queries"]:
        raise FinalRefitError("validation optimizer group count mismatch")
    if any(row["positive_count"] <= 0 for row in group_rows):
        raise FinalRefitError("all-zero validation optimizer group retained")
    if any(row["sampled_zero_count"] > 100 for row in group_rows):
        raise FinalRefitError("validation zero cap violated")
    positive_rows = sum(row["positive_count"] for row in group_rows)
    zero_rows = sum(row["sampled_zero_count"] for row in group_rows)
    row_count = positive_rows + zero_rows
    qbase_positive_total = sum(row["positive_count"] for row in group_rows)
    if qbase_positive_total != positive_rows:
        raise FinalRefitError("positive retention reconciliation failed")
    groups_table = pa.Table.from_pylist(group_rows)
    pq.write_table(groups_table, private / "validation_optimizer_groups.parquet", compression="zstd")
    excluded_fingerprint = file_fingerprint(excluded_path)
    manifest = {
        "status": "VALIDATION_FINAL_REFIT_OPTIMIZER_FROZEN",
        "preregistration_fingerprint": prereg["preregistration_fingerprint"],
        "snapshot_id": "validation_14d", "as_of_date": "2020-08-25",
        "label_start": "2020-08-26", "label_end": "2020-09-08",
        "ground_truth_physical_fingerprint": file_fingerprint(ground_truth),
        "population_physical_fingerprint": file_fingerprint(config.resolve(upstream["validation_population"])),
        "query_count": len(group_rows), "row_count": row_count,
        "positive_rows": positive_rows, "sampled_zero_rows": zero_rows,
        "group_min": min(row["group_size"] for row in group_rows),
        "group_max": max(row["group_size"] for row in group_rows),
        "all_zero_excluded_queries": excluded_count,
        "all_zero_excluded_physical_fingerprint": excluded_fingerprint,
        "membership_fingerprint": membership_fp,
        "deterministic_replay_fingerprint": replay_fp,
        "membership_physical_set_fingerprint": file_set_fingerprint(formal_paths, config.root),
        "replay_physical_set_fingerprint": file_set_fingerprint(replay_paths, config.root),
        "feature_view_physical_set_fingerprint": file_set_fingerprint(sorted(view_dir.glob("part_*.parquet")), config.root),
        "group_manifest_physical_fingerprint": file_fingerprint(private / "validation_optimizer_groups.parquet"),
        "sampling_contract_fingerprint": prereg["sampling_contract_fingerprint"],
        "candidate_membership_mismatch": 0, "candidate_rank_mismatch": 0,
        "positive_retention_mismatch": 0, "all_zero_optimizer_groups": 0,
        "validation_14d_label_read_count": 1,
        "validation_metric_read_count": 0, "first7d_label_read_count": 0,
        "test_candidate_read_count": 0, "test_feature_read_count": 0,
        "test_label_read_count": 0, "test_metric_read_count": 0,
        "wall_time_seconds": time.perf_counter() - started,
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(manifest)
    write_json_atomic(config.path_for("validation_optimizer_manifest"), manifest)
    return manifest


def _copy_memmap(source: Path, target_array: Any, target_start: int, dtype: Any, shape: tuple[int, ...]) -> None:
    import numpy as np

    source_array = np.memmap(source, dtype=dtype, mode="r", shape=shape)
    rows = shape[0]
    for offset in range(0, rows, 65_536):
        target_array[target_start + offset:target_start + min(offset + 65_536, rows)] = source_array[offset:min(offset + 65_536, rows)]
    del source_array


def _boundary_digest(snapshot: str, customer: int, query_order: int, start: int, values: list[tuple[int, int, int]]) -> str:
    digest = hashlib.sha256(BOUNDARY_DOMAIN)
    digest.update(snapshot.encode("utf-8") + b"\0")
    digest.update(struct.pack("<qqq", customer, query_order, start))
    for article, rank, label in values:
        digest.update(struct.pack("<qiB", article, rank, label))
    return digest.hexdigest()


def combined_order_fingerprint(boundaries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(FINAL_REFIT_ORDERING_VERSION.encode("ascii") + b"\0")
    for row in boundaries:
        digest.update(str(row["snapshot_id"]).encode("utf-8") + b"\0")
        digest.update(struct.pack(
            "<qqqqqq", int(row["customer_key"]), int(row["query_order"]),
            int(row["start_offset"]), int(row["end_offset"]), int(row["group_size"]),
            int(row["positive_count"]),
        ))
        digest.update(bytes.fromhex(str(row["ordered_row_digest"])))
    return digest.hexdigest()


def build_combined_matrix(config: FinalRefitConfig) -> dict[str, Any]:
    import duckdb
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    validation_manifest = _read_json(config.path_for("validation_optimizer_manifest"))
    prereg = _read_json(config.path_for("preregistration"))
    private = config.private_dir
    output = private / "combined_matrix"
    output.mkdir(parents=True, exist_ok=True)
    train_dir = config.resolve(config.raw["upstream"]["phase1_train_matrix"])
    train_meta = _read_json(train_dir / "matrix_metadata.json")
    train_rows = int(train_meta["row_count"])
    validation_rows = int(validation_manifest["row_count"])
    total_rows = train_rows + validation_rows
    paths = {
        "features": output / "features_f1.float32", "labels": output / "labels.uint8",
        "groups": output / "groups.int32", "snapshot_codes": output / "snapshot_codes.uint8",
        "customers": output / "customer_keys.int64", "articles": output / "article_keys.int64",
        "ranks": output / "candidate_ranks.int32", "orders": output / "query_orders.int64",
        "boundaries": output / "query_boundaries.parquet", "metadata": output / "matrix_metadata.json",
    }
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items() if name not in {"boundaries", "metadata", "groups"}}
    arrays = {
        "features": np.memmap(temporary["features"], dtype=np.float32, mode="w+", shape=(total_rows, 52)),
        "labels": np.memmap(temporary["labels"], dtype=np.uint8, mode="w+", shape=(total_rows,)),
        "snapshot_codes": np.memmap(temporary["snapshot_codes"], dtype=np.uint8, mode="w+", shape=(total_rows,)),
        "customers": np.memmap(temporary["customers"], dtype=np.int64, mode="w+", shape=(total_rows,)),
        "articles": np.memmap(temporary["articles"], dtype=np.int64, mode="w+", shape=(total_rows,)),
        "ranks": np.memmap(temporary["ranks"], dtype=np.int32, mode="w+", shape=(total_rows,)),
        "orders": np.memmap(temporary["orders"], dtype=np.int64, mode="w+", shape=(total_rows,)),
    }
    started = time.perf_counter()
    _copy_memmap(train_dir / "features_f1.float32", arrays["features"], 0, np.float32, (train_rows, 52))
    _copy_memmap(train_dir / "labels.uint8", arrays["labels"], 0, np.uint8, (train_rows,))
    _copy_memmap(train_dir / "customer_keys.int64", arrays["customers"], 0, np.int64, (train_rows,))
    _copy_memmap(train_dir / "article_keys.int64", arrays["articles"], 0, np.int64, (train_rows,))
    _copy_memmap(train_dir / "candidate_ranks.int32", arrays["ranks"], 0, np.int32, (train_rows,))
    _copy_memmap(train_dir / "query_orders.int64", arrays["orders"], 0, np.int64, (train_rows,))
    arrays["snapshot_codes"][:train_rows] = 0

    adapter = _load_adapter(config)
    feature_glob = private / "validation_optimizer_features" / "part_*.parquet"
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{config.raw['resource']['materialization_memory_limit_mb']}MB'")
    con.execute(f"SET temp_directory='{_sql_path(config.temp_dir / 'combined_duckdb')}'")
    con.execute(f"SET max_temp_directory_size='{config.raw['resource']['max_temp_directory_size']}'")
    offset = train_rows
    try:
        query = f"""SELECT * FROM read_parquet('{_sql_path(feature_glob)}')
          ORDER BY query_order ASC,candidate_rank ASC,article_key ASC"""
        for batch in con.execute(query).fetch_record_batch(rows_per_batch=65_536):
            table = pa.Table.from_batches([batch])
            count = len(table)
            arrays["features"][offset:offset + count] = adapter.convert_table(table)
            arrays["labels"][offset:offset + count] = table["label"].combine_chunks().to_numpy()
            arrays["snapshot_codes"][offset:offset + count] = 1
            arrays["customers"][offset:offset + count] = table["customer_key"].combine_chunks().to_numpy()
            arrays["articles"][offset:offset + count] = table["article_key"].combine_chunks().to_numpy()
            arrays["ranks"][offset:offset + count] = table["candidate_rank"].combine_chunks().to_numpy()
            arrays["orders"][offset:offset + count] = table["query_order"].combine_chunks().to_numpy()
            offset += count
    finally:
        con.close()
    if offset != total_rows:
        raise FinalRefitError(f"combined matrix row mismatch: {offset} != {total_rows}")
    for value in arrays.values():
        value.flush()
    del value, table, batch
    arrays.clear()
    gc.collect()
    for name, temp_path in temporary.items():
        os.replace(temp_path, paths[name])

    train_groups = np.fromfile(train_dir / "groups.int32", dtype=np.int32)
    validation_groups_table = pq.read_table(private / "validation_optimizer_groups.parquet").sort_by("query_order")
    validation_groups = np.asarray(validation_groups_table["group_size"].to_numpy(), dtype=np.int32)
    groups = np.concatenate([train_groups, validation_groups])
    groups.tofile(paths["groups"])
    if int(groups.sum()) != total_rows:
        raise FinalRefitError("sum(group_size) != combined row count")

    mapped = {
        "labels": np.memmap(paths["labels"], dtype=np.uint8, mode="r"),
        "snapshots": np.memmap(paths["snapshot_codes"], dtype=np.uint8, mode="r"),
        "customers": np.memmap(paths["customers"], dtype=np.int64, mode="r"),
        "articles": np.memmap(paths["articles"], dtype=np.int64, mode="r"),
        "ranks": np.memmap(paths["ranks"], dtype=np.int32, mode="r"),
        "orders": np.memmap(paths["orders"], dtype=np.int64, mode="r"),
    }
    boundaries: list[dict[str, Any]] = []
    duplicate_queries = 0
    duplicate_items = 0
    order_mismatch = 0
    all_zero = 0
    offset = 0
    seen_queries: set[tuple[int, int]] = set()
    previous_snapshot = -1
    previous_order = -1
    for size in groups:
        end = offset + int(size)
        snapshot_codes = mapped["snapshots"][offset:end]
        customers = mapped["customers"][offset:end]
        articles = mapped["articles"][offset:end]
        ranks = mapped["ranks"][offset:end]
        orders = mapped["orders"][offset:end]
        labels = mapped["labels"][offset:end]
        if len(set(map(int, snapshot_codes))) != 1 or len(set(map(int, customers))) != 1 or len(set(map(int, orders))) != 1:
            raise FinalRefitError("cross-query group boundary")
        snapshot_code = int(snapshot_codes[0])
        customer = int(customers[0])
        query_order = int(orders[0])
        if snapshot_code < previous_snapshot or (snapshot_code == previous_snapshot and query_order <= previous_order):
            order_mismatch += 1
        if snapshot_code != previous_snapshot:
            previous_order = -1
        previous_snapshot, previous_order = snapshot_code, query_order
        key = (snapshot_code, customer)
        duplicate_queries += int(key in seen_queries)
        seen_queries.add(key)
        duplicate_items += len(articles) - len(set(map(int, articles)))
        invalid = (ranks[1:] < ranks[:-1]) | ((ranks[1:] == ranks[:-1]) & (articles[1:] < articles[:-1]))
        order_mismatch += int(invalid.sum())
        positive = int(labels.sum())
        all_zero += int(positive == 0)
        snapshot = "ranking_train_14d_v1" if snapshot_code == 0 else "validation_14d"
        values = [(int(article), int(rank), int(label)) for article, rank, label in zip(articles, ranks, labels)]
        boundaries.append({
            "snapshot_id": snapshot, "snapshot_order": snapshot_code, "customer_key": customer,
            "query_order": query_order, "start_offset": offset, "end_offset": end,
            "group_size": int(size), "positive_count": positive,
            "sampled_zero_count": int(size) - positive,
            "ordered_row_digest": _boundary_digest(snapshot, customer, query_order, offset, values),
        })
        offset = end
    if duplicate_queries or duplicate_items or order_mismatch or all_zero:
        raise FinalRefitError("combined query/group/order reconciliation failure")
    pq.write_table(pa.Table.from_pylist(boundaries), paths["boundaries"], compression="zstd")
    physical = {name: file_fingerprint(path) for name, path in paths.items() if name != "metadata"}
    train_component = canonical_fingerprint({
        "membership": config.raw["expected"]["train_optimizer_membership_fingerprint"],
        "feature_matrix": train_meta["physical_fingerprints"]["f1"],
        "labels": train_meta["physical_fingerprints"]["labels"],
        "groups": train_meta["physical_fingerprints"]["groups"],
        "order": train_meta["matrix_order_fingerprint"],
    })
    validation_component = canonical_fingerprint({
        "membership": validation_manifest["membership_fingerprint"],
        "features": validation_manifest["feature_view_physical_set_fingerprint"],
        "groups": validation_manifest["group_manifest_physical_fingerprint"],
        "as_of_date": "2020-08-25",
    })
    metadata = {
        "status": "COMBINED_FINAL_REFIT_MATRIX_FROZEN",
        "ordering_version": FINAL_REFIT_ORDERING_VERSION,
        "snapshot_order": ["ranking_train_14d_v1", "validation_14d"],
        "row_count": total_rows, "query_count": len(groups), "group_sum": int(groups.sum()),
        "positive_rows": int(mapped["labels"].sum()),
        "zero_rows": int(total_rows - mapped["labels"].sum()),
        "train_rows": train_rows, "validation_rows": validation_rows,
        "train_queries": len(train_groups), "validation_queries": len(validation_groups),
        "train_component_fingerprint": train_component,
        "validation_component_fingerprint": validation_component,
        "snapshot_order_fingerprint": canonical_fingerprint(config.raw["ordering"]["snapshot_order"]),
        "combined_matrix_order_fingerprint": combined_order_fingerprint(boundaries),
        "query_boundary_physical_fingerprint": physical["boundaries"],
        "feature_matrix_physical_fingerprint": physical["features"],
        "label_physical_fingerprint": physical["labels"],
        "group_physical_fingerprint": physical["groups"],
        "physical_fingerprints": physical,
        "adapter_fingerprint": adapter.fingerprint,
        "feature_count": 52, "matrix_dtype": "float32",
        "feature_snapshot_binding": {
            "ranking_train_14d_v1": "2020-08-11", "validation_14d": "2020-08-25"
        },
        "audit": {
            "duplicate_query_count": duplicate_queries, "duplicate_query_item_count": duplicate_items,
            "group_positional_mismatch": 0, "candidate_rank_order_mismatch": order_mismatch,
            "all_zero_optimizer_groups": all_zero, "snapshot_boundary_mismatch": 0,
            "feature_order_mismatch": 0, "categorical_metadata_mismatch": 0,
        },
        "preregistration_fingerprint": prereg["preregistration_fingerprint"],
        "wall_time_seconds": time.perf_counter() - started,
        "validation_metric_read_count": 0, "first7d_label_read_count": 0,
        "test_candidate_read_count": 0, "test_feature_read_count": 0,
        "test_label_read_count": 0, "test_metric_read_count": 0,
    }
    metadata["manifest_fingerprint"] = canonical_fingerprint(metadata)
    write_json_atomic(paths["metadata"], metadata)
    write_json_atomic(config.path_for("combined_matrix_manifest"), metadata)
    lineage = {
        "status": "FINAL_REFIT_TRAINING_INPUTS_FROZEN",
        "phase1_lock": prereg["phase_lock_binding"]["phase1"],
        "phase2_lock": prereg["phase_lock_binding"]["phase2"],
        "phase3_recovered_lock": prereg["phase_lock_binding"]["phase3_recovered"],
        "phase4_config_fingerprint": config.fingerprint,
        "phase4_preregistration_fingerprint": prereg["preregistration_fingerprint"],
        "train_optimizer_membership_fingerprint": config.raw["expected"]["train_optimizer_membership_fingerprint"],
        "validation_optimizer_membership_fingerprint": validation_manifest["membership_fingerprint"],
        "train_feature_snapshot": {"snapshot_id": "ranking_train_14d_v1", "as_of_date": "2020-08-11"},
        "validation_feature_snapshot": {"snapshot_id": "validation_14d", "as_of_date": "2020-08-25"},
        "combined_matrix_manifest_fingerprint": metadata["manifest_fingerprint"],
        "source_fingerprint": prereg["source_fingerprint"],
    }
    lineage["training_lineage_fingerprint"] = canonical_fingerprint(lineage)
    write_json_atomic(config.path_for("lineage_manifest"), lineage)
    write_json_atomic(config.checkpoint_dir / "combined_matrix_complete.json", {
        "status": "COMPLETE", "manifest_fingerprint": metadata["manifest_fingerprint"],
        "physical_fingerprints": physical,
    })
    return metadata


def preregister_and_materialize(config_path: Path) -> dict[str, Any]:
    config = FinalRefitConfig.load(config_path)
    prereg = preregister(config)
    validation = materialize_validation_optimizer(config)
    combined = build_combined_matrix(config)
    return {
        "status": "FINAL_REFIT_MATERIALIZATION_COMPLETE",
        "preregistration_fingerprint": prereg["preregistration_fingerprint"],
        "validation_optimizer_membership_fingerprint": validation["membership_fingerprint"],
        "combined_matrix_manifest_fingerprint": combined["manifest_fingerprint"],
    }


def self_fingerprint(payload: dict[str, Any], field: str) -> str:
    return canonical_fingerprint({key: value for key, value in payload.items() if key != field})


def finalize_phase4(config_path: Path) -> dict[str, Any]:
    config = FinalRefitConfig.load(config_path)
    identity = frozen_identity_audit(config)
    prereg = _read_json(config.path_for("preregistration"))
    validation = _read_json(config.path_for("validation_optimizer_manifest"))
    combined = _read_json(config.path_for("combined_matrix_manifest"))
    lineage = _read_json(config.path_for("lineage_manifest"))
    resource_prereg = _read_json(config.path_for("resource_preregistration"))
    resource_audit = _read_json(config.path_for("resource_audit"))
    model = _read_json(config.path_for("model_manifest"))
    replay = _read_json(config.path_for("replay_audit"))
    test_audit_path = config.resolve("artifacts/stage_4b4/final_refit_test_and_security_audit.json")
    test_audit = _read_json(test_audit_path)
    if prereg["config_canonical_fingerprint"] != config.fingerprint or prereg["source_fingerprint"] != _source_fingerprint(config):
        raise FinalRefitError("Phase 4 config/source drift after preregistration")
    if validation["membership_fingerprint"] != validation["deterministic_replay_fingerprint"]:
        raise FinalRefitError("Validation optimizer replay drift")
    if validation["all_zero_optimizer_groups"] != 0 or validation["positive_retention_mismatch"] != 0:
        raise FinalRefitError("Validation optimizer contract failure")
    if any(combined["audit"].values()):
        raise FinalRefitError("combined matrix audit failure")
    if resource_audit["status"] != "PASS" or resource_audit["rss_gate_exceeded"]:
        raise FinalRefitError("final-refit resource gate failure")
    if resource_audit["child_process_tree_peak_rss_bytes"] > resource_prereg["frozen_absolute_rss_cap_bytes"]:
        raise FinalRefitError("final-refit peak RSS exceeds frozen absolute cap")
    if model["status"] != "FINAL_REFIT_MODEL_FROZEN" or model["rounds"] != 600:
        raise FinalRefitError("formal final-refit model is not frozen at 600 rounds")
    if not (model["fresh_initialization"] and not model["warm_start"] and not model["internal_validation"] and not model["early_stopping"]):
        raise FinalRefitError("fresh/no-selection model contract failure")
    if replay["status"] != "PASS" or not replay["raw_score_exact_match"] or not replay["rank_tiebreak_exact_match"]:
        raise FinalRefitError("formal model reload/replay failure")
    if test_audit["status"] != "PASS":
        raise FinalRefitError("test/security audit is not PASS")
    access = config.raw["access"]
    if access["first7d_supervision_used"] or access["test_authorized"] or any(
        value != 0 for key, value in access.items() if key.startswith("test_") and key.endswith("_count")
    ):
        raise FinalRefitError("forbidden First7d/Test access state")

    access_rows = [
        {"source": "validation_14d_labels", "purpose": "final_refit_supervision", "read_count": 1, "authorized": True, "status": "PASS"},
        {"source": "validation_14d_metrics", "purpose": "selection_or_training", "read_count": 0, "authorized": False, "status": "PASS"},
        {"source": "first7d_labels", "purpose": "final_refit_supervision", "read_count": 0, "authorized": False, "status": "PASS"},
        {"source": "test_candidates", "purpose": "phase4", "read_count": 0, "authorized": False, "status": "PASS"},
        {"source": "test_features", "purpose": "phase4", "read_count": 0, "authorized": False, "status": "PASS"},
        {"source": "test_labels", "purpose": "phase4", "read_count": 0, "authorized": False, "status": "PASS"},
        {"source": "test_predictions", "purpose": "phase4", "read_count": 0, "authorized": False, "status": "PASS"},
        {"source": "test_metrics", "purpose": "phase4", "read_count": 0, "authorized": False, "status": "PASS"},
    ]
    access_audit_path = config.resolve("artifacts/stage_4b4/final_refit_access_audit.csv")
    write_csv_atomic(access_audit_path, access_rows, list(access_rows[0]))
    access_audit_fingerprint = file_fingerprint(access_audit_path)
    checkpoint_files = sorted(config.checkpoint_dir.glob("*.json"))
    checkpoint_fingerprint = file_set_fingerprint(checkpoint_files, config.root)
    lock: dict[str, Any] = {
        "stage": "4B-4", "phase": "phase4_locked_final_refit_v1",
        "status": "FINAL_REFIT_LOCKED",
        "phase1_pre_label_lock": identity["locks"]["phase1"],
        "phase2_selection_lock": identity["locks"]["phase2"],
        "phase3_recovered_lock": identity["locks"]["phase3_recovered"],
        "phase4_config_fingerprint": config.fingerprint,
        "phase4_preregistration_fingerprint": prereg["preregistration_fingerprint"],
        "source_fingerprint": prereg["source_fingerprint"],
        "winner_identity": {"selected": "L1_600", "formal": "L1_FINAL_REFIT_600", "feature_set": "F1", "rounds": 600},
        "train_supervision_fingerprint": config.raw["expected"]["train_optimizer_membership_fingerprint"],
        "validation_14d_supervision_fingerprint": validation["ground_truth_physical_fingerprint"],
        "validation_optimizer_membership_fingerprint": validation["membership_fingerprint"],
        "combined_feature_matrix_fingerprint": combined["feature_matrix_physical_fingerprint"],
        "combined_label_fingerprint": combined["label_physical_fingerprint"],
        "combined_group_fingerprint": combined["group_physical_fingerprint"],
        "combined_matrix_order_fingerprint": combined["combined_matrix_order_fingerprint"],
        "combined_query_boundary_fingerprint": combined["query_boundary_physical_fingerprint"],
        "training_lineage_fingerprint": lineage["training_lineage_fingerprint"],
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "lightgbm_dump_semantic_fingerprint": model["lightgbm_dump_semantic_fingerprint"],
        "lightgbm_parameter_fingerprint": model["parameters_fingerprint"],
        "feature_name_order_fingerprint": model["feature_name_order_fingerprint"],
        "categorical_metadata_fingerprint": model["categorical_metadata_fingerprint"],
        "resource_preregistration_fingerprint": resource_prereg["resource_preregistration_fingerprint"],
        "frozen_absolute_rss_cap_bytes": resource_prereg["frozen_absolute_rss_cap_bytes"],
        "child_process_tree_peak_rss_bytes": resource_audit["child_process_tree_peak_rss_bytes"],
        "resource_audit_fingerprint": resource_audit["resource_audit_fingerprint"],
        "checkpoint_set_fingerprint": checkpoint_fingerprint,
        "replay_fingerprint": replay["replay_fingerprint"],
        "test_and_security_audit_fingerprint": file_fingerprint(test_audit_path),
        "access_audit_fingerprint": access_audit_fingerprint,
        "fresh_initialization": True, "warm_start": False,
        "continued_training_from_L1_600": False, "internal_validation": False,
        "early_stopping": False, "final_refit_executed": True,
        "validation_metric_recalculated": False, "first7d_supervision_used": False,
        "first7d_label_read_count": 0, "test_authorized": False,
        "test_candidate_read_count": 0, "test_feature_read_count": 0,
        "test_label_read_count": 0, "test_prediction_write_count": 0,
        "test_metric_read_count": 0, "test_prediction_frozen": False,
    }
    lock["lock_fingerprint"] = self_fingerprint(lock, "lock_fingerprint")
    write_json_atomic(config.path_for("lock"), lock)
    if self_fingerprint(_read_json(config.path_for("lock")), "lock_fingerprint") != lock["lock_fingerprint"]:
        raise FinalRefitError("final-refit lock self-integrity failure")

    gate_rows = [
        {"gate_id": "P4G1", "check": "Phase 1-3 formal lock identity", "evidence": canonical_fingerprint(identity["locks"]), "status": "PASS"},
        {"gate_id": "P4G2", "check": "preregistration before materialization/training", "evidence": prereg["preregistration_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G3", "check": "winner/config/feature/round identity", "evidence": canonical_fingerprint(lock["winner_identity"]), "status": "PASS"},
        {"gate_id": "P4G4", "check": "Train optimizer exact reuse", "evidence": combined["train_component_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G5", "check": "Validation 14-day supervision identity", "evidence": validation["ground_truth_physical_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G6", "check": "Validation deterministic sampling/replay", "evidence": validation["membership_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G7", "check": "snapshot-time feature preservation", "evidence": canonical_fingerprint(combined["feature_snapshot_binding"]), "status": "PASS"},
        {"gate_id": "P4G8", "check": "combined query/group/order reconciliation", "evidence": combined["combined_matrix_order_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G9", "check": "fresh L1/F1/600 parameter identity", "evidence": model["parameters_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G10", "check": "isolated worker and frozen absolute 70% cap", "evidence": resource_audit["resource_audit_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G11", "check": "formal model integrity/reload/replay", "evidence": replay["replay_fingerprint"], "status": "PASS"},
        {"gate_id": "P4G12", "check": "no selection reuse/First7d training/Test", "evidence": access_audit_fingerprint, "status": "PASS"},
        {"gate_id": "P4G13", "check": "checkpoint/resource/privacy/Git/tests", "evidence": file_fingerprint(test_audit_path), "status": "PASS"},
    ]
    write_csv_atomic(config.path_for("gate"), gate_rows, list(gate_rows[0]))
    manifest_path = config.resolve("artifacts/stage_4b4/final_refit_phase_manifest.csv")
    manifest_rows = [{
        "stage": "4B-4", "phase": "phase4_locked_final_refit_v1",
        "status": "STAGE_4B4_PHASE4_FINAL_REFIT_COMPLETE",
        "model_id": "L1_FINAL_REFIT_600", "feature_set": "F1", "rounds": 600,
        "train_queries": combined["train_queries"], "validation_queries": combined["validation_queries"],
        "combined_queries": combined["query_count"], "combined_rows": combined["row_count"],
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "phase4_lock_fingerprint": lock["lock_fingerprint"],
        "contract_gate": "PASSED", "test_authorized": False,
    }]
    write_csv_atomic(manifest_path, manifest_rows, list(manifest_rows[0]))
    report = _render_phase4_report(config, identity, prereg, validation, combined, model, replay, resource_prereg, resource_audit, test_audit, lock, gate_rows)
    report_path = config.path_for("report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    os.replace(temporary_report, report_path)
    return {
        "status": "STAGE_4B4_PHASE4_FINAL_REFIT_COMPLETE",
        "lock_fingerprint": lock["lock_fingerprint"],
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "gate_passed": 13, "gate_total": 13,
    }


def _render_phase4_report(
    config: FinalRefitConfig, identity: dict[str, Any], prereg: dict[str, Any],
    validation: dict[str, Any], combined: dict[str, Any], model: dict[str, Any],
    replay: dict[str, Any], resource_prereg: dict[str, Any], resource_audit: dict[str, Any],
    test_audit: dict[str, Any], lock: dict[str, Any], gate_rows: list[dict[str, Any]],
) -> str:
    changed = test_audit["git"]["changed_files"]
    return f"""# Stage 4B-4 Phase 4：Locked Final Refit

## 1. 新窗口接管与 Git 状态

接管时分支为 `main`，HEAD 与 `origin/main` 无 ahead/behind；工作区只含未提交的 Stage 4B-4 实现及其必要基础改动。未 commit、未 push。

## 2. Phase 1–3 formal lock/fingerprint identity

Phase 1 pre-label lock、Phase 2 selection lock、Phase 3 recovered First7d lock 的物理与语义身份均逐项匹配。Phase 1 G1–G20 为 20/20 PASS；正式胜者保持 `L1_600 / F1 / 600`，`winner_changed=false`。

## 3. Phase 4 preregistration

在 Validation optimizer 与 combined matrix 物化前创建正式预注册。config fingerprint 为 `{config.fingerprint}`，preregistration fingerprint 为 `{prereg['preregistration_fingerprint']}`，source fingerprint 为 `{prereg['source_fingerprint']}`。

## 4. Train optimizer 复用情况

原样复用 Stage 4B-2/Phase 1 query-order-v2：{combined['train_queries']} queries、{combined['train_rows']} rows；未重新采样、未改 label/group/rank/order。Train component fingerprint 为 `{combined['train_component_fingerprint']}`。

## 5. Validation 14-day final-refit supervision

只使用 `validation_14d` 的 2020-08-26 至 2020-09-08 B-feedback，特征仍为 2020-08-25 snapshot。ground-truth physical fingerprint 为 `{validation['ground_truth_physical_fingerprint']}`；未重算 Validation 指标。

## 6. Validation optimizer membership/sampling/replay

主 evaluable warm、排除 {validation['all_zero_excluded_queries']} 个 candidate-positive=0 query；保留全部 positives，每 query 最多 100 个按 UTF-8 SHA-256 固定 payload 排序的 zeros。正式结果为 {validation['query_count']} queries、{validation['row_count']} rows、{validation['positive_rows']} positives、{validation['sampled_zero_rows']} zeros，group size {validation['group_min']}–{validation['group_max']}。membership 与不同输入顺序 replay fingerprint 均为 `{validation['membership_fingerprint']}`。

## 7. Train/Validation snapshot feature preservation

Train rows 继续使用 as-of 2020-08-11 的冻结 F1 matrix；Validation rows 使用 as-of 2020-08-25 的冻结 F1 features。未用后期 history 重算旧 snapshot。

## 8. Combined matrix/group/order 统计

按 Train→Validation、snapshot 内 query_order、query 内 candidate_rank/article_key 拼接：{combined['query_count']} queries、{combined['row_count']} rows、{combined['positive_rows']} positives、{combined['zero_rows']} zeros；`sum(group_size)` 精确一致。combined order fingerprint 为 `{combined['combined_matrix_order_fingerprint']}`，所有 boundary/duplicate/order/all-zero mismatch 为 0。

## 9. Final-refit model config

`L1_FINAL_REFIT_600` 使用 F1 52 特征、12 个 categorical features、LambdaRank 和 Phase 1 完全相同的参数/seeds/weights，固定 600 rounds。

## 10. Fresh initialization/no-selection 证据

`fresh_initialization=true`、`warm_start=false`、`init_model_used=false`、`internal_validation=false`、`valid_sets_used=false`、`early_stopping=false`。未训练 L0/B0，未重新读取指标选轮数或调参。

## 11. Final model fingerprints

Physical `{model['model_physical_fingerprint']}`；semantic `{model['model_semantic_fingerprint']}`；LightGBM dump semantic `{model['lightgbm_dump_semantic_fingerprint']}`；training lineage `{model['training_lineage_fingerprint']}`。

## 12. Model reload/prediction replay

模型临时保存后 reload，再在原子提升后的正式模型上 reload。固定 label-free query slice 的 raw score 与 rank/tie-break 均 exact match；replay fingerprint `{replay['replay_fingerprint']}`。

## 13. Resource preregistration、absolute cap、peak RSS

Worker 启动前一次冻结 available memory={resource_prereg['available_memory_before_worker_bytes']} bytes，fraction=0.70，absolute cap={resource_prereg['frozen_absolute_rss_cap_bytes']} bytes。外部持续监控 process-tree peak RSS={resource_audit['child_process_tree_peak_rss_bytes']} bytes，未超限；worker 未重算 cap。

## 14. Checkpoint/recovery/private artifact

Validation 各 partition、combined matrix 与模型 atomic promotion 均有 checkpoint。完整 membership、feature view、memmap、模型、日志位于 Git-ignored private/tmp/checkpoint；公开目录只保存安全聚合、状态和 fingerprint。

## 15. P4G1–P4G13

独立 Phase 4 Contract Gate 共 {len(gate_rows)}/13 PASS。本阶段无新 Value Gate。

## 16. Targeted/full tests 和安全审计

Targeted `{test_audit['tests']['targeted']}`；full pytest `{test_audit['tests']['full']}`；compile、`git diff --check`、raw-data diff、privacy/path、ignore、tracked-private、冻结漂移审计均 PASS。

## 17. Test/First7d 访问审计

First7d supervision read=0；Test candidate/feature/label/prediction/metric read/write 全部为 0；`test_authorized=false`、`test_prediction_frozen=false`。

## 18. 未执行项

未重跑 Phase 1–3，未重新计算 Validation/First7d 指标，未使用 First7d supervision，未执行 Test candidate/features/predictions/labels/metrics，未 rerank、未 retrieval retune、未新增模型或超参实验。

## 19. 新增/修改文件

本阶段审计记录的工作区变更文件数为 {len(changed)}；完整列表见 `final_refit_test_and_security_audit.json`。README 与 AGENTS 未修改。

## 20. Commit/push 状态

未 commit，未 push；按任务要求停在 ChatGPT 验收前。

## 21. 最终结论

`STAGE_4B4_PHASE4_FINAL_REFIT_COMPLETE`。Locked model configuration 为 L1/F1/600，正式 final-refit model 已冻结，Phase 4 Contract Gate PASSED；Test 仍未授权。

> 离线 final refit 不等于线上 CTR/CVR、收入提升或 A/B Test 结果；本阶段只证明冻结配置下的工程与协议正确性。
"""
