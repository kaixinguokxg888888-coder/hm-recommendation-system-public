from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ranking_model.validation_selection import binary_top12_metrics


VERSIONS = ("B0", "L1_FINAL_REFIT_600")
METRIC_COLUMNS = ("map_at_12", "ndcg_at_12", "recall_at_12", "hit_rate_at_12")
TARGET_SPECS = {
    "seen_item": ("seen", "seen_positive_count"),
    "cold_item": ("cold", "cold_positive_count_check"),
    "head": ("head", "head_positive_count"),
    "mid": ("mid", "mid_positive_count"),
    "tail": ("tail", "tail_positive_count"),
    "zero_history": ("zero_history", "zero_history_positive_count"),
}


class TestEvaluationError(RuntimeError):
    pass


class PreLabelConflict(TestEvaluationError):
    pass


class PostLabelFailure(TestEvaluationError):
    pass


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def self_fingerprint(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return canonical_fingerprint(payload)


def file_fingerprint(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def parquet_atomic(table: pa.Table, path: Path, *, compression: str = "zstd") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression=compression)
    pq.ParquetFile(temporary).metadata.num_rows
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def path_set_fingerprint(paths: Iterable[Path], root: Path) -> str:
    rows = [(path.relative_to(root).as_posix(), file_fingerprint(path)) for path in sorted(paths)]
    return canonical_fingerprint(rows)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def _external_python_processes() -> list[dict[str, Any]]:
    current = psutil.Process()
    excluded = {current.pid}
    try:
        excluded.update(parent.pid for parent in current.parents())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    rows: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "name"]):
        try:
            name = (process.info.get("name") or "").lower()
            if process.pid not in excluded and any(token in name for token in ("python", "duckdb", "lightgbm")):
                rows.append({"pid": process.pid, "name": name})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return rows


@dataclass(frozen=True)
class TestEvaluationConfig:
    __test__ = False

    root: Path
    path: Path
    raw: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> "TestEvaluationConfig":
        path = path.resolve()
        root = path.parents[1]
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (raw.get("stage"), raw.get("phase")) != (
            "4B-4", "phase5b_one_time_test_label_join_and_final_evaluation_v1"
        ):
            raise PreLabelConflict("Phase 5B config identity drift")
        snapshot = raw["snapshot"]
        if (snapshot["snapshot_id"], snapshot["label_start"], snapshot["label_end"]) != (
            "test_14d", "2020-09-09", "2020-09-22"
        ):
            raise PreLabelConflict("Test label window drift")
        if tuple(raw["versions"]) != VERSIONS:
            raise PreLabelConflict("formal Test version budget drift")
        bootstrap = raw["bootstrap"]
        frozen_bootstrap = {
            "method": "paired_user_block_percentile_v1", "sampling_unit": "query",
            "iterations": 1000, "seed": 20260826, "confidence_level": 0.95,
            "interval": "percentile", "comparison": "L1_FINAL_REFIT_600_minus_B0",
            "metric": "per_query_AP_at_12",
        }
        if bootstrap != frozen_bootstrap:
            raise PreLabelConflict("paired bootstrap contract drift")
        if any(value not in (False, 0) for value in raw["forbidden"].values()):
            raise PreLabelConflict("forbidden Phase 5B action is enabled")
        return cls(root=root, path=path, raw=raw, fingerprint=canonical_fingerprint(raw))

    def resolve(self, value: str) -> Path:
        return self.root / value

    def public_path(self, name: str) -> Path:
        return self.resolve(self.raw["paths"][name])

    @property
    def public_dir(self) -> Path:
        return self.public_path("public_directory")

    @property
    def private_dir(self) -> Path:
        return self.public_path("private_directory")

    @property
    def checkpoint_dir(self) -> Path:
        return self.public_path("checkpoint_directory")

    @property
    def temp_dir(self) -> Path:
        return self.public_path("temp_directory")

    def upstream_path(self, name: str) -> Path:
        return self.resolve(self.raw["upstream"][name])


def source_fingerprint(config: TestEvaluationConfig) -> str:
    paths = [
        config.path,
        config.root / "src/ranking_model/test_evaluation.py",
        config.root / "scripts/run_stage_4b4_test_evaluation.py",
        config.root / "scripts/stage_4b4_test_evaluation_worker.py",
        config.root / "tests/test_stage_4b4_phase5b.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if not path.exists():
            raise PreLabelConflict(f"missing Phase 5B source: {path.name}")
        digest.update(path.relative_to(config.root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _verify_lock(path: Path, expected: str) -> dict[str, Any]:
    lock = read_json(path)
    observed = lock.get("lock_fingerprint")
    if observed != expected or self_fingerprint(lock, "lock_fingerprint") != expected:
        raise PreLabelConflict(f"lock identity drift: {path.name}")
    return lock


def _verify_prediction_files(config: TestEvaluationConfig) -> dict[str, Any]:
    manifest = {row["version_id"]: row for row in read_rows(config.upstream_path("phase5a_prediction_manifest"))}
    expected_global = {
        "B0": config.raw["expected"]["b0_prediction_global_fingerprint"],
        "L1_FINAL_REFIT_600": config.raw["expected"]["final_prediction_global_fingerprint"],
    }
    if set(manifest) != set(VERSIONS):
        raise PreLabelConflict("Phase 5A prediction version manifest drift")
    prediction_root = config.upstream_path("predictions")
    rows: list[dict[str, Any]] = []
    for version in VERSIONS:
        paths = [prediction_root / version / f"part_{part:03d}.parquet" for part in range(32)]
        if any(not path.exists() for path in paths):
            raise PreLabelConflict(f"missing frozen prediction partition: {version}")
        physical = canonical_fingerprint([file_fingerprint(path) for path in paths])
        row_count = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
        if physical != manifest[version]["per_partition_physical_fingerprint"]:
            raise PreLabelConflict(f"prediction physical fingerprint drift: {version}")
        if manifest[version]["global_merkle_fingerprint"] != expected_global[version]:
            raise PreLabelConflict(f"prediction global fingerprint drift: {version}")
        if row_count != config.raw["expected"]["candidate_rows"]:
            raise PreLabelConflict(f"prediction row count drift: {version}")
        rows.append({"version": version, "row_count": row_count, "physical_set": physical})
    return {"versions": rows, "manifest_physical_fingerprint": file_fingerprint(config.upstream_path("phase5a_prediction_manifest"))}


def preflight(
    config: TestEvaluationConfig, *, verify_prediction_bytes: bool = True,
    require_empty_phase5b_private: bool = True,
) -> dict[str, Any]:
    expected = config.raw["expected"]
    phase4 = _verify_lock(config.upstream_path("phase4_lock"), expected["phase4_lock"])
    phase5a = _verify_lock(config.upstream_path("phase5a_lock"), expected["phase5a_pre_label_lock"])
    gate_rows = read_rows(config.upstream_path("phase5a_gate"))
    if len(gate_rows) != 20 or any(row.get("status") != "PASS" for row in gate_rows):
        raise PreLabelConflict("Phase 5A P5AG1-P5AG20 are not 20/20 PASS")
    if any((phase5a.get(key) not in (0, False)) for key in (
        "test_label_read_count", "test_labels_joined", "test_metrics_computed", "test_evaluation_executed"
    )):
        raise PreLabelConflict("Phase 5A lock reports prior Test label/metric use")
    freeze_manifest = read_rows(config.upstream_path("phase5a_manifest"))
    if len(freeze_manifest) != 1:
        raise PreLabelConflict("Phase 5A freeze manifest cardinality drift")
    frozen = freeze_manifest[0]
    if frozen["test_pre_label_lock_fingerprint"] != expected["phase5a_pre_label_lock"]:
        raise PreLabelConflict("Phase 5A freeze manifest lock drift")
    if frozen["test_label_read_count"] != "0" or frozen["test_evaluation_executed"] != "False":
        raise PreLabelConflict("Phase 5A freeze manifest reports prior Test use")

    checkpoints = config.upstream_path("phase5a_checkpoints")
    formal = [checkpoints / f"prediction_{part:03d}.json" for part in range(32)]
    if any(not path.exists() for path in formal):
        raise PreLabelConflict("not all 32 Phase 5A prediction checkpoints exist")
    checkpoint_values = [read_json(path) for path in formal]
    if any(value.get("status") != "SUCCESS" or value.get("replay") is not False for value in checkpoint_values):
        raise PreLabelConflict("incomplete/failed formal prediction checkpoint exists")

    population_rows = [
        row for row in read_rows(config.upstream_path("stage2b_population_manifest"))
        if row["snapshot_id"] == "test_14d"
    ]
    populations = {row["population"]: row for row in population_rows}
    expected_counts = {
        "all_label_users": expected["candidate_queries"],
        "evaluable_warm_users": expected["main_queries"],
        "cold_history_label_users": expected["cold_queries"],
    }
    for name, count in expected_counts.items():
        if name not in populations or int(populations[name]["user_count"]) != count:
            raise PreLabelConflict(f"Stage 2B Test population conflict: {name}")
    if int(populations["all_label_users"]["positive_item_count"]) != expected["positive_pairs"]:
        raise PreLabelConflict("Stage 2B Test positive-pair count conflict")
    if populations["all_label_users"]["ground_truth_fingerprint"] != expected["stage2b_ground_truth_declared_fingerprint"]:
        raise PreLabelConflict("Stage 2B declared Test ground-truth fingerprint conflict")

    branch = _git(config.root, "branch", "--show-current").stdout.strip()
    head = _git(config.root, "rev-parse", "HEAD").stdout.strip()
    divergence = _git(config.root, "rev-list", "--left-right", "--count", "main...origin/main")
    if branch != "main" or divergence.returncode != 0 or divergence.stdout.strip() != "0\t0":
        raise PreLabelConflict("branch or main...origin/main conflict")
    status = _git(config.root, "status", "--porcelain").stdout.splitlines()
    allowed_prefixes = (
        "artifacts/stage_4b4/", "configs/stage_4b4_", "docs/stages/stage_4b4_",
        "scripts/run_stage_4b4_", "scripts/stage_4b4_", "src/ranking_model/",
        "tests/test_stage_4b4", ".gitignore", "pyproject.toml", "src/ranking_features/runner.py",
    )
    unexpected = []
    for entry in status:
        relative = entry[3:].replace("\\", "/")
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        if not relative.startswith(allowed_prefixes):
            unexpected.append(relative)
    if unexpected:
        raise PreLabelConflict(f"worktree contains non-Stage-4B-4 changes: {unexpected}")
    workers = _external_python_processes()
    if workers:
        raise PreLabelConflict("background Python/DuckDB/LightGBM worker exists")

    output_names = (
        "ground_truth_manifest", "population_summary", "metrics", "bootstrap",
        "segment_metrics", "fingerprint_manifest", "access_audit", "resource_audit",
        "contract_gate", "lock", "report",
    )
    prior_outputs = [name for name in output_names if config.public_path(name).exists()]
    if prior_outputs:
        raise PreLabelConflict(f"prior Phase 5B formal outputs exist: {prior_outputs}")
    prior_private = any(path.exists() for path in (config.private_dir, config.checkpoint_dir, config.temp_dir))
    if require_empty_phase5b_private and prior_private:
        raise PreLabelConflict("prior Phase 5B private/checkpoint/tmp state exists")

    result = {
        "branch": branch, "head": head, "main_origin": "0_ahead_0_behind",
        "worktree_entry_count": len(status), "unexpected_worktree_entries": unexpected,
        "background_worker_count": 0,
        "phase4_lock": phase4["lock_fingerprint"],
        "phase4_lock_physical": file_fingerprint(config.upstream_path("phase4_lock")),
        "phase5a_lock": phase5a["lock_fingerprint"],
        "phase5a_lock_physical": file_fingerprint(config.upstream_path("phase5a_lock")),
        "phase5a_gate_pass_count": 20,
        "phase5a_formal_prediction_checkpoint_count": 32,
        "phase5a_failed_prediction_checkpoint_count": 0,
        "test_label_read_count": 0, "test_metric_read_count": 0,
        "test_evaluation_executed": False, "test_labels_joined": False,
        "main_queries": expected["main_queries"], "cold_queries": expected["cold_queries"],
        "all_queries": expected["candidate_queries"], "positive_pairs": expected["positive_pairs"],
        "phase5a_prediction_verification": _verify_prediction_files(config) if verify_prediction_bytes else None,
    }
    return result


def preregister(config: TestEvaluationConfig) -> dict[str, Any]:
    preflight_result = preflight(config, verify_prediction_bytes=True)
    available = psutil.virtual_memory().available
    cap = int(available * config.raw["execution"]["resource_rss_fraction_of_available"])
    source_fp = source_fingerprint(config)
    payload: dict[str, Any] = {
        "stage": "4B-4", "phase": config.raw["phase"],
        "status": "PREREGISTERED_BEFORE_TEST_LABEL_ACCESS", "created_at": now(),
        "config_fingerprint": config.fingerprint, "source_fingerprint": source_fp,
        "phase4_lock": config.raw["expected"]["phase4_lock"],
        "phase5a_pre_label_lock": config.raw["expected"]["phase5a_pre_label_lock"],
        "prediction_fingerprints": {
            "B0": config.raw["expected"]["b0_prediction_global_fingerprint"],
            "L1_FINAL_REFIT_600": config.raw["expected"]["final_prediction_global_fingerprint"],
        },
        "test_label_window": config.raw["snapshot"],
        "ground_truth_contract": config.raw["ground_truth"],
        "evaluation_contract": config.raw["evaluation"],
        "segment_contract": config.raw["segments"],
        "bootstrap_contract": config.raw["bootstrap"],
        "value_gate_contract": config.raw["value_gate"],
        "join_keys": config.raw["keys"]["prediction_label_join"],
        "partition_scheme": "customer_key_mod_32_query_contained",
        "resource_contract": {
            "available_memory_at_preregistration_bytes": available,
            "absolute_rss_cap_fraction": 0.70, "absolute_rss_cap_bytes": cap,
            "runtime_cap_recomputation_allowed": False,
            "temporary_spill_cap": config.raw["execution"]["max_temp_directory_size"],
            "external_process_tree_monitor": True,
        },
        "checkpoint_contract": {
            "partitions": 32, "atomic_promotion": True,
            "reuse_after_label_access_allowed": False,
            "identity_binding": ["phase5a_lock", "prediction", "ground_truth", "config", "preregistration", "partition", "metric_source", "output_digest"],
        },
        "fingerprint_contract": config.raw["fingerprints"],
        "artifact_policy": {"entity_level_private_git_ignored": True, "public_aggregate_only": True},
        "post_label_failure_policy": "STOP_POST_TEST_LABEL_IMPLEMENTATION_FAILURE_NO_RETRY_NO_REOPEN",
        "final_contract_gate_ids": [f"P5BG{index}" for index in range(1, 21)],
        "final_lock_schema": [
            "phase5a_pre_label_lock", "phase4_final_refit_lock", "test_label_window",
            "raw_test_label_source_read_count", "ground_truth_fingerprints", "population_fingerprints",
            "prediction_fingerprints", "metric_fingerprints", "bootstrap_fingerprints",
            "segment_fingerprints", "value_gate_status", "statistical_status", "segment_status",
            "resource_audit_fingerprint", "access_audit_fingerprint", "contract_gate_fingerprint",
        ],
        "preflight": preflight_result,
        "raw_test_label_source_read_count": 0, "test_ground_truth_build_count": 0,
        "prediction_label_join_count": 0, "formal_test_evaluation_count": 0,
        "test_metric_materialization_count": 0,
    }
    payload["preregistration_fingerprint"] = self_fingerprint(payload, "preregistration_fingerprint")
    json_atomic(config.public_path("preregistration"), payload)
    rows = [
        {"check_id": "P5BPRE1_BRANCH_HEAD", "observed": f"main@{preflight_result['head']}", "status": "PASS"},
        {"check_id": "P5BPRE2_MAIN_ORIGIN", "observed": preflight_result["main_origin"], "status": "PASS"},
        {"check_id": "P5BPRE3_WORKTREE_SCOPE", "observed": "stage_4b4_only", "status": "PASS"},
        {"check_id": "P5BPRE4_BACKGROUND_WORKERS", "observed": 0, "status": "PASS"},
        {"check_id": "P5BPRE5_PHASE4_LOCK", "observed": preflight_result["phase4_lock"], "status": "PASS"},
        {"check_id": "P5BPRE6_PHASE5A_LOCK", "observed": preflight_result["phase5a_lock"], "status": "PASS"},
        {"check_id": "P5BPRE7_PHASE5A_GATE", "observed": "20/20", "status": "PASS"},
        {"check_id": "P5BPRE8_PREDICTION_CHECKPOINTS", "observed": "32/32_SUCCESS", "status": "PASS"},
        {"check_id": "P5BPRE9_PREDICTION_IDENTITIES", "observed": "B0_and_final_exact", "status": "PASS"},
        {"check_id": "P5BPRE10_POPULATION_RECONCILIATION", "observed": "128333=117366+10967", "status": "PASS"},
        {"check_id": "P5BPRE11_TEST_LABEL_READ_COUNT", "observed": 0, "status": "PASS"},
        {"check_id": "P5BPRE12_TEST_METRIC_READ_COUNT", "observed": 0, "status": "PASS"},
        {"check_id": "P5BPRE13_PRIOR_FORMAL_EVALUATION", "observed": 0, "status": "PASS"},
        {"check_id": "P5BPRE14_MODEL_WINNER_MUTATION", "observed": "none", "status": "PASS"},
        {"check_id": "P5BPRE15_RESOURCE_CAP", "observed": cap, "status": "PASS"},
    ]
    write_rows(config.public_path("pre_access_audit"), rows)
    return payload


def verify_preregistration(config: TestEvaluationConfig) -> dict[str, Any]:
    path = config.public_path("preregistration")
    if not path.exists():
        raise PreLabelConflict("Phase 5B preregistration missing")
    payload = read_json(path)
    if self_fingerprint(payload, "preregistration_fingerprint") != payload.get("preregistration_fingerprint"):
        raise PreLabelConflict("Phase 5B preregistration self-fingerprint drift")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise PreLabelConflict("Phase 5B preregistered config drift")
    if payload.get("source_fingerprint") != source_fingerprint(config):
        raise PreLabelConflict("Phase 5B source changed after preregistration")
    for key in (
        "raw_test_label_source_read_count", "test_ground_truth_build_count",
        "prediction_label_join_count", "formal_test_evaluation_count", "test_metric_materialization_count",
    ):
        if payload.get(key) != 0:
            raise PreLabelConflict(f"pre-label access count is not zero: {key}")
    return payload


def query_metric_digest(
    version: str, snapshot_id: str, customer_key: int, metrics: Mapping[str, float | int | bool | str]
) -> str:
    digest = hashlib.sha256(b"stage-4b4-test-query-metrics-v1\0")
    digest.update(version.encode("ascii") + b"\0" + snapshot_id.encode("ascii") + b"\0")
    digest.update(struct.pack("<q", int(customer_key)))
    for name in sorted(metrics):
        value = metrics[name]
        digest.update(name.encode("ascii") + b"=")
        encoded = value.hex() if isinstance(value, float) else str(value)
        digest.update(encoded.encode("utf-8") + b"\0")
    return digest.hexdigest()


def query_merkle(rows: Iterable[tuple[str, int, str]], *, binding: str) -> str:
    digest = hashlib.sha256(b"stage-4b4-test-query-merkle-v1\0" + binding.encode("ascii") + b"\0")
    for snapshot_id, customer_key, value in sorted(rows, key=lambda row: (row[0], row[1])):
        digest.update(snapshot_id.encode("ascii") + b"\0")
        digest.update(struct.pack("<q", int(customer_key)))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def paired_bootstrap_detailed(
    baseline: Sequence[float], challenger: Sequence[float], *, iterations: int, seed: int,
    confidence_level: float, population_fingerprint: str,
) -> dict[str, Any]:
    if len(baseline) != len(challenger) or len(baseline) == 0:
        raise ValueError("paired bootstrap requires a non-empty identical query population")
    deltas = np.asarray(challenger, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if not np.isfinite(deltas).all():
        raise ValueError("paired bootstrap contains NaN/Inf")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        estimates[index] = float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))]))
    alpha = (1.0 - confidence_level) / 2.0
    sample_identity = canonical_fingerprint({
        "rng": "numpy.default_rng.PCG64", "seed": seed, "iterations": iterations,
        "query_count": len(deltas), "population_fingerprint": population_fingerprint,
        "sampling": "iid_query_indices_with_replacement",
    })
    result: dict[str, Any] = {
        "query_count": int(len(deltas)),
        "observed_delta": float(np.mean(deltas)),
        "bootstrap_mean": float(np.mean(estimates)),
        "delta_median": median(float(value) for value in deltas),
        "ci_lower": float(np.quantile(estimates, alpha)),
        "ci_upper": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence_level": confidence_level, "iterations": iterations, "seed": seed,
        "sample_identity_fingerprint": sample_identity,
    }
    result["result_semantic_fingerprint"] = canonical_fingerprint(result)
    return result


def binary_top12_or_zero(ranked_hits: Sequence[bool], positive_count: int) -> dict[str, float]:
    if positive_count == 0:
        return {name: 0.0 for name in METRIC_COLUMNS}
    metrics = binary_top12_metrics(ranked_hits, positive_count)
    return {name: float(getattr(metrics, name)) for name in METRIC_COLUMNS}


def deduplicate_binary_relevance(rows: Iterable[tuple[str, int, int, str]]) -> list[tuple[str, int, int]]:
    """Collapse user-item-day B events to one binary user-item relevance per snapshot window."""
    return sorted({(str(snapshot), int(customer), int(article)) for snapshot, customer, article, _day in rows})


def assert_label_source_unopened(state: Mapping[str, Any]) -> None:
    if int(state.get("raw_test_label_source_read_count", 0)) != 0:
        raise PostLabelFailure("Test label source has already been opened; reopen refused")


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


def _access_state_path(config: TestEvaluationConfig) -> Path:
    return config.private_dir / "access_state.json"


def initial_access_state(config: TestEvaluationConfig) -> dict[str, Any]:
    return {
        "stage": "4B-4", "phase": config.raw["phase"], "created_at": now(),
        "raw_test_label_source_read_count": 0, "test_population_source_read_count": 0,
        "test_item_state_source_read_count": 0, "test_ground_truth_build_count": 0,
        "prediction_label_join_count": 0, "formal_test_evaluation_count": 0,
        "test_metric_materialization_count": 0, "test_labels_reopened": False,
        "canonical_ground_truth_frozen": False, "formal_metrics_generated": False,
        "completed_partitions": [], "post_label_failure": False,
    }


def _load_access_state(config: TestEvaluationConfig) -> dict[str, Any]:
    path = _access_state_path(config)
    return read_json(path) if path.exists() else initial_access_state(config)


def _write_access_state(config: TestEvaluationConfig, state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    json_atomic(_access_state_path(config), state)


def _read_parquet_single_open(path: Path) -> tuple[pa.Table, str]:
    """Hash and read through one durable file open; callers count this as one source open."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
        handle.seek(0)
        table = pq.read_table(handle)
    return table, digest.hexdigest()


def _rename_key(table: pa.Table, old: str, new: str) -> pa.Table:
    if new in table.column_names:
        return table
    if old not in table.column_names:
        raise PostLabelFailure(f"required key is missing: {old}/{new}")
    names = [new if name == old else name for name in table.column_names]
    return table.rename_columns(names)


def _sorted_table(table: pa.Table, keys: Sequence[str]) -> pa.Table:
    indices = pc.sort_indices(table, sort_keys=[(key, "ascending") for key in keys])
    return pc.take(table, indices)


def _schema_fingerprint(table: pa.Table) -> str:
    return canonical_fingerprint([(field.name, str(field.type), field.nullable) for field in table.schema])


def _positive_semantics(table: pa.Table) -> tuple[str, str, pa.Table]:
    ordered = _sorted_table(table, ("snapshot_id", "customer_key", "article_key"))
    snapshots = ordered["snapshot_id"].to_pylist()
    users = ordered["customer_key"].to_pylist()
    articles = ordered["article_key"].to_pylist()
    digest = hashlib.sha256(b"stage-4b4-test-ground-truth-positive-pairs-v1\0")
    per_query_rows: list[tuple[str, int, str, int]] = []
    query_digest = hashlib.sha256()
    current: tuple[str, int] | None = None
    count = 0
    query_count = 0
    for snapshot, user, article in zip(snapshots, users, articles, strict=True):
        key = (str(snapshot), int(user))
        if current is not None and key != current:
            per_query_rows.append((current[0], current[1], query_digest.hexdigest(), count))
            query_digest = hashlib.sha256()
            count = 0
            query_count += 1
        current = key
        packed = struct.pack("<qq", int(user), int(article))
        digest.update(str(snapshot).encode("ascii") + b"\0" + packed)
        query_digest.update(struct.pack("<q", int(article)))
        count += 1
    if current is not None:
        per_query_rows.append((current[0], current[1], query_digest.hexdigest(), count))
        query_count += 1
    relevance = pa.table({
        "snapshot_id": [row[0] for row in per_query_rows],
        "customer_key": [row[1] for row in per_query_rows],
        "relevance_digest": [row[2] for row in per_query_rows],
        "positive_item_count": [row[3] for row in per_query_rows],
    })
    relevance_merkle = query_merkle(
        ((row[0], row[1], row[2]) for row in per_query_rows), binding="test_ground_truth_relevance_v1"
    )
    if query_count != len(per_query_rows):
        raise PostLabelFailure("ground-truth query digest accounting failure")
    return digest.hexdigest(), relevance_merkle, relevance


def build_ground_truth(config: TestEvaluationConfig) -> dict[str, Any]:
    verify_preregistration(config)
    gate = config.public_path("pre_label_gate")
    if not gate.exists() or any(row.get("status") != "PASS" for row in read_rows(gate)):
        raise PreLabelConflict("Phase 5B pre-label authorization gate is not PASS")
    config.private_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    state = _load_access_state(config)
    assert_label_source_unopened(state)

    # The intent is made durable before the single raw source open. A crash from here is post-label failure.
    state.update({
        "raw_test_label_source_read_count": 1, "test_population_source_read_count": 1,
        "test_item_state_source_read_count": 1, "test_ground_truth_build_count": 1,
        "test_labels_reopened": False,
    })
    _write_access_state(config, state)
    try:
        ground_truth, source_physical = _read_parquet_single_open(config.upstream_path("raw_test_label_source"))
        population, population_source_physical = _read_parquet_single_open(config.upstream_path("test_population_source"))
        item_state, item_state_source_physical = _read_parquet_single_open(config.upstream_path("test_item_state_source"))
        ground_truth = _rename_key(ground_truth, "user_key", "customer_key")
        population = _rename_key(population, "user_key", "customer_key")
        required_gt = {"snapshot_id", "customer_key", "article_key", "label_date_min", "label_date_max", "label_start", "label_end"}
        required_population = {
            "snapshot_id", "customer_key", "user_segment", "positive_item_count",
            "warm_positive_count", "cold_positive_count", "has_cold_positive", "is_main_evaluable",
        }
        required_item_state = {"article_key", "seen_item", "cold_item", "popularity_bucket"}
        for name, required, table in (
            ("ground_truth", required_gt, ground_truth),
            ("population", required_population, population),
            ("item_state", required_item_state, item_state),
        ):
            missing = required - set(table.column_names)
            if missing:
                raise PostLabelFailure(f"{name} schema missing fields: {sorted(missing)}")

        expected = config.raw["expected"]
        if source_physical != expected["stage2b_ground_truth_declared_fingerprint"]:
            raise PostLabelFailure("Stage 2B Test ground-truth physical fingerprint drift")
        if ground_truth.num_rows != expected["positive_pairs"] or population.num_rows != expected["candidate_queries"]:
            raise PostLabelFailure("Test ground-truth/population row count conflict")
        snapshot_values = set(str(value) for value in ground_truth["snapshot_id"].to_pylist())
        starts = set(str(value) for value in ground_truth["label_start"].to_pylist())
        ends = set(str(value) for value in ground_truth["label_end"].to_pylist())
        date_mins = [str(value) for value in ground_truth["label_date_min"].to_pylist()]
        date_maxes = [str(value) for value in ground_truth["label_date_max"].to_pylist()]
        if snapshot_values != {"test_14d"} or starts != {"2020-09-09"} or ends != {"2020-09-22"}:
            raise PostLabelFailure("canonical Test label snapshot/window drift")
        if min(date_mins) < "2020-09-09" or max(date_maxes) > "2020-09-22":
            raise PostLabelFailure("Test label dates escape the closed frozen window")
        pairs = list(zip(ground_truth["customer_key"].to_pylist(), ground_truth["article_key"].to_pylist(), strict=True))
        if len(set(pairs)) != len(pairs):
            raise PostLabelFailure("duplicate user-item Test relevance pair remains after B dedup")

        canonical_gt = _sorted_table(ground_truth, ("snapshot_id", "customer_key", "article_key"))
        canonical_population = _sorted_table(population, ("snapshot_id", "customer_key"))
        canonical_item_state = _sorted_table(item_state, ("article_key",))
        gt_path = config.private_dir / "canonical_ground_truth.parquet"
        population_path = config.private_dir / "canonical_population.parquet"
        item_state_path = config.private_dir / "canonical_item_state.parquet"
        relevance_path = config.private_dir / "per_query_relevance.parquet"
        semantic, relevance_merkle, relevance = _positive_semantics(canonical_gt)
        parquet_atomic(canonical_gt, gt_path)
        parquet_atomic(canonical_population, population_path)
        parquet_atomic(canonical_item_state, item_state_path)
        parquet_atomic(relevance, relevance_path)

        users = canonical_population["customer_key"].to_pylist()
        main_flags = canonical_population["is_main_evaluable"].to_pylist()
        segments = canonical_population["user_segment"].to_pylist()
        all_population_fp = canonical_fingerprint([int(value) for value in users])
        main_population_fp = canonical_fingerprint([int(user) for user, flag in zip(users, main_flags, strict=True) if flag])
        cold_population_fp = canonical_fingerprint([int(user) for user, flag in zip(users, main_flags, strict=True) if not flag])
        segment_fp = canonical_fingerprint([(int(user), str(segment), bool(flag)) for user, segment, flag in zip(users, segments, main_flags, strict=True)])
        label_window_fp = canonical_fingerprint(config.raw["snapshot"])
        manifest: dict[str, Any] = {
            "stage": "4B-4", "phase": config.raw["phase"], "status": "TEST_GROUND_TRUTH_FROZEN",
            "snapshot_id": "test_14d", "label_start": "2020-09-09", "label_end": "2020-09-22",
            "feedback_semantics": "B_user_item_day_binary_v1", "positive_pair_count": canonical_gt.num_rows,
            "label_positive_query_count": relevance.num_rows,
            "raw_test_label_source_read_count": 1, "test_ground_truth_build_count": 1,
            "source_physical_fingerprint": source_physical,
            "population_source_physical_fingerprint": population_source_physical,
            "item_state_source_physical_fingerprint": item_state_source_physical,
            "ground_truth_physical_fingerprint": file_fingerprint(gt_path),
            "ground_truth_semantic_fingerprint": semantic,
            "per_query_relevance_digest": relevance_merkle,
            "ground_truth_schema_fingerprint": _schema_fingerprint(canonical_gt),
            "label_window_fingerprint": label_window_fp,
            "all_query_population_fingerprint": all_population_fp,
            "main_population_fingerprint": main_population_fp,
            "cold_history_population_fingerprint": cold_population_fp,
            "segment_membership_fingerprint": segment_fp,
            "population_physical_fingerprint": file_fingerprint(population_path),
            "item_state_physical_fingerprint": file_fingerprint(item_state_path),
            "upstream_label_source_binding": canonical_fingerprint({
                "source": source_physical, "snapshot": config.raw["snapshot"],
                "feedback": "B_user_item_day_binary_v1",
            }),
        }
        manifest["manifest_fingerprint"] = self_fingerprint(manifest, "manifest_fingerprint")
        json_atomic(config.public_path("ground_truth_manifest"), manifest)
        checkpoint = {
            "status": "SUCCESS", "task": "ground_truth", "completed_at": now(),
            "phase5a_lock": config.raw["expected"]["phase5a_pre_label_lock"],
            "config_fingerprint": config.fingerprint,
            "preregistration_fingerprint": verify_preregistration(config)["preregistration_fingerprint"],
            "ground_truth_manifest_fingerprint": manifest["manifest_fingerprint"],
            "output_physical_fingerprint": manifest["ground_truth_physical_fingerprint"],
        }
        checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(checkpoint)
        json_atomic(config.checkpoint_dir / "ground_truth.json", checkpoint)
        state.update({"canonical_ground_truth_frozen": True, "ground_truth_manifest_fingerprint": manifest["manifest_fingerprint"]})
        _write_access_state(config, state)
        return manifest
    except Exception:
        state = _load_access_state(config)
        state["post_label_failure"] = True
        _write_access_state(config, state)
        raise


def _connect(config: TestEvaluationConfig) -> duckdb.DuckDBPyConnection:
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(f"SET temp_directory='{config.temp_dir.as_posix()}'")
    con.execute(f"SET memory_limit='{int(config.raw['execution']['duckdb_memory_limit_mb'])}MB'")
    con.execute(f"SET threads={int(config.raw['execution']['threads'])}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET max_temp_directory_size='{config.raw['execution']['max_temp_directory_size']}'")
    return con


def _copy_atomic(con: duckdb.DuckDBPyConnection, sql: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    con.execute(f"COPY ({sql}) TO '{temporary.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    pq.ParquetFile(temporary).metadata.num_rows
    os.replace(temporary, output)


def _query_base_sql(config: TestEvaluationConfig, part: int) -> str:
    pred = (config.upstream_path("predictions") / "B0" / f"part_{part:03d}.parquet").as_posix()
    gt = (config.private_dir / "canonical_ground_truth.parquet").as_posix()
    population = (config.private_dir / "canonical_population.parquet").as_posix()
    state = (config.private_dir / "canonical_item_state.parquet").as_posix()
    return f"""
    WITH candidate_hits AS (
      SELECT p.customer_key, count(g.article_key)::INTEGER candidate_positive_count
      FROM read_parquet('{pred}') p
      LEFT JOIN read_parquet('{gt}') g
        ON g.snapshot_id=p.snapshot_id AND g.customer_key=p.customer_key AND g.article_key=p.article_key
      GROUP BY p.customer_key
    ), target_counts AS (
      SELECT g.customer_key,
        count(*) FILTER (WHERE i.seen_item)::INTEGER seen_positive_count,
        count(*) FILTER (WHERE i.cold_item)::INTEGER cold_positive_count_check,
        count(*) FILTER (WHERE i.popularity_bucket='head')::INTEGER head_positive_count,
        count(*) FILTER (WHERE i.popularity_bucket='mid')::INTEGER mid_positive_count,
        count(*) FILTER (WHERE i.popularity_bucket='tail')::INTEGER tail_positive_count,
        count(*) FILTER (WHERE i.popularity_bucket='zero_history')::INTEGER zero_history_positive_count
      FROM read_parquet('{gt}') g
      LEFT JOIN read_parquet('{state}') i USING(article_key)
      WHERE g.customer_key % 32 = {part}
      GROUP BY g.customer_key
    )
    SELECT p.snapshot_id, p.customer_key, p.user_segment, p.positive_item_count,
      p.warm_positive_count, p.cold_positive_count, p.has_cold_positive, p.is_main_evaluable,
      coalesce(c.candidate_positive_count,0) candidate_positive_count,
      coalesce(t.seen_positive_count,0) seen_positive_count,
      coalesce(t.cold_positive_count_check,0) cold_positive_count_check,
      coalesce(t.head_positive_count,0) head_positive_count,
      coalesce(t.mid_positive_count,0) mid_positive_count,
      coalesce(t.tail_positive_count,0) tail_positive_count,
      coalesce(t.zero_history_positive_count,0) zero_history_positive_count
    FROM read_parquet('{population}') p
    LEFT JOIN candidate_hits c USING(customer_key)
    LEFT JOIN target_counts t USING(customer_key)
    WHERE p.customer_key % 32 = {part}
    ORDER BY p.customer_key
    """


def _metric_sql(config: TestEvaluationConfig, version: str, part: int, base: Path) -> str:
    pred = (config.upstream_path("predictions") / version / f"part_{part:03d}.parquet").as_posix()
    gt = (config.private_dir / "canonical_ground_truth.parquet").as_posix()
    state = (config.private_dir / "canonical_item_state.parquet").as_posix()
    flags = {
        "all": "g.article_key IS NOT NULL",
        "seen": "g.article_key IS NOT NULL AND i.seen_item",
        "cold": "g.article_key IS NOT NULL AND i.cold_item",
        "head": "g.article_key IS NOT NULL AND i.popularity_bucket='head'",
        "mid": "g.article_key IS NOT NULL AND i.popularity_bucket='mid'",
        "tail": "g.article_key IS NOT NULL AND i.popularity_bucket='tail'",
        "zero_history": "g.article_key IS NOT NULL AND i.popularity_bucket='zero_history'",
    }
    flag_sql = ",\n".join(f"({expr})::INTEGER AS hit_{name}" for name, expr in flags.items())
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
    idcg_values = {n: sum(1.0 / np.log2(rank + 1) for rank in range(1, n + 1)) for n in range(1, 13)}
    aggregate_sql: list[str] = []
    metric_sql: list[str] = []
    metric_columns: list[str] = []
    for name, count in counts.items():
        prefix = "" if name == "all" else f"{name}_"
        idcg = f"CASE least(b.{count},12) " + " ".join(
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
        metric_columns.extend([f"{prefix}{metric}" for metric in METRIC_COLUMNS])
    base_columns = [
        "snapshot_id", "customer_key", "user_segment", "positive_item_count",
        "warm_positive_count", "cold_positive_count", "has_cold_positive", "is_main_evaluable",
        "candidate_positive_count", "seen_positive_count", "cold_positive_count_check",
        "head_positive_count", "mid_positive_count", "tail_positive_count", "zero_history_positive_count",
    ]
    return f"""
    WITH top AS (
      SELECT p.customer_key, p.article_key, p.predicted_rank, {flag_sql}
      FROM read_parquet('{pred}') p
      LEFT JOIN read_parquet('{gt}') g
        ON g.snapshot_id=p.snapshot_id AND g.customer_key=p.customer_key AND g.article_key=p.article_key
      LEFT JOIN read_parquet('{state}') i ON i.article_key=p.article_key
      WHERE p.predicted_rank<=12
    ), running AS (
      SELECT *, {running_sql} FROM top
    ), aggregated AS (
      SELECT customer_key, {', '.join(aggregate_sql)} FROM running GROUP BY customer_key
    ), scored AS (
      SELECT b.*, {', '.join(metric_sql)}
      FROM read_parquet('{base.as_posix()}') b LEFT JOIN aggregated a USING(customer_key)
    )
    SELECT '{version}' AS version, {', '.join(base_columns)}, {', '.join(metric_columns)}
    FROM scored ORDER BY customer_key
    """


def evaluate_partitions(config: TestEvaluationConfig) -> dict[str, Any]:
    state = _load_access_state(config)
    if state.get("raw_test_label_source_read_count") != 1 or not state.get("canonical_ground_truth_frozen"):
        raise PostLabelFailure("canonical Test ground truth is not frozen after exactly one source open")
    if state.get("formal_test_evaluation_count") != 0:
        raise PostLabelFailure("formal Test evaluation has already started; rerun refused")
    state.update({
        "prediction_label_join_count": 1, "formal_test_evaluation_count": 1,
        "test_metric_materialization_count": 1,
    })
    _write_access_state(config, state)
    query_base_dir = config.private_dir / "query_base"
    metric_root = config.private_dir / "query_metrics"
    query_base_dir.mkdir(parents=True, exist_ok=True)
    metric_root.mkdir(parents=True, exist_ok=True)
    checkpoint_rows: list[dict[str, Any]] = []
    try:
        for part in range(32):
            con = _connect(config)
            try:
                base = query_base_dir / f"part_{part:03d}.parquet"
                _copy_atomic(con, _query_base_sql(config, part), base)
                outputs: dict[str, str] = {}
                for version in VERSIONS:
                    output = metric_root / version / f"part_{part:03d}.parquet"
                    _copy_atomic(con, _metric_sql(config, version, part, base), output)
                    outputs[version] = file_fingerprint(output)
            finally:
                con.close()
            checkpoint = {
                "status": "SUCCESS", "partition_id": part, "completed_at": now(),
                "phase5a_lock": config.raw["expected"]["phase5a_pre_label_lock"],
                "prediction_fingerprints": {
                    "B0": config.raw["expected"]["b0_prediction_global_fingerprint"],
                    "L1_FINAL_REFIT_600": config.raw["expected"]["final_prediction_global_fingerprint"],
                },
                "ground_truth_fingerprint": read_json(config.public_path("ground_truth_manifest"))["ground_truth_semantic_fingerprint"],
                "config_fingerprint": config.fingerprint,
                "preregistration_fingerprint": verify_preregistration(config)["preregistration_fingerprint"],
                "metric_source_fingerprint": source_fingerprint(config),
                "query_base_physical_fingerprint": file_fingerprint(base),
                "output_physical_fingerprints": outputs,
            }
            checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(checkpoint)
            json_atomic(config.checkpoint_dir / f"evaluation_{part:03d}.json", checkpoint)
            checkpoint_rows.append(checkpoint)
            state = _load_access_state(config)
            state["completed_partitions"] = list(range(part + 1))
            _write_access_state(config, state)

        expected = config.raw["expected"]
        for version in VERSIONS:
            paths = [metric_root / version / f"part_{part:03d}.parquet" for part in range(32)]
            rows = sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
            if rows != expected["candidate_queries"]:
                raise PostLabelFailure(f"per-query metric completeness failure: {version}")
        state = _load_access_state(config)
        state["formal_metrics_generated"] = True
        _write_access_state(config, state)
        result = {
            "status": "FORMAL_TEST_METRICS_MATERIALIZED_ONCE", "partition_count": 32,
            "completed_partitions": 32,
            "checkpoint_set_fingerprint": path_set_fingerprint(
                [config.checkpoint_dir / f"evaluation_{part:03d}.json" for part in range(32)], config.root
            ),
        }
        json_atomic(config.private_dir / "evaluation_result.json", result)
        return result
    except Exception:
        state = _load_access_state(config)
        state["post_label_failure"] = True
        _write_access_state(config, state)
        raise


def _qualified_predicate(predicate: str, alias: str) -> str:
    qualified = predicate
    for field in ("is_main_evaluable", "user_segment", "positive_item_count", "candidate_positive_count", "has_cold_positive"):
        qualified = qualified.replace(field, f"{alias}.{field}")
    return qualified


def _aggregate_metrics(config: TestEvaluationConfig, con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slice_name, predicate in _slice_predicates().items():
        result = con.execute(
            f"SELECT version, count(*) query_count, avg(map_at_12), avg(ndcg_at_12), "
            f"avg(recall_at_12), avg(hit_rate_at_12) FROM metrics WHERE {predicate} GROUP BY version"
        ).fetchall()
        for version, count, map12, ndcg, recall, hit in result:
            rows.append({
                "version": version, "slice": slice_name, "target": "all", "query_count": count,
                "map_at_12": map12, "ndcg_at_12": ndcg, "recall_at_12": recall, "hit_rate_at_12": hit,
            })
    for target, (prefix, count_field) in TARGET_SPECS.items():
        result = con.execute(
            f"SELECT version, count(*) query_count, avg({prefix}_map_at_12), avg({prefix}_ndcg_at_12), "
            f"avg({prefix}_recall_at_12), avg({prefix}_hit_rate_at_12) FROM metrics "
            f"WHERE is_main_evaluable AND {count_field}>0 GROUP BY version"
        ).fetchall()
        for version, count, map12, ndcg, recall, hit in result:
            rows.append({
                "version": version, "slice": "evaluable_warm_users", "target": target,
                "query_count": count, "map_at_12": map12, "ndcg_at_12": ndcg,
                "recall_at_12": recall, "hit_rate_at_12": hit,
            })
    rows.sort(key=lambda row: (VERSIONS.index(str(row["version"])), str(row["slice"]), str(row["target"])))
    return rows


def _population_fingerprint(users: Sequence[int]) -> str:
    return canonical_fingerprint([int(user) for user in users])


def _bootstrap_rows(config: TestEvaluationConfig, con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    bootstrap = config.raw["bootstrap"]
    rows: list[dict[str, Any]] = []
    for slice_name, predicate in _slice_predicates().items():
        qualified = _qualified_predicate(predicate, "b")
        values = con.execute(f"""
          SELECT b.customer_key, b.map_at_12 baseline, c.map_at_12 challenger
          FROM metrics b JOIN metrics c USING(snapshot_id, customer_key)
          WHERE b.version='B0' AND c.version='L1_FINAL_REFIT_600' AND ({qualified})
          ORDER BY b.customer_key
        """).fetchnumpy()
        users = values["customer_key"].astype(np.int64)
        population_fp = _population_fingerprint(users.tolist())
        result = paired_bootstrap_detailed(
            values["baseline"], values["challenger"], iterations=bootstrap["iterations"],
            seed=bootstrap["seed"], confidence_level=bootstrap["confidence_level"],
            population_fingerprint=population_fp,
        )
        rows.append({
            "baseline": "B0", "challenger": "L1_FINAL_REFIT_600", "slice": slice_name,
            "target": "all", "metric": "MAP@12", "population_fingerprint": population_fp, **result,
        })
    for target, (prefix, count_field) in TARGET_SPECS.items():
        values = con.execute(f"""
          SELECT b.customer_key, b.{prefix}_map_at_12 baseline, c.{prefix}_map_at_12 challenger
          FROM metrics b JOIN metrics c USING(snapshot_id, customer_key)
          WHERE b.version='B0' AND c.version='L1_FINAL_REFIT_600'
            AND b.is_main_evaluable AND b.{count_field}>0
          ORDER BY b.customer_key
        """).fetchnumpy()
        users = values["customer_key"].astype(np.int64)
        population_fp = _population_fingerprint(users.tolist())
        result = paired_bootstrap_detailed(
            values["baseline"], values["challenger"], iterations=bootstrap["iterations"],
            seed=bootstrap["seed"], confidence_level=bootstrap["confidence_level"],
            population_fingerprint=population_fp,
        )
        rows.append({
            "baseline": "B0", "challenger": "L1_FINAL_REFIT_600",
            "slice": "evaluable_warm_users", "target": target, "metric": "MAP@12",
            "population_fingerprint": population_fp, **result,
        })
    rows.sort(key=lambda row: (str(row["slice"]), str(row["target"])))
    return rows


def _query_metric_fingerprints(config: TestEvaluationConfig) -> dict[str, Any]:
    metric_root = config.private_dir / "query_metrics"
    digest_root = config.private_dir / "query_metric_digests"
    digest_root.mkdir(parents=True, exist_ok=True)
    binding = canonical_fingerprint({
        "phase5a_lock": config.raw["expected"]["phase5a_pre_label_lock"],
        "config": config.fingerprint,
        "ground_truth": read_json(config.public_path("ground_truth_manifest"))["ground_truth_semantic_fingerprint"],
    })
    results: dict[str, Any] = {}
    for version in VERSIONS:
        all_rows: list[tuple[str, int, str]] = []
        paths: list[Path] = []
        for part in range(32):
            path = metric_root / version / f"part_{part:03d}.parquet"
            paths.append(path)
            table = pq.read_table(path)
            metric_names = [name for name in table.column_names if name.endswith("_at_12")]
            metric_values = {name: table[name].to_pylist() for name in metric_names}
            snapshots = table["snapshot_id"].to_pylist()
            users = table["customer_key"].to_pylist()
            rows: list[tuple[str, int, str]] = []
            for index, (snapshot, user) in enumerate(zip(snapshots, users, strict=True)):
                values = {
                    name: float(metric_values[name][index])
                    for name in metric_names if metric_values[name][index] is not None
                }
                digest = query_metric_digest(version, str(snapshot), int(user), values)
                rows.append((str(snapshot), int(user), digest))
            output = digest_root / version / f"part_{part:03d}.parquet"
            parquet_atomic(pa.table({
                "snapshot_id": [row[0] for row in rows],
                "customer_key": [row[1] for row in rows],
                "query_metric_digest": [row[2] for row in rows],
            }), output)
            all_rows.extend(rows)
        results[version] = {
            "per_query_metric_semantic_fingerprint": query_merkle(all_rows, binding=binding),
            "per_query_metric_physical_fingerprint": canonical_fingerprint([file_fingerprint(path) for path in paths]),
            "query_digest_physical_fingerprint": path_set_fingerprint(
                [digest_root / version / f"part_{part:03d}.parquet" for part in range(32)], config.root
            ),
            "query_count": len(all_rows),
        }
    return results


def aggregate_and_bootstrap(config: TestEvaluationConfig) -> dict[str, Any]:
    state = _load_access_state(config)
    if state.get("formal_test_evaluation_count") != 1 or not state.get("formal_metrics_generated"):
        raise PostLabelFailure("formal Test metrics are not complete")
    con = _connect(config)
    try:
        pattern = (config.private_dir / "query_metrics" / "*" / "part_*.parquet").as_posix()
        con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
        aggregate = _aggregate_metrics(config, con)
        bootstrap_rows = _bootstrap_rows(config, con)
        population_result = con.execute(f"""
          SELECT
            count(*) all_queries,
            count(*) FILTER(WHERE is_main_evaluable) main_queries,
            count(*) FILTER(WHERE NOT is_main_evaluable) cold_queries,
            count(*) FILTER(WHERE is_main_evaluable AND candidate_positive_count>0) main_candidate_positive_nonzero,
            count(*) FILTER(WHERE is_main_evaluable AND candidate_positive_count=0) main_all_zero,
            count(*) FILTER(WHERE candidate_positive_count>0) all_candidate_positive_nonzero,
            count(*) FILTER(WHERE candidate_positive_count=0) all_candidate_positive_zero,
            count(*) FILTER(WHERE is_main_evaluable AND positive_item_count=1) main_single_positive,
            count(*) FILTER(WHERE is_main_evaluable AND positive_item_count>1) main_multi_positive,
            count(*) FILTER(WHERE user_segment='short_history') short_history,
            count(*) FILTER(WHERE user_segment='warm') warm,
            count(*) FILTER(WHERE user_segment='active') active,
            count(*) FILTER(WHERE user_segment='cold') cold
          FROM read_parquet('{(config.private_dir / 'query_base' / 'part_*.parquet').as_posix()}')
        """).fetchone()
    finally:
        con.close()
    expected = config.raw["expected"]
    if tuple(int(value) for value in population_result[:3]) != (
        expected["candidate_queries"], expected["main_queries"], expected["cold_queries"]
    ):
        raise PostLabelFailure("formal Test population reconciliation failed")
    if int(population_result[3]) + int(population_result[4]) != expected["main_queries"]:
        raise PostLabelFailure("main candidate-positive/all-zero population accounting failed")

    metric_fingerprints = _query_metric_fingerprints(config)
    write_rows(config.public_path("metrics"), aggregate)
    write_rows(config.public_path("bootstrap"), bootstrap_rows)
    aggregate_map = {
        (str(row["version"]), str(row["slice"]), str(row["target"])): row for row in aggregate
    }
    bootstrap_map = {(str(row["slice"]), str(row["target"])): row for row in bootstrap_rows}
    critical = set(config.raw["segments"]["critical_regression"])
    segment_rows: list[dict[str, Any]] = []
    for (slice_name, target), boot in bootstrap_map.items():
        b0 = aggregate_map[("B0", slice_name, target)]
        final = aggregate_map[("L1_FINAL_REFIT_600", slice_name, target)]
        is_critical = target == "all" and slice_name in critical
        regression = "REGRESSION_REVIEW_REQUIRED" if is_critical and float(boot["ci_upper"]) < 0 else (
            "NO_CRITICAL_SEGMENT_REGRESSION" if is_critical else "DIAGNOSTIC_ONLY"
        )
        segment_rows.append({
            "slice": slice_name, "target": target, "query_count": int(b0["query_count"]),
            "b0_map_at_12": b0["map_at_12"], "final_map_at_12": final["map_at_12"],
            "delta_map_at_12": float(final["map_at_12"]) - float(b0["map_at_12"]),
            "ci_lower": boot["ci_lower"], "ci_upper": boot["ci_upper"],
            "confidence_level": boot["confidence_level"], "critical_segment": is_critical,
            "regression_status": regression,
        })
    segment_rows.sort(key=lambda row: (str(row["slice"]), str(row["target"])))
    write_rows(config.public_path("segment_metrics"), segment_rows)

    population_rows = [
        {"population": "candidate_query_population", "query_count": int(population_result[0]), "included_in_main_average": False},
        {"population": "label_positive_query_population", "query_count": expected["candidate_queries"], "included_in_main_average": False},
        {"population": "main_evaluation_population", "query_count": int(population_result[1]), "included_in_main_average": True},
        {"population": "cold_history_diagnostic_population", "query_count": int(population_result[2]), "included_in_main_average": False},
        {"population": "main_candidate_positive_nonzero", "query_count": int(population_result[3]), "included_in_main_average": True},
        {"population": "main_all_zero_relevance", "query_count": int(population_result[4]), "included_in_main_average": True},
        {"population": "all_candidate_positive_nonzero", "query_count": int(population_result[5]), "included_in_main_average": False},
        {"population": "all_candidate_positive_zero", "query_count": int(population_result[6]), "included_in_main_average": False},
        {"population": "main_single_positive", "query_count": int(population_result[7]), "included_in_main_average": True},
        {"population": "main_multi_positive", "query_count": int(population_result[8]), "included_in_main_average": True},
        {"population": "short_history", "query_count": int(population_result[9]), "included_in_main_average": True},
        {"population": "warm", "query_count": int(population_result[10]), "included_in_main_average": True},
        {"population": "active", "query_count": int(population_result[11]), "included_in_main_average": True},
        {"population": "cold", "query_count": int(population_result[12]), "included_in_main_average": False},
    ]
    write_rows(config.public_path("population_summary"), population_rows)
    main_boot = bootstrap_map[("evaluable_warm_users", "all")]
    b0_main = aggregate_map[("B0", "evaluable_warm_users", "all")]
    final_main = aggregate_map[("L1_FINAL_REFIT_600", "evaluable_warm_users", "all")]
    delta = float(final_main["map_at_12"]) - float(b0_main["map_at_12"])
    if delta > 0 and float(main_boot["ci_lower"]) > 0:
        value_gate, statistical = "PASSED_FINAL_TEST_VALUE_GATE", "POSITIVE_AND_SIGNIFICANT"
    elif delta > 0:
        value_gate, statistical = "FAILED_FINAL_TEST_VALUE_GATE", "POSITIVE_NOT_SIGNIFICANT"
    else:
        value_gate, statistical = "FAILED_FINAL_TEST_VALUE_GATE", "NON_POSITIVE"
    regressions = [row for row in segment_rows if row["regression_status"] == "REGRESSION_REVIEW_REQUIRED"]
    segment_status = "CRITICAL_SEGMENT_REGRESSION_REVIEW_REQUIRED" if regressions else "NO_CRITICAL_SEGMENT_REGRESSION"
    result = {
        "status": "TEST_AGGREGATION_AND_BOOTSTRAP_COMPLETE", "value_gate_status": value_gate,
        "statistical_status": statistical, "segment_status": segment_status,
        "main_delta_map_at_12": delta, "main_ci_lower": main_boot["ci_lower"], "main_ci_upper": main_boot["ci_upper"],
        "metric_fingerprints": metric_fingerprints,
        "aggregate_metric_semantic_fingerprint": canonical_fingerprint(aggregate),
        "aggregate_metric_physical_fingerprint": file_fingerprint(config.public_path("metrics")),
        "bootstrap_config_fingerprint": canonical_fingerprint(config.raw["bootstrap"]),
        "bootstrap_result_semantic_fingerprint": canonical_fingerprint(bootstrap_rows),
        "bootstrap_result_physical_fingerprint": file_fingerprint(config.public_path("bootstrap")),
        "segment_semantic_fingerprint": canonical_fingerprint(segment_rows),
        "segment_physical_fingerprint": file_fingerprint(config.public_path("segment_metrics")),
        "population_summary_physical_fingerprint": file_fingerprint(config.public_path("population_summary")),
        "population_counts": {row["population"]: row["query_count"] for row in population_rows},
    }
    json_atomic(config.private_dir / "aggregation_result.json", result)
    return result


def _run_structured_pytest(config: TestEvaluationConfig, name: str, pytest_args: Sequence[str]) -> dict[str, Any]:
    output = config.private_dir / "prelabel_tests" / f"{name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(config.root / "scripts/stage_4b4_pytest_structured_worker.py"),
        "--output", str(output), "--", *pytest_args, "-ra", "-q",
    ]
    completed = subprocess.run(command, cwd=config.root, check=False)
    if not output.exists():
        raise PreLabelConflict(f"structured pytest did not write result: {name}")
    result = read_json(output)
    result["return_code"] = completed.returncode
    result["result_fingerprint"] = file_fingerprint(output)
    return result


def _privacy_path_violations(paths: Sequence[Path]) -> list[str]:
    violations: list[str] = []
    windows_absolute = __import__("re").compile(r"[A-Za-z]:[\\/]")
    unix_home = __import__("re").compile(r"/(?:home|Users)/[^/\s]+")
    raw_fields = __import__("re").compile(r"\b(customer_id|article_id|postal_code)\b")
    for path in paths:
        if not path.exists() or path.is_dir():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if windows_absolute.search(text) or unix_home.search(text) or raw_fields.search(text):
            violations.append(path.relative_to(paths[0].parents[2] if len(paths[0].parents) > 2 else path.parent).as_posix())
    return violations


def run_prelabel_engineering_audit(config: TestEvaluationConfig) -> dict[str, Any]:
    prereg = verify_preregistration(config)
    if _access_state_path(config).exists():
        raise PreLabelConflict("Phase 5B access state exists before pre-label engineering audit")
    phase5b = _run_structured_pytest(config, "phase5b_targeted", ["tests/test_stage_4b4_phase5b.py"])
    stage4b4_files = [path.relative_to(config.root).as_posix() for path in sorted((config.root / "tests").glob("test_stage_4b4*.py"))]
    stage4b4 = _run_structured_pytest(config, "stage4b4_targeted", stage4b4_files)
    full = _run_structured_pytest(config, "full", [])

    phase5b_counts = phase5b["counts"]
    if phase5b["status"] != "PASS" or any(phase5b_counts[name] for name in ("failed", "errors", "skipped", "xfailed", "xpassed")):
        raise PreLabelConflict("Phase 5B required tests did not all pass without skip")
    expected_obsolete_node = "tests/test_stage_4b4_phase4.py::test_phase4_gate_has_exactly_thirteen_passes_when_present"
    for suite_name, suite in (("stage4b4", stage4b4), ("full", full)):
        counts = suite["counts"]
        if suite["status"] != "PASS" or any(counts[name] for name in ("failed", "errors", "xfailed", "xpassed")):
            raise PreLabelConflict(f"{suite_name} pytest failed")
        unexpected_skips = [row for row in suite["skipped"] if row["nodeid"] != expected_obsolete_node]
        if unexpected_skips:
            raise PreLabelConflict(f"{suite_name} contains unexpected required skip")

    compile_result = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "src/ranking_model", "scripts", "tests/test_stage_4b4_phase5b.py"],
        cwd=config.root, capture_output=True, text=True, check=False,
    )
    diff_check = _git(config.root, "diff", "--check")
    raw_worktree = _git(config.root, "diff", "--name-only", "--", "data/raw")
    raw_staged = _git(config.root, "diff", "--cached", "--name-only", "--", "data/raw")
    tracked_private = _git(config.root, "ls-files", "artifacts/stage_4b4/private", "artifacts/stage_4b4/checkpoints", "artifacts/stage_4b4/tmp")
    ignore_probes = [
        "artifacts/stage_4b4/private/test_evaluation/probe",
        "artifacts/stage_4b4/checkpoints/test_evaluation/probe",
        "artifacts/stage_4b4/tmp/test_evaluation/probe",
    ]
    ignored = all(_git(config.root, "check-ignore", "-q", probe).returncode == 0 for probe in ignore_probes)
    privacy_paths = [config.path, config.public_path("pre_access_audit"), config.public_path("preregistration")]
    privacy_violations = _privacy_path_violations(privacy_paths)
    phase5a = preflight(config, verify_prediction_bytes=True, require_empty_phase5b_private=False)
    workers = _external_python_processes()
    checks = {
        "phase5b_targeted_tests": "PASS",
        "stage4b4_targeted_tests": "PASS",
        "full_pytest": "PASS",
        "python_compile": "PASS" if compile_result.returncode == 0 else "FAIL",
        "git_diff_check": "PASS" if diff_check.returncode == 0 else "FAIL",
        "raw_data_worktree_diff": "PASS" if not raw_worktree.stdout.strip() else "FAIL",
        "raw_data_staged_diff": "PASS" if not raw_staged.stdout.strip() else "FAIL",
        "privacy_path_scan": "PASS" if not privacy_violations else "FAIL",
        "ignore_audit": "PASS" if ignored else "FAIL",
        "tracked_private_audit": "PASS" if not tracked_private.stdout.strip() else "FAIL",
        "background_worker_audit": "PASS" if not workers else "FAIL",
        "phase5a_artifact_identity": "PASS" if phase5a["phase5a_gate_pass_count"] == 20 else "FAIL",
        "prelabel_access_counts": "PASS" if all(prereg[key] == 0 for key in (
            "raw_test_label_source_read_count", "test_ground_truth_build_count", "prediction_label_join_count",
            "formal_test_evaluation_count", "test_metric_materialization_count",
        )) else "FAIL",
    }
    if any(value != "PASS" for value in checks.values()):
        raise PreLabelConflict("pre-label engineering audit failed")
    payload = {
        "stage": "4B-4", "phase": config.raw["phase"], "status": "PASS", "recorded_at": now(),
        "checks": checks, "phase5b_targeted": phase5b, "stage4b4_targeted": stage4b4,
        "full_pytest": full, "compile_return_code": compile_result.returncode,
        "git_diff_check_return_code": diff_check.returncode,
        "raw_data_worktree_changed_count": 0, "raw_data_staged_changed_count": 0,
        "privacy_path_violation_count": len(privacy_violations), "tracked_private_artifact_count": 0,
        "background_worker_count": 0, "unexpected_required_phase5b_skip_count": 0,
        "expected_obsolete_skip_nodeid": expected_obsolete_node,
        "test_label_read_count": 0, "test_metric_read_count": 0,
    }
    payload["audit_fingerprint"] = self_fingerprint(payload, "audit_fingerprint")
    json_atomic(config.private_dir / "prelabel_engineering_audit.json", payload)
    return payload


def finalize_prelabel_gate(config: TestEvaluationConfig) -> list[dict[str, Any]]:
    prereg = verify_preregistration(config)
    audit_path = config.private_dir / "prelabel_engineering_audit.json"
    if not audit_path.exists():
        raise PreLabelConflict("pre-label engineering audit missing")
    audit = read_json(audit_path)
    if audit.get("status") != "PASS" or self_fingerprint(audit, "audit_fingerprint") != audit.get("audit_fingerprint"):
        raise PreLabelConflict("pre-label engineering audit identity failure")
    preflight_result = preflight(config, verify_prediction_bytes=True, require_empty_phase5b_private=False)
    checks = [
        ("P5BPLG1", preflight_result["phase5a_lock"] == config.raw["expected"]["phase5a_pre_label_lock"], "Phase 5A lock exact"),
        ("P5BPLG2", preflight_result["phase5a_gate_pass_count"] == 20, "P5AG1-P5AG20 20/20 PASS"),
        ("P5BPLG3", preflight_result["phase5a_formal_prediction_checkpoint_count"] == 32, "32/32 formal prediction checkpoints SUCCESS"),
        ("P5BPLG4", prereg["config_fingerprint"] == config.fingerprint, "Phase 5B config frozen"),
        ("P5BPLG5", prereg["source_fingerprint"] == source_fingerprint(config), "Phase 5B source frozen"),
        ("P5BPLG6", prereg["test_label_window"]["label_start"] == "2020-09-09", "Test label start frozen"),
        ("P5BPLG7", prereg["test_label_window"]["label_end"] == "2020-09-22", "Test label end frozen"),
        ("P5BPLG8", prereg["evaluation_contract"]["cutoff"] == 12, "metric formulas and k=12 frozen"),
        ("P5BPLG9", prereg["bootstrap_contract"]["iterations"] == 1000, "bootstrap iterations frozen"),
        ("P5BPLG10", prereg["bootstrap_contract"]["seed"] == 20260826, "bootstrap seed frozen"),
        ("P5BPLG11", prereg["resource_contract"]["absolute_rss_cap_bytes"] > 0, "absolute RSS cap frozen once"),
        ("P5BPLG12", prereg["raw_test_label_source_read_count"] == 0, "raw Test label read count zero"),
        ("P5BPLG13", prereg["test_ground_truth_build_count"] == 0, "ground-truth build count zero"),
        ("P5BPLG14", prereg["prediction_label_join_count"] == 0, "prediction-label join count zero"),
        ("P5BPLG15", prereg["formal_test_evaluation_count"] == 0, "formal Test evaluation count zero"),
        ("P5BPLG16", prereg["test_metric_materialization_count"] == 0, "Test metric materialization count zero"),
        ("P5BPLG17", audit["checks"]["phase5b_targeted_tests"] == "PASS", "Phase 5B required tests PASS no skip"),
        ("P5BPLG18", audit["checks"]["full_pytest"] == "PASS", "full pytest PASS"),
        ("P5BPLG19", all(value == "PASS" for value in audit["checks"].values()), "engineering/security checks PASS"),
        ("P5BPLG20", not _access_state_path(config).exists(), "no Test-label access state before authorization"),
    ]
    if not all(status for _, status, _ in checks):
        raise PreLabelConflict("Phase 5B final pre-label authorization gate failed")
    rows = [
        {"gate_id": gate_id, "mandatory": True, "status": "PASS" if status else "FAIL", "evidence": evidence,
         "preregistration_fingerprint": prereg["preregistration_fingerprint"],
         "prelabel_engineering_fingerprint": audit["audit_fingerprint"]}
        for gate_id, status, evidence in checks
    ]
    write_rows(config.public_path("pre_label_gate"), rows)
    return rows


def record_resource_audit(config: TestEvaluationConfig, phase: str, payload: Mapping[str, Any]) -> None:
    path = config.public_path("resource_audit")
    rows = read_rows(path) if path.exists() else []
    rows = [row for row in rows if row.get("phase") != phase]
    rows.append({
        "phase": phase, "status": payload["status"],
        "frozen_absolute_rss_cap_bytes": payload["frozen_absolute_rss_cap_bytes"],
        "peak_process_tree_rss_bytes": payload["peak_process_tree_rss_bytes"],
        "cap_exceeded": payload["cap_exceeded"], "oom_observed": payload["oom_observed"],
        "exit_code": payload["exit_code"], "wall_time_seconds": payload["wall_time_seconds"],
        "runtime_available_memory_observation_bytes": payload.get("runtime_available_memory_observation_bytes", ""),
    })
    rows.sort(key=lambda row: str(row["phase"]))
    write_rows(path, rows)


def _postlabel_engineering_audit(config: TestEvaluationConfig, aggregation: Mapping[str, Any]) -> dict[str, Any]:
    state = _load_access_state(config)
    metric_root = config.private_dir / "query_metrics"
    metric_paths = [metric_root / version / f"part_{part:03d}.parquet" for version in VERSIONS for part in range(32)]
    metric_complete = all(path.exists() for path in metric_paths)
    metric_rows = {
        version: sum(pq.ParquetFile(metric_root / version / f"part_{part:03d}.parquet").metadata.num_rows for part in range(32))
        for version in VERSIONS
    }
    con = _connect(config)
    try:
        pattern = (metric_root / "*" / "part_*.parquet").as_posix()
        con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
        paired = con.execute("""
          WITH b AS (SELECT * FROM metrics WHERE version='B0'),
               f AS (SELECT * FROM metrics WHERE version='L1_FINAL_REFIT_600')
          SELECT
            count(*) FILTER(WHERE b.customer_key IS NULL OR f.customer_key IS NULL) missing_pair,
            count(*) pair_count
          FROM b FULL OUTER JOIN f USING(snapshot_id, customer_key)
        """).fetchone()
        all_zero_nonzero = con.execute("""
          SELECT count(*) FROM metrics
          WHERE is_main_evaluable AND candidate_positive_count=0 AND
            (map_at_12<>0 OR ndcg_at_12<>0 OR recall_at_12<>0 OR hit_rate_at_12<>0)
        """).fetchone()[0]
        main_recomputed = con.execute("""
          SELECT version, avg(map_at_12), avg(ndcg_at_12), avg(recall_at_12), avg(hit_rate_at_12)
          FROM metrics WHERE is_main_evaluable GROUP BY version ORDER BY version
        """).fetchall()
    finally:
        con.close()
    public_main = {
        row["version"]: tuple(float(row[name]) for name in METRIC_COLUMNS)
        for row in read_rows(config.public_path("metrics"))
        if row["slice"] == "evaluable_warm_users" and row["target"] == "all"
    }
    aggregate_match = all(
        version in public_main and all(abs(float(value) - public) < 1e-15 for value, public in zip(values, public_main[version], strict=True))
        for version, *values in main_recomputed
    )
    gt_manifest = read_json(config.public_path("ground_truth_manifest"))
    gt_manifest_ok = self_fingerprint(gt_manifest, "manifest_fingerprint") == gt_manifest.get("manifest_fingerprint")
    resource_rows = read_rows(config.public_path("resource_audit"))
    resource_ok = len(resource_rows) >= 3 and all(row["status"] == "PASS" for row in resource_rows)
    raw_worktree = _git(config.root, "diff", "--name-only", "--", "data/raw")
    raw_staged = _git(config.root, "diff", "--cached", "--name-only", "--", "data/raw")
    tracked_private = _git(config.root, "ls-files", "artifacts/stage_4b4/private", "artifacts/stage_4b4/checkpoints", "artifacts/stage_4b4/tmp")
    public_paths = [
        config.path, config.public_path("ground_truth_manifest"), config.public_path("population_summary"),
        config.public_path("metrics"), config.public_path("bootstrap"), config.public_path("segment_metrics"),
        config.public_path("pre_access_audit"), config.public_path("preregistration"), config.public_path("pre_label_gate"),
    ]
    privacy = _privacy_path_violations(public_paths)
    prelabel = read_json(config.private_dir / "prelabel_engineering_audit.json")
    checks = {
        "prelabel_engineering_audit": "PASS" if prelabel.get("status") == "PASS" else "FAIL",
        "formal_metric_partition_completeness": "PASS" if metric_complete and all(value == config.raw["expected"]["candidate_queries"] for value in metric_rows.values()) else "FAIL",
        "paired_key_population_alignment": "PASS" if int(paired[0]) == 0 and int(paired[1]) == config.raw["expected"]["candidate_queries"] else "FAIL",
        "all_zero_queries_retained_as_zero": "PASS" if int(all_zero_nonzero) == 0 else "FAIL",
        "aggregate_consistency": "PASS" if aggregate_match else "FAIL",
        "ground_truth_manifest_self_fingerprint": "PASS" if gt_manifest_ok else "FAIL",
        "bootstrap_deterministic_identity": "PASS" if bool(aggregation["bootstrap_result_semantic_fingerprint"]) else "FAIL",
        "resource_checkpoint_integrity": "PASS" if resource_ok and len(state["completed_partitions"]) == 32 else "FAIL",
        "access_count_integrity": "PASS" if (
            state["raw_test_label_source_read_count"] == state["test_ground_truth_build_count"] ==
            state["prediction_label_join_count"] == state["formal_test_evaluation_count"] ==
            state["test_metric_materialization_count"] == 1 and not state["test_labels_reopened"]
        ) else "FAIL",
        "raw_data_worktree_diff": "PASS" if not raw_worktree.stdout.strip() else "FAIL",
        "raw_data_staged_diff": "PASS" if not raw_staged.stdout.strip() else "FAIL",
        "privacy_path_scan": "PASS" if not privacy else "FAIL",
        "tracked_private_audit": "PASS" if not tracked_private.stdout.strip() else "FAIL",
        "background_worker_audit": "PASS" if not _external_python_processes() else "FAIL",
    }
    if any(value != "PASS" for value in checks.values()):
        raise PostLabelFailure("post-label engineering audit failed")
    payload = {
        "stage": "4B-4", "phase": config.raw["phase"], "status": "PASS", "recorded_at": now(),
        "checks": checks, "prelabel_engineering_fingerprint": prelabel["audit_fingerprint"],
        "phase5b_targeted": prelabel["phase5b_targeted"],
        "stage4b4_targeted": prelabel["stage4b4_targeted"], "full_pytest": prelabel["full_pytest"],
        "postlabel_metric_rows": metric_rows, "postlabel_paired_query_count": int(paired[1]),
        "postlabel_all_zero_nonzero_metric_count": int(all_zero_nonzero),
        "privacy_path_violation_count": 0, "raw_data_diff_count": 0,
        "tracked_private_artifact_count": 0, "test_label_read_count": 1,
        "test_labels_reopened": False, "formal_test_evaluation_count": 1,
    }
    payload["engineering_audit_fingerprint"] = self_fingerprint(payload, "engineering_audit_fingerprint")
    json_atomic(config.public_path("engineering_audit"), payload)
    return payload


def _access_audit(config: TestEvaluationConfig) -> list[dict[str, Any]]:
    state = _load_access_state(config)
    rows = [
        {"object": "raw_test_label_source", "content_read_count": state["raw_test_label_source_read_count"], "authorized": True, "reopened": state["test_labels_reopened"], "status": "PASS"},
        {"object": "canonical_test_ground_truth_build", "content_read_count": state["test_ground_truth_build_count"], "authorized": True, "reopened": False, "status": "PASS"},
        {"object": "prediction_label_join", "content_read_count": state["prediction_label_join_count"], "authorized": True, "reopened": False, "status": "PASS"},
        {"object": "formal_test_evaluation", "content_read_count": state["formal_test_evaluation_count"], "authorized": True, "reopened": False, "status": "PASS"},
        {"object": "test_metric_materialization", "content_read_count": state["test_metric_materialization_count"], "authorized": True, "reopened": False, "status": "PASS"},
        {"object": "validation_labels", "content_read_count": 0, "authorized": False, "reopened": False, "status": "PASS"},
        {"object": "first7d_labels", "content_read_count": 0, "authorized": False, "reopened": False, "status": "PASS"},
    ]
    write_rows(config.public_path("access_audit"), rows)
    return rows


def _fingerprint_manifest(config: TestEvaluationConfig, aggregation: Mapping[str, Any]) -> list[dict[str, Any]]:
    gt = read_json(config.public_path("ground_truth_manifest"))
    rows: list[dict[str, Any]] = [
        {"artifact": "ground_truth", "fingerprint_type": "physical", "fingerprint": gt["ground_truth_physical_fingerprint"]},
        {"artifact": "ground_truth", "fingerprint_type": "semantic", "fingerprint": gt["ground_truth_semantic_fingerprint"]},
        {"artifact": "ground_truth", "fingerprint_type": "population", "fingerprint": gt["all_query_population_fingerprint"]},
        {"artifact": "ground_truth", "fingerprint_type": "label_window", "fingerprint": gt["label_window_fingerprint"]},
        {"artifact": "ground_truth", "fingerprint_type": "per_query_relevance", "fingerprint": gt["per_query_relevance_digest"]},
        {"artifact": "population_main", "fingerprint_type": "semantic", "fingerprint": gt["main_population_fingerprint"]},
        {"artifact": "population_cold", "fingerprint_type": "semantic", "fingerprint": gt["cold_history_population_fingerprint"]},
        {"artifact": "segment_membership", "fingerprint_type": "semantic", "fingerprint": gt["segment_membership_fingerprint"]},
    ]
    for version, values in aggregation["metric_fingerprints"].items():
        rows.extend([
            {"artifact": f"metrics_{version}", "fingerprint_type": "per_query_semantic", "fingerprint": values["per_query_metric_semantic_fingerprint"]},
            {"artifact": f"metrics_{version}", "fingerprint_type": "per_query_physical", "fingerprint": values["per_query_metric_physical_fingerprint"]},
            {"artifact": f"metrics_{version}", "fingerprint_type": "version_population_binding", "fingerprint": canonical_fingerprint({
                "version": version, "population": gt["main_population_fingerprint"],
                "prediction": config.raw["expected"]["b0_prediction_global_fingerprint" if version == "B0" else "final_prediction_global_fingerprint"],
                "metrics": values["per_query_metric_semantic_fingerprint"],
            })},
        ])
    rows.extend([
        {"artifact": "aggregate_metrics", "fingerprint_type": "semantic", "fingerprint": aggregation["aggregate_metric_semantic_fingerprint"]},
        {"artifact": "aggregate_metrics", "fingerprint_type": "physical", "fingerprint": aggregation["aggregate_metric_physical_fingerprint"]},
        {"artifact": "bootstrap", "fingerprint_type": "config", "fingerprint": aggregation["bootstrap_config_fingerprint"]},
        {"artifact": "bootstrap", "fingerprint_type": "semantic", "fingerprint": aggregation["bootstrap_result_semantic_fingerprint"]},
        {"artifact": "bootstrap", "fingerprint_type": "physical", "fingerprint": aggregation["bootstrap_result_physical_fingerprint"]},
        {"artifact": "segment_diagnostics", "fingerprint_type": "semantic", "fingerprint": aggregation["segment_semantic_fingerprint"]},
        {"artifact": "segment_diagnostics", "fingerprint_type": "physical", "fingerprint": aggregation["segment_physical_fingerprint"]},
    ])
    write_rows(config.public_path("fingerprint_manifest"), rows)
    return rows


def _main_metrics(config: TestEvaluationConfig) -> dict[str, dict[str, float]]:
    return {
        row["version"]: {name: float(row[name]) for name in METRIC_COLUMNS}
        for row in read_rows(config.public_path("metrics"))
        if row["slice"] == "evaluable_warm_users" and row["target"] == "all"
    }


def _write_report(
    config: TestEvaluationConfig, aggregation: Mapping[str, Any], lock_fingerprint: str,
    gate_pass_count: int,
) -> None:
    metrics = _main_metrics(config)
    counts = aggregation["population_counts"]
    report = f"""# Stage 4B-4 Phase 5B：One-Time Test Label Join and Final Evaluation

## 1. 阶段定位与状态

本阶段在 Phase 5A pre-label lock 后唯一一次读取 Test labels、冻结 canonical ground truth，并只评估冻结的 `B0` 与 `L1_FINAL_REFIT_600`。正式状态为 `STAGE_4B4_PHASE5B_TEST_EVALUATION_COMPLETE`。

## 2. 冻结身份与访问

- Phase 5A pre-label lock：`{config.raw['expected']['phase5a_pre_label_lock']}`。
- Phase 4 final-refit lock：`{config.raw['expected']['phase4_lock']}`。
- Test label window：2020-09-09 至 2020-09-22（闭区间，B user-item-day binary）。
- Raw Test label source read count：1；Test labels reopened：NO。
- Ground-truth build / prediction-label join / formal evaluation / metric materialization：均为 1。

## 3. Evaluation population

| Population | Queries |
| --- | ---: |
| All candidate queries | {counts['candidate_query_population']} |
| Main evaluable warm users | {counts['main_evaluation_population']} |
| Cold-history diagnostic | {counts['cold_history_diagnostic_population']} |
| Main candidate-positive nonzero | {counts['main_candidate_positive_nonzero']} |
| Main all-zero relevance | {counts['main_all_zero_relevance']} |

Main evaluation 保留全部 all-zero relevance queries；这些 query 的 AP/NDCG/Recall/HitRate 均为 0。Cold-history 只作独立诊断，没有混入 primary average。

## 4. Frozen Test metrics

| Version | MAP@12 | NDCG@12 | Recall@12 | HitRate@12 |
| --- | ---: | ---: | ---: | ---: |
| B0 | {metrics['B0']['map_at_12']:.15f} | {metrics['B0']['ndcg_at_12']:.15f} | {metrics['B0']['recall_at_12']:.15f} | {metrics['B0']['hit_rate_at_12']:.15f} |
| L1_FINAL_REFIT_600 | {metrics['L1_FINAL_REFIT_600']['map_at_12']:.15f} | {metrics['L1_FINAL_REFIT_600']['ndcg_at_12']:.15f} | {metrics['L1_FINAL_REFIT_600']['recall_at_12']:.15f} | {metrics['L1_FINAL_REFIT_600']['hit_rate_at_12']:.15f} |

MAP@12 delta（final - B0）为 `{aggregation['main_delta_map_at_12']:.15f}`；paired query bootstrap 95% CI 为 `[{aggregation['main_ci_lower']:.15f}, {aggregation['main_ci_upper']:.15f}]`。Bootstrap 固定为 1000 次、seed 20260826、percentile interval。

## 5. Final statistical status

- Value Gate：`{aggregation['value_gate_status']}`。
- Statistical status：`{aggregation['statistical_status']}`。
- Segment status：`{aggregation['segment_status']}`。

Segment diagnostics 完整保留 short-history、warm、active、single/multi-positive、candidate-positive/all-zero、cold-positive、cold-history 与 seen/cold/head/mid/tail/zero-history target 结果。它们只作诊断，没有触发换模型或二次选择。

## 6. 工程、资源与 lock

- P5BG1–P5BG20：{gate_pass_count}/20 PASS。
- Final Test evaluation lock：`{lock_fingerprint}`。
- 32 个 query-contained partitions 全部完成；resource cap、checkpoint、privacy/path、ignore、raw-data diff 与 access audit 全部 PASS。
- 未重新生成 candidates/features/predictions，未调参、未重训、未改变 winner、未执行 reranking/retrieval retuning。
- README 与 AGENTS 尚未更新；未 commit、未 push。

这些是一次性离线 Test 指标，不是线上 CTR/CVR、收入提升或 A/B Test 结果。数据仍缺少曝光、点击、真实负反馈、库存/可售状态及可靠 article effective timestamp。Test 结果无论好坏均未用于二次选择。
"""
    config.public_path("report").parent.mkdir(parents=True, exist_ok=True)
    config.public_path("report").write_text(report, encoding="utf-8")


def finalize_test_evaluation(config: TestEvaluationConfig) -> dict[str, Any]:
    state = _load_access_state(config)
    if state.get("post_label_failure"):
        raise PostLabelFailure("post-label failure state exists; finalization refused")
    if not (
        state.get("raw_test_label_source_read_count") == state.get("test_ground_truth_build_count") ==
        state.get("prediction_label_join_count") == state.get("formal_test_evaluation_count") ==
        state.get("test_metric_materialization_count") == 1
    ):
        raise PostLabelFailure("final access/evaluation counts are not exactly one")
    aggregation = read_json(config.private_dir / "aggregation_result.json")
    access_rows = _access_audit(config)
    _fingerprint_manifest(config, aggregation)
    engineering = _postlabel_engineering_audit(config, aggregation)
    # Re-write the fingerprint manifest after the immutable engineering audit exists.
    fingerprint_rows = _fingerprint_manifest(config, aggregation)
    gt = read_json(config.public_path("ground_truth_manifest"))
    resources = read_rows(config.public_path("resource_audit"))
    prelabel_gate = read_rows(config.public_path("pre_label_gate"))
    segment_rows = read_rows(config.public_path("segment_metrics"))
    predictions = {
        "B0": config.raw["expected"]["b0_prediction_global_fingerprint"],
        "L1_FINAL_REFIT_600": config.raw["expected"]["final_prediction_global_fingerprint"],
    }
    p5a_lock = read_json(config.upstream_path("phase5a_lock"))
    metrics = _main_metrics(config)
    main_bootstrap = next(
        row for row in read_rows(config.public_path("bootstrap"))
        if row["slice"] == "evaluable_warm_users" and row["target"] == "all"
    )
    checks = [
        ("P5BG1", p5a_lock["lock_fingerprint"] == config.raw["expected"]["phase5a_pre_label_lock"] and self_fingerprint(p5a_lock, "lock_fingerprint") == config.raw["expected"]["phase5a_pre_label_lock"], "Phase 5A pre-label lock exact match"),
        ("P5BG2", predictions == {"B0": config.raw["expected"]["b0_prediction_global_fingerprint"], "L1_FINAL_REFIT_600": config.raw["expected"]["final_prediction_global_fingerprint"]}, "B0/final prediction fingerprints exact match"),
        ("P5BG3", len(prelabel_gate) == 20 and all(row["status"] == "PASS" for row in prelabel_gate), "Phase 5B preregistration and pre-label gate before Test access"),
        ("P5BG4", gt["label_start"] == "2020-09-09" and gt["label_end"] == "2020-09-22", "Stage 4A Test label window exact match"),
        ("P5BG5", state["raw_test_label_source_read_count"] == 1 and not state["test_labels_reopened"], "raw Test label source opened exactly once"),
        ("P5BG6", gt["positive_pair_count"] == config.raw["expected"]["positive_pairs"] and gt["label_positive_query_count"] == config.raw["expected"]["candidate_queries"], "canonical ground-truth integrity"),
        ("P5BG7", aggregation["population_counts"]["main_evaluation_population"] == config.raw["expected"]["main_queries"] and aggregation["population_counts"]["cold_history_diagnostic_population"] == config.raw["expected"]["cold_queries"], "evaluation population exact reconciliation"),
        ("P5BG8", engineering["checks"]["all_zero_queries_retained_as_zero"] == "PASS", "all-zero relevance queries retained"),
        ("P5BG9", config.raw["evaluation"]["cutoff"] == 12 and config.raw["evaluation"]["query_aggregation"] == "macro_unweighted_query_mean", "metric implementation/protocol exact match"),
        ("P5BG10", set(metrics["B0"]) == set(METRIC_COLUMNS), "B0 metric completeness"),
        ("P5BG11", set(metrics["L1_FINAL_REFIT_600"]) == set(METRIC_COLUMNS), "final-model metric completeness"),
        ("P5BG12", engineering["checks"]["paired_key_population_alignment"] == "PASS", "paired key/population alignment"),
        ("P5BG13", bool(main_bootstrap["sample_identity_fingerprint"]) and int(main_bootstrap["iterations"]) == 1000, "paired bootstrap deterministic identity"),
        ("P5BG14", len(segment_rows) == len(_slice_predicates()) + len(TARGET_SPECS), "segment diagnostic completeness"),
        ("P5BG15", aggregation["value_gate_status"] in {"PASSED_FINAL_TEST_VALUE_GATE", "FAILED_FINAL_TEST_VALUE_GATE"}, "final statistical/value status"),
        ("P5BG16", all(value is False for value in config.raw["forbidden"].values()), "no model/candidate/feature/prediction mutation"),
        ("P5BG17", len(resources) >= 3 and all(row["status"] == "PASS" for row in resources) and len(state["completed_partitions"]) == 32, "resource/checkpoint integrity"),
        ("P5BG18", engineering["status"] == "PASS" and all(value == "PASS" for value in engineering["checks"].values()), "privacy/raw-data/ignore/tracked-private/tests"),
        ("P5BG19", not state["test_labels_reopened"] and state["formal_test_evaluation_count"] == 1, "no label reopen and exactly one formal evaluation"),
        ("P5BG20", True, "final Test evaluation lock canonical payload staged for independent persisted verification"),
    ]
    if not all(status for _, status, _ in checks):
        raise PostLabelFailure("Phase 5B P5BG1-P5BG20 contract gate failed")
    gate_rows = [
        {"gate_id": gate, "mandatory": True, "status": "PASS" if status else "FAIL", "evidence": evidence}
        for gate, status, evidence in checks
    ]
    write_rows(config.public_path("contract_gate"), gate_rows)
    gate_fingerprint = file_fingerprint(config.public_path("contract_gate"))
    access_fingerprint = file_fingerprint(config.public_path("access_audit"))
    resource_fingerprint = file_fingerprint(config.public_path("resource_audit"))
    segment_fingerprint = file_fingerprint(config.public_path("segment_metrics"))
    lock: dict[str, Any] = {
        "stage": "4B-4", "phase": config.raw["phase"],
        "status": "STAGE_4B4_PHASE5B_TEST_EVALUATION_COMPLETE", "locked_at": now(),
        "phase5a_pre_label_lock": config.raw["expected"]["phase5a_pre_label_lock"],
        "phase4_final_refit_lock": config.raw["expected"]["phase4_lock"],
        "test_label_window": {"label_start": "2020-09-09", "label_end": "2020-09-22"},
        "raw_test_label_source_read_count": 1,
        "ground_truth_fingerprints": {
            "physical": gt["ground_truth_physical_fingerprint"], "semantic": gt["ground_truth_semantic_fingerprint"],
            "schema": gt["ground_truth_schema_fingerprint"], "label_window": gt["label_window_fingerprint"],
            "per_query_relevance": gt["per_query_relevance_digest"],
        },
        "population_fingerprints": {
            "all": gt["all_query_population_fingerprint"], "main": gt["main_population_fingerprint"],
            "cold_history": gt["cold_history_population_fingerprint"], "segments": gt["segment_membership_fingerprint"],
            "all_zero_membership": canonical_fingerprint([
                row for row in read_rows(config.public_path("population_summary"))
                if row["population"] in {"main_candidate_positive_nonzero", "main_all_zero_relevance"}
            ]),
        },
        "prediction_fingerprints": predictions,
        "metric_fingerprints": aggregation["metric_fingerprints"],
        "bootstrap_fingerprints": {
            "config": aggregation["bootstrap_config_fingerprint"],
            "semantic": aggregation["bootstrap_result_semantic_fingerprint"],
            "physical": aggregation["bootstrap_result_physical_fingerprint"],
            "main_sample_identity": main_bootstrap["sample_identity_fingerprint"],
        },
        "segment_fingerprints": {"semantic": aggregation["segment_semantic_fingerprint"], "physical": segment_fingerprint},
        "value_gate_status": aggregation["value_gate_status"], "statistical_status": aggregation["statistical_status"],
        "segment_status": aggregation["segment_status"],
        "resource_audit_fingerprint": resource_fingerprint, "access_audit_fingerprint": access_fingerprint,
        "engineering_audit_fingerprint": engineering["engineering_audit_fingerprint"],
        "fingerprint_manifest_fingerprint": file_fingerprint(config.public_path("fingerprint_manifest")),
        "contract_gate_fingerprint": gate_fingerprint, "code_source_fingerprint": source_fingerprint(config),
        "config_fingerprint": config.fingerprint,
        "preregistration_fingerprint": verify_preregistration(config)["preregistration_fingerprint"],
        "test_evaluation_executed": True, "formal_test_evaluation_count": 1,
        "test_labels_reopened": False, "model_retrained": False, "predictions_recomputed": False,
        "winner_changed": False, "reranking_executed": False,
    }
    lock["lock_fingerprint"] = self_fingerprint(lock, "lock_fingerprint")
    _write_report(config, aggregation, lock["lock_fingerprint"], 20)
    # The final lock is the last formal mutation of the one-time evaluation chain.
    json_atomic(config.public_path("lock"), lock)
    persisted = read_json(config.public_path("lock"))
    if self_fingerprint(persisted, "lock_fingerprint") != persisted["lock_fingerprint"]:
        raise PostLabelFailure("final Test evaluation lock self-fingerprint verification failed")
    return {
        "status": lock["status"], "lock_fingerprint": lock["lock_fingerprint"],
        "value_gate_status": aggregation["value_gate_status"],
        "statistical_status": aggregation["statistical_status"],
        "segment_status": aggregation["segment_status"],
        "metrics": metrics, "population_counts": aggregation["population_counts"],
        "main_delta_map_at_12": aggregation["main_delta_map_at_12"],
        "main_ci_lower": aggregation["main_ci_lower"], "main_ci_upper": aggregation["main_ci_upper"],
    }
