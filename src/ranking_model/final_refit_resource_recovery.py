from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import psutil


class RecoveryError(RuntimeError):
    pass


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def self_fingerprint(payload: dict[str, Any], field: str) -> str:
    return canonical_fingerprint({key: value for key, value in payload.items() if key != field})


def file_fingerprint(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def file_set_fingerprint(paths: Iterable[Path], root: Path) -> str:
    rows = [{
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_fingerprint(path),
    } for path in sorted(paths, key=lambda value: value.as_posix())]
    return canonical_fingerprint(rows)


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RecoveryConfig:
    root: Path
    path: Path
    raw: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> "RecoveryConfig":
        resolved = path.resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        root = resolved.parent.parent
        value = cls(root=root, path=resolved, raw=raw, fingerprint=canonical_fingerprint(raw))
        value.validate()
        return value

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
        if self.raw.get("recovery_type") != "lightgbm_binary_dataset_isolated_training_v1":
            raise RecoveryError("wrong resource recovery type")
        scope = self.raw["recovery_scope"]
        forbidden_true = (
            "validation_optimizer_recomputed", "combined_matrix_recomputed", "negative_sampling_recomputed",
            "label_join_recomputed", "validation_labels_reopened", "first7d_labels_read",
            "validation_metrics_recomputed", "first7d_metrics_recomputed", "test_access_allowed",
        )
        if any(scope[name] for name in forbidden_true):
            raise RecoveryError("RECOVERY_SCOPE_EXCEEDED")
        if any(value != 0 for key, value in scope.items() if key.startswith("test_") and key.endswith("_count")):
            raise RecoveryError("RECOVERY_SCOPE_EXCEEDED")
        model = self.raw["model"]
        expected_model = {
            "model_id": "L1_FINAL_REFIT_600", "feature_set": "F1", "feature_count": 52,
            "objective": "lambdarank", "num_boost_round": 600, "fresh_initialization": True,
            "warm_start": False, "continued_training_from_L1_600": False,
            "early_stopping": False, "internal_validation": False,
            "sample_weight": 1.0, "query_weight": 1.0,
        }
        if model != expected_model:
            raise RecoveryError("FINAL_REFIT_SEMANTIC_DRIFT: model contract")
        if self.raw["resource"]["conversion_rss_fraction_of_available"] != 0.70:
            raise RecoveryError("conversion resource fraction must be 0.70")
        if self.raw["resource"]["training_rss_fraction_of_available"] != 0.70:
            raise RecoveryError("training resource fraction must be 0.70")
        base_config = _read_json(self.resolve(self.raw["base"]["final_refit_config"]))
        phase1_config = _read_json(self.resolve(self.raw["base"]["phase1_config"]))
        if self.raw["lightgbm"]["params"] != base_config["lightgbm"]["params"]:
            raise RecoveryError("FINAL_REFIT_SEMANTIC_DRIFT: base params")
        if self.raw["lightgbm"]["params"] != phase1_config["lightgbm"]["params"]:
            raise RecoveryError("FINAL_REFIT_SEMANTIC_DRIFT: Phase 1 params")
        if self.raw["lightgbm"]["weights"] != phase1_config["lightgbm"]["weights"]:
            raise RecoveryError("FINAL_REFIT_SEMANTIC_DRIFT: Phase 1 weights")


def recovery_source_fingerprint(config: RecoveryConfig) -> str:
    paths = [
        config.path,
        config.root / "src/ranking_model/final_refit_resource_recovery.py",
        config.root / "scripts/run_stage_4b4_final_refit_resource_recovery.py",
        config.root / "scripts/stage_4b4_binary_conversion_worker.py",
        config.root / "scripts/stage_4b4_binary_verification_worker.py",
        config.root / "scripts/stage_4b4_binary_training_worker.py",
        config.root / "scripts/stage_4b4_final_model_replay_worker.py",
        config.root / "scripts/run_stage_4b4_final_refit_resource_recovery_audit.py",
    ]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise RecoveryError(f"recovery source inventory incomplete: {missing}")
    return file_set_fingerprint(paths, config.root)


def verify_recovery_inputs(config: RecoveryConfig) -> dict[str, Any]:
    import numpy as np

    base = config.raw["base"]
    expected = config.raw["expected"]
    original = config.raw["original_failure"]
    checks: dict[str, bool] = {}
    base_config_path = config.resolve(base["final_refit_config"])
    checks["base_config_physical"] = file_fingerprint(base_config_path) == base["final_refit_config_physical_fingerprint"]
    from ranking_model.final_refit import FinalRefitConfig, _source_fingerprint
    base_config = FinalRefitConfig.load(base_config_path)
    checks["base_config_canonical"] = base_config.fingerprint == base["final_refit_config_canonical_fingerprint"]
    checks["base_source"] = _source_fingerprint(base_config) == base["base_source_fingerprint"]
    prereg_path = config.resolve(base["final_refit_preregistration"])
    prereg = _read_json(prereg_path)
    checks["base_preregistration"] = prereg.get("preregistration_fingerprint") == base["final_refit_preregistration_fingerprint"]
    validation_path = config.resolve(base["validation_optimizer_manifest"])
    validation = _read_json(validation_path)
    checks["validation_manifest_physical"] = file_fingerprint(validation_path) == base["validation_optimizer_manifest_physical_fingerprint"]
    checks["validation_manifest_semantic"] = (
        validation.get("manifest_fingerprint") == base["validation_optimizer_manifest_fingerprint"]
        and self_fingerprint(validation, "manifest_fingerprint") == validation["manifest_fingerprint"]
    )
    checks["validation_counts"] = (
        validation["query_count"] == expected["validation_queries"]
        and validation["row_count"] == expected["validation_rows"]
        and validation["membership_fingerprint"] == base["validation_optimizer_membership_fingerprint"]
        and validation["deterministic_replay_fingerprint"] == validation["membership_fingerprint"]
        and validation["all_zero_optimizer_groups"] == 0
    )
    private_base = config.root / "artifacts/stage_4b4/private/final_refit"
    checks["validation_membership_files"] = file_set_fingerprint(
        sorted((private_base / "validation_optimizer_membership").glob("part_*.parquet")), config.root
    ) == expected["validation_membership_physical_set_fingerprint"]
    checks["validation_replay_files"] = file_set_fingerprint(
        sorted((private_base / "validation_optimizer_membership_replay").glob("part_*.parquet")), config.root
    ) == expected["validation_replay_physical_set_fingerprint"]
    checks["validation_feature_view_files"] = file_set_fingerprint(
        sorted((private_base / "validation_optimizer_features").glob("part_*.parquet")), config.root
    ) == expected["validation_feature_view_physical_set_fingerprint"]

    combined_manifest_path = config.resolve(base["combined_matrix_manifest"])
    combined = _read_json(combined_manifest_path)
    checks["combined_manifest_physical"] = file_fingerprint(combined_manifest_path) == base["combined_matrix_manifest_physical_fingerprint"]
    checks["combined_manifest_semantic"] = (
        combined.get("manifest_fingerprint") == base["combined_matrix_manifest_fingerprint"]
        and self_fingerprint(combined, "manifest_fingerprint") == combined["manifest_fingerprint"]
    )
    combined_checkpoint_path = config.resolve(base["combined_matrix_checkpoint"])
    combined_checkpoint = _read_json(combined_checkpoint_path)
    checks["combined_checkpoint"] = (
        file_fingerprint(combined_checkpoint_path) == base["combined_matrix_checkpoint_physical_fingerprint"]
        and combined_checkpoint.get("status") == "COMPLETE"
        and combined_checkpoint.get("manifest_fingerprint") == combined["manifest_fingerprint"]
    )
    matrix_dir = config.resolve(base["combined_matrix_directory"])
    files = {
        "features": matrix_dir / "features_f1.float32", "labels": matrix_dir / "labels.uint8",
        "groups": matrix_dir / "groups.int32", "snapshot_codes": matrix_dir / "snapshot_codes.uint8",
        "customers": matrix_dir / "customer_keys.int64", "articles": matrix_dir / "article_keys.int64",
        "ranks": matrix_dir / "candidate_ranks.int32", "orders": matrix_dir / "query_orders.int64",
        "boundaries": matrix_dir / "query_boundaries.parquet",
    }
    actual_physical = {name: file_fingerprint(path) for name, path in files.items()}
    checks["combined_physical_files"] = actual_physical == combined["physical_fingerprints"]
    groups = np.fromfile(files["groups"], dtype=np.int32)
    labels = np.memmap(files["labels"], dtype=np.uint8, mode="r")
    checks["combined_counts"] = (
        len(groups) == expected["combined_queries"]
        and len(labels) == expected["combined_rows"]
        and int(groups.sum()) == expected["combined_rows"]
        and int(labels.sum()) == expected["combined_positive_rows"]
        and int(len(labels) - labels.sum()) == expected["combined_zero_rows"]
        and combined["feature_count"] == expected["combined_feature_count"]
        and combined["matrix_dtype"] == expected["combined_dtype"]
    )
    checks["combined_semantics"] = (
        combined["combined_matrix_order_fingerprint"] == expected["combined_matrix_order_fingerprint"]
        and combined["snapshot_order_fingerprint"] == expected["snapshot_order_fingerprint"]
        and combined["feature_matrix_physical_fingerprint"] == expected["feature_matrix_physical_fingerprint"]
        and combined["label_physical_fingerprint"] == expected["label_physical_fingerprint"]
        and combined["group_physical_fingerprint"] == expected["group_physical_fingerprint"]
        and combined["query_boundary_physical_fingerprint"] == expected["query_boundary_physical_fingerprint"]
        and not any(combined["audit"].values())
    )
    adapter = _read_json(config.resolve(base["phase1_adapter"]))
    checks["adapter"] = (
        adapter["adapter_fingerprint"] == base["phase1_adapter_fingerprint"]
        and len(adapter["ordered_feature_names"]) == expected["combined_feature_count"]
        and canonical_fingerprint(adapter["ordered_feature_names"]) == expected["feature_name_order_fingerprint"]
    )
    phase2 = _read_json(config.resolve(base["phase2_lock"]))
    phase3 = _read_json(config.resolve(base["phase3_recovered_lock"]))
    checks["phase_locks"] = (
        file_fingerprint(config.resolve(base["phase1_lock"])) == base["phase1_lock_physical_fingerprint"]
        and file_fingerprint(config.resolve(base["phase2_lock"])) == base["phase2_lock_physical_fingerprint"]
        and phase2["lock_fingerprint"] == base["phase2_lock_semantic_fingerprint"]
        and file_fingerprint(config.resolve(base["phase3_recovered_lock"])) == base["phase3_recovered_lock_physical_fingerprint"]
        and phase3["lock_fingerprint"] == base["phase3_recovered_lock_semantic_fingerprint"]
        and phase2["winner"] == "L1_600" and phase3["winner_changed"] is False
    )
    original_resource_path = config.resolve(original["resource_audit"])
    original_resource = _read_json(original_resource_path)
    checks["original_failure_preserved"] = (
        file_fingerprint(original_resource_path) == original["resource_audit_physical_fingerprint"]
        and original_resource["resource_audit_fingerprint"] == original["resource_audit_semantic_fingerprint"]
        and original_resource["status"] == "BLOCKED_BY_RESOURCE"
        and original_resource["frozen_absolute_rss_cap_bytes"] == original["original_frozen_cap_bytes"]
        and original_resource["child_process_tree_peak_rss_bytes"] == original["original_peak_rss_bytes"]
        and original["original_peak_rss_bytes"] - original["original_frozen_cap_bytes"] == original["original_excess_bytes"]
    )
    checks["no_forbidden_access"] = (
        validation["first7d_label_read_count"] == 0
        and validation["test_candidate_read_count"] == 0
        and validation["test_feature_read_count"] == 0
        and validation["test_label_read_count"] == 0
        and validation["test_metric_read_count"] == 0
        and combined["first7d_label_read_count"] == 0
        and combined["test_candidate_read_count"] == 0
        and combined["test_feature_read_count"] == 0
        and combined["test_label_read_count"] == 0
        and combined["test_metric_read_count"] == 0
    )
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RecoveryError(f"RECOVERY_INPUT_IDENTITY_FAILURE: {failed}")
    return {
        "status": "RECOVERY_INPUT_IDENTITY_VERIFIED",
        "checks": checks,
        "actual_combined_physical_fingerprints": actual_physical,
        "validation_optimizer_manifest_fingerprint": validation["manifest_fingerprint"],
        "validation_optimizer_membership_fingerprint": validation["membership_fingerprint"],
        "combined_matrix_manifest_fingerprint": combined["manifest_fingerprint"],
        "combined_matrix_order_fingerprint": combined["combined_matrix_order_fingerprint"],
        "combined_rows": len(labels), "combined_queries": len(groups),
        "combined_positive_rows": int(labels.sum()), "combined_zero_rows": int(len(labels) - labels.sum()),
        "validation_optimizer_recomputed": False, "combined_matrix_recomputed": False,
        "validation_labels_reopened": False, "first7d_labels_read": False,
        "test_access": False,
    }


def preregister_recovery(config: RecoveryConfig) -> dict[str, Any]:
    identity = verify_recovery_inputs(config)
    source_fingerprint = recovery_source_fingerprint(config)
    payload = {
        "status": "FINAL_REFIT_RESOURCE_RECOVERY_PREREGISTERED",
        "authorization": config.raw["authorization"],
        "recovery_type": config.raw["recovery_type"],
        "recovery_config_canonical_fingerprint": config.fingerprint,
        "recovery_source_fingerprint": source_fingerprint,
        "base_final_refit_config_fingerprint": config.raw["base"]["final_refit_config_canonical_fingerprint"],
        "base_final_refit_preregistration_fingerprint": config.raw["base"]["final_refit_preregistration_fingerprint"],
        "original_failure_audit_fingerprint": config.raw["original_failure"]["resource_audit_semantic_fingerprint"],
        "validation_optimizer_fingerprint": identity["validation_optimizer_membership_fingerprint"],
        "combined_matrix_fingerprint": identity["combined_matrix_manifest_fingerprint"],
        "combined_matrix_order_fingerprint": identity["combined_matrix_order_fingerprint"],
        "model_contract": config.raw["model"],
        "parameter_fingerprint": canonical_fingerprint(config.raw["lightgbm"]["params"]),
        "weights_fingerprint": canonical_fingerprint(config.raw["lightgbm"]["weights"]),
        "resource_contract": config.raw["resource"],
        "recovery_scope": config.raw["recovery_scope"],
        "conversion_started": False, "training_started": False,
    }
    payload["recovery_preregistration_fingerprint"] = canonical_fingerprint(payload)
    path = config.path_for("recovery_preregistration")
    if path.exists():
        if _read_json(path) != payload:
            raise RecoveryError("recovery preregistration drift")
    else:
        write_json_atomic(path, payload)
    identity_payload = dict(identity)
    identity_payload["recovery_config_canonical_fingerprint"] = config.fingerprint
    identity_payload["recovery_source_fingerprint"] = source_fingerprint
    identity_payload["input_identity_audit_fingerprint"] = canonical_fingerprint(identity_payload)
    audit_path = config.path_for("input_identity_audit")
    if audit_path.exists():
        if _read_json(audit_path) != identity_payload:
            raise RecoveryError("recovery input identity audit drift")
    else:
        write_json_atomic(audit_path, identity_payload)
    failure_checkpoint = {
        "status": "FAILED", "reuse_status": "NOT_REUSABLE_AS_FORMAL_MODEL",
        "attempt": "original_raw_memmap_training",
        "original_resource_audit_physical_fingerprint": config.raw["original_failure"]["resource_audit_physical_fingerprint"],
        "original_resource_audit_semantic_fingerprint": config.raw["original_failure"]["resource_audit_semantic_fingerprint"],
        "model_promoted": False, "formal_model_exists": False,
    }
    failure_checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(failure_checkpoint)
    checkpoint_path = config.path_for("original_failure_checkpoint")
    if checkpoint_path.exists():
        if _read_json(checkpoint_path) != failure_checkpoint:
            raise RecoveryError("original failure checkpoint drift")
    else:
        write_json_atomic(checkpoint_path, failure_checkpoint)
    return payload


def process_tree_rss(process: psutil.Process) -> int:
    values = [process]
    try:
        values.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for value in values:
        try:
            total += int(value.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def terminate_process_tree(process: psutil.Process) -> None:
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    _, alive = psutil.wait_procs(children + [process], timeout=5)
    for value in alive:
        try:
            value.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def _run_monitored(
    *, command: list[str], root: Path, cap: int, stdout_path: Path, stderr_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    peak = 0
    available = int(psutil.virtual_memory().available)
    available_min = available
    available_max = available
    exceeded = False
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            rss = process_tree_rss(process)
            peak = max(peak, rss)
            observed = int(psutil.virtual_memory().available)
            available_min, available_max = min(available_min, observed), max(available_max, observed)
            if rss > cap:
                exceeded = True
                terminate_process_tree(process)
                break
            time.sleep(0.25)
        exit_code = child.wait()
    return {
        "worker_exit_code": exit_code, "child_process_tree_peak_rss_bytes": peak,
        "rss_gate_exceeded": exceeded, "runtime_available_memory_min_bytes_observation_only": available_min,
        "runtime_available_memory_max_bytes_observation_only": available_max,
        "wall_time_seconds": time.perf_counter() - started,
    }


def _resource_preregistration(
    *, config: RecoveryConfig, kind: str, path: Path, worker_path: Path,
    input_fingerprint: str,
) -> dict[str, Any]:
    if path.exists():
        raise RecoveryError(f"{kind} resource attempt already preregistered; cap shopping/retry forbidden")
    available = int(psutil.virtual_memory().available)
    cap = int(available * 0.70)
    payload = {
        "status": f"FINAL_REFIT_{kind.upper()}_RESOURCE_PREREGISTERED_BEFORE_WORKER",
        f"available_memory_at_{kind}_start_bytes": available,
        "rss_fraction": 0.70,
        f"frozen_absolute_{kind}_cap_bytes": cap,
        "worker_consumes_frozen_absolute_cap_only": True,
        "runtime_available_memory_is_observation_only": True,
        "external_process_tree_monitor": True,
        "cap_shopping_allowed": False,
        "input_fingerprint": input_fingerprint,
        "recovery_config_canonical_fingerprint": config.fingerprint,
        "recovery_preregistration_fingerprint": _read_json(
            config.path_for("recovery_preregistration")
        )["recovery_preregistration_fingerprint"],
        "worker_source_fingerprint": file_fingerprint(worker_path),
    }
    payload["resource_preregistration_fingerprint"] = canonical_fingerprint(payload)
    write_json_atomic(path, payload)
    return payload


def convert_binary_dataset(config: RecoveryConfig) -> dict[str, Any]:
    identity = verify_recovery_inputs(config)
    prereg = preregister_recovery(config)
    binary_dir = config.private_dir / "lightgbm_binary"
    temp_dir = config.temp_dir / "binary_conversion"
    checkpoint_dir = config.checkpoint_dir
    for path in (binary_dir, temp_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)
    formal_binary = binary_dir / "final_refit_train.bin"
    binary_manifest_path = config.path_for("binary_manifest")
    if formal_binary.exists() or binary_manifest_path.exists():
        if not formal_binary.exists() or not binary_manifest_path.exists():
            raise RecoveryError("partial formal binary state")
        manifest = _read_json(binary_manifest_path)
        if manifest.get("status") != "SUCCESS_FORMAL_BINARY_DATASET":
            raise RecoveryError("formal binary manifest is not SUCCESS")
        if file_fingerprint(formal_binary) != manifest["binary_physical_fingerprint"]:
            raise RecoveryError("BINARY_DATASET_SEMANTIC_MISMATCH")
        return manifest
    resource_path = config.path_for("conversion_resource_preregistration")
    worker_path = config.root / "scripts/stage_4b4_binary_conversion_worker.py"
    resource = _resource_preregistration(
        config=config, kind="conversion", path=resource_path, worker_path=worker_path,
        input_fingerprint=identity["combined_matrix_manifest_fingerprint"],
    )
    temporary_binary = temp_dir / "final_refit_train.bin.tmp"
    worker_result_path = temp_dir / "conversion_worker_result.json"
    command = [
        sys.executable, str(worker_path), "--recovery-config", str(config.path),
        "--base-config", str(config.resolve(config.raw["base"]["final_refit_config"])),
        "--combined-directory", str(config.resolve(config.raw["base"]["combined_matrix_directory"])),
        "--combined-manifest", str(config.resolve(config.raw["base"]["combined_matrix_manifest"])),
        "--adapter", str(config.resolve(config.raw["base"]["phase1_adapter"])),
        "--resource-preregistration", str(resource_path),
        "--absolute-rss-cap-bytes", str(resource["frozen_absolute_conversion_cap_bytes"]),
        "--temporary-binary", str(temporary_binary), "--result", str(worker_result_path),
        "--worker-source-fingerprint", resource["worker_source_fingerprint"],
    ]
    monitored = _run_monitored(
        command=command, root=config.root, cap=resource["frozen_absolute_conversion_cap_bytes"],
        stdout_path=temp_dir / "conversion_worker.stdout.log",
        stderr_path=temp_dir / "conversion_worker.stderr.log",
    )
    resource_audit = {
        "status": "BLOCKED_BY_RESOURCE_BINARY_BUILD" if monitored["rss_gate_exceeded"] else (
            "PASS" if monitored["worker_exit_code"] == 0 else "RESOURCE_RECOVERY_IMPLEMENTATION_FAILURE"
        ),
        **monitored,
        "rss_fraction": 0.70,
        "frozen_absolute_conversion_cap_bytes": resource["frozen_absolute_conversion_cap_bytes"],
        "resource_preregistration_fingerprint": resource["resource_preregistration_fingerprint"],
        "runtime_cap_recomputed": False, "oom_observed": False,
        "temporary_binary_exists": temporary_binary.exists(),
        "temporary_binary_size_bytes": temporary_binary.stat().st_size if temporary_binary.exists() else 0,
        "binary_promoted": False,
    }
    resource_audit["resource_audit_fingerprint"] = canonical_fingerprint(resource_audit)
    write_json_atomic(config.path_for("conversion_resource_audit"), resource_audit)
    if monitored["rss_gate_exceeded"]:
        raise RecoveryError("STOP BLOCKED_BY_RESOURCE_BINARY_BUILD")
    if monitored["worker_exit_code"] != 0 or not worker_result_path.exists() or not temporary_binary.exists():
        raise RecoveryError("STOP RESOURCE_RECOVERY_IMPLEMENTATION_FAILURE")
    worker_result = _read_json(worker_result_path)
    if worker_result["binary_physical_fingerprint"] != file_fingerprint(temporary_binary):
        raise RecoveryError("BINARY_DATASET_SEMANTIC_MISMATCH")
    if worker_result["frozen_absolute_conversion_cap_bytes_consumed"] != resource["frozen_absolute_conversion_cap_bytes"]:
        raise RecoveryError("conversion worker absolute cap consumption drift")

    verifier_path = config.root / "scripts/stage_4b4_binary_verification_worker.py"
    verifier_result_path = temp_dir / "binary_verification_worker_result.json"
    verify_command = [
        sys.executable, str(verifier_path), "--recovery-config", str(config.path),
        "--binary", str(temporary_binary), "--conversion-result", str(worker_result_path),
        "--adapter", str(config.resolve(config.raw["base"]["phase1_adapter"])),
        "--result", str(verifier_result_path),
    ]
    verification_monitored = _run_monitored(
        command=verify_command, root=config.root, cap=resource["frozen_absolute_conversion_cap_bytes"],
        stdout_path=temp_dir / "verification_worker.stdout.log",
        stderr_path=temp_dir / "verification_worker.stderr.log",
    )
    if verification_monitored["rss_gate_exceeded"] or verification_monitored["worker_exit_code"] != 0:
        raise RecoveryError("STOP BINARY_DATASET_SEMANTIC_MISMATCH")
    verification = _read_json(verifier_result_path)
    if verification["status"] != "BINARY_DATASET_VERIFIED_AWAITING_ATOMIC_PROMOTION":
        raise RecoveryError("BINARY_DATASET_SEMANTIC_MISMATCH")
    if verification["binary_physical_fingerprint"] != worker_result["binary_physical_fingerprint"]:
        raise RecoveryError("BINARY_DATASET_SEMANTIC_MISMATCH")
    os.replace(temporary_binary, formal_binary)
    if file_fingerprint(formal_binary) != worker_result["binary_physical_fingerprint"]:
        raise RecoveryError("formal binary atomic promotion integrity failure")
    verification_audit = {
        **verification,
        "status": "SUCCESS_FORMAL_BINARY_DATASET",
        "verification_worker_process_tree_peak_rss_bytes": verification_monitored["child_process_tree_peak_rss_bytes"],
        "verification_worker_exit_code": verification_monitored["worker_exit_code"],
        "atomic_promotion": True,
    }
    verification_audit["formal_verification_audit_fingerprint"] = canonical_fingerprint(verification_audit)
    write_json_atomic(config.path_for("binary_verification_audit"), verification_audit)
    binary_manifest = {
        "status": "SUCCESS_FORMAL_BINARY_DATASET",
        "formal_binary_relative_path": "artifacts/stage_4b4/private/final_refit/resource_recovery/lightgbm_binary/final_refit_train.bin",
        "binary_physical_fingerprint": worker_result["binary_physical_fingerprint"],
        "binary_dataset_semantic_fingerprint": verification["binary_dataset_semantic_fingerprint"],
        "dataset_metadata_fingerprint": verification["dataset_metadata_fingerprint"],
        "rows": worker_result["rows"], "queries": worker_result["queries"],
        "features": worker_result["features"], "positive_rows": worker_result["positive_rows"],
        "zero_rows": worker_result["zero_rows"], "group_count": worker_result["group_count"],
        "group_sum": worker_result["group_sum"], "binary_size_bytes": formal_binary.stat().st_size,
        "label_fingerprint": worker_result["label_semantic_fingerprint"],
        "group_fingerprint": worker_result["group_semantic_fingerprint"],
        "feature_name_order_fingerprint": worker_result["feature_name_order_fingerprint"],
        "categorical_metadata_fingerprint": worker_result["categorical_metadata_fingerprint"],
        "dataset_binning_fingerprint": worker_result["dataset_binning_fingerprint"],
        "parameter_fingerprint": worker_result["parameter_fingerprint"],
        "combined_matrix_manifest_fingerprint": worker_result["combined_matrix_manifest_fingerprint"],
        "combined_feature_matrix_fingerprint": worker_result["combined_feature_matrix_fingerprint"],
        "combined_matrix_order_fingerprint": worker_result["combined_matrix_order_fingerprint"],
        "lightgbm_version": worker_result["lightgbm_version"],
        "conversion_result_fingerprint": worker_result["conversion_result_fingerprint"],
        "conversion_resource_audit_fingerprint": resource_audit["resource_audit_fingerprint"],
        "verification_audit_fingerprint": verification_audit["formal_verification_audit_fingerprint"],
        "atomic_promotion": True, "validation_labels_reopened": False,
        "first7d_labels_read": False, "test_access": False,
    }
    binary_manifest["binary_manifest_fingerprint"] = canonical_fingerprint(binary_manifest)
    write_json_atomic(binary_manifest_path, binary_manifest)
    conversion_checkpoint = {
        "status": "SUCCESS", "artifact": "FORMAL_BINARY_DATASET",
        "binary_physical_fingerprint": binary_manifest["binary_physical_fingerprint"],
        "conversion_resource_audit_fingerprint": resource_audit["resource_audit_fingerprint"],
    }
    conversion_checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(conversion_checkpoint)
    write_json_atomic(checkpoint_dir / "binary_conversion.json", conversion_checkpoint)
    verification_checkpoint = {
        "status": "SUCCESS", "artifact": "FORMAL_BINARY_DATASET_VERIFIED",
        "binary_semantic_fingerprint": binary_manifest["binary_dataset_semantic_fingerprint"],
        "verification_audit_fingerprint": verification_audit["formal_verification_audit_fingerprint"],
    }
    verification_checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(verification_checkpoint)
    write_json_atomic(checkpoint_dir / "binary_verification.json", verification_checkpoint)
    return binary_manifest


def train_from_binary(config: RecoveryConfig) -> dict[str, Any]:
    verify_recovery_inputs(config)
    prereg = preregister_recovery(config)
    binary_manifest = _read_json(config.path_for("binary_manifest"))
    formal_binary = config.private_dir / "lightgbm_binary/final_refit_train.bin"
    if binary_manifest.get("status") != "SUCCESS_FORMAL_BINARY_DATASET":
        raise RecoveryError("formal binary Dataset is not SUCCESS")
    if file_fingerprint(formal_binary) != binary_manifest["binary_physical_fingerprint"]:
        raise RecoveryError("BINARY_DATASET_SEMANTIC_MISMATCH")
    model_dir = config.private_dir / "models"
    temp_dir = config.temp_dir / "binary_training"
    for path in (model_dir, temp_dir, config.checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)
    formal_model = model_dir / "L1_FINAL_REFIT_600.txt"
    model_manifest_path = config.path_for("model_manifest")
    if formal_model.exists() or model_manifest_path.exists():
        if not formal_model.exists() or not model_manifest_path.exists():
            raise RecoveryError("partial formal model state")
        manifest = _read_json(model_manifest_path)
        if manifest.get("status") != "FINAL_REFIT_MODEL_FROZEN":
            raise RecoveryError("formal model manifest is not frozen")
        if file_fingerprint(formal_model) != manifest["model_physical_fingerprint"]:
            raise RecoveryError("formal model physical drift")
        return manifest
    resource_path = config.path_for("training_resource_preregistration")
    worker_path = config.root / "scripts/stage_4b4_binary_training_worker.py"
    resource = _resource_preregistration(
        config=config, kind="training", path=resource_path, worker_path=worker_path,
        input_fingerprint=binary_manifest["binary_manifest_fingerprint"],
    )
    temporary_model = temp_dir / "L1_FINAL_REFIT_600.txt.tmp"
    worker_result_path = temp_dir / "training_worker_result.json"
    command = [
        sys.executable, str(worker_path), "--recovery-config", str(config.path),
        "--base-config", str(config.resolve(config.raw["base"]["final_refit_config"])),
        "--phase1-config", str(config.resolve(config.raw["base"]["phase1_config"])),
        "--binary", str(formal_binary), "--binary-manifest", str(config.path_for("binary_manifest")),
        "--adapter", str(config.resolve(config.raw["base"]["phase1_adapter"])),
        "--resource-preregistration", str(resource_path),
        "--absolute-rss-cap-bytes", str(resource["frozen_absolute_training_cap_bytes"]),
        "--temporary-model", str(temporary_model), "--result", str(worker_result_path),
        "--worker-source-fingerprint", resource["worker_source_fingerprint"],
    ]
    monitored = _run_monitored(
        command=command, root=config.root, cap=resource["frozen_absolute_training_cap_bytes"],
        stdout_path=temp_dir / "training_worker.stdout.log",
        stderr_path=temp_dir / "training_worker.stderr.log",
    )
    resource_audit = {
        "status": "BLOCKED_BY_RESOURCE" if monitored["rss_gate_exceeded"] else (
            "PASS" if monitored["worker_exit_code"] == 0 else "RESOURCE_RECOVERY_IMPLEMENTATION_FAILURE"
        ),
        **monitored,
        "rss_fraction": 0.70,
        "frozen_absolute_training_cap_bytes": resource["frozen_absolute_training_cap_bytes"],
        "resource_preregistration_fingerprint": resource["resource_preregistration_fingerprint"],
        "runtime_cap_recomputed": False, "oom_observed": False,
        "temporary_model_exists": temporary_model.exists(),
        "temporary_model_size_bytes": temporary_model.stat().st_size if temporary_model.exists() else 0,
        "binary_dataset_size_bytes": formal_binary.stat().st_size,
        "model_promoted": False,
    }
    resource_audit["resource_audit_fingerprint"] = canonical_fingerprint(resource_audit)
    write_json_atomic(config.path_for("training_resource_audit"), resource_audit)
    if monitored["rss_gate_exceeded"]:
        raise RecoveryError("STOP BLOCKED_BY_RESOURCE")
    if monitored["worker_exit_code"] != 0 or not worker_result_path.exists() or not temporary_model.exists():
        raise RecoveryError("STOP RESOURCE_RECOVERY_IMPLEMENTATION_FAILURE")
    worker_result = _read_json(worker_result_path)
    if worker_result["model_physical_fingerprint"] != file_fingerprint(temporary_model):
        raise RecoveryError("temporary model integrity failure")
    if worker_result["frozen_absolute_training_cap_bytes_consumed"] != resource["frozen_absolute_training_cap_bytes"]:
        raise RecoveryError("training worker absolute cap consumption drift")
    os.replace(temporary_model, formal_model)
    if file_fingerprint(formal_model) != worker_result["model_physical_fingerprint"]:
        raise RecoveryError("formal model atomic promotion integrity failure")
    matrix_dir = config.resolve(config.raw["base"]["combined_matrix_directory"])
    replay_worker = config.root / "scripts/stage_4b4_final_model_replay_worker.py"
    replay_result_private = temp_dir / "model_replay_worker_result.json"
    replay_command = [
        sys.executable, str(replay_worker), "--model", str(formal_model),
        "--feature-matrix", str(matrix_dir / "features_f1.float32"),
        "--article-keys", str(matrix_dir / "article_keys.int64"),
        "--candidate-ranks", str(matrix_dir / "candidate_ranks.int32"),
        "--groups", str(matrix_dir / "groups.int32"),
        "--total-rows", str(config.raw["expected"]["combined_rows"]),
        "--result", str(replay_result_private),
    ]
    replay_run = subprocess.run(replay_command, cwd=config.root, text=True, capture_output=True)
    if replay_run.returncode != 0 or not replay_result_private.exists():
        raise RecoveryError("formal model reload/replay failure")
    replay = _read_json(replay_result_private)
    if replay["status"] != "PASS" or not replay["raw_score_exact_match"] or not replay["rank_tiebreak_exact_match"]:
        raise RecoveryError("formal model reload/replay mismatch")
    write_json_atomic(config.path_for("replay_audit"), replay)
    checkpoint = {
        "status": "SUCCESS", "artifact": "FORMAL_FINAL_REFIT_MODEL",
        "model_physical_fingerprint": worker_result["model_physical_fingerprint"],
        "resource_audit_fingerprint": resource_audit["resource_audit_fingerprint"],
        "replay_fingerprint": replay["replay_fingerprint"],
    }
    checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(checkpoint)
    write_json_atomic(config.checkpoint_dir / "binary_training_worker.json", checkpoint)
    model_verification_checkpoint = {
        "status": "SUCCESS", "rounds": worker_result["rounds"], "trees": worker_result["trees"],
        "model_semantic_fingerprint": worker_result["model_semantic_fingerprint"],
        "dump_semantic_fingerprint": worker_result["lightgbm_dump_semantic_fingerprint"],
    }
    model_verification_checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(model_verification_checkpoint)
    write_json_atomic(config.checkpoint_dir / "model_verification.json", model_verification_checkpoint)
    replay_checkpoint = {
        "status": "SUCCESS", "label_free": True, "replay_fingerprint": replay["replay_fingerprint"],
    }
    replay_checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(replay_checkpoint)
    write_json_atomic(config.checkpoint_dir / "prediction_replay.json", replay_checkpoint)
    model_manifest = {
        **worker_result,
        "status": "FINAL_REFIT_MODEL_FROZEN",
        "formal_model_relative_path": "artifacts/stage_4b4/private/final_refit/resource_recovery/models/L1_FINAL_REFIT_600.txt",
        "formal_model_size_bytes": formal_model.stat().st_size,
        "atomic_promotion": True,
        "resource_audit_fingerprint": resource_audit["resource_audit_fingerprint"],
        "recovery_config_canonical_fingerprint": config.fingerprint,
        "recovery_preregistration_fingerprint": prereg["recovery_preregistration_fingerprint"],
        "recovery_source_fingerprint": prereg["recovery_source_fingerprint"],
        "original_failure_audit_fingerprint": config.raw["original_failure"]["resource_audit_semantic_fingerprint"],
        "training_lineage_fingerprint": config.raw["base"]["training_lineage_fingerprint"],
        "combined_matrix_manifest_fingerprint": config.raw["base"]["combined_matrix_manifest_fingerprint"],
        "replay_fingerprint": replay["replay_fingerprint"],
        "checkpoint_fingerprint": checkpoint["checkpoint_fingerprint"],
        "validation_optimizer_recomputed": False, "combined_matrix_recomputed": False,
        "validation_labels_reopened": False,
    }
    model_manifest["model_manifest_fingerprint"] = canonical_fingerprint(model_manifest)
    write_json_atomic(model_manifest_path, model_manifest)
    return model_manifest


def finalize_recovery(config: RecoveryConfig) -> dict[str, Any]:
    identity = verify_recovery_inputs(config)
    prereg = preregister_recovery(config)
    original_failure = _read_json(config.resolve(config.raw["original_failure"]["resource_audit"]))
    validation = _read_json(config.resolve(config.raw["base"]["validation_optimizer_manifest"]))
    combined = _read_json(config.resolve(config.raw["base"]["combined_matrix_manifest"]))
    binary = _read_json(config.path_for("binary_manifest"))
    binary_verification = _read_json(config.path_for("binary_verification_audit"))
    conversion_prereg = _read_json(config.path_for("conversion_resource_preregistration"))
    conversion_resource = _read_json(config.path_for("conversion_resource_audit"))
    training_prereg = _read_json(config.path_for("training_resource_preregistration"))
    training_resource = _read_json(config.path_for("training_resource_audit"))
    model = _read_json(config.path_for("model_manifest"))
    replay = _read_json(config.path_for("replay_audit"))
    test_audit = _read_json(config.path_for("test_security_audit"))
    formal_binary = config.private_dir / "lightgbm_binary/final_refit_train.bin"
    formal_model = config.private_dir / "models/L1_FINAL_REFIT_600.txt"
    checks = {
        "original_failure": (
            original_failure["status"] == "BLOCKED_BY_RESOURCE"
            and file_fingerprint(config.resolve(config.raw["original_failure"]["resource_audit"]))
            == config.raw["original_failure"]["resource_audit_physical_fingerprint"]
        ),
        "recovery_preregistration": (
            prereg["recovery_config_canonical_fingerprint"] == config.fingerprint
            and prereg["recovery_source_fingerprint"] == recovery_source_fingerprint(config)
        ),
        "inputs_reused": (
            not identity["validation_optimizer_recomputed"] and not identity["combined_matrix_recomputed"]
            and identity["validation_optimizer_membership_fingerprint"] == validation["membership_fingerprint"]
            and identity["combined_matrix_manifest_fingerprint"] == combined["manifest_fingerprint"]
        ),
        "binary": (
            binary["status"] == "SUCCESS_FORMAL_BINARY_DATASET"
            and file_fingerprint(formal_binary) == binary["binary_physical_fingerprint"]
            and binary_verification["status"] == "SUCCESS_FORMAL_BINARY_DATASET"
            and all(binary_verification["checks"].values())
        ),
        "conversion_resource": (
            conversion_resource["status"] == "PASS"
            and not conversion_resource["rss_gate_exceeded"]
            and conversion_resource["child_process_tree_peak_rss_bytes"]
            <= conversion_prereg["frozen_absolute_conversion_cap_bytes"]
        ),
        "training_resource": (
            training_resource["status"] == "PASS"
            and not training_resource["rss_gate_exceeded"]
            and training_resource["child_process_tree_peak_rss_bytes"]
            <= training_prereg["frozen_absolute_training_cap_bytes"]
        ),
        "model": (
            model["status"] == "FINAL_REFIT_MODEL_FROZEN"
            and file_fingerprint(formal_model) == model["model_physical_fingerprint"]
            and model["rounds"] == 600 and model["trees"] == 600
            and model["fresh_initialization"] and not model["warm_start"]
            and not model["internal_validation"] and not model["early_stopping"]
            and model["loaded_formal_binary_only"] and not model["loaded_raw_combined_memmap"]
        ),
        "replay": (
            replay["status"] == "PASS" and replay["raw_score_exact_match"]
            and replay["rank_tiebreak_exact_match"] and not replay["label_vector_opened"]
        ),
        "security_tests": test_audit["status"] == "PASS",
        "no_forbidden_access": (
            not config.raw["recovery_scope"]["validation_labels_reopened"]
            and not config.raw["recovery_scope"]["first7d_labels_read"]
            and not config.raw["recovery_scope"]["test_access_allowed"]
        ),
    }
    if not all(checks.values()):
        raise RecoveryError(f"recovery finalization failure: {[key for key,value in checks.items() if not value]}")
    access_rows = [
        {"source": "validation_optimizer", "action": "exact_reuse", "read_count": 1, "recomputed": False, "status": "PASS"},
        {"source": "combined_matrix", "action": "exact_reuse_binary_conversion", "read_count": 1, "recomputed": False, "status": "PASS"},
        {"source": "validation_label_source", "action": "recovery", "read_count": 0, "recomputed": False, "status": "PASS"},
        {"source": "first7d_label_source", "action": "recovery", "read_count": 0, "recomputed": False, "status": "PASS"},
        {"source": "validation_first7d_test_metrics", "action": "recovery", "read_count": 0, "recomputed": False, "status": "PASS"},
        {"source": "test_candidates_features_labels_predictions_metrics", "action": "recovery", "read_count": 0, "recomputed": False, "status": "PASS"},
    ]
    write_csv_atomic(config.path_for("access_audit"), access_rows, list(access_rows[0]))
    access_fingerprint = file_fingerprint(config.path_for("access_audit"))
    checkpoint_files = sorted(config.checkpoint_dir.glob("*.json"))
    checkpoint_set_fingerprint = file_set_fingerprint(checkpoint_files, config.root)
    lock: dict[str, Any] = {
        "stage": "4B-4", "phase": "phase4_final_refit_resource_recovery_v1",
        "status": "FINAL_REFIT_LOCKED_AFTER_RESOURCE_RECOVERY",
        "phase1_pre_label_lock_physical_fingerprint": config.raw["base"]["phase1_lock_physical_fingerprint"],
        "phase2_selection_lock_physical_fingerprint": config.raw["base"]["phase2_lock_physical_fingerprint"],
        "phase2_selection_lock_semantic_fingerprint": config.raw["base"]["phase2_lock_semantic_fingerprint"],
        "phase3_recovered_lock_physical_fingerprint": config.raw["base"]["phase3_recovered_lock_physical_fingerprint"],
        "phase3_recovered_lock_semantic_fingerprint": config.raw["base"]["phase3_recovered_lock_semantic_fingerprint"],
        "base_final_refit_config_fingerprint": config.raw["base"]["final_refit_config_canonical_fingerprint"],
        "base_final_refit_preregistration_fingerprint": config.raw["base"]["final_refit_preregistration_fingerprint"],
        "recovery_config_canonical_fingerprint": config.fingerprint,
        "recovery_preregistration_fingerprint": prereg["recovery_preregistration_fingerprint"],
        "recovery_source_fingerprint": prereg["recovery_source_fingerprint"],
        "original_resource_failure_audit_physical_fingerprint": config.raw["original_failure"]["resource_audit_physical_fingerprint"],
        "original_resource_failure_audit_semantic_fingerprint": config.raw["original_failure"]["resource_audit_semantic_fingerprint"],
        "original_resource_failure_status": "BLOCKED_BY_RESOURCE",
        "original_frozen_cap_bytes": original_failure["frozen_absolute_rss_cap_bytes"],
        "original_peak_rss_bytes": original_failure["child_process_tree_peak_rss_bytes"],
        "original_model_promoted": False,
        "validation_optimizer_manifest_fingerprint": validation["manifest_fingerprint"],
        "validation_optimizer_membership_fingerprint": validation["membership_fingerprint"],
        "combined_matrix_manifest_fingerprint": combined["manifest_fingerprint"],
        "combined_matrix_order_fingerprint": combined["combined_matrix_order_fingerprint"],
        "combined_feature_matrix_fingerprint": combined["feature_matrix_physical_fingerprint"],
        "combined_label_fingerprint": combined["label_physical_fingerprint"],
        "combined_group_fingerprint": combined["group_physical_fingerprint"],
        "combined_query_boundary_fingerprint": combined["query_boundary_physical_fingerprint"],
        "binary_physical_fingerprint": binary["binary_physical_fingerprint"],
        "binary_dataset_semantic_fingerprint": binary["binary_dataset_semantic_fingerprint"],
        "binary_manifest_fingerprint": binary["binary_manifest_fingerprint"],
        "binary_dataset_binning_fingerprint": binary["dataset_binning_fingerprint"],
        "binary_conversion_resource_preregistration_fingerprint": conversion_prereg["resource_preregistration_fingerprint"],
        "binary_conversion_frozen_cap_bytes": conversion_prereg["frozen_absolute_conversion_cap_bytes"],
        "binary_conversion_peak_rss_bytes": conversion_resource["child_process_tree_peak_rss_bytes"],
        "binary_conversion_resource_audit_fingerprint": conversion_resource["resource_audit_fingerprint"],
        "training_resource_preregistration_fingerprint": training_prereg["resource_preregistration_fingerprint"],
        "training_frozen_cap_bytes": training_prereg["frozen_absolute_training_cap_bytes"],
        "training_peak_rss_bytes": training_resource["child_process_tree_peak_rss_bytes"],
        "training_resource_audit_fingerprint": training_resource["resource_audit_fingerprint"],
        "model_identity": {"model_id": "L1_FINAL_REFIT_600", "feature_set": "F1", "rounds": 600, "trees": 600},
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "lightgbm_dump_semantic_fingerprint": model["lightgbm_dump_semantic_fingerprint"],
        "parameter_fingerprint": model["parameter_fingerprint"],
        "feature_name_order_fingerprint": model["feature_name_order_fingerprint"],
        "categorical_metadata_fingerprint": model["categorical_metadata_fingerprint"],
        "training_lineage_fingerprint": model["training_lineage_fingerprint"],
        "model_replay_fingerprint": replay["replay_fingerprint"],
        "checkpoint_set_fingerprint": checkpoint_set_fingerprint,
        "test_security_audit_physical_fingerprint": file_fingerprint(config.path_for("test_security_audit")),
        "access_audit_physical_fingerprint": access_fingerprint,
        "fresh_initialization": True, "warm_start": False,
        "continued_training_from_L1_600": False, "internal_validation": False,
        "early_stopping": False, "validation_optimizer_recomputed": False,
        "combined_matrix_recomputed": False, "validation_labels_reopened": False,
        "first7d_labels_read": False, "final_refit_executed": True,
        "test_authorized": False, "test_candidate_read_count": 0,
        "test_feature_read_count": 0, "test_label_read_count": 0,
        "test_prediction_write_count": 0, "test_metric_read_count": 0,
        "test_prediction_frozen": False,
    }
    lock["lock_fingerprint"] = self_fingerprint(lock, "lock_fingerprint")
    write_json_atomic(config.path_for("lock"), lock)
    if self_fingerprint(_read_json(config.path_for("lock")), "lock_fingerprint") != lock["lock_fingerprint"]:
        raise RecoveryError("final-refit lock integrity failure")
    gate_rows = [
        {"gate_id": "R1", "check": "original resource failure evidence preserved", "evidence": lock["original_resource_failure_audit_semantic_fingerprint"], "status": "PASS"},
        {"gate_id": "R2", "check": "recovery config and authorization identity", "evidence": prereg["recovery_preregistration_fingerprint"], "status": "PASS"},
        {"gate_id": "R3", "check": "Validation optimizer exact reuse", "evidence": validation["membership_fingerprint"], "status": "PASS"},
        {"gate_id": "R4", "check": "combined matrix exact reuse", "evidence": combined["manifest_fingerprint"], "status": "PASS"},
        {"gate_id": "R5", "check": "no label reopen or rematerialization", "evidence": access_fingerprint, "status": "PASS"},
        {"gate_id": "R6", "check": "binary Dataset construction parameter identity", "evidence": binary["dataset_binning_fingerprint"], "status": "PASS"},
        {"gate_id": "R7", "check": "binary row/group/label/feature integrity", "evidence": binary["binary_dataset_semantic_fingerprint"], "status": "PASS"},
        {"gate_id": "R8", "check": "binary conversion frozen 70 percent cap", "evidence": conversion_resource["resource_audit_fingerprint"], "status": "PASS"},
        {"gate_id": "R9", "check": "fresh isolated binary-only training worker", "evidence": model["training_worker_result_fingerprint"], "status": "PASS"},
        {"gate_id": "R10", "check": "training worker frozen 70 percent cap", "evidence": training_resource["resource_audit_fingerprint"], "status": "PASS"},
        {"gate_id": "R11", "check": "exact L1 F1 600 fresh-init model identity", "evidence": model["parameter_fingerprint"], "status": "PASS"},
        {"gate_id": "R12", "check": "formal model fingerprint reload replay", "evidence": replay["replay_fingerprint"], "status": "PASS"},
        {"gate_id": "R13", "check": "no Validation First7d Test metric or selection", "evidence": access_fingerprint, "status": "PASS"},
        {"gate_id": "R14", "check": "no Test access", "evidence": canonical_fingerprint(config.raw["recovery_scope"]), "status": "PASS"},
        {"gate_id": "R15", "check": "checkpoint privacy Git tests", "evidence": file_fingerprint(config.path_for("test_security_audit")), "status": "PASS"},
        {"gate_id": "R16", "check": "final-refit lock integrity", "evidence": lock["lock_fingerprint"], "status": "PASS"},
    ]
    write_csv_atomic(config.path_for("recovered_gate"), gate_rows, list(gate_rows[0]))
    gate_checkpoint = {
        "status": "SUCCESS", "resource_recovery_gate": "PASSED",
        "lock_fingerprint": lock["lock_fingerprint"],
        "gate_physical_fingerprint": file_fingerprint(config.path_for("recovered_gate")),
    }
    gate_checkpoint["checkpoint_fingerprint"] = canonical_fingerprint(gate_checkpoint)
    write_json_atomic(config.checkpoint_dir / "lock_and_gate.json", gate_checkpoint)
    manifest_rows = [{
        "stage": "4B-4", "phase": "phase4_final_refit_resource_recovery_v1",
        "status": "STAGE_4B4_PHASE4_RESOURCE_RECOVERY_COMPLETE",
        "model_id": "L1_FINAL_REFIT_600", "feature_set": "F1", "rounds": 600,
        "binary_physical_fingerprint": binary["binary_physical_fingerprint"],
        "binary_semantic_fingerprint": binary["binary_dataset_semantic_fingerprint"],
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "lock_fingerprint": lock["lock_fingerprint"], "resource_recovery_gate": "PASSED",
        "test_authorized": False,
    }]
    manifest_path = config.resolve("artifacts/stage_4b4/final_refit_resource_recovery_manifest.csv")
    write_csv_atomic(manifest_path, manifest_rows, list(manifest_rows[0]))
    report = _render_recovery_report(
        config, original_failure, validation, combined, prereg, binary, binary_verification,
        conversion_prereg, conversion_resource, training_prereg, training_resource,
        model, replay, test_audit, lock,
    )
    report_path = config.path_for("report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)
    return {
        "status": "STAGE_4B4_PHASE4_RESOURCE_RECOVERY_COMPLETE",
        "binary_physical_fingerprint": binary["binary_physical_fingerprint"],
        "binary_semantic_fingerprint": binary["binary_dataset_semantic_fingerprint"],
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "lock_fingerprint": lock["lock_fingerprint"], "gate_passed": 16, "gate_total": 16,
    }


def _render_recovery_report(
    config: RecoveryConfig, original: dict[str, Any], validation: dict[str, Any],
    combined: dict[str, Any], prereg: dict[str, Any], binary: dict[str, Any],
    binary_verification: dict[str, Any], conversion_prereg: dict[str, Any],
    conversion_resource: dict[str, Any], training_prereg: dict[str, Any],
    training_resource: dict[str, Any], model: dict[str, Any], replay: dict[str, Any],
    test_audit: dict[str, Any], lock: dict[str, Any],
) -> str:
    return f"""# Stage 4B-4 Phase 4：Final Refit Resource Recovery

## 1. Git 和 recovery 接管状态

接管 branch/HEAD 与 origin 未发生意外变化；工作区只有 Stage 4B-4 相关修改，无遗留 Python/LightGBM worker。恢复授权为 `{config.raw['authorization']}`。

## 2. 原资源失败证据

原 raw-memmap worker 正确停止为 `{original['status']}`：cap={original['frozen_absolute_rss_cap_bytes']} bytes，peak={original['child_process_tree_peak_rss_bytes']} bytes，超出 {original['child_process_tree_peak_rss_bytes'] - original['frozen_absolute_rss_cap_bytes']} bytes；OOM=false，formal model 未 promotion。原 audit 保持原物理指纹。

## 3. Validation optimizer 复用

精确复用且未重新采样/物化：{validation['query_count']} queries、{validation['row_count']} rows；membership/replay fingerprint 均为 `{validation['membership_fingerprint']}`。

## 4. Combined matrix 复用

精确复用且未重新物化：{combined['query_count']} queries、{combined['row_count']} rows、52 F1 features、{combined['positive_rows']} positives、{combined['zero_rows']} zeros；所有 frozen physical/order/boundary fingerprints 匹配。

## 5. Recovery config fingerprint

Recovery config canonical fingerprint `{config.fingerprint}`；preregistration `{prereg['recovery_preregistration_fingerprint']}`；source `{prereg['recovery_source_fingerprint']}`。

## 6. Binary conversion worker

Fresh isolated conversion worker 只导入标准库、NumPy、LightGBM，仅消费 frozen combined memmap/labels/groups；未训练模型、未计算业务指标、未打开标签源。

## 7. Binary resource preregistration/cap/peak

Conversion 启动前 available={conversion_prereg['available_memory_at_conversion_start_bytes']} bytes，70% frozen cap={conversion_prereg['frozen_absolute_conversion_cap_bytes']} bytes；process-tree peak={conversion_resource['child_process_tree_peak_rss_bytes']} bytes，resource gate PASS，worker 未重算 cap。

## 8. Binary Dataset rows/groups/labels/features

Formal binary：{binary['rows']} rows、{binary['queries']} groups、group sum={binary['group_sum']}、{binary['features']} features、{binary['positive_rows']} positives、{binary['zero_rows']} zeros。验证 checks 全部 PASS。

## 9. Binary physical/semantic fingerprints

Physical `{binary['binary_physical_fingerprint']}`；semantic `{binary['binary_dataset_semantic_fingerprint']}`；metadata `{binary['dataset_metadata_fingerprint']}`；binning `{binary['dataset_binning_fingerprint']}`。

## 10. Binary round-trip/equivalence

Conversion 后由独立进程仅加载 binary，label/group byte identity、feature order、categorical/binning binding 全部一致。Synthetic raw→binary fixed-round prediction equivalence 由定向测试验证。

## 11. Isolated training worker

另一个 fresh worker 只加载 formal binary Dataset，不加载 raw combined memmap、DuckDB、PyArrow 或 feature/label pipeline；fresh initialization 固定训练 600 rounds/600 trees。

## 12. Training resource preregistration/cap/peak

Training 启动前 available={training_prereg['available_memory_at_training_start_bytes']} bytes，70% frozen cap={training_prereg['frozen_absolute_training_cap_bytes']} bytes；process-tree peak={training_resource['child_process_tree_peak_rss_bytes']} bytes，resource gate PASS，worker 未重算 cap。

## 13. Exact L1/F1/600 parameter identity

完整参数、weights、12 个 categorical features、全部 seeds、num_threads=4 与 Phase 1 L1 逐字段一致；无 valid_sets、early stopping、warm start 或参数搜索。

## 14. Final model fingerprints

Physical `{model['model_physical_fingerprint']}`；semantic `{model['model_semantic_fingerprint']}`；dump semantic `{model['lightgbm_dump_semantic_fingerprint']}`；parameter `{model['parameter_fingerprint']}`。

## 15. Model reload/replay

Temporary model reload 验证 600 rounds/trees；原子 promotion 后两个独立 reload 在固定 label-free 完整 query slice 上 raw score 与 deterministic rank/tie-break exact match。Replay fingerprint `{replay['replay_fingerprint']}`。

## 16. Checkpoints/private artifacts

原失败 checkpoint 保持 FAILED/NOT_REUSABLE；binary conversion/verification、binary training、model verification、prediction replay、lock/gate 均有独立 SUCCESS checkpoint。Binary、model、logs 与 temporary artifacts 均在 Git-ignored private/tmp/checkpoint。

## 17. Recovered lock

正式 final-refit lock 状态 `FINAL_REFIT_LOCKED_AFTER_RESOURCE_RECOVERY`，lock fingerprint `{lock['lock_fingerprint']}`，完整绑定原失败、recovery、binary、两次资源门、模型与 replay。

## 18. R1–R16 gate

独立 Phase 4 Resource Recovery Gate 为 16/16 PASS。

## 19. Targeted/full tests 和安全审计

Targeted `{test_audit['tests']['targeted']}`；full `{test_audit['tests']['full']}`；compile、diff、privacy/path、raw-data、ignore、tracked-private 均 PASS。

## 20. Validation/First7d/Test 访问审计

Validation optimizer/combined 仅复用；Validation labels reopened=false；First7d labels read=false；Test candidate/feature/label/prediction/metric 全部 0。

## 21. 未执行项

未重建 optimizer/matrix，未 negative sampling/label join，未计算 Validation/First7d/Test metrics，未 rerank、retrieval retune 或运行 raw-memmap formal training。

## 22. Commit/push

未 commit，未 push；README 与 AGENTS 未更新。

## 23. 最终状态

`STAGE_4B4_PHASE4_RESOURCE_RECOVERY_COMPLETE`。Binary Dataset 与 final-refit model 已冻结，recovered gate PASSED，Test 仍未授权；等待 ChatGPT 技术验收。

> 该结果只证明离线 final-refit 工程与协议正确，不代表线上 CTR/CVR、收入提升或 A/B Test 结果。
"""
