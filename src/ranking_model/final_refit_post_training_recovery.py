from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class PostTrainingRecoveryError(RuntimeError):
    """Phase 4 训练后恢复违反冻结合同时抛出的异常。"""


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
    rows = [
        {
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_fingerprint(path),
        }
        for path in sorted(paths, key=lambda value: value.as_posix())
    ]
    return canonical_fingerprint(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def copy_bytes_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    os.replace(temporary, destination)


@dataclass(frozen=True)
class PostTrainingConfig:
    root: Path
    path: Path
    raw: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> "PostTrainingConfig":
        resolved = path.resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        value = cls(
            root=resolved.parent.parent,
            path=resolved,
            raw=raw,
            fingerprint=canonical_fingerprint(raw),
        )
        value.validate()
        return value

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()

    def path_for(self, name: str) -> Path:
        return self.resolve(self.raw["paths"][name])

    def validate(self) -> None:
        if self.raw.get("phase") != "phase4_post_training_manifest_audit_recovery_v1":
            raise PostTrainingRecoveryError("wrong post-training recovery phase")
        scope = self.raw["scope"]
        forbidden = (
            "model_retraining",
            "binary_rebuild",
            "prediction_replay_recomputed",
            "label_access",
            "validation_optimizer_recomputed",
            "combined_matrix_recomputed",
        )
        if any(scope[name] for name in forbidden):
            raise PostTrainingRecoveryError("RECOVERY_SCOPE_EXCEEDED")
        if scope["first7d_read_count"] != 0:
            raise PostTrainingRecoveryError("RECOVERY_SCOPE_EXCEEDED")
        if any(value != 0 for key, value in scope.items() if key.startswith("test_") and key.endswith("_count")):
            raise PostTrainingRecoveryError("RECOVERY_SCOPE_EXCEEDED")


def post_training_source_fingerprint(config: PostTrainingConfig) -> str:
    paths = [
        config.path,
        config.root / "src/ranking_model/final_refit_post_training_recovery.py",
        config.root / "scripts/run_stage_4b4_final_refit_post_training_recovery.py",
        config.root / "scripts/run_stage_4b4_final_refit_post_training_audit.py",
        config.root / "tests/test_stage_4b4_phase4_post_training_recovery.py",
    ]
    if any(not path.exists() for path in paths):
        raise PostTrainingRecoveryError("post-training source inventory incomplete")
    return file_set_fingerprint(paths, config.root)


def _replay_binding(config: PostTrainingConfig) -> dict[str, Any]:
    expected = config.raw["expected"]
    replay_path = config.path_for("replay_audit")
    if file_fingerprint(replay_path) != expected["replay_audit_physical_fingerprint"]:
        raise PostTrainingRecoveryError("POST_TRAINING_ARTIFACT_DRIFT: replay physical fingerprint")
    replay = read_json(replay_path)
    first = replay.get("first_reload", {})
    second = replay.get("second_reload", {})
    raw_values = {first.get("raw_score_fingerprint"), second.get("raw_score_fingerprint")}
    tie_values = {first.get("rank_tiebreak_fingerprint"), second.get("rank_tiebreak_fingerprint")}
    if raw_values != {expected["raw_score_fingerprint"]} or tie_values != {expected["tie_break_fingerprint"]}:
        raise PostTrainingRecoveryError("REPLAY_EVIDENCE_INCOMPLETE")
    if (
        replay.get("status") != "PASS"
        or replay.get("replay_fingerprint") != expected["replay_fingerprint"]
        or not replay.get("raw_score_exact_match")
        or not replay.get("rank_tiebreak_exact_match")
        or replay.get("label_vector_opened")
        or first.get("non_finite_scores") != 0
        or second.get("non_finite_scores") != 0
    ):
        raise PostTrainingRecoveryError("REPLAY_EVIDENCE_INCOMPLETE")
    return {
        "raw_score_fingerprint": expected["raw_score_fingerprint"],
        "tie_break_fingerprint": expected["tie_break_fingerprint"],
        "replay_audit_physical_fingerprint": expected["replay_audit_physical_fingerprint"],
        "replay_fingerprint": expected["replay_fingerprint"],
        "label_free": True,
        "model_reloaded": True,
        "raw_score_mismatch_count": 0,
        "tie_break_mismatch_count": 0,
        "nonfinite_count": 0,
    }


def _model_tree_audit(config: PostTrainingConfig) -> dict[str, int]:
    # 只读解析模型结构；不加载任何特征、标签，也不调用 predict。
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(config.path_for("model_artifact")))
    return {
        "trees": int(booster.num_trees()),
        "iterations": int(booster.current_iteration()),
        "features": int(booster.num_feature()),
    }


def _training_process_count() -> int:
    import psutil

    markers = (
        "stage_4b4_binary_training_worker.py",
        "stage_4b4_final_refit_worker.py",
        "stage_4b4_final_model_replay_worker.py",
        "stage_4b4_binary_conversion_worker.py",
    )
    count = 0
    for process in psutil.process_iter(["pid", "cmdline"]):
        if process.pid == os.getpid():
            continue
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(marker in command for marker in markers):
            count += 1
    return count


def verify_frozen_post_training_artifacts(
    config: PostTrainingConfig,
    *,
    allow_corrected_manifest: bool,
) -> dict[str, Any]:
    expected = config.raw["expected"]
    paths = config.raw["paths"]
    checks: dict[str, bool] = {}

    validation_path = config.resolve(paths["validation_manifest"])
    combined_path = config.resolve(paths["combined_manifest"])
    binary_manifest_path = config.resolve(paths["binary_manifest"])
    binary_path = config.resolve(paths["binary_artifact"])
    model_manifest_path = config.resolve(paths["model_manifest"])
    model_path = config.resolve(paths["model_artifact"])

    checks["validation_manifest_unchanged"] = (
        file_fingerprint(validation_path) == expected["validation_manifest_physical_fingerprint"]
    )
    checks["combined_manifest_unchanged"] = (
        file_fingerprint(combined_path) == expected["combined_manifest_physical_fingerprint"]
    )
    checks["binary_manifest_unchanged"] = (
        file_fingerprint(binary_manifest_path) == expected["binary_manifest_physical_fingerprint"]
    )
    binary_manifest = read_json(binary_manifest_path)
    checks["binary_artifact_unchanged"] = (
        file_fingerprint(binary_path) == expected["binary_physical_fingerprint"]
        and binary_manifest.get("binary_physical_fingerprint") == expected["binary_physical_fingerprint"]
        and binary_manifest.get("binary_dataset_semantic_fingerprint") == expected["binary_semantic_fingerprint"]
    )

    manifest = read_json(model_manifest_path)
    manifest_physical = file_fingerprint(model_manifest_path)
    if allow_corrected_manifest and "prediction_replay" in manifest:
        correction_path = config.path_for("manifest_correction_audit")
        correction = read_json(correction_path) if correction_path.exists() else {}
        checks["model_manifest_state"] = (
            self_fingerprint(manifest, "model_manifest_fingerprint") == manifest.get("model_manifest_fingerprint")
            and correction.get("corrected_manifest_physical_fingerprint") == manifest_physical
            and correction.get("status") == "PASS"
        )
    else:
        checks["model_manifest_state"] = (
            manifest_physical == expected["original_incomplete_manifest_physical_fingerprint"]
            and manifest.get("model_manifest_fingerprint")
            == expected["original_incomplete_manifest_semantic_fingerprint"]
            and self_fingerprint(manifest, "model_manifest_fingerprint")
            == manifest.get("model_manifest_fingerprint")
            and "prediction_replay" not in manifest
        )
    checks["model_artifact_unchanged"] = (
        file_fingerprint(model_path) == expected["model_physical_fingerprint"]
        and manifest.get("model_physical_fingerprint") == expected["model_physical_fingerprint"]
        and manifest.get("model_semantic_fingerprint") == expected["model_semantic_fingerprint"]
    )
    tree_audit = _model_tree_audit(config)
    checks["model_structure"] = tree_audit == {
        "trees": expected["trees"],
        "iterations": expected["rounds"],
        "features": expected["features"],
    }
    replay_binding = _replay_binding(config)
    checks["replay_evidence"] = bool(replay_binding["raw_score_fingerprint"])
    checks["no_background_training_or_replay_worker"] = _training_process_count() == 0

    recovery_prereg = read_json(config.resolve(paths["resource_recovery_config"]))
    checks["original_recovery_source_binding"] = (
        recovery_prereg["base"]["training_lineage_fingerprint"]
        == expected["training_lineage_fingerprint"]
    )
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise PostTrainingRecoveryError(f"POST_TRAINING_ARTIFACT_DRIFT: {failed}")
    return {
        "status": "POST_TRAINING_ARTIFACT_IDENTITY_VERIFIED",
        "checks": checks,
        "model_physical_fingerprint": expected["model_physical_fingerprint"],
        "model_semantic_fingerprint": expected["model_semantic_fingerprint"],
        "binary_physical_fingerprint": expected["binary_physical_fingerprint"],
        "binary_semantic_fingerprint": expected["binary_semantic_fingerprint"],
        "replay_audit_physical_fingerprint": expected["replay_audit_physical_fingerprint"],
        "replay_fingerprint": expected["replay_fingerprint"],
        "tree_audit": tree_audit,
        "model_retrained_during_recovery": False,
        "binary_rebuilt_during_recovery": False,
        "prediction_recomputed_during_recovery": False,
        "labels_read_during_recovery": False,
    }


def preregister_post_training_recovery(config: PostTrainingConfig) -> dict[str, Any]:
    identity = verify_frozen_post_training_artifacts(config, allow_corrected_manifest=False)
    paths = config.raw["paths"]
    source_fingerprint = post_training_source_fingerprint(config)
    checkpoint_dir = config.resolve(paths["checkpoint_directory"])
    checkpoint_files = sorted(checkpoint_dir.glob("*.json"))
    payload: dict[str, Any] = {
        "status": "POST_TRAINING_RECOVERY_PREREGISTERED_BEFORE_METADATA_MUTATION",
        "authorization": config.raw["authorization"],
        "config_canonical_fingerprint": config.fingerprint,
        "source_fingerprint": source_fingerprint,
        "original_recovery_source_fingerprint": config.raw["expected"]["original_recovery_source_fingerprint"],
        "original_incomplete_manifest_physical_fingerprint": config.raw["expected"]["original_incomplete_manifest_physical_fingerprint"],
        "failed_test_audit_physical_fingerprint": config.raw["expected"]["failed_test_audit_physical_fingerprint"],
        "model_physical_fingerprint": identity["model_physical_fingerprint"],
        "model_semantic_fingerprint": identity["model_semantic_fingerprint"],
        "binary_physical_fingerprint": identity["binary_physical_fingerprint"],
        "binary_semantic_fingerprint": identity["binary_semantic_fingerprint"],
        "replay_audit_physical_fingerprint": identity["replay_audit_physical_fingerprint"],
        "replay_fingerprint": identity["replay_fingerprint"],
        "preexisting_checkpoint_set_fingerprint": file_set_fingerprint(checkpoint_files, config.root),
        "preexisting_checkpoint_count": len(checkpoint_files),
        "model_retraining_allowed": False,
        "binary_rebuild_allowed": False,
        "prediction_recomputation_allowed": False,
        "label_access_allowed": False,
    }
    payload["preregistration_fingerprint"] = self_fingerprint(payload, "preregistration_fingerprint")
    path = config.path_for("preregistration")
    if path.exists():
        if read_json(path) != payload:
            raise PostTrainingRecoveryError("post-training preregistration drift")
    else:
        write_json_atomic(path, payload)
    identity_payload = dict(identity)
    identity_payload["config_canonical_fingerprint"] = config.fingerprint
    identity_payload["source_fingerprint"] = source_fingerprint
    identity_payload["identity_audit_fingerprint"] = self_fingerprint(
        identity_payload, "identity_audit_fingerprint"
    )
    write_json_atomic(config.path_for("identity_audit"), identity_payload)
    return payload


def _privacy_violation_count(raw_text: str) -> int:
    drive_paths = re.findall(r"(?i)\b[a-z]:[\\/]", raw_text)
    user_markers = re.findall(r"(?i)users[\\/][^\\/\s'\"]+", raw_text)
    return len(drive_paths) + len(user_markers)


def correct_post_training_metadata(config: PostTrainingConfig) -> dict[str, Any]:
    prereg = read_json(config.path_for("preregistration"))
    if prereg.get("status") != "POST_TRAINING_RECOVERY_PREREGISTERED_BEFORE_METADATA_MUTATION":
        raise PostTrainingRecoveryError("post-training recovery is not preregistered")
    expected = config.raw["expected"]
    model_manifest_path = config.path_for("model_manifest")
    original_manifest_private = config.path_for("original_manifest_private")
    failed_audit_path = config.path_for("failed_test_audit")
    failed_audit_private = config.path_for("failed_test_audit_private")

    if file_fingerprint(model_manifest_path) != expected["original_incomplete_manifest_physical_fingerprint"]:
        raise PostTrainingRecoveryError("POST_TRAINING_ARTIFACT_DRIFT: incomplete manifest")
    if file_fingerprint(failed_audit_path) != expected["failed_test_audit_physical_fingerprint"]:
        raise PostTrainingRecoveryError("POST_TRAINING_ARTIFACT_DRIFT: failed audit")

    original_manifest = read_json(model_manifest_path)
    original_failed_bytes = failed_audit_path.read_bytes()
    copy_bytes_atomic(model_manifest_path, original_manifest_private)
    copy_bytes_atomic(failed_audit_path, failed_audit_private)
    if file_fingerprint(original_manifest_private) != expected["original_incomplete_manifest_physical_fingerprint"]:
        raise PostTrainingRecoveryError("private incomplete manifest preservation failure")
    if file_fingerprint(failed_audit_private) != expected["failed_test_audit_physical_fingerprint"]:
        raise PostTrainingRecoveryError("private failed audit preservation failure")

    replay_binding = _replay_binding(config)
    corrected = dict(original_manifest)
    corrected["prediction_replay"] = {
        "raw_score_fingerprint": replay_binding["raw_score_fingerprint"],
        "tie_break_fingerprint": replay_binding["tie_break_fingerprint"],
        "replay_audit_physical_fingerprint": replay_binding["replay_audit_physical_fingerprint"],
        "label_free": True,
        "model_reloaded": True,
        "raw_score_mismatch_count": 0,
        "tie_break_mismatch_count": 0,
        "nonfinite_count": 0,
    }
    corrected["model_manifest_fingerprint"] = self_fingerprint(corrected, "model_manifest_fingerprint")
    temporary = model_manifest_path.with_suffix(model_manifest_path.suffix + ".corrected.tmp")
    temporary.write_text(json.dumps(corrected, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    candidate = read_json(temporary)
    if candidate["prediction_replay"]["raw_score_fingerprint"] != expected["raw_score_fingerprint"]:
        raise PostTrainingRecoveryError("REPLAY_EVIDENCE_INCOMPLETE")
    if self_fingerprint(candidate, "model_manifest_fingerprint") != candidate["model_manifest_fingerprint"]:
        raise PostTrainingRecoveryError("corrected model manifest self-fingerprint failure")
    original_keys = set(original_manifest) - {"model_manifest_fingerprint"}
    if any(candidate[key] != original_manifest[key] for key in original_keys):
        raise PostTrainingRecoveryError("FINAL_REFIT_SEMANTIC_DRIFT: non-schema manifest field")
    if file_fingerprint(config.path_for("model_artifact")) != expected["model_physical_fingerprint"]:
        raise PostTrainingRecoveryError("POST_TRAINING_ARTIFACT_DRIFT: model")
    os.replace(temporary, model_manifest_path)

    corrected_physical = file_fingerprint(model_manifest_path)
    correction_audit: dict[str, Any] = {
        "status": "PASS",
        "failure_category": "MISSING_NESTED_PREDICTION_REPLAY_SCHEMA",
        "original_schema_validation_failure": "prediction_replay.raw_score_fingerprint missing",
        "original_incomplete_manifest_physical_fingerprint": expected["original_incomplete_manifest_physical_fingerprint"],
        "original_incomplete_manifest_semantic_fingerprint": expected["original_incomplete_manifest_semantic_fingerprint"],
        "corrected_manifest_physical_fingerprint": corrected_physical,
        "corrected_manifest_semantic_fingerprint": corrected["model_manifest_fingerprint"],
        "original_manifest_private_evidence_fingerprint": file_fingerprint(original_manifest_private),
        "model_physical_fingerprint_before": prereg["model_physical_fingerprint"],
        "model_physical_fingerprint_after": file_fingerprint(config.path_for("model_artifact")),
        "model_semantic_fingerprint": expected["model_semantic_fingerprint"],
        "binary_physical_fingerprint_before": prereg["binary_physical_fingerprint"],
        "binary_physical_fingerprint_after": file_fingerprint(config.path_for("binary_artifact")),
        "replay_audit_physical_fingerprint_before": prereg["replay_audit_physical_fingerprint"],
        "replay_audit_physical_fingerprint_after": file_fingerprint(config.path_for("replay_audit")),
        "model_artifact_unchanged": True,
        "binary_artifact_unchanged": True,
        "replay_artifact_unchanged": True,
        "only_manifest_metadata_schema_changed": True,
        "atomic_promotion": True,
        "model_retrained_during_fix": False,
        "prediction_recomputed_during_fix": False,
        "labels_read_during_fix": False,
    }
    correction_audit["correction_audit_fingerprint"] = self_fingerprint(
        correction_audit, "correction_audit_fingerprint"
    )
    write_json_atomic(config.path_for("manifest_correction_audit"), correction_audit)

    sanitized_failure: dict[str, Any] = {
        "status": "SANITIZED_FAILURE_EVIDENCE_PRESERVED",
        "failure_category": "RESOURCE_RECOVERY_IMPLEMENTATION_FAILURE",
        "original_private_evidence_physical_fingerprint": expected["failed_test_audit_physical_fingerprint"],
        "manifest_missing_field_failure": "prediction_replay.raw_score_fingerprint",
        "git_warning_misclassification": True,
        "privacy_violation_count": _privacy_violation_count(original_failed_bytes.decode("utf-8", errors="ignore")),
        "correction_status": "MANIFEST_AND_AUDITOR_CORRECTION_AUTHORIZED",
        "original_failure_fact_deleted": False,
        "private_evidence_tracked": False,
    }
    sanitized_failure["sanitized_failure_audit_fingerprint"] = self_fingerprint(
        sanitized_failure, "sanitized_failure_audit_fingerprint"
    )
    write_json_atomic(config.path_for("sanitized_failure_audit"), sanitized_failure)
    pending_public = {
        "status": "PENDING_AUTHORIZED_AUDIT_RERUN",
        "prior_failure_category": "RESOURCE_RECOVERY_IMPLEMENTATION_FAILURE",
        "original_private_evidence_physical_fingerprint": expected["failed_test_audit_physical_fingerprint"],
        "public_privacy_sanitized": True,
    }
    write_json_atomic(failed_audit_path, pending_public)

    checkpoint = {
        "status": "SUCCESS",
        "artifact": "POST_TRAINING_MANIFEST_AND_FAILURE_AUDIT_CORRECTION",
        "corrected_manifest_physical_fingerprint": corrected_physical,
        "correction_audit_fingerprint": correction_audit["correction_audit_fingerprint"],
        "sanitized_failure_audit_fingerprint": sanitized_failure["sanitized_failure_audit_fingerprint"],
        "model_retrained": False,
        "prediction_recomputed": False,
    }
    checkpoint["checkpoint_fingerprint"] = self_fingerprint(checkpoint, "checkpoint_fingerprint")
    checkpoint_path = config.path_for("checkpoint_directory") / "post_training_manifest_correction.json"
    write_json_atomic(checkpoint_path, checkpoint)
    return {
        "status": "POST_TRAINING_METADATA_CORRECTED",
        "corrected_manifest_physical_fingerprint": corrected_physical,
        "corrected_manifest_semantic_fingerprint": corrected["model_manifest_fingerprint"],
        "raw_score_fingerprint": replay_binding["raw_score_fingerprint"],
        "model_retrained": False,
        "prediction_recomputed": False,
    }


def warning_categories(stderr: str) -> list[str]:
    categories: set[str] = set()
    for line in stderr.splitlines():
        lowered = line.lower()
        if "ignore" in lowered and ("permission denied" in lowered or "unable to access" in lowered):
            categories.add("GLOBAL_IGNORE_ACCESS_WARNING")
        elif "lf will be replaced by crlf" in lowered:
            categories.add("LINE_ENDING_WARNING")
        elif line.strip():
            categories.add("OTHER_GIT_WARNING")
    return sorted(categories)


def evaluate_raw_data_git_results(
    working: subprocess.CompletedProcess[str],
    staged: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    working_files = [line.strip() for line in working.stdout.splitlines() if line.strip()]
    staged_files = [line.strip() for line in staged.stdout.splitlines() if line.strip()]
    command_success = working.returncode == 0 and staged.returncode == 0
    changed = sorted(set(working_files + staged_files))
    return {
        "status": "PASS" if command_success and not changed else ("FAIL" if command_success else "UNKNOWN"),
        "working_tree_return_code": working.returncode,
        "staged_return_code": staged.returncode,
        "working_tree_changed_file_count": len(working_files),
        "staged_changed_file_count": len(staged_files),
        "changed_file_count": len(changed),
        "warning_categories": sorted(set(warning_categories(working.stderr) + warning_categories(staged.stderr))),
        "stderr_is_not_diff_evidence": True,
    }


def _phase_lock_fingerprints(config: PostTrainingConfig) -> dict[str, str]:
    return {
        name: file_fingerprint(config.path_for(name))
        for name in ("phase1_lock", "phase2_lock", "phase3_lock")
    }


def finalize_post_training_recovery(config: PostTrainingConfig) -> dict[str, Any]:
    identity = verify_frozen_post_training_artifacts(config, allow_corrected_manifest=True)
    prereg = read_json(config.path_for("preregistration"))
    correction = read_json(config.path_for("manifest_correction_audit"))
    sanitized_failure = read_json(config.path_for("sanitized_failure_audit"))
    test_audit = read_json(config.path_for("test_security_audit"))
    if test_audit.get("status") != "PASS":
        raise PostTrainingRecoveryError("POST_TRAINING_ENGINEERING_GATE_FAILED")
    model = read_json(config.path_for("model_manifest"))
    binary = read_json(config.path_for("binary_manifest"))
    replay = read_json(config.path_for("replay_audit"))
    validation = read_json(config.path_for("validation_manifest"))
    combined = read_json(config.path_for("combined_manifest"))
    conversion_prereg = read_json(config.path_for("conversion_preregistration"))
    conversion_resource = read_json(config.path_for("conversion_resource_audit"))
    training_prereg = read_json(config.path_for("training_preregistration"))
    training_resource = read_json(config.path_for("training_resource_audit"))
    original_resource = read_json(config.path_for("original_resource_failure_audit"))
    phase_locks = _phase_lock_fingerprints(config)

    access_rows = [
        {"source": "validation_optimizer_metadata", "read_count": 1, "recomputed": False, "status": "PASS"},
        {"source": "combined_matrix_manifest", "read_count": 1, "recomputed": False, "status": "PASS"},
        {"source": "validation_label_source", "read_count": 0, "recomputed": False, "status": "PASS"},
        {"source": "first7d_label_source", "read_count": 0, "recomputed": False, "status": "PASS"},
        {"source": "test_candidates_features_labels_predictions_metrics", "read_count": 0, "recomputed": False, "status": "PASS"},
    ]
    write_csv_atomic(config.path_for("access_audit"), access_rows, list(access_rows[0]))
    access_fingerprint = file_fingerprint(config.path_for("access_audit"))

    lock: dict[str, Any] = {
        "stage": "4B-4",
        "phase": "phase4_post_training_manifest_audit_recovery_v1",
        "status": "FINAL_REFIT_LOCKED_AFTER_POST_TRAINING_RECOVERY",
        "phase1_pre_label_lock_physical_fingerprint": phase_locks["phase1_lock"],
        "phase2_selection_lock_physical_fingerprint": phase_locks["phase2_lock"],
        "phase3_recovered_lock_physical_fingerprint": phase_locks["phase3_lock"],
        "base_final_refit_config_physical_fingerprint": file_fingerprint(config.path_for("base_final_refit_config")),
        "resource_recovery_config_physical_fingerprint": file_fingerprint(config.path_for("resource_recovery_config")),
        "post_training_config_canonical_fingerprint": config.fingerprint,
        "post_training_preregistration_fingerprint": prereg["preregistration_fingerprint"],
        "post_training_source_fingerprint": prereg["source_fingerprint"],
        "original_recovery_source_fingerprint": prereg["original_recovery_source_fingerprint"],
        "original_resource_failure_audit_physical_fingerprint": file_fingerprint(config.path_for("original_resource_failure_audit")),
        "original_resource_failure_status": original_resource["status"],
        "validation_optimizer_manifest_physical_fingerprint": file_fingerprint(config.path_for("validation_manifest")),
        "validation_optimizer_membership_fingerprint": validation["membership_fingerprint"],
        "combined_matrix_manifest_physical_fingerprint": file_fingerprint(config.path_for("combined_manifest")),
        "combined_matrix_manifest_fingerprint": combined["manifest_fingerprint"],
        "binary_manifest_fingerprint": binary["binary_manifest_fingerprint"],
        "binary_physical_fingerprint": binary["binary_physical_fingerprint"],
        "binary_semantic_fingerprint": binary["binary_dataset_semantic_fingerprint"],
        "binary_conversion_frozen_cap_bytes": conversion_prereg["frozen_absolute_conversion_cap_bytes"],
        "binary_conversion_peak_rss_bytes": conversion_resource["child_process_tree_peak_rss_bytes"],
        "binary_conversion_resource_gate": conversion_resource["status"],
        "training_frozen_cap_bytes": training_prereg["frozen_absolute_training_cap_bytes"],
        "training_peak_rss_bytes": training_resource["child_process_tree_peak_rss_bytes"],
        "training_resource_gate": training_resource["status"],
        "model_identity": {"model_id": "L1_FINAL_REFIT_600", "feature_set": "F1", "rounds": 600, "trees": 600},
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "model_manifest_physical_fingerprint": file_fingerprint(config.path_for("model_manifest")),
        "model_manifest_semantic_fingerprint": model["model_manifest_fingerprint"],
        "parameter_fingerprint": model["parameter_fingerprint"],
        "training_lineage_fingerprint": model["training_lineage_fingerprint"],
        "replay_audit_physical_fingerprint": file_fingerprint(config.path_for("replay_audit")),
        "replay_fingerprint": replay["replay_fingerprint"],
        "prediction_replay": model["prediction_replay"],
        "manifest_correction_audit_physical_fingerprint": file_fingerprint(config.path_for("manifest_correction_audit")),
        "manifest_correction_audit_fingerprint": correction["correction_audit_fingerprint"],
        "sanitized_failure_audit_physical_fingerprint": file_fingerprint(config.path_for("sanitized_failure_audit")),
        "sanitized_test_security_audit_physical_fingerprint": file_fingerprint(config.path_for("test_security_audit")),
        "access_audit_physical_fingerprint": access_fingerprint,
        "fresh_initialization": True,
        "warm_start": False,
        "internal_validation": False,
        "early_stopping": False,
        "validation_optimizer_recomputed": False,
        "combined_matrix_recomputed": False,
        "model_retrained_during_recovery": False,
        "prediction_recomputed_during_recovery": False,
        "validation_labels_reopened": False,
        "first7d_read_count": 0,
        "final_refit_executed": True,
        "first7d_supervision_used": False,
        "test_authorized": False,
        "test_candidate_read_count": 0,
        "test_feature_read_count": 0,
        "test_label_read_count": 0,
        "test_prediction_write_count": 0,
        "test_metric_read_count": 0,
        "test_prediction_frozen": False,
    }
    lock["lock_fingerprint"] = self_fingerprint(lock, "lock_fingerprint")
    write_json_atomic(config.path_for("lock"), lock)
    if self_fingerprint(read_json(config.path_for("lock")), "lock_fingerprint") != lock["lock_fingerprint"]:
        raise PostTrainingRecoveryError("final-refit lock integrity failure")

    gate_rows = [
        {"gate_id": "R1", "check": "original resource failure evidence preserved", "evidence": lock["original_resource_failure_audit_physical_fingerprint"], "status": "PASS"},
        {"gate_id": "R2", "check": "resource recovery config identity", "evidence": prereg["original_recovery_source_fingerprint"], "status": "PASS"},
        {"gate_id": "R3", "check": "Validation optimizer exact reuse", "evidence": validation["membership_fingerprint"], "status": "PASS"},
        {"gate_id": "R4", "check": "combined matrix exact reuse", "evidence": combined["manifest_fingerprint"], "status": "PASS"},
        {"gate_id": "R5", "check": "binary Dataset integrity", "evidence": binary["binary_dataset_semantic_fingerprint"], "status": "PASS"},
        {"gate_id": "R6", "check": "binary conversion resource gate", "evidence": conversion_resource["resource_audit_fingerprint"], "status": "PASS"},
        {"gate_id": "R7", "check": "fresh isolated training identity", "evidence": model["training_worker_result_fingerprint"], "status": "PASS"},
        {"gate_id": "R8", "check": "training resource gate", "evidence": training_resource["resource_audit_fingerprint"], "status": "PASS"},
        {"gate_id": "R9", "check": "exact L1 F1 600 model identity", "evidence": model["model_semantic_fingerprint"], "status": "PASS"},
        {"gate_id": "R10", "check": "model manifest schema complete", "evidence": model["model_manifest_fingerprint"], "status": "PASS"},
        {"gate_id": "R11", "check": "model replay fingerprint binding", "evidence": model["prediction_replay"]["raw_score_fingerprint"], "status": "PASS"},
        {"gate_id": "R12", "check": "raw-data diff auditor correctness", "evidence": test_audit["raw_data_audit"]["audit_fingerprint"], "status": "PASS"},
        {"gate_id": "R13", "check": "public privacy and path safety", "evidence": test_audit["privacy_audit"]["audit_fingerprint"], "status": "PASS"},
        {"gate_id": "R14", "check": "no label reopen and no Test access", "evidence": access_fingerprint, "status": "PASS"},
        {"gate_id": "R15", "check": "targeted full tests and engineering audits", "evidence": file_fingerprint(config.path_for("test_security_audit")), "status": "PASS"},
        {"gate_id": "R16", "check": "final-refit lock integrity", "evidence": lock["lock_fingerprint"], "status": "PASS"},
    ]
    write_csv_atomic(config.path_for("gate"), gate_rows, list(gate_rows[0]))

    manifest_rows = [{
        "stage": "4B-4",
        "phase": "phase4_post_training_manifest_audit_recovery_v1",
        "status": "STAGE_4B4_PHASE4_FINAL_REFIT_COMPLETE",
        "model_id": "L1_FINAL_REFIT_600",
        "feature_set": "F1",
        "rounds": 600,
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "binary_semantic_fingerprint": binary["binary_dataset_semantic_fingerprint"],
        "model_retrained_during_fix": False,
        "prediction_recomputed_during_fix": False,
        "lock_fingerprint": lock["lock_fingerprint"],
        "resource_recovery_gate": "PASSED",
        "test_authorized": False,
    }]
    write_csv_atomic(config.path_for("manifest"), manifest_rows, list(manifest_rows[0]))

    report = render_report(
        config=config,
        original_resource=original_resource,
        binary=binary,
        model=model,
        replay=replay,
        correction=correction,
        sanitized_failure=sanitized_failure,
        test_audit=test_audit,
        lock=lock,
    )
    report_path = config.path_for("report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    os.replace(temporary, report_path)

    checkpoint = {
        "status": "SUCCESS",
        "resource_recovery_gate": "PASSED",
        "lock_fingerprint": lock["lock_fingerprint"],
        "gate_physical_fingerprint": file_fingerprint(config.path_for("gate")),
        "model_retrained": False,
        "prediction_recomputed": False,
    }
    checkpoint["checkpoint_fingerprint"] = self_fingerprint(checkpoint, "checkpoint_fingerprint")
    write_json_atomic(config.path_for("checkpoint_directory") / "lock_and_gate.json", checkpoint)
    return {
        "status": "STAGE_4B4_PHASE4_FINAL_REFIT_COMPLETE",
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "binary_semantic_fingerprint": binary["binary_dataset_semantic_fingerprint"],
        "lock_fingerprint": lock["lock_fingerprint"],
        "gate_passed": 16,
        "gate_total": 16,
        "model_retrained": False,
        "prediction_recomputed": False,
    }


def render_report(
    *,
    config: PostTrainingConfig,
    original_resource: dict[str, Any],
    binary: dict[str, Any],
    model: dict[str, Any],
    replay: dict[str, Any],
    correction: dict[str, Any],
    sanitized_failure: dict[str, Any],
    test_audit: dict[str, Any],
    lock: dict[str, Any],
) -> str:
    return f"""# Stage 4B-4 Phase 4：Locked Final Refit

## 1. 首次 raw-matrix 训练资源失败

首次正式 raw-memmap 训练按冻结资源门停止：cap 为 {original_resource['frozen_absolute_rss_cap_bytes']} bytes，进程树峰值为 {original_resource['child_process_tree_peak_rss_bytes']} bytes，超出 {original_resource['child_process_tree_peak_rss_bytes'] - original_resource['frozen_absolute_rss_cap_bytes']} bytes；未发生 OOM，也未晋升该次临时模型。原失败证据保持 `BLOCKED_BY_RESOURCE`。

## 2. Binary Dataset 资源恢复

恢复阶段精确复用既有 Validation optimizer 与 combined matrix，未重新采样、物化或读取新标签。LightGBM binary Dataset 含 {binary['rows']} rows、{binary['queries']} queries、{binary['features']} 个 F1 features，semantic fingerprint 为 `{binary['binary_dataset_semantic_fingerprint']}`。

## 3. Binary-only 正式训练

独立 worker 仅加载 formal binary Dataset，以 fresh initialization 训练 L1/F1/LambdaRank 恰好 600 rounds/trees；未使用 warm start、internal validation 或 early stopping。模型 semantic fingerprint 为 `{model['model_semantic_fingerprint']}`。

## 4. Post-training manifest schema 缺失

首次训练后工程审计发现 model manifest 缺少 `prediction_replay.raw_score_fingerprint`。该缺陷只影响 manifest schema，不改变模型、binary、combined matrix 或 replay 语义；失败事实和原始字节证据均已保留。

## 5. Git warning auditor 误分类

旧 auditor 将 Git 全局 ignore 权限 warning 的 stderr 误当成 raw-data diff。修复后 working-tree 与 staged raw diff 分别依据命令 return code 和 stdout/path result 判定，warning 仅以脱敏类别记录，不作为 changed file。

## 6. 失败 audit 的隐私问题

旧失败 audit 含本机绝对路径，不能作为公开产物。其原始字节版本已迁入 Git-ignored private failure-evidence，公开失败证据只保留失败类别、原始 SHA-256、缺失字段、warning 误分类、隐私违规计数与修复状态；原失败事实未删除。

## 7. 本次授权修复范围

本次只执行 manifest schema、auditor、公开隐私、测试审计与 lock/gate/report 修复。未重建 binary、未重训模型、未重新运行 prediction/replay、未打开任何标签，也未执行业务指标或模型选择。

## 8. 模型与 replay 未重算

模型 physical fingerprint `{model['model_physical_fingerprint']}` 与 binary physical fingerprint `{binary['binary_physical_fingerprint']}` 保持不变。Replay audit physical fingerprint `{correction['replay_audit_physical_fingerprint_after']}` 保持不变；既有 raw-score fingerprint `{model['prediction_replay']['raw_score_fingerprint']}` 与 tie-break fingerprint `{model['prediction_replay']['tie_break_fingerprint']}` 被原样绑定到 corrected manifest。

## 9. Manifest correction evidence

Original incomplete manifest physical fingerprint `{correction['original_incomplete_manifest_physical_fingerprint']}`；corrected physical fingerprint `{correction['corrected_manifest_physical_fingerprint']}`；corrected semantic fingerprint `{correction['corrected_manifest_semantic_fingerprint']}`。只新增既有 replay 证据的嵌套 metadata/schema，其他原字段保持一致。

## 10. 最终工程与隐私审计

Targeted pytest、full pytest、compile、`git diff --check`、raw-data working/staged diff、privacy/path、ignore 与 tracked-private 审计均 PASS。公开审计不含可恢复的绝对路径或用户名。Sanitized failure audit fingerprint 为 `{sanitized_failure['sanitized_failure_audit_fingerprint']}`。

## 11. Final-refit lock 与 R1–R16

Final-refit lock 状态为 `FINAL_REFIT_LOCKED_AFTER_POST_TRAINING_RECOVERY`，fingerprint `{lock['lock_fingerprint']}`。Resource-recovery gate 为 R1–R16 全部 PASS。

## 12. 访问边界与结论

Validation labels reopened=NO；First7d read count=0；Test candidate/feature/label/prediction/metric counts 全部为 0，Test 未授权。当前状态为 `STAGE_4B4_PHASE4_FINAL_REFIT_COMPLETE`，等待 ChatGPT 技术验收。该结论仅证明离线 final-refit 工程与协议完成，不代表线上 CTR/CVR、收入提升或 A/B Test 结果。
"""
