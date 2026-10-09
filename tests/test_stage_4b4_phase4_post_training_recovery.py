from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from ranking_model.final_refit_post_training_recovery import (
    PostTrainingConfig,
    evaluate_raw_data_git_results,
    file_fingerprint,
    self_fingerprint,
    warning_categories,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stage_4b4_final_refit_post_training_recovery.yaml"


def _config() -> PostTrainingConfig:
    return PostTrainingConfig.load(CONFIG_PATH)


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["git"], returncode, stdout, stderr)


def test_post_training_scope_forbids_retraining_rebuild_prediction_and_labels() -> None:
    scope = _config().raw["scope"]
    assert scope["model_retraining"] is False
    assert scope["binary_rebuild"] is False
    assert scope["prediction_replay_recomputed"] is False
    assert scope["label_access"] is False
    assert scope["validation_optimizer_recomputed"] is False
    assert scope["combined_matrix_recomputed"] is False


def test_post_training_runner_exposes_no_train_convert_or_replay_step() -> None:
    source = (ROOT / "scripts/run_stage_4b4_final_refit_post_training_recovery.py").read_text(encoding="utf-8")
    assert 'choices=("preregister", "correct", "finalize")' in source
    assert "train_from_binary" not in source
    assert "convert_binary_dataset" not in source
    assert "final_model_replay" not in source


def test_raw_git_warning_is_not_a_diff() -> None:
    warning = "warning: unable to access a global ignore file: Permission denied\n"
    audit = evaluate_raw_data_git_results(_completed(0, stderr=warning), _completed(0, stderr=warning))
    assert audit["status"] == "PASS"
    assert audit["changed_file_count"] == 0
    assert audit["warning_categories"] == ["GLOBAL_IGNORE_ACCESS_WARNING"]
    assert audit["stderr_is_not_diff_evidence"] is True


def test_raw_git_nonzero_return_is_unknown_not_pass() -> None:
    audit = evaluate_raw_data_git_results(_completed(1, stderr="fatal"), _completed(0))
    assert audit["status"] == "UNKNOWN"


def test_actual_working_or_staged_raw_diff_fails() -> None:
    assert evaluate_raw_data_git_results(_completed(0, "data/raw/example\n"), _completed(0))["status"] == "FAIL"
    assert evaluate_raw_data_git_results(_completed(0), _completed(0, "data/raw/example\n"))["status"] == "FAIL"


def test_warning_categories_never_return_original_warning_text() -> None:
    values = warning_categories("warning: unable to access a global ignore file: Permission denied")
    assert values == ["GLOBAL_IGNORE_ACCESS_WARNING"]
    assert all("/" not in value and "\\" not in value for value in values)


def test_corrected_manifest_has_nested_formal_replay_binding_when_present() -> None:
    config = _config()
    manifest = json.loads(config.path_for("model_manifest").read_text(encoding="utf-8"))
    if "prediction_replay" not in manifest:
        pytest.skip("manifest correction has not run yet")
    replay = json.loads(config.path_for("replay_audit").read_text(encoding="utf-8"))
    nested = manifest["prediction_replay"]
    assert nested["raw_score_fingerprint"] == replay["first_reload"]["raw_score_fingerprint"]
    assert nested["raw_score_fingerprint"] == replay["second_reload"]["raw_score_fingerprint"]
    assert nested["tie_break_fingerprint"] == replay["first_reload"]["rank_tiebreak_fingerprint"]
    assert nested["replay_audit_physical_fingerprint"] == file_fingerprint(config.path_for("replay_audit"))
    assert nested["label_free"] is True and nested["model_reloaded"] is True
    assert nested["raw_score_mismatch_count"] == nested["tie_break_mismatch_count"] == 0
    assert nested["nonfinite_count"] == 0


def test_corrected_manifest_self_fingerprint_when_present() -> None:
    path = _config().path_for("model_manifest")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "prediction_replay" not in manifest:
        pytest.skip("manifest correction has not run yet")
    assert self_fingerprint(manifest, "model_manifest_fingerprint") == manifest["model_manifest_fingerprint"]


def test_manifest_correction_proves_model_binary_replay_unchanged_when_present() -> None:
    config = _config()
    path = config.path_for("manifest_correction_audit")
    if not path.exists():
        pytest.skip("correction audit is generated with the authorized correction")
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert audit["status"] == "PASS"
    assert audit["model_artifact_unchanged"] is True
    assert audit["binary_artifact_unchanged"] is True
    assert audit["replay_artifact_unchanged"] is True
    assert audit["only_manifest_metadata_schema_changed"] is True
    assert audit["model_retrained_during_fix"] is False
    assert audit["prediction_recomputed_during_fix"] is False


def test_model_and_binary_physical_identity_are_frozen() -> None:
    config = _config()
    expected = config.raw["expected"]
    assert file_fingerprint(config.path_for("model_artifact")) == expected["model_physical_fingerprint"]
    assert file_fingerprint(config.path_for("binary_artifact")) == expected["binary_physical_fingerprint"]
    assert file_fingerprint(config.path_for("replay_audit")) == expected["replay_audit_physical_fingerprint"]


def test_private_failure_evidence_is_ignored_when_present() -> None:
    config = _config()
    paths = [config.path_for("original_manifest_private"), config.path_for("failed_test_audit_private")]
    if not all(path.exists() for path in paths):
        pytest.skip("private evidence is created during correction")
    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "-q", path.relative_to(ROOT).as_posix()],
            cwd=ROOT,
            capture_output=True,
        )
        assert result.returncode == 0


def test_public_sanitized_failure_audit_contains_no_absolute_path_when_present() -> None:
    path = _config().path_for("sanitized_failure_audit")
    if not path.exists():
        pytest.skip("sanitized failure audit is created during correction")
    text = path.read_text(encoding="utf-8")
    assert ":\\" not in text and ":/" not in text
    payload = json.loads(text)
    assert payload["original_failure_fact_deleted"] is False
    assert payload["privacy_violation_count"] > 0


def test_formal_test_audit_is_sanitized_when_passed() -> None:
    path = _config().path_for("test_security_audit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        pytest.skip("formal fixed audit has not run yet")
    text = path.read_text(encoding="utf-8")
    assert ":\\" not in text and ":/" not in text
    assert payload["raw_data_audit"]["status"] == "PASS"
    assert payload["privacy_audit"]["status"] == "PASS"


def test_final_lock_binds_post_training_recovery_when_present() -> None:
    config = _config()
    path = config.path_for("lock")
    if not path.exists():
        pytest.skip("final lock is generated after all audits pass")
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert self_fingerprint(lock, "lock_fingerprint") == lock["lock_fingerprint"]
    assert lock["model_retrained_during_recovery"] is False
    assert lock["prediction_recomputed_during_recovery"] is False
    assert lock["prediction_replay"]["raw_score_fingerprint"] == config.raw["expected"]["raw_score_fingerprint"]
    assert lock["validation_labels_reopened"] is False
    assert lock["first7d_read_count"] == 0
    assert lock["test_authorized"] is False


def test_r1_to_r16_gate_is_exact_when_present() -> None:
    path = _config().path_for("gate")
    if not path.exists():
        pytest.skip("R1-R16 gate is generated after all audits pass")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["gate_id"] for row in rows] == [f"R{index}" for index in range(1, 17)]
    assert all(row["status"] == "PASS" for row in rows)


def test_no_label_file_path_or_prediction_call_in_post_training_recovery() -> None:
    sources = [
        ROOT / "src/ranking_model/final_refit_post_training_recovery.py",
        ROOT / "scripts/run_stage_4b4_final_refit_post_training_recovery.py",
        ROOT / "scripts/run_stage_4b4_final_refit_post_training_audit.py",
    ]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert "labels.uint8" not in text
        assert ".predict(" not in text
        assert "train_from_binary" not in text
        assert "convert_binary_dataset" not in text
