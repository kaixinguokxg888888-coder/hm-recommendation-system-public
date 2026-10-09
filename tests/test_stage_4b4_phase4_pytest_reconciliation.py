from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ranking_model.pytest_skip_reconciliation import (
    EXPECTED_SKIP_NODEID,
    ReconciliationConfig,
    ReconciliationError,
    canonical_pytest_command,
    classify_skip,
    self_fingerprint,
    validate_structured_result,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stage_4b4_pytest_skip_reconciliation.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> ReconciliationConfig:
    return ReconciliationConfig.load(CONFIG_PATH)


def _valid_result() -> dict[str, object]:
    return {
        "status": "PASS",
        "return_code": 0,
        "counts": {
            "collected": 2,
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "skipped": 1,
            "xfailed": 0,
            "xpassed": 0,
            "outcome_total": 2,
        },
        "skipped": [{"nodeid": EXPECTED_SKIP_NODEID, "reason": "formal gate is created during finalization"}],
    }


def test_reconciliation_config_freezes_parser_error_and_non_mutating_scope() -> None:
    config = _config()
    assert config.raw["root_cause"] == "AUDIT_PARSER_ERROR"
    assert config.raw["historical_reported_counts"] == {
        "targeted_skipped": 1,
        "full_skipped": 2,
        "source": "prior_narrative_post_lock_rerun_summary",
    }
    assert config.raw["historical_formal_audit_displayed_progress_markers"]["counts_are_formal_statistics"] is False
    assert config.raw["commands"]["targeted_pytest_args"][-2:] == ["-ra", "-q"]
    assert config.raw["commands"]["full_pytest_args"] == ["-ra", "-q"]


def test_structured_result_accepts_exact_counts_and_rejects_summary_drift() -> None:
    payload = _valid_result()
    validate_structured_result(payload)
    payload["counts"]["outcome_total"] = 1  # type: ignore[index]
    with pytest.raises(ReconciliationError, match="outcome total mismatch"):
        validate_structured_result(payload)


def test_structured_result_rejects_failures_and_missing_skip_reasons() -> None:
    failed = _valid_result()
    failed["counts"]["passed"] = 0  # type: ignore[index]
    failed["counts"]["failed"] = 1  # type: ignore[index]
    with pytest.raises(ReconciliationError, match="ENGINEERING_GATE_FAILED"):
        validate_structured_result(failed)
    missing_reason = _valid_result()
    missing_reason["skipped"][0]["reason"] = ""  # type: ignore[index]
    with pytest.raises(ReconciliationError, match="empty skip reason"):
        validate_structured_result(missing_reason)


def test_only_frozen_legacy_gate_skip_is_noncritical() -> None:
    expected = classify_skip(
        EXPECTED_SKIP_NODEID,
        "formal gate is created during finalization",
        "targeted",
        "tests/test_stage_4b4_phase4.py:190",
    )
    assert expected["classification"] == "EXPECTED_OBSOLETE_ARTIFACT_SKIP"
    assert expected["is_expected"] is True
    assert expected["is_phase4_required_gate"] is False
    unknown = classify_skip("tests/test_other.py::test_required", "missing", "full", "tests/test_other.py:1")
    assert unknown["classification"] == "UNKNOWN_SKIP"
    assert unknown["is_expected"] is False
    assert unknown["is_phase4_required_gate"] is True


def test_public_command_is_canonical_and_path_sanitized() -> None:
    command = canonical_pytest_command(["tests/test_stage_4b4_phase4.py", "-ra", "-q"], "targeted.json")
    assert command[:2] == ["python", "scripts/stage_4b4_pytest_structured_worker.py"]
    assert command[2:4] == ["--output", "<private-results>/targeted.json"]
    assert str(ROOT) not in " ".join(command)


def test_original_or_reconciled_artifact_chain_is_internally_consistent() -> None:
    config = _config()
    expected = config.raw["expected"]
    reconciliation_path = config.path_for("reconciliation")
    audit_path = config.path_for("test_audit")
    if not reconciliation_path.exists():
        assert _sha256(audit_path) == expected["original_test_audit_physical_fingerprint"]
        return
    reconciliation = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    failed_attempt2 = config.path_for("failed_attempt2")
    if (
        failed_attempt2.exists()
        and _sha256(audit_path) == expected["failed_attempt2_test_audit_physical_fingerprint"]
    ):
        failure = json.loads(failed_attempt2.read_text(encoding="utf-8"))
        assert failure["status"] == "FAILED_ATTEMPT2_SUPERSEDED_BEFORE_FINAL_ACCEPTANCE"
        assert failure["failed_test_audit_physical_fingerprint"] == _sha256(audit_path)
        return
    if failed_attempt2.exists() and _sha256(audit_path) == expected["original_test_audit_physical_fingerprint"]:
        assert reconciliation["corrected_test_audit_physical_fingerprint"] == expected[
            "failed_attempt2_test_audit_physical_fingerprint"
        ]
        return
    assert reconciliation["status"] == "PYTEST_SKIP_AUDIT_RECONCILED"
    assert reconciliation["corrected_test_audit_physical_fingerprint"] == _sha256(audit_path)
    assert audit["parser_version"] == "pytest_hook_report_v1"
    assert audit["critical_phase4_test_skipped"] is False
    assert audit["raw_data_audit"]["status"] == "PASS"
    assert audit["privacy_audit"]["status"] == "PASS"
    assert self_fingerprint(audit, "test_security_audit_fingerprint") == audit["test_security_audit_fingerprint"]


def test_current_or_original_lock_keeps_model_binary_replay_and_access_identity() -> None:
    config = _config()
    expected = config.raw["expected"]
    lock = json.loads(config.path_for("lock").read_text(encoding="utf-8"))
    assert self_fingerprint(lock, "lock_fingerprint") == lock["lock_fingerprint"]
    assert lock["validation_labels_reopened"] is False
    assert lock["first7d_read_count"] == 0
    for field in (
        "test_candidate_read_count",
        "test_feature_read_count",
        "test_label_read_count",
        "test_prediction_write_count",
        "test_metric_read_count",
    ):
        assert lock[field] == 0
    assert _sha256(config.path_for("model_artifact")) == expected["model_physical_fingerprint"]
    assert _sha256(config.path_for("binary_artifact")) == expected["binary_physical_fingerprint"]
    assert _sha256(config.path_for("replay_audit")) == expected["replay_audit_physical_fingerprint"]


def test_reconciliation_sources_do_not_call_training_prediction_or_label_readers() -> None:
    sources = [
        ROOT / "src/ranking_model/pytest_skip_reconciliation.py",
        ROOT / "scripts/run_stage_4b4_pytest_skip_reconciliation.py",
        ROOT / "scripts/stage_4b4_pytest_structured_worker.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for forbidden in ("train_from_binary(", ".predict(", "labels.uint8", "test_labels"):
        assert forbidden not in combined


def test_structured_worker_restores_repository_root_for_script_imports() -> None:
    worker = (ROOT / "scripts/stage_4b4_pytest_structured_worker.py").read_text(encoding="utf-8")
    assert "sys.path.insert(0, str(root))" in worker
