from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ranking_model.first7d_resource_recovery import (
    First7dResourceRecoveryRunner,
    RecoveryArtifactDrift,
    RecoveryScopeExceeded,
    canonical_fingerprint,
    self_fingerprint,
    validate_access_state,
    validate_recovery_permissions,
    validate_winner_state,
)
from ranking_model.resource_contract import (
    ResourceContractError,
    absolute_cap,
    consume_frozen_rss_contract,
    resource_gate_passes,
    validate_registered_fraction,
)


def _preregistration() -> dict[str, object]:
    return {
        "preregistration_fingerprint": "historical",
        "resource_contract": {
            "available_memory_at_preregistration_bytes": 1_857_146_880,
            "rss_hard_limit_bytes": 1_207_145_472,
            "duckdb_memory_limit_mb": 531,
        },
    }


def _recovery() -> dict[str, object]:
    return {
        "available_memory_at_original_preregistration_bytes": 1_857_146_880,
        "invalid_preregistered_cap_bytes": 1_207_145_472,
        "formal_required_fraction": 0.70,
        "corrected_formal_cap_bytes": 1_300_002_816,
    }


def test_runtime_available_change_does_not_change_recovery_cap() -> None:
    first = consume_frozen_rss_contract(
        _preregistration(), runtime_available_memory_bytes=900_000_000, recovery=_recovery()
    )
    second = consume_frozen_rss_contract(
        _preregistration(), runtime_available_memory_bytes=9_000_000_000, recovery=_recovery()
    )
    assert first.frozen_absolute_rss_cap_bytes == second.frozen_absolute_rss_cap_bytes == 1_300_002_816
    assert first.runtime_available_memory_bytes != second.runtime_available_memory_bytes


def test_registered_65_percent_is_detected_against_required_70_percent() -> None:
    with pytest.raises(ResourceContractError):
        validate_registered_fraction(0.65)
    with pytest.raises(ResourceContractError):
        consume_frozen_rss_contract(_preregistration(), runtime_available_memory_bytes=1)
    validate_registered_fraction(0.70)
    assert absolute_cap(1_857_146_880, 0.70) == 1_300_002_816


def test_peak_equal_to_frozen_cap_passes() -> None:
    assert resource_gate_passes(1_300_002_816, 1_300_002_816)


def test_peak_above_frozen_cap_fails() -> None:
    assert not resource_gate_passes(1_300_002_817, 1_300_002_816)


def test_invalid_runtime_recomputed_cap_cannot_replace_frozen_cap() -> None:
    contract = consume_frozen_rss_contract(
        _preregistration(), runtime_available_memory_bytes=1_879_483_246, recovery=_recovery()
    )
    assert contract.frozen_absolute_rss_cap_bytes == 1_300_002_816
    assert contract.frozen_absolute_rss_cap_bytes != 1_315_638_272
    assert contract.source == "authorized_recovery_absolute_cap"


def test_recovery_mode_has_no_label_or_metric_computation_calls() -> None:
    root = Path(__file__).resolve().parents[1]
    source_path = root / "src/ranking_model/first7d_resource_recovery.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    called = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert imported.isdisjoint({"duckdb", "pyarrow"})
    assert called.isdisjoint({
        "_materialize_first7d_once", "_build_query_base", "_build_metrics",
        "_aggregate", "_bootstrap", "evaluate",
    })


def test_recovery_access_and_permissions_forbid_reopen_and_recompute() -> None:
    permissions = {
        "first7d_label_reopen_allowed": False,
        "ground_truth_rematerialization_allowed": False,
        "population_rematerialization_allowed": False,
        "metric_recomputation_allowed": False,
        "bootstrap_recomputation_allowed": False,
        "prediction_recomputation_allowed": False,
        "winner_change_allowed": False,
        "methodology_changed": False,
    }
    validate_recovery_permissions(permissions)
    with pytest.raises(RecoveryScopeExceeded):
        validate_recovery_permissions({**permissions, "metric_recomputation_allowed": True})
    validate_access_state({
        "first7d_read_count": 1, "validation_14d_source_content_open_count": 1,
        "test_candidate_read_count": 0, "test_feature_read_count": 0,
        "test_label_read_count": 0, "test_metric_read_count": 0,
    })
    with pytest.raises(RecoveryArtifactDrift):
        validate_access_state({
            "first7d_read_count": 2, "validation_14d_source_content_open_count": 1,
            "test_candidate_read_count": 0, "test_feature_read_count": 0,
            "test_label_read_count": 0, "test_metric_read_count": 0,
        })


def test_recovery_cannot_change_winner_or_direction() -> None:
    valid = {
        "winner": "L1_600", "winner_feature_set": "F1", "winner_iteration": 600,
        "winner_changed": False, "direction_status": "ROBUSTNESS_DIRECTION_PASS",
        "confidence_status": "POSITIVE_AND_SIGNIFICANT",
        "segment_regression_review": "NO_FIRST7D_CRITICAL_SEGMENT_REGRESSION",
    }
    validate_winner_state(valid)
    with pytest.raises(RecoveryArtifactDrift):
        validate_winner_state({**valid, "winner": "B0"})


def test_original_config_is_preserved_and_recovery_config_is_versioned() -> None:
    root = Path(__file__).resolve().parents[1]
    original = json.loads((root / "configs/stage_4b4_first7d_diagnostic.yaml").read_text(encoding="utf-8"))
    recovery = json.loads((root / "configs/stage_4b4_first7d_resource_recovery.yaml").read_text(encoding="utf-8"))
    assert original["execution"]["rss_hard_fraction_of_available_at_preregistration"] == 0.65
    assert recovery["recovery_type"] == "post_first7d_resource_contract_only"
    assert recovery["resource_correction"]["formal_required_fraction"] == 0.70
    assert canonical_fingerprint(original) == recovery["base_phase3_config_fingerprint"]


def test_recovered_lock_self_fingerprint_contract() -> None:
    lock = {"status": "FIRST7D_RESOURCE_CONTRACT_RECOVERED", "winner": "L1_600"}
    lock["lock_fingerprint"] = canonical_fingerprint(lock)
    assert self_fingerprint(lock, "lock_fingerprint") == lock["lock_fingerprint"]


def test_supersession_terms_are_explicit_in_recovery_source() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ranking_model/first7d_resource_recovery.py").read_text(encoding="utf-8")
    assert "SUPERSEDED" in source
    assert "NOT_FORMAL_DUE_TO_RESOURCE_CONTRACT_IDENTITY_FAILURE" in source
    # Construction only: this must not call preflight or touch formal artifacts.
    runner = First7dResourceRecoveryRunner(
        root, root / "configs/stage_4b4_first7d_resource_recovery.yaml"
    )
    assert runner.raw["authorization"] == "chatgpt_post_first7d_resource_recovery_2026-08-18"
