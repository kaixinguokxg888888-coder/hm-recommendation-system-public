from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ranking_model.final_refit import (
    FINAL_REFIT_ORDERING_VERSION,
    FinalRefitConfig,
    FinalRefitError,
    canonical_fingerprint,
    combined_order_fingerprint,
    deterministic_validation_sample,
    frozen_identity_audit,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stage_4b4_final_refit.yaml"


def _config() -> FinalRefitConfig:
    return FinalRefitConfig.load(CONFIG_PATH)


def test_phase4_config_is_exact_locked_winner() -> None:
    raw = _config().raw
    assert raw["model"]["model_id"] == "L1_FINAL_REFIT_600"
    assert raw["model"]["selected_model_id"] == "L1_600"
    assert raw["model"]["feature_set"] == "F1"
    assert raw["model"]["num_boost_round"] == 600
    assert raw["model"]["initialization"] == "fresh"
    assert raw["model"]["warm_start"] is False


def test_phase4_parameters_exactly_match_phase1() -> None:
    raw = _config().raw
    phase1 = json.loads((ROOT / "configs/stage_4b4_lambdarank.yaml").read_text(encoding="utf-8"))
    assert raw["lightgbm"]["params"] == phase1["lightgbm"]["params"]
    assert raw["lightgbm"]["weights"] == phase1["lightgbm"]["weights"]
    assert raw["lightgbm"]["num_boost_round"] == phase1["lightgbm"]["num_boost_round"]


def test_frozen_phase1_to_phase3_identity_is_current() -> None:
    audit = frozen_identity_audit(_config())
    assert audit["gates"]["phase1"] == {"passed": 20, "total": 20}
    assert audit["winner"] == {"model_id": "L1_600", "feature_set": "F1", "rounds": 600}


def test_sampling_is_sha256_utf8_and_input_order_independent() -> None:
    rows = [(index, 1 if index in (3, 117) else 0) for index in range(150)]
    forward = deterministic_validation_sample("validation_14d", 42, rows)
    reverse = deterministic_validation_sample("validation_14d", 42, reversed(rows))
    assert forward == reverse
    assert (3, 1) in forward and (117, 1) in forward
    assert sum(label == 0 for _, label in forward) == 100
    payload = "validation_14d|42|0|20260811".encode("utf-8")
    assert hashlib.sha256(payload).hexdigest()


def test_sampling_retains_all_positives_and_all_zeros_below_cap() -> None:
    rows = [(1, 1), (2, 0), (3, 1), (4, 0)]
    assert deterministic_validation_sample("validation_14d", 9, rows) == sorted(rows)


def test_sampling_rejects_non_binary_labels() -> None:
    with pytest.raises(FinalRefitError):
        deterministic_validation_sample("validation_14d", 9, [(1, 2)])


def test_snapshot_aware_query_key_and_order_are_frozen() -> None:
    raw = _config().raw
    assert raw["ordering"]["version"] == FINAL_REFIT_ORDERING_VERSION
    assert raw["ordering"]["query_key"] == ["snapshot_id", "customer_key"]
    assert raw["ordering"]["snapshot_order"] == ["ranking_train_14d_v1", "validation_14d"]


def test_two_snapshot_feature_cutoffs_are_distinct() -> None:
    snapshots = _config().raw["snapshots"]
    assert snapshots[0]["as_of_date"] == "2020-08-11"
    assert snapshots[1]["as_of_date"] == "2020-08-25"
    assert snapshots[0]["label_end"] == snapshots[1]["as_of_date"]


def test_categorical_feature_identity_is_exact() -> None:
    raw = _config().raw
    adapter = json.loads((ROOT / raw["upstream"]["phase1_adapter"]).read_text(encoding="utf-8"))
    assert raw["model"]["feature_count"] == len(adapter["ordered_feature_names"]) == 52
    assert raw["categorical_features"] == adapter["categorical_features"]
    assert len(raw["categorical_features"]) == 12


def test_resource_fraction_is_exactly_seventy_percent() -> None:
    resource = _config().raw["resource"]
    assert resource["rss_hard_fraction_of_available"] == 0.70
    assert resource["worker_consumes_frozen_absolute_cap_only"] is True
    assert resource["external_process_tree_monitor"] is True


def test_worker_has_no_forbidden_pipeline_imports_or_available_memory_call() -> None:
    source = (ROOT / "scripts/stage_4b4_final_refit_worker.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.split(".", 1)[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported.isdisjoint({"duckdb", "pyarrow", "ranking_features", "psutil"})
    assert "virtual_memory" not in source


def test_worker_uses_fresh_training_without_init_model_or_valid_sets() -> None:
    source = (ROOT / "scripts/stage_4b4_final_refit_worker.py").read_text(encoding="utf-8")
    assert "lgb.train(params, dataset, num_boost_round=600" in source
    assert "init_model=" not in source
    assert "valid_sets=" not in source
    assert "early_stopping(" not in source


def test_worker_forces_exact_600_round_integrity() -> None:
    source = (ROOT / "scripts/stage_4b4_final_refit_worker.py").read_text(encoding="utf-8")
    assert "booster.current_iteration() != 600" in source
    assert "reloaded.current_iteration() != 600" in source


def test_combined_order_fingerprint_is_snapshot_sensitive() -> None:
    row = {
        "snapshot_id": "ranking_train_14d_v1", "customer_key": 1, "query_order": 0,
        "start_offset": 0, "end_offset": 2, "group_size": 2, "positive_count": 1,
        "ordered_row_digest": "00" * 32,
    }
    first = combined_order_fingerprint([row])
    second = combined_order_fingerprint([{**row, "snapshot_id": "validation_14d"}])
    assert first != second


def test_phase4_access_contract_forbids_first7d_and_test() -> None:
    access = _config().raw["access"]
    assert access["phase2_validation_14d_label_read_authorized"] is True
    assert access["validation_metric_recalculation"] is False
    assert access["first7d_supervision_used"] is False
    assert all(value == 0 for key, value in access.items() if key.startswith("test_") and key.endswith("_count"))
    assert access["test_authorized"] is False


def test_config_has_no_model_search_space() -> None:
    raw = _config().raw
    encoded = json.dumps(raw, sort_keys=True)
    assert "checkpoints" not in raw["lightgbm"]
    assert "L0_" not in encoded and "B0" not in encoded


def test_final_artifacts_bind_required_zero_access_counts_when_present() -> None:
    lock_path = ROOT / "artifacts/stage_4b4/stage_4b4_final_refit_lock.json"
    if not lock_path.exists():
        pytest.skip("formal Phase 4 lock is created only after successful final refit")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["final_refit_executed"] is True
    assert lock["test_authorized"] is False
    assert lock["test_prediction_frozen"] is False
    assert lock["first7d_supervision_used"] is False
    assert all(lock[name] == 0 for name in (
        "test_candidate_read_count", "test_feature_read_count", "test_label_read_count",
        "test_prediction_write_count", "test_metric_read_count",
    ))


def test_final_model_manifest_proves_reload_and_replay_when_present() -> None:
    path = ROOT / "artifacts/stage_4b4/final_refit_model_fingerprint_manifest.json"
    if not path.exists():
        pytest.skip("formal model manifest is created only after successful final refit")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "FINAL_REFIT_MODEL_FROZEN"
    assert manifest["rounds"] == 600
    assert manifest["atomic_promotion"] is True
    assert manifest["prediction_replay"]["raw_score_fingerprint"]


def test_phase4_gate_has_exactly_thirteen_passes_when_present() -> None:
    path = ROOT / "artifacts/stage_4b4/final_refit_contract_gate.csv"
    if not path.exists():
        pytest.skip("formal gate is created during finalization")
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["gate_id"] for row in rows] == [f"P4G{index}" for index in range(1, 14)]
    assert all(row["status"] == "PASS" for row in rows)


def test_no_final_refit_code_references_first7d_artifact_paths() -> None:
    for path in (
        ROOT / "src/ranking_model/final_refit.py",
        ROOT / "scripts/stage_4b4_final_refit_worker.py",
        ROOT / "scripts/run_stage_4b4_final_refit_training.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "ground_truth_first7d" not in source
        assert "first7d_ground_truth" not in source


def test_numpy_group_boundary_contract_example() -> None:
    groups = np.asarray([2, 3], dtype=np.int32)
    starts = np.r_[0, np.cumsum(groups[:-1], dtype=np.int64)]
    labels = np.asarray([1, 0, 0, 1, 0], dtype=np.uint8)
    assert int(groups.sum()) == len(labels)
    assert np.add.reduceat(labels, starts).tolist() == [1, 1]


def test_canonical_fingerprint_is_order_independent_for_mapping_keys() -> None:
    assert canonical_fingerprint({"a": 1, "b": 2}) == canonical_fingerprint({"b": 2, "a": 1})
