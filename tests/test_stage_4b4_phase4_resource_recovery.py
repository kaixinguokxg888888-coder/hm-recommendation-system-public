from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from ranking_model.final_refit_resource_recovery import (
    RecoveryConfig,
    canonical_fingerprint,
    file_fingerprint,
    recovery_source_fingerprint,
    self_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stage_4b4_final_refit_resource_recovery.yaml"


def _config() -> RecoveryConfig:
    return RecoveryConfig.load(CONFIG_PATH)


def test_recovery_scope_is_binary_only_and_non_rematerializing() -> None:
    raw = _config().raw
    assert raw["recovery_type"] == "lightgbm_binary_dataset_isolated_training_v1"
    scope = raw["recovery_scope"]
    assert scope["validation_optimizer_recomputed"] is False
    assert scope["combined_matrix_recomputed"] is False
    assert scope["label_join_recomputed"] is False
    assert scope["validation_labels_reopened"] is False
    assert scope["first7d_labels_read"] is False
    assert scope["test_access_allowed"] is False


def test_recovery_model_and_params_exactly_match_phase1() -> None:
    raw = _config().raw
    phase1 = json.loads((ROOT / raw["base"]["phase1_config"]).read_text(encoding="utf-8"))
    base = json.loads((ROOT / raw["base"]["final_refit_config"]).read_text(encoding="utf-8"))
    assert raw["lightgbm"]["params"] == phase1["lightgbm"]["params"] == base["lightgbm"]["params"]
    assert raw["lightgbm"]["weights"] == phase1["lightgbm"]["weights"]
    assert raw["model"]["model_id"] == "L1_FINAL_REFIT_600"
    assert raw["model"]["feature_set"] == "F1"
    assert raw["model"]["num_boost_round"] == 600


def test_original_resource_failure_is_preserved() -> None:
    raw = _config().raw
    path = ROOT / raw["original_failure"]["resource_audit"]
    audit = json.loads(path.read_text(encoding="utf-8"))
    assert file_fingerprint(path) == raw["original_failure"]["resource_audit_physical_fingerprint"]
    assert audit["status"] == "BLOCKED_BY_RESOURCE"
    assert audit["frozen_absolute_rss_cap_bytes"] == 2_800_775_577
    assert audit["child_process_tree_peak_rss_bytes"] == 2_818_605_056
    assert audit["child_process_tree_peak_rss_bytes"] - audit["frozen_absolute_rss_cap_bytes"] == 17_829_479


def test_validation_optimizer_and_combined_manifest_identity_without_label_source_read() -> None:
    raw = _config().raw
    validation = json.loads((ROOT / raw["base"]["validation_optimizer_manifest"]).read_text(encoding="utf-8"))
    combined = json.loads((ROOT / raw["base"]["combined_matrix_manifest"]).read_text(encoding="utf-8"))
    assert validation["query_count"] == 62_296 and validation["row_count"] == 6_332_014
    assert validation["membership_fingerprint"] == validation["deterministic_replay_fingerprint"]
    assert combined["query_count"] == 113_793 and combined["row_count"] == 11_568_070
    assert combined["feature_count"] == 52 and combined["matrix_dtype"] == "float32"
    assert combined["group_sum"] == combined["row_count"]
    assert not any(combined["audit"].values())


def test_recovery_source_inventory_is_stable() -> None:
    assert recovery_source_fingerprint(_config())


def test_conversion_and_training_caps_are_exactly_seventy_percent() -> None:
    resource = _config().raw["resource"]
    assert resource["conversion_rss_fraction_of_available"] == 0.70
    assert resource["training_rss_fraction_of_available"] == 0.70
    assert resource["worker_recalculates_available_memory"] is False
    assert resource["cap_shopping_allowed"] is False
    assert int(4_001_107_968 * 0.70) == 2_800_775_577


@pytest.mark.parametrize("name", [
    "stage_4b4_binary_conversion_worker.py",
    "stage_4b4_binary_verification_worker.py",
    "stage_4b4_binary_training_worker.py",
    "stage_4b4_final_model_replay_worker.py",
])
def test_isolated_workers_do_not_import_forbidden_pipeline_modules(name: str) -> None:
    source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
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
    assert imported.isdisjoint({"duckdb", "pyarrow", "ranking_features"})
    assert "virtual_memory" not in source


def test_binary_training_worker_has_no_raw_matrix_input() -> None:
    source = (ROOT / "scripts/stage_4b4_binary_training_worker.py").read_text(encoding="utf-8")
    assert "--combined-directory" not in source
    assert "--feature-matrix" not in source
    assert "np.memmap" not in source
    assert "loaded_formal_binary_only" in source


def test_binary_training_is_fresh_exact_600_without_selection() -> None:
    source = (ROOT / "scripts/stage_4b4_binary_training_worker.py").read_text(encoding="utf-8")
    assert "lgb.train(params, dataset, num_boost_round=600" in source
    assert "init_model=" not in source
    assert "valid_sets=" not in source
    assert "early_stopping(" not in source
    assert "booster.num_trees() != 600" in source
    assert "len(dump[\"tree_info\"]) != 600" in source


def test_binary_round_trip_preserves_labels_groups_features_and_prediction(tmp_path: Path) -> None:
    features = np.asarray([
        [0.0, 0.0, np.nan], [1.0, 1.0, 0.5], [2.0, 0.0, 1.0],
        [3.0, 1.0, 1.5], [4.0, 2.0, 2.0], [5.0, 0.0, 2.5],
    ], dtype=np.float32)
    labels = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.uint8)
    groups = np.asarray([3, 3], dtype=np.int32)
    params = {
        "objective": "lambdarank", "metric": "None", "label_gain": [0, 1],
        "max_bin": 255, "data_random_seed": 20260811, "seed": 20260811,
        "min_data_in_leaf": 1, "num_leaves": 3, "deterministic": True,
        "force_col_wise": True, "verbosity": -1,
    }
    names = ["numeric", "category", "nullable"]
    raw = lgb.Dataset(
        features, label=labels, group=groups, feature_name=names,
        categorical_feature=[1], params=params,
    )
    raw.construct()
    binary_path = tmp_path / "fixture.bin"
    raw.save_binary(binary_path)
    loaded = lgb.Dataset(str(binary_path), params=params)
    loaded.construct()
    assert loaded.num_data() == len(labels)
    assert loaded.num_feature() == len(names)
    assert loaded.feature_name == names
    assert np.array_equal(np.asarray(loaded.get_label(), dtype=np.uint8), labels)
    assert np.array_equal(np.asarray(loaded.get_group(), dtype=np.int32), groups)
    raw_model = lgb.train(params, raw, num_boost_round=4)
    binary_model = lgb.train(params, loaded, num_boost_round=4)
    assert np.array_equal(raw_model.predict(features), binary_model.predict(features))


def test_binary_semantic_fingerprint_binds_source_and_metadata() -> None:
    first = canonical_fingerprint({"binary": "a", "combined": "b", "labels": "c", "groups": "d"})
    second = canonical_fingerprint({"binary": "a", "combined": "changed", "labels": "c", "groups": "d"})
    assert first != second


def test_atomic_promotions_are_guarded_by_resource_and_integrity_checks() -> None:
    source = (ROOT / "src/ranking_model/final_refit_resource_recovery.py").read_text(encoding="utf-8")
    assert source.count("os.replace(temporary_binary, formal_binary)") == 1
    assert source.count("os.replace(temporary_model, formal_model)") == 1
    assert source.index("if monitored[\"rss_gate_exceeded\"]") < source.index("os.replace(temporary_binary, formal_binary)")


def test_model_replay_worker_has_no_label_argument_or_label_vector_open() -> None:
    source = (ROOT / "scripts/stage_4b4_final_model_replay_worker.py").read_text(encoding="utf-8")
    assert "--labels" not in source
    assert "labels.uint8" not in source
    assert '"label_vector_opened": False' in source


def test_formal_binary_manifest_when_present() -> None:
    path = ROOT / "artifacts/stage_4b4/final_refit_binary_dataset_manifest.json"
    if not path.exists():
        pytest.skip("formal binary is created after conversion resource gate")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "SUCCESS_FORMAL_BINARY_DATASET"
    assert manifest["rows"] == 11_568_070 and manifest["queries"] == 113_793
    assert manifest["features"] == 52 and manifest["group_sum"] == manifest["rows"]


def test_formal_model_manifest_when_present() -> None:
    path = ROOT / "artifacts/stage_4b4/final_refit_model_fingerprint_manifest.json"
    if not path.exists():
        pytest.skip("formal model is created after binary-only training resource gate")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["status"] == "FINAL_REFIT_MODEL_FROZEN"
    assert manifest["rounds"] == manifest["trees"] == 600
    assert manifest["fresh_initialization"] is True and manifest["warm_start"] is False
    assert manifest["loaded_formal_binary_only"] is True


def test_recovered_lock_and_gate_when_present() -> None:
    raw = _config().raw
    lock_path = ROOT / raw["paths"]["lock"]
    gate_path = ROOT / raw["paths"]["recovered_gate"]
    if not lock_path.exists() or not gate_path.exists():
        pytest.skip("recovered lock/gate are created during finalization")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert self_fingerprint(lock, "lock_fingerprint") == lock["lock_fingerprint"]
    with gate_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["gate_id"] for row in rows] == [f"R{index}" for index in range(1, 17)]
    assert all(row["status"] == "PASS" for row in rows)


def test_no_test_access_contract() -> None:
    scope = _config().raw["recovery_scope"]
    assert scope["test_access_allowed"] is False
    assert all(value == 0 for key, value in scope.items() if key.startswith("test_") and key.endswith("_count"))


def test_original_failure_checkpoint_when_present_is_not_reusable() -> None:
    path = ROOT / _config().raw["paths"]["original_failure_checkpoint"]
    if not path.exists():
        pytest.skip("failure checkpoint is written during recovery preregistration")
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    assert checkpoint["status"] == "FAILED"
    assert checkpoint["reuse_status"] == "NOT_REUSABLE_AS_FORMAL_MODEL"


def test_recovery_config_contains_no_search_space() -> None:
    raw = _config().raw
    assert raw["model"]["num_boost_round"] == 600
    assert raw["lightgbm"]["params"]["num_threads"] == 4
    assert "checkpoints" not in raw["lightgbm"]
