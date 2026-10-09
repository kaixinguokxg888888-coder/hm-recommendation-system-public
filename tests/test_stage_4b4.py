from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from ranking_features.schema import feature_schema_rows
from ranking_model.adapter import (
    ModelInputAdapter,
    assert_adapter_deterministic,
    assert_query_group_order,
)
from ranking_model.config import Stage4B4Config, Stage4B4ConfigError
from ranking_model.fingerprint import canonical_score, merkle_fingerprint, prediction_query_digest
from ranking_model.ordering import (
    ORDERING_VERSION,
    TrainingOrderError,
    audit_ordered_arrays,
    matrix_order_fingerprint,
)
from scripts.stage_4b4_recovery import process_tree_rss, source_fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage_4b4_lambdarank.yaml"


def load_raw() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def adapters() -> tuple[ModelInputAdapter, ModelInputAdapter]:
    raw = load_raw()
    stage4b3 = json.loads((ROOT / raw["upstream"]["stage_4b3_config"]).read_text(encoding="utf-8"))
    rows = feature_schema_rows(stage4b3["canonical_fields"])
    public = [{**row, "model_input_eligible": str(row["model_input_eligible"]),
               "feature_set": str(row["feature_set"])} for row in rows]
    mapping = raw["adapter"]["user_segment_mapping"]
    return (ModelInputAdapter.from_schema(public, "F0", mapping),
            ModelInputAdapter.from_schema(public, "F1", mapping))


def test_formal_config_and_budget() -> None:
    config = Stage4B4Config.load(CONFIG)
    assert config.raw["predictions"]["versions"] == [
        "B0", "L0_100", "L0_300", "L0_600", "L1_100", "L1_300", "L1_600"
    ]
    assert config.raw["lightgbm"]["num_boost_round"] == 600
    assert config.raw["lightgbm"]["params"]["objective"] == "lambdarank"
    assert config.raw["lightgbm"]["params"]["metric"] == "None"
    assert config.raw["lightgbm"]["params"]["lambdarank_truncation_level"] == 15


def test_config_rejects_label_access_and_extra_checkpoint(tmp_path: Path) -> None:
    for mutation in ("label", "checkpoint"):
        raw = load_raw()
        if mutation == "label":
            raw["access_lock"]["validation_labels_read"] = True
        else:
            raw["lightgbm"]["checkpoints"] = [100, 200, 300, 600]
        path = tmp_path / f"{mutation}.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(Stage4B4ConfigError):
            Stage4B4Config.load(path)


def test_adapter_feature_order_and_categorical_contract() -> None:
    f0, f1 = adapters()
    assert len(f0.ordered_features) == 30
    assert len(f1.ordered_features) == 52
    assert f1.ordered_features[:30] == f0.ordered_features
    assert f0.categorical_features == ("user_segment",)
    assert len(f1.categorical_features) == 12
    assert "primary_source" not in f1.ordered_features
    assert not any(name in f1.ordered_features for name in ("label", "query_order", "sampling_digest"))


def test_adapter_preserves_nan_and_maps_boolean_segment() -> None:
    f0, _ = adapters()
    values = {}
    for name, dtype in zip(f0.ordered_features, f0.dtypes):
        if name == "user_segment":
            values[name] = pa.array(["cold", "active"])
        elif dtype == "bool":
            values[name] = pa.array([False, True])
        elif dtype == "float64":
            values[name] = pa.array([None, 1.5], type=pa.float64())
        elif dtype == "int64":
            values[name] = pa.array([0, 2], type=pa.int64())
        else:
            values[name] = pa.array([None, 2], type=pa.int32())
    table = pa.table(values)
    matrix = f0.convert_table(table)
    assert matrix.dtype == np.float32
    assert matrix[0, f0.ordered_features.index("user_segment")] == 0
    assert matrix[1, f0.ordered_features.index("user_segment")] == 3
    assert np.isnan(matrix[0, f0.ordered_features.index("user_recency_days")])
    assert matrix[0, f0.ordered_features.index("ui_seen_before")] == 0
    assert_adapter_deterministic(f0, table)


def test_prediction_tie_break_and_order_invariance() -> None:
    rows = [(9, 2, 0.5), (7, 1, 0.5), (8, 3, 0.7)]
    left = prediction_query_digest("L0_100", "validation_14d", 4, rows)
    right = prediction_query_digest("L0_100", "validation_14d", 4, list(reversed(rows)))
    assert left == right
    assert left[0] != left[1]


def test_score_serialization_rejects_nonfinite() -> None:
    assert canonical_score(-0.0) == "-0x0.0p+0"
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_score(value)


def test_merkle_is_schedule_invariant_and_binding_sensitive() -> None:
    rows = [("validation_14d", 2, "00" * 32), ("validation_14d", 1, "11" * 32)]
    assert merkle_fingerprint(rows, "x", "a") == merkle_fingerprint(reversed(rows), "x", "a")
    assert merkle_fingerprint(rows, "x", "a") != merkle_fingerprint(rows, "x", "b")


def test_no_validation_or_test_paths_in_training_inputs() -> None:
    raw = load_raw()
    train_roles = ("train_optimizer", "train_groups", "train_optimizer_features")
    assert all("validation" not in raw["upstream"][role].lower() for role in train_roles)
    assert all("test" not in raw["upstream"][role].lower() for role in train_roles)
    assert "label" not in raw["upstream"]["validation_candidates"].lower()


def test_query_group_contract_rejects_order_drift() -> None:
    groups = [("ranking_train_14d_v1", 10, 1, 101), ("ranking_train_14d_v1", 20, 2, 102)]
    canonical = [("ranking_train_14d_v1", 10, 101), ("ranking_train_14d_v1", 20, 102)]
    assert_query_group_order(canonical, groups)
    with pytest.raises(ValueError):
        assert_query_group_order(list(reversed(canonical)), groups)


def test_isolated_worker_has_narrow_import_and_access_surface() -> None:
    worker = (ROOT / "scripts" / "stage_4b4_isolated_worker.py").read_text(encoding="utf-8")
    assert "duckdb" not in worker
    assert "pyarrow" not in worker
    assert "validation_candidates" not in worker
    assert "validation_history" not in worker
    assert "test_candidate" not in worker
    assert "subprocess" not in worker


def test_ranking_model_package_lazily_loads_heavy_runner() -> None:
    package = (ROOT / "src" / "ranking_model" / "__init__.py").read_text(encoding="utf-8")
    assert "from .runner import Stage4B4Runner" not in package.split("def __getattr__", 1)[0]


def ordered_fixture() -> tuple[dict[str, np.ndarray], list[tuple[str, int, int, int, int, int]]]:
    arrays = {
        "customers": np.array([10, 10, 20, 20], dtype=np.int64),
        "orders": np.array([1, 1, 2, 2], dtype=np.int64),
        "articles": np.array([100, 101, 200, 201], dtype=np.int64),
        "ranks": np.array([1, 2, 1, 2], dtype=np.int32),
        "labels": np.array([1, 0, 1, 0], dtype=np.uint8),
    }
    expected = [("s", 10, 1, 2, 1, 1), ("s", 20, 2, 2, 1, 1)]
    return arrays, expected


def run_order_audit(arrays: dict[str, np.ndarray], expected):
    return audit_ordered_arrays(
        snapshot_id="s", customer_keys=arrays["customers"], query_orders=arrays["orders"],
        article_keys=arrays["articles"], candidate_ranks=arrays["ranks"], labels=arrays["labels"],
        expected_queries=expected,
    )


def test_global_query_order_fixture_passes_and_is_schedule_invariant() -> None:
    arrays, expected = ordered_fixture()
    audit, boundaries = run_order_audit(arrays, expected)
    assert audit["query_position_mismatch"] == 0
    assert matrix_order_fingerprint(boundaries) == matrix_order_fingerprint(list(boundaries))


@pytest.mark.parametrize("mutation", ("partition_major", "same_set_wrong_position", "rank_order"))
def test_position_and_within_query_drift_are_rejected(mutation: str) -> None:
    arrays, expected = ordered_fixture()
    if mutation in {"partition_major", "same_set_wrong_position"}:
        order = np.array([2, 3, 0, 1])
        arrays = {key: value[order] for key, value in arrays.items()}
    else:
        arrays["articles"][:2] = [101, 100]
        arrays["ranks"][:2] = [2, 1]
    with pytest.raises(TrainingOrderError):
        run_order_audit(arrays, expected)


def test_group_size_position_and_pseudo_all_zero_are_rejected() -> None:
    arrays, _ = ordered_fixture()
    expected = [("s", 10, 1, 1, 1, 0), ("s", 20, 2, 3, 1, 2)]
    with pytest.raises(TrainingOrderError) as error:
        run_order_audit(arrays, expected)
    payload = json.loads(str(error.value))
    assert payload["query_position_mismatch"] > 0 or payload["all_zero_groups"] > 0


def test_feature_and_label_order_share_matrix_fingerprint_binding() -> None:
    arrays, expected = ordered_fixture()
    _, boundaries = run_order_audit(arrays, expected)
    original = matrix_order_fingerprint(boundaries)
    changed = [dict(row) for row in boundaries]
    changed[0]["ordered_row_digest"] = "00" * 32
    assert original != matrix_order_fingerprint(changed)


def test_worker_requires_v2_ordering_metadata() -> None:
    worker = (ROOT / "scripts" / "stage_4b4_isolated_worker.py").read_text(encoding="utf-8")
    assert "worker rejects non-v2 matrix ordering" in worker
    assert ORDERING_VERSION in (ROOT / "src" / "ranking_model" / "ordering.py").read_text(encoding="utf-8")


def test_recovery_source_and_external_tree_monitor_are_available() -> None:
    assert len(source_fingerprint(ROOT)) == 64
    assert process_tree_rss(-1) == 0
