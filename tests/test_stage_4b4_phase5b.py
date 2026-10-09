from __future__ import annotations

import json
import math
import copy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ranking_model.test_evaluation import (
    METRIC_COLUMNS,
    PostLabelFailure,
    TARGET_SPECS,
    TestEvaluationConfig,
    _connect,
    _copy_atomic,
    _metric_sql,
    _query_base_sql,
    _qualified_predicate,
    _slice_predicates,
    assert_label_source_unopened,
    binary_top12_or_zero,
    canonical_fingerprint,
    deduplicate_binary_relevance,
    initial_access_state,
    paired_bootstrap_detailed,
    query_merkle,
    query_metric_digest,
    self_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_4b4_test_evaluation.yaml"


def test_phase5b_config_freezes_one_time_test_contract() -> None:
    config = TestEvaluationConfig.load(CONFIG)
    assert config.raw["snapshot"] == {
        "snapshot_id": "test_14d", "as_of_date": "2020-09-08",
        "request_date": "2020-09-09", "label_start": "2020-09-09",
        "label_end": "2020-09-22", "label_window_days": 14,
        "feedback_semantics": "B_user_item_day_binary_v1", "label_window_closed": "both",
    }
    assert config.raw["versions"] == ["B0", "L1_FINAL_REFIT_600"]
    assert config.raw["evaluation"]["cutoff"] == 12
    assert config.raw["execution"]["formal_test_evaluation_limit"] == 1
    assert config.raw["execution"]["checkpoint_reuse_after_label_access_allowed"] is False
    assert all(value is False for value in config.raw["forbidden"].values())


def test_ap_ndcg_recall_hitrate_single_positive_golden() -> None:
    result = binary_top12_or_zero([False, True, False], 1)
    assert result == {
        "map_at_12": 0.5,
        "ndcg_at_12": 1.0 / math.log2(3),
        "recall_at_12": 1.0,
        "hit_rate_at_12": 1.0,
    }


def test_ap_ndcg_recall_hitrate_multi_positive_and_full_denominators() -> None:
    result = binary_top12_or_zero([True, False, True], 4)
    expected_ap = (1.0 + 2.0 / 3.0) / 4.0
    expected_dcg = 1.0 + 1.0 / math.log2(4)
    expected_idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, 5))
    assert result["map_at_12"] == pytest.approx(expected_ap)
    assert result["ndcg_at_12"] == pytest.approx(expected_dcg / expected_idcg)
    assert result["recall_at_12"] == 0.5
    assert result["hit_rate_at_12"] == 1.0


def test_all_zero_relevance_query_retained_as_four_zeros() -> None:
    assert binary_top12_or_zero([True, True], 0) == {name: 0.0 for name in METRIC_COLUMNS}


def test_candidate_positive_zero_does_not_change_future_positive_denominator() -> None:
    result = binary_top12_or_zero([False] * 12, 3)
    assert result == {name: 0.0 for name in METRIC_COLUMNS}


def test_user_item_day_binary_dedup_and_duplicate_label_handling() -> None:
    rows = [
        ("test_14d", 1, 10, "2020-09-09"),
        ("test_14d", 1, 10, "2020-09-09"),
        ("test_14d", 1, 10, "2020-09-10"),
        ("test_14d", 1, 11, "2020-09-10"),
    ]
    assert deduplicate_binary_relevance(rows) == [("test_14d", 1, 10), ("test_14d", 1, 11)]


def test_prediction_label_join_keys_are_snapshot_user_item() -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert raw["keys"]["prediction_label_join"] == ["snapshot_id", "customer_key", "article_key"]
    assert raw["ground_truth"]["positive_injection_allowed"] is False


def test_paired_bootstrap_is_deterministic_and_query_paired() -> None:
    baseline = np.array([0.0, 0.2, 0.4, 0.0], dtype=np.float64)
    final = np.array([0.1, 0.3, 0.2, 0.0], dtype=np.float64)
    kwargs = {
        "iterations": 1000, "seed": 20260826, "confidence_level": 0.95,
        "population_fingerprint": canonical_fingerprint([1, 2, 3, 4]),
    }
    first = paired_bootstrap_detailed(baseline, final, **kwargs)
    second = paired_bootstrap_detailed(baseline, final, **kwargs)
    assert first == second
    assert first["query_count"] == 4
    assert first["observed_delta"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="identical query population"):
        paired_bootstrap_detailed(baseline, final[:-1], **kwargs)


def test_bootstrap_rejects_nan_inf() -> None:
    with pytest.raises(ValueError, match="NaN/Inf"):
        paired_bootstrap_detailed(
            [0.0, np.nan], [0.1, 0.2], iterations=10, seed=1, confidence_level=0.95,
            population_fingerprint="a" * 64,
        )


def test_user_and_target_segment_contract_is_frozen() -> None:
    predicates = _slice_predicates()
    assert set(predicates) == {
        "evaluable_warm_users", "short_history", "warm", "active",
        "positive_count_eq_1", "positive_count_gt_1", "candidate_positive_zero",
        "candidate_positive_nonzero", "has_cold_positive", "no_cold_positive",
        "cold_history_label_users",
    }
    assert set(TARGET_SPECS) == {"seen_item", "cold_item", "head", "mid", "tail", "zero_history"}
    qualified = _qualified_predicate(predicates["candidate_positive_zero"], "b")
    assert qualified == "b.is_main_evaluable AND b.candidate_positive_count=0"


def test_query_metric_digest_and_merkle_are_order_stable_and_value_sensitive() -> None:
    values = {"map_at_12": 0.1, "ndcg_at_12": 0.2}
    digest = query_metric_digest("B0", "test_14d", 7, values)
    assert digest == query_metric_digest("B0", "test_14d", 7, dict(reversed(list(values.items()))))
    assert digest != query_metric_digest("B0", "test_14d", 7, {"map_at_12": 0.2, "ndcg_at_12": 0.2})
    rows = [("test_14d", 7, digest), ("test_14d", 3, digest)]
    assert query_merkle(rows, binding="x") == query_merkle(list(reversed(rows)), binding="x")


def test_canonical_json_self_fingerprint_detects_mutation() -> None:
    payload = {"status": "LOCKED", "count": 1}
    payload["lock_fingerprint"] = self_fingerprint(payload, "lock_fingerprint")
    assert self_fingerprint(payload, "lock_fingerprint") == payload["lock_fingerprint"]
    payload["count"] = 2
    assert self_fingerprint(payload, "lock_fingerprint") != payload["lock_fingerprint"]


def test_label_access_guard_refuses_any_reopen() -> None:
    config = TestEvaluationConfig.load(CONFIG)
    state = initial_access_state(config)
    assert_label_source_unopened(state)
    state["raw_test_label_source_read_count"] = 1
    with pytest.raises(PostLabelFailure, match="reopen refused"):
        assert_label_source_unopened(state)


def test_phase5a_lock_and_prediction_identities_match_new_config_without_label_access() -> None:
    config = TestEvaluationConfig.load(CONFIG)
    lock = json.loads(config.upstream_path("phase5a_lock").read_text(encoding="utf-8"))
    assert lock["lock_fingerprint"] == config.raw["expected"]["phase5a_pre_label_lock"]
    assert self_fingerprint(lock, "lock_fingerprint") == lock["lock_fingerprint"]
    assert lock["test_label_read_count"] == 0
    assert lock["test_evaluation_executed"] is False
    assert lock["prediction_fingerprints"]["B0"]["global_merkle"] == config.raw["expected"]["b0_prediction_global_fingerprint"]
    assert lock["prediction_fingerprints"]["L1_FINAL_REFIT_600"]["global_merkle"] == config.raw["expected"]["final_prediction_global_fingerprint"]


def test_postlabel_worker_source_has_no_raw_source_reopen_loop() -> None:
    source = (ROOT / "scripts/stage_4b4_test_evaluation_worker.py").read_text(encoding="utf-8")
    assert source.count("build_ground_truth(config)") == 1
    assert "ground_truth" in source and "evaluation" in source and "bootstrap" in source
    module = (ROOT / "src/ranking_model/test_evaluation.py").read_text(encoding="utf-8")
    assert "assert_label_source_unopened(state)" in module
    assert "checkpoint_reuse_after_label_access_allowed\": false" in CONFIG.read_text(encoding="utf-8")


def test_partition_sql_retains_all_zero_and_full_positive_denominators(tmp_path: Path) -> None:
    raw = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    raw["upstream"]["predictions"] = "predictions"
    raw["paths"]["private_directory"] = "private"
    raw["paths"]["temp_directory"] = "tmp"
    config = TestEvaluationConfig(tmp_path, tmp_path / "config.json", raw, canonical_fingerprint(raw))
    (tmp_path / "predictions/B0").mkdir(parents=True)
    prediction = pa.table({
        "snapshot_id": ["test_14d"] * 4,
        "customer_key": pa.array([32, 32, 64, 64], type=pa.int64()),
        "article_key": pa.array([100, 101, 102, 103], type=pa.int64()),
        "candidate_rank": pa.array([1, 2, 1, 2], type=pa.int32()),
        "raw_score": [0.0] * 4,
        "predicted_rank": pa.array([1, 2, 1, 2], type=pa.int32()),
        "version_id": ["B0"] * 4,
        "partition_id": pa.array([0] * 4, type=pa.int32()),
    })
    pq.write_table(prediction, tmp_path / "predictions/B0/part_000.parquet")
    config.private_dir.mkdir(parents=True)
    pq.write_table(pa.table({
        "snapshot_id": ["test_14d"] * 3,
        "customer_key": pa.array([32, 32, 64], type=pa.int64()),
        "article_key": pa.array([100, 999, 500], type=pa.int64()),
    }), config.private_dir / "canonical_ground_truth.parquet")
    pq.write_table(pa.table({
        "snapshot_id": ["test_14d", "test_14d"],
        "customer_key": pa.array([32, 64], type=pa.int64()),
        "user_segment": ["warm", "active"],
        "positive_item_count": pa.array([2, 1], type=pa.int32()),
        "warm_positive_count": pa.array([1, 1], type=pa.int32()),
        "cold_positive_count": pa.array([1, 0], type=pa.int32()),
        "has_cold_positive": [True, False],
        "is_main_evaluable": [True, True],
    }), config.private_dir / "canonical_population.parquet")
    pq.write_table(pa.table({
        "article_key": pa.array([100, 101, 102, 103, 500, 999], type=pa.int64()),
        "seen_item": [True, True, True, True, True, False],
        "cold_item": [False, False, False, False, False, True],
        "popularity_bucket": ["head", "mid", "tail", "head", "mid", "zero_history"],
    }), config.private_dir / "canonical_item_state.parquet")
    con = _connect(config)
    try:
        base = config.private_dir / "base.parquet"
        output = config.private_dir / "metrics.parquet"
        _copy_atomic(con, _query_base_sql(config, 0), base)
        _copy_atomic(con, _metric_sql(config, "B0", 0, base), output)
    finally:
        con.close()
    rows = pq.read_table(output).to_pylist()
    assert rows[0]["customer_key"] == 32
    assert rows[0]["candidate_positive_count"] == 1
    assert rows[0]["map_at_12"] == 0.5
    assert rows[0]["recall_at_12"] == 0.5
    assert rows[1]["customer_key"] == 64
    assert rows[1]["candidate_positive_count"] == 0
    assert all(rows[1][name] == 0.0 for name in METRIC_COLUMNS)


def test_private_public_paths_and_no_raw_identifier_fields() -> None:
    config = TestEvaluationConfig.load(CONFIG)
    assert config.private_dir.as_posix().endswith("artifacts/stage_4b4/private/test_evaluation")
    assert config.checkpoint_dir.as_posix().endswith("artifacts/stage_4b4/checkpoints/test_evaluation")
    public_names = {Path(value).name for key, value in config.raw["paths"].items() if key not in {
        "private_directory", "checkpoint_directory", "temp_directory", "public_directory"
    }}
    assert "stage_4b4_test_evaluation_lock.json" in public_names
    text = CONFIG.read_text(encoding="utf-8")
    assert "customer_id" not in text and "article_id" not in text and "postal_code" not in text
