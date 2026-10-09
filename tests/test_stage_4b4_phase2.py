from __future__ import annotations

import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

from ranking_model.validation_runner import ValidationSelectionRunner
from ranking_model.validation_selection import (
    VERSIONS,
    binary_top12_metrics,
    paired_bootstrap,
    select_winner,
    value_gate,
)
from retrieval.evaluation import user_metrics


def test_sql_metric_formula_matches_canonical_evaluator() -> None:
    ranked = [7, 3, 9, 5]
    truth = {3, 5, 11}
    canonical = user_metrics(ranked, truth, ks=(12,))
    actual = binary_top12_metrics([item in truth for item in ranked], len(truth))
    assert actual.map_at_12 == canonical.ap_at_12
    assert math.isclose(actual.ndcg_at_12, (1 / math.log2(3) + 1 / math.log2(5)) / sum(
        1 / math.log2(rank + 1) for rank in range(1, 4)
    ))
    assert actual.recall_at_12 == canonical.recall[12]
    assert actual.hit_rate_at_12 == canonical.hit_rate[12]


def test_retrieval_miss_is_zero_but_keeps_denominator() -> None:
    metrics = binary_top12_metrics([], 4)
    assert metrics.map_at_12 == metrics.ndcg_at_12 == 0.0
    assert metrics.recall_at_12 == metrics.hit_rate_at_12 == 0.0


def test_recall_denominator_includes_missed_positives() -> None:
    metrics = binary_top12_metrics([True, False, True], 5)
    assert metrics.recall_at_12 == 2 / 5


def test_baseline_rank_equivalence() -> None:
    truth = {12}
    candidate_rank = [10, 11, 12]
    assert user_metrics(candidate_rank, truth, ks=(12,)).ap_at_12 == 1 / 3


def test_winner_lexicographic_and_low_complexity_tie_break() -> None:
    rows = {
        version: {"map_at_12": 1.0, "ndcg_at_12": 1.0, "recall_at_12": 1.0, "hit_rate_at_12": 1.0}
        for version in VERSIONS
    }
    assert select_winner(rows) == "B0"
    rows["L1_600"]["map_at_12"] = 1.1
    assert select_winner(rows) == "L1_600"


def test_bootstrap_is_paired_and_replayable() -> None:
    args = ([0.0, 1.0, 0.0], [1.0, 1.0, 1.0])
    first = paired_bootstrap(*args, iterations=1000, seed=20260826)
    second = paired_bootstrap(*args, iterations=1000, seed=20260826)
    assert first == second
    assert math.isclose(first["delta_mean"], 2 / 3)
    array_result = paired_bootstrap(
        np.asarray(args[0]), np.asarray(args[1]), iterations=1000, seed=20260826
    )
    assert array_result == first


def test_value_gate_rules() -> None:
    assert value_gate("B0", 0.0, 0.0) == "FAILED_VALUE_GATE"
    assert value_gate("L0_100", 0.01, -0.001) == "INCONCLUSIVE_VALUE_GATE"
    assert value_gate("L0_100", 0.01, 0.001) == "PASSED_VALUE_GATE"


def test_partition_metric_sql_handles_hits_and_retrieval_miss(tmp_path: Path) -> None:
    prediction_root = tmp_path / "predictions"
    (prediction_root / "B0").mkdir(parents=True)
    pq.write_table(pa.table({
        "customer_key": [1, 1, 2, 2], "article_key": [10, 11, 20, 21],
        "predicted_rank": [1, 2, 1, 2],
    }), prediction_root / "B0" / "part_000.parquet")
    gt = tmp_path / "ground_truth.parquet"
    pq.write_table(pa.table({"user_key": [1, 1, 2], "article_key": [10, 12, 22]}), gt)
    state = tmp_path / "item_state.parquet"
    pq.write_table(pa.table({
        "article_key": [10, 11, 12, 20, 21, 22],
        "seen_item": [True, True, False, True, True, False],
        "cold_item": [False, False, True, False, False, True],
        "popularity_bucket": ["head", "mid", "zero_history", "tail", "tail", "zero_history"],
    }), state)
    base = tmp_path / "base.parquet"
    pq.write_table(pa.table({
        "snapshot_id": ["validation_14d", "validation_14d"], "user_key": [1, 2],
        "user_segment": ["warm", "short_history"], "positive_item_count": [2, 1],
        "warm_positive_count": [1, 0], "cold_positive_count": [1, 1],
        "has_cold_positive": [True, True], "is_main_evaluable": [True, True],
        "candidate_positive_count": [1, 0], "seen_positive_count": [1, 0],
        "cold_positive_count_check": [1, 1], "head_positive_count": [1, 0],
        "mid_positive_count": [0, 0], "tail_positive_count": [0, 0],
        "zero_history_positive_count": [1, 1],
    }), base)
    runner = object.__new__(ValidationSelectionRunner)
    runner.paths = {"predictions": prediction_root, "ground_truth": gt, "item_state": state}
    runner.temp = tmp_path / "tmp"
    runner.memory_mb = 512
    runner.raw = {"execution": {"threads": 1, "max_temp_directory_size": "1GB"}}
    output = tmp_path / "metrics.parquet"
    runner._build_metrics("B0", 0, base, output)
    rows = pq.read_table(output).to_pylist()
    assert len(rows) == 2
    assert rows[0]["map_at_12"] == 0.5
    assert rows[0]["recall_at_12"] == 0.5
    assert rows[1]["map_at_12"] == rows[1]["recall_at_12"] == 0.0
