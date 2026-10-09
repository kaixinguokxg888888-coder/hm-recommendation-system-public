from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ranking_model.first7d_diagnostic import (
    FIRST7D_VERSIONS,
    classify_robustness,
    segment_review_required,
)
from ranking_model.first7d_runner import First7dDiagnosticRunner
from ranking_model.resource_contract import ResourceContractError
from ranking_model.validation_selection import canonical_fingerprint


def test_direction_and_confidence_boundaries() -> None:
    assert classify_robustness(0.01, 0.001, 0.02).direction_status == "ROBUSTNESS_DIRECTION_PASS"
    assert classify_robustness(0.01, 0.001, 0.02).confidence_status == "POSITIVE_AND_SIGNIFICANT"
    assert classify_robustness(0.0, 0.0, 0.0).confidence_status == "NONNEGATIVE_BUT_INCONCLUSIVE"
    assert classify_robustness(-0.01, -0.02, -0.001).direction_status == "VALIDATION_ROBUSTNESS_NOT_MET"
    assert classify_robustness(-0.01, -0.02, -0.001).confidence_status == "SIGNIFICANT_NEGATIVE"
    assert classify_robustness(-0.01, -0.02, 0.001).confidence_status == "NEGATIVE_BUT_INCONCLUSIVE"


def test_critical_segment_rule() -> None:
    assert segment_review_required([0.0, -0.0001])
    assert not segment_review_required([0.0, 0.1])


def test_config_registers_only_locked_pair_and_exact_window(repo_root: Path = Path(__file__).resolve().parents[1]) -> None:
    raw = json.loads((repo_root / "configs/stage_4b4_first7d_diagnostic.yaml").read_text(encoding="utf-8"))
    assert tuple(raw["versions"]) == FIRST7D_VERSIONS
    assert raw["diagnostic"]["snapshot_id"] == "validation_14d"
    assert raw["diagnostic"]["label_start"] == "2020-08-26"
    assert raw["diagnostic"]["label_end"] == "2020-09-01"
    assert raw["diagnostic"]["selection_use"] is False
    assert raw["locked_winner"] == {
        "model_id": "L1_600", "feature_set": "F1", "iteration": 600, "winner_changed": False
    }
    # Historical evidence is preserved: this original config contains the
    # confirmed 65% resource-contract defect and is no longer executable.
    assert raw["execution"]["rss_hard_fraction_of_available_at_preregistration"] == 0.65
    with pytest.raises(ResourceContractError):
        First7dDiagnosticRunner(
            repo_root, repo_root / "configs/stage_4b4_first7d_diagnostic.yaml"
        )


def _synthetic_runner(tmp_path: Path) -> First7dDiagnosticRunner:
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "configs/stage_4b4_first7d_diagnostic.yaml").read_text(encoding="utf-8"))
    raw["execution"]["rss_hard_fraction_of_available_at_preregistration"] = 0.70
    corrected = tmp_path / "synthetic_first7d_config.json"
    corrected.write_text(json.dumps(raw), encoding="utf-8")
    runner = First7dDiagnosticRunner(root, corrected)
    runner.public = tmp_path / "public"
    runner.private = tmp_path / "private"
    runner.checkpoints = tmp_path / "checkpoints"
    runner.temp = tmp_path / "tmp"
    runner.paths.update({
        "validation_ground_truth": tmp_path / "validation_gt.parquet",
        "validation_population": tmp_path / "validation_population.parquet",
        "validation_item_state": tmp_path / "item_state.parquet",
        "predictions": tmp_path / "predictions",
    })
    runner._prereg = {"preregistration_fingerprint": "synthetic"}
    return runner


def _write_synthetic_inputs(runner: First7dDiagnosticRunner) -> None:
    pq.write_table(pa.table({
        "snapshot_id": ["validation_14d"] * 4,
        "user_key": [32, 32, 64, 96], "article_key": [10, 11, 20, 30], "label": [1, 1, 1, 1],
        "label_date_min": [date(2020, 8, 26), date(2020, 9, 2), date(2020, 9, 1), date(2020, 9, 8)],
        "label_date_max": [date(2020, 8, 26), date(2020, 9, 2), date(2020, 9, 1), date(2020, 9, 8)],
        "as_of_date": [date(2020, 8, 25)] * 4,
        "label_start": [date(2020, 8, 26)] * 4,
        "label_end": [date(2020, 9, 8)] * 4,
        "feedback_version": ["B-primary_C-log1p-p99-v1"] * 4,
    }), runner.paths["validation_ground_truth"])
    pq.write_table(pa.table({
        "snapshot_id": ["validation_14d"] * 3, "user_key": [32, 64, 96],
        "user_segment": ["warm", "cold", "active"],
        "history_interaction_count": [7, 0, 30], "history_distinct_item_count": [5, 0, 20],
        "positive_item_count": [2, 1, 1], "warm_positive_count": [1, 0, 1],
        "cold_positive_count": [1, 1, 0], "has_cold_positive": [True, True, False],
        "is_main_evaluable": [True, False, True],
        "as_of_date": [date(2020, 8, 25)] * 3,
    }), runner.paths["validation_population"])
    pq.write_table(pa.table({
        "article_key": [10, 11, 20, 30],
        "seen_item": [True, False, False, True], "cold_item": [False, True, True, False],
        "popularity_bucket": ["head", "zero_history", "zero_history", "tail"],
    }), runner.paths["validation_item_state"])


def test_first7d_materialization_filters_same_validation_ground_truth_once(tmp_path: Path) -> None:
    runner = _synthetic_runner(tmp_path)
    _write_synthetic_inputs(runner)
    gt, population, checkpoint = runner._materialize_first7d_once()
    gt_rows = pq.read_table(gt).to_pylist()
    pop_rows = pq.read_table(population).to_pylist()
    assert [(row["user_key"], row["article_key"]) for row in gt_rows] == [(32, 10), (64, 20)]
    assert all(str(row["label_end"]) == "2020-09-01" for row in gt_rows)
    assert [(row["user_key"], row["positive_item_count"]) for row in pop_rows] == [(32, 1), (64, 1)]
    assert checkpoint["query_count"] == 2
    access = json.loads((runner.public / "first7d_diagnostic_access_audit.json").read_text(encoding="utf-8"))
    assert access["first7d_read_count"] == 1
    assert access["test_label_read_count"] == 0


def test_first7d_metric_keeps_missed_positive_denominator_and_b0_rank(tmp_path: Path) -> None:
    runner = _synthetic_runner(tmp_path)
    _write_synthetic_inputs(runner)
    gt, population, _ = runner._materialize_first7d_once()
    for version in FIRST7D_VERSIONS:
        directory = runner.paths["predictions"] / version
        directory.mkdir(parents=True)
        if version == "B0":
            articles = [99, 10, 98, 97]
            ranks = [1, 2, 1, 2]
        else:
            articles = [10, 99, 20, 98]
            ranks = [1, 2, 1, 2]
        pq.write_table(pa.table({
            "customer_key": [32, 32, 64, 64], "article_key": articles,
            "candidate_rank": [1, 2, 1, 2], "predicted_rank": ranks,
        }), directory / "part_000.parquet")
    base = tmp_path / "base.parquet"
    runner._build_query_base(0, gt, population, base)
    helper = runner._metric_helper(gt)
    output = tmp_path / "metrics.parquet"
    helper._build_metrics("B0", 0, base, output)
    rows = pq.read_table(output).to_pylist()
    assert rows
    first = next(row for row in rows if row["customer_key"] == 32)
    assert first["map_at_12"] == 0.5
    assert first["recall_at_12"] == 1.0


def test_self_excluding_fingerprint() -> None:
    payload = {"a": 1}
    payload["fingerprint"] = canonical_fingerprint(payload)
    runner = object.__new__(First7dDiagnosticRunner)
    assert runner._self_fingerprint(payload, "fingerprint") == payload["fingerprint"]
