from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ranking_model.fingerprint import prediction_query_digest
from ranking_model.test_prediction_freeze import (
    EXPECTED_PHASE4_LOCK,
    VERSIONS,
    FreezeConfig,
    canonical_fingerprint,
    self_fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_4b4_test_prediction_freeze.yaml"


def test_phase5a_config_freezes_only_two_versions_and_full500() -> None:
    config = FreezeConfig.load(CONFIG)
    assert tuple(config.raw["versions"]) == VERSIONS
    assert config.raw["prediction"]["full_candidate_count_per_query"] == 500
    assert config.raw["prediction"]["optimizer_sampled_view_allowed"] is False


def test_phase5a_test_calendar_and_phase4_lock_are_exact() -> None:
    config = FreezeConfig.load(CONFIG)
    assert config.raw["snapshot"] == {
        "snapshot_id": "test_14d", "as_of_date": "2020-09-08", "request_date": "2020-09-09"
    }
    assert config.raw["expected"]["phase4_lock_fingerprint"] == EXPECTED_PHASE4_LOCK


def test_phase5a_resource_cap_is_frozen_seventy_percent() -> None:
    config = FreezeConfig.load(CONFIG)
    resource = config.raw["resource"]
    assert resource["absolute_rss_cap_bytes"] == int(
        resource["available_memory_at_preregistration_bytes"] * 0.70
    )
    assert resource["runtime_cap_recomputation_allowed"] is False


def test_phase5a_preregistration_is_self_fingerprinted_and_label_free() -> None:
    config = FreezeConfig.load(CONFIG)
    prereg = json.loads(config.path_for("preregistration").read_text(encoding="utf-8"))
    assert prereg["preregistration_fingerprint"] == self_fingerprint(
        prereg, "preregistration_fingerprint"
    )
    assert prereg["test_label_content_read_count"] == 0
    assert prereg["test_metric_read_count"] == 0
    assert prereg["prediction_versions"] == list(VERSIONS)


def test_prediction_digest_preserves_frozen_tie_break() -> None:
    rows = [(30, 3, 0.5), (10, 1, 0.5), (20, 2, 0.8)]
    left = prediction_query_digest("L1_FINAL_REFIT_600", "test_14d", 7, rows)
    right = prediction_query_digest("L1_FINAL_REFIT_600", "test_14d", 7, list(reversed(rows)))
    assert left == right
    assert left[0] != left[1]


def test_phase5a_output_schema_excludes_labels_and_metrics() -> None:
    config = FreezeConfig.load(CONFIG)
    output = set(config.raw["prediction"]["output_schema"])
    forbidden = {"label", "relevance", "hit", "ap", "dcg", "ndcg", "recall", "hitrate"}
    assert output.isdisjoint(forbidden)
    assert output == {
        "snapshot_id", "customer_key", "article_key", "candidate_rank",
        "raw_score", "predicted_rank", "version_id", "partition_id",
    }


def test_canonical_fingerprint_is_order_invariant_for_mappings() -> None:
    assert canonical_fingerprint({"b": 2, "a": 1}) == canonical_fingerprint({"a": 1, "b": 2})


def test_phase5a_lock_self_fingerprint_when_present() -> None:
    config = FreezeConfig.load(CONFIG)
    lock_path = config.path_for("pre_label_lock")
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        assert lock["lock_fingerprint"] == self_fingerprint(lock, "lock_fingerprint")
        assert lock["test_label_read_count"] == 0
        assert lock["test_metrics_computed"] is False
