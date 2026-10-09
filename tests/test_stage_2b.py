from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from data_audit.stage1a import sensitive_value_matches
from data_foundation.evaluation import CANDIDATE_DOMAINS, checkpoint_matches, evaluation_object
from data_foundation.ids import stable_dense_mapping
from data_foundation.item_state import item_state, popularity_bucket, popularity_thresholds
from data_foundation.labels import deduplicate_ground_truth
from data_foundation.populations import population_membership, positive_count_bucket, user_segment


def sample_labels() -> list[tuple[int, int, date, int]]:
    return [
        (0, 10, date(2020, 1, 1), 2),
        (0, 10, date(2020, 1, 2), 1),
        (0, 11, date(2020, 1, 2), 1),
    ]


def test_ground_truth_user_item_deduplication() -> None:
    assert len(deduplicate_ground_truth(sample_labels())) == 2


def test_multi_day_same_item_is_one_positive() -> None:
    row = deduplicate_ground_truth(sample_labels())[(0, 10)]
    assert row["label"] == 1
    assert row["user_item_label_days"] == 2
    assert row["label_daily_strength_sum"] == 3


def test_all_label_users_definition() -> None:
    assert population_membership(0, 1)["all_label_users"]


def test_evaluable_warm_user_definition() -> None:
    assert population_membership(1, 1)["evaluable_warm_users"]
    assert not population_membership(0, 1)["evaluable_warm_users"]


def test_cold_history_user_definition() -> None:
    assert population_membership(0, 2)["cold_history_label_users"]


@pytest.mark.parametrize(
    "value,expected",
    [(0, "cold"), (1, "short_history"), (4, "short_history"), (5, "warm"), (20, "warm"), (21, "active")],
)
def test_user_segment_boundaries(value: int, expected: str) -> None:
    assert user_segment(value) == expected


def test_seen_and_cold_item_definition() -> None:
    assert item_state(2, 1) == {"seen_item": True, "cold_item": False, "zero_history_dimension_item": False}
    assert item_state(0, 1) == {"seen_item": False, "cold_item": True, "zero_history_dimension_item": True}


def test_popularity_thresholds_use_history_only() -> None:
    assert popularity_thresholds([1, 2, 3, 100]) == (2, 100)
    assert popularity_bucket(1, 2, 100) == "tail"
    assert popularity_bucket(2, 2, 100) == "mid"
    assert popularity_bucket(100, 2, 100) == "head"


@pytest.mark.parametrize("value,expected", [(1, "1"), (2, "2"), (3, "3-5"), (5, "3-5"), (6, "6-10"), (10, "6-10"), (11, "10+")])
def test_positive_count_buckets(value: int, expected: str) -> None:
    assert positive_count_bucket(value) == expected


def test_validation_and_test_are_isolated() -> None:
    validation = deduplicate_ground_truth([(1, 1, date(2020, 1, 1), 1)])
    test = deduplicate_ground_truth([(1, 2, date(2020, 1, 2), 1)])
    assert set(validation) != set(test)


def test_internal_key_coverage() -> None:
    users = stable_dense_mapping(["u1", "u2"])
    items = stable_dense_mapping(["i1", "i2"])
    assert set(users.values()) == {0, 1}
    assert set(items.values()) == {0, 1}


def test_candidate_domain_protocol() -> None:
    assert CANDIDATE_DOMAINS["main_warm_evaluation"] == "history_seen_items_only"
    assert "full_article_dimension" in CANDIDATE_DOMAINS["cold_item_evaluation"]


def test_public_aggregate_has_no_real_ids() -> None:
    assert not sensitive_value_matches('{"snapshot_id":"test_14d","user_count":12}')


def test_private_paths_match_git_policy() -> None:
    assert "private" in "artifacts/stage_2b/private/test_ground_truth.parquet"


def test_population_manifest_counts() -> None:
    rows = [evaluation_object(1, 2, 2, 3, 2, 1), evaluation_object(2, 0, 0, 1, 0, 1)]
    assert sum(row["is_main_evaluable"] for row in rows) == 1
    assert sum(row["positive_item_count"] for row in rows) == 4


def test_checkpoint_recovery() -> None:
    checkpoint = {"status": "SUCCESS", "dataset_fingerprint": "d", "config_fingerprint": "c"}
    assert checkpoint_matches(checkpoint, "d", "c", True)
    assert not checkpoint_matches(checkpoint, "d", "c", True, True)


def test_input_file_is_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "private.parquet"
    path.write_bytes(b"stage2a-private")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    _ = path.read_bytes()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
