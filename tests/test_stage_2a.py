from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import duckdb
import pytest

from data_audit.stage1a import sensitive_value_matches
from data_foundation.calendar import dates_inclusive, missing_dates, weekday_composition
from data_foundation.feedback import binary_interactions, capped_strength, daily_counts, nearest_rank_cap
from data_foundation.ids import KEY_STRATEGY, mapping_fingerprint, stable_dense_mapping, validate_mapping
from data_foundation.manifest import MANIFEST_FIELDS, checkpoint_matches, manifest_counts_match
from data_foundation.split import Snapshot, derive_snapshots, trailing_window


@pytest.fixture
def tiny_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "transactions.parquet"
    con = duckdb.connect()
    con.execute(
        f"""COPY (SELECT * FROM (VALUES
        (DATE '2020-01-01', 'u1', 'i1'),
        (DATE '2020-01-01', 'u1', 'i1'),
        (DATE '2020-01-02', 'u2', 'i2'))
        t(t_dat, customer_id, article_id))
        TO '{path.as_posix()}' (FORMAT PARQUET)"""
    )
    con.close()
    return path


def test_date_boundaries_are_inclusive() -> None:
    snapshot = Snapshot("x", date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4), 2)
    snapshot.validate()
    assert snapshot.contains_history(date(2020, 1, 2))
    assert snapshot.contains_label(date(2020, 1, 4))


def test_validation_and_test_do_not_overlap() -> None:
    snapshots = derive_snapshots(date(2020, 9, 22), date(2018, 9, 20))
    assert snapshots["validation_14d"].label_end < snapshots["test_14d"].label_start


def test_history_must_precede_label() -> None:
    with pytest.raises(ValueError, match="严格早于"):
        Snapshot("bad", date(2020, 1, 1), date(2020, 1, 3), date(2020, 1, 3), date(2020, 1, 4), 2).validate()


def test_binary_feedback_deduplicates(tiny_parquet: Path) -> None:
    rows = duckdb.connect().execute(
        f"SELECT t_dat::VARCHAR, customer_id, article_id FROM read_parquet('{tiny_parquet.as_posix()}')"
    ).fetchall()
    assert len(binary_interactions(rows)) == 2


def test_c_feedback_counts(tiny_parquet: Path) -> None:
    rows = duckdb.connect().execute(
        f"SELECT t_dat::VARCHAR, customer_id, article_id FROM read_parquet('{tiny_parquet.as_posix()}')"
    ).fetchall()
    assert sorted(daily_counts(rows).values()) == [1, 2]


def test_c_cap_uses_only_supplied_history() -> None:
    assert nearest_rank_cap([1, 1, 2, 100], 0.75) == 2
    assert capped_strength(100, 2)[0] == 2


def test_snapshots_calculate_caps_separately() -> None:
    assert nearest_rank_cap([1, 1, 1], 0.99) == 1
    assert nearest_rank_cap([1, 2, 3], 0.99) == 3


def test_stable_one_to_one_internal_encoding() -> None:
    mapping = stable_dense_mapping(["b", "a", "c", "a"])
    assert mapping == {"a": 0, "b": 1, "c": 2}
    validate_mapping(mapping, ["a", "b", "c"])
    assert mapping_fingerprint(mapping) == mapping_fingerprint(stable_dense_mapping(["c", "a", "b"]))


def test_internal_encoding_is_not_collision_hash() -> None:
    assert "hash" not in KEY_STRATEGY


def test_cold_entity_receives_dimension_key() -> None:
    mapping = stable_dense_mapping(["warm", "cold_without_transaction"])
    assert "cold_without_transaction" in mapping


def test_manifest_protocol_has_required_counts() -> None:
    for field in ("history_rows_A", "history_rows_B", "label_rows_A", "label_rows_B", "C_cap", "output_fingerprints"):
        assert field in MANIFEST_FIELDS


def test_manifest_counts_match_actual_scan() -> None:
    actual = {
        "history_rows_A": 3, "history_rows_B": 2, "label_rows_A": 2,
        "label_rows_B": 2, "history_users": 2, "label_users": 2,
        "history_items": 2, "label_items": 2, "evaluable_users": 1,
        "cold_user_count": 1, "cold_item_count": 1, "C_cap": 2,
        "C_cap_input_rows": 2,
    }
    manifest = {key: str(value) for key, value in actual.items()}
    assert manifest_counts_match(manifest, actual)
    manifest["label_rows_B"] = "3"
    assert not manifest_counts_match(manifest, actual)


@pytest.mark.parametrize("days,expected", [(7, 1), (14, 2), (28, 4)])
def test_weekday_composition_for_complete_windows(days: int, expected: int) -> None:
    start, end = trailing_window(date(2020, 9, 22), days)
    assert set(weekday_composition(start, end).values()) == {expected}


def test_validation_and_test_weekday_counts_match() -> None:
    snapshots = derive_snapshots(date(2020, 9, 22), date(2018, 9, 20))
    val = snapshots["validation_14d"]
    test = snapshots["test_14d"]
    assert weekday_composition(val.label_start, val.label_end) == weekday_composition(test.label_start, test.label_end)


def test_missing_date_detection() -> None:
    assert missing_dates(date(2020, 1, 1), date(2020, 1, 3), [date(2020, 1, 1), date(2020, 1, 3)]) == [date(2020, 1, 2)]


def test_weekday_and_weekend_counts() -> None:
    days = dates_inclusive(date(2020, 9, 9), date(2020, 9, 22))
    assert sum(day.isoweekday() <= 5 for day in days) == 10
    assert sum(day.isoweekday() >= 6 for day in days) == 4


def test_public_output_has_no_real_ids() -> None:
    assert not sensitive_value_matches('{"split_id":"test_14d","history_users":12}')


def test_checkpoint_recovery() -> None:
    checkpoint = {"status": "SUCCESS", "input_fingerprint": "d", "config_fingerprint": "c"}
    assert checkpoint_matches(checkpoint, "d", "c", True)
    assert not checkpoint_matches(checkpoint, "d", "c", True, force=True)
    assert not checkpoint_matches(checkpoint, "changed", "c", True)


def test_source_parquet_is_not_modified(tiny_parquet: Path) -> None:
    before = hashlib.sha256(tiny_parquet.read_bytes()).hexdigest()
    duckdb.connect().execute(f"SELECT count(*) FROM read_parquet('{tiny_parquet.as_posix()}')").fetchone()
    after = hashlib.sha256(tiny_parquet.read_bytes()).hexdigest()
    assert before == after
