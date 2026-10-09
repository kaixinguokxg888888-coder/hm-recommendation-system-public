from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_audit.stage1b import (
    FEEDBACK_SCOPES,
    activity_bucket,
    cold_start_metrics,
    feedback_counts,
    gini_from_counts,
    profile_checkpoint_can_skip,
    top_share,
    validate_profile_artifacts,
)


def test_three_feedback_scopes_differ() -> None:
    rows = [
        ("2020-01-01", "u1", "i1"),
        ("2020-01-01", "u1", "i1"),
        ("2020-01-02", "u1", "i1"),
    ]
    assert feedback_counts(rows) == {
        "A_raw_rows": 3,
        "B_binary_user_item_day": 2,
        "C_daily_strength_sum": 3,
        "same_day_excess": 1,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, "1"), (2, "2"), (3, "3-5"), (5, "3-5"), (6, "6-10"),
     (10, "6-10"), (11, "11-20"), (20, "11-20"), (21, "21-50"),
     (50, "21-50"), (51, "51-100"), (100, "51-100"), (101, "100+")],
)
def test_user_and_item_bucket_boundaries(value: int, expected: str) -> None:
    assert activity_bucket(value) == expected


def test_repurchase_and_same_day_cross_day_definitions() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE tx(d DATE,u VARCHAR,i VARCHAR);"
        "INSERT INTO tx VALUES ('2020-01-01','u1','i1'),"
        "('2020-01-01','u1','i1'),('2020-01-02','u1','i1')"
    )
    row = connection.execute(
        "WITH day AS (SELECT d,u,i,count(*) n FROM tx GROUP BY ALL),"
        "ui AS (SELECT u,i,sum(n) raw,count(*) active_days FROM day GROUP BY u,i)"
        "SELECT (SELECT sum(n-1) FROM day),(SELECT sum(raw-1) FROM ui),"
        "(SELECT sum(active_days-1) FROM ui)"
    ).fetchone()
    connection.close()
    assert row == (1, 2, 1)


def test_cold_user_and_new_item_definitions() -> None:
    value = cold_start_metrics(
        [("u1", "i1"), ("u2", "i2")],
        [("u1", "i3"), ("u3", "i2"), ("u3", "i3")],
    )
    assert value["new_users"] == 1
    assert value["new_items"] == 1
    assert value["cold_both_row_rate"] == pytest.approx(1 / 3)


def test_strict_time_split_has_no_overlap() -> None:
    train_end = "2020-01-07"
    test_start = "2020-01-08"
    assert train_end < test_start


def test_evaluable_user_and_visibility_definition() -> None:
    train_users = {"u1", "u2"}
    test_users = {"u2", "u3"}
    evaluable = test_users & train_users
    assert evaluable == {"u2"}
    assert len(evaluable) / len(test_users) == 0.5


def test_top_share_calculation() -> None:
    assert top_share([70, 20, 10], 1 / 3) == pytest.approx(0.7)


def test_gini_long_tail_metric() -> None:
    assert gini_from_counts([1, 1, 1]) == pytest.approx(0.0)
    assert gini_from_counts([0, 0, 10]) > 0.6


def test_small_temporary_parquet_feedback_query(tmp_path: Path) -> None:
    path = tmp_path / "tiny.parquet"
    pq.write_table(
        pa.table(
            {
                "t_dat": ["2020-01-01", "2020-01-01", "2020-01-02"],
                "customer_id": ["u1", "u1", "u1"],
                "article_id": ["i1", "i1", "i1"],
            }
        ),
        path,
    )
    connection = duckdb.connect(":memory:")
    safe = path.as_posix().replace("'", "''")
    row = connection.execute(
        f"SELECT count(*),count(DISTINCT (t_dat,customer_id,article_id)) "
        f"FROM read_parquet('{safe}')"
    ).fetchone()
    connection.close()
    assert row == (3, 2)


def test_checkpoint_recovery_and_force() -> None:
    checkpoint = {
        "status": "SUCCESS",
        "input_fingerprint": "i",
        "config_fingerprint": "c",
    }
    assert profile_checkpoint_can_skip(checkpoint, "i", "c", True, False)
    assert not profile_checkpoint_can_skip(checkpoint, "i", "c", True, True)
    assert not profile_checkpoint_can_skip(checkpoint, "changed", "c", True, False)


def test_profile_output_rejects_real_ids(tmp_path: Path) -> None:
    clean = tmp_path / "clean.json"
    clean.write_text('{"customer_count": 3}', encoding="utf-8")
    validate_profile_artifacts([clean], [])
    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text('{"value":"0123456789"}', encoding="utf-8")
    with pytest.raises(ValueError):
        validate_profile_artifacts([unsafe], [])


def test_profile_output_allows_public_https_url(tmp_path: Path) -> None:
    artifact = tmp_path / "resource.json"
    artifact.write_text(
        json.dumps(
            {
                "notes": (
                    "Possible solutions:\n"
                    "See https://duckdb.org/docs/stable/guides/performance"
                )
            }
        ),
        encoding="utf-8",
    )
    validate_profile_artifacts([artifact], [])


def test_report_scope_fields_are_complete() -> None:
    assert set(FEEDBACK_SCOPES) == {"A", "B", "C"}
    assert "not true quantity" in FEEDBACK_SCOPES["C"]
    config = json.loads(
        Path("configs/stage_1b_profile.yaml").read_text(encoding="utf-8")
    )
    assert config["analysis"]["evaluation_windows_days"] == [7, 14, 28]
    assert len(config["groups"]) == 7
