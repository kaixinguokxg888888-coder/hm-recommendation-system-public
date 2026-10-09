from __future__ import annotations

import copy
import inspect
import json
import random
from datetime import date
from pathlib import Path

import duckdb
import pytest

from ranking_features.config import Stage4B3Config, Stage4B3ConfigError
from ranking_features.contracts import (
    LEAKAGE_DENYLIST_REQUIRED,
    SnapshotSpec,
    assert_label_free_model_schema,
    assert_projection_identity,
    category_affinity,
    category_code,
    deterministic_category_vocabulary,
    inclusive_window_start,
    recency_days,
    repeat_ratio,
    segment_from_count,
)
from ranking_features.fingerprint import canonical_value, reference_semantic_fingerprint
from ranking_features.materialization import query_digest_select
from ranking_features.runner import (
    CHECKPOINT_VERSION,
    CURRENT_SOURCE_REPRODUCTION_AUDIT_FIELDS,
    CURRENT_SOURCE_REPRODUCTION_PARTITION,
    checkpoint_is_reusable,
)
from ranking_features.schema import feature_schema_rows, model_feature_names, schema_fingerprint


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_4b3_ranking_features.yaml"


def config() -> Stage4B3Config:
    return Stage4B3Config.load(CONFIG)


def schema() -> list[dict[str, object]]:
    return feature_schema_rows(config().raw["canonical_fields"])


def test_frozen_snapshot_and_partition_contract() -> None:
    value = config()
    assert value.raw["snapshot"] == {
        "snapshot_id": "ranking_train_14d_v1",
        "as_of_date": "2020-08-11",
        "request_date": "2020-08-12",
    }
    assert value.raw["execution"]["partitions"] == 32
    assert value.raw["execution"]["query_partition_expression"] == "customer_key % 32"


def test_formal_calendar_drift_is_rejected(tmp_path: Path) -> None:
    raw = copy.deepcopy(config().raw)
    raw["snapshot"]["request_date"] = "2020-08-13"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(Stage4B3ConfigError):
        Stage4B3Config.load(path)


def test_snapshot_parameterized_builder_supports_synthetic_but_preserves_cutoff_pair() -> None:
    value = SnapshotSpec("fake", date(2020, 1, 2), date(2020, 1, 3), "h", "c", "v", 2)
    value.validate()
    with pytest.raises(ValueError):
        SnapshotSpec("fake", date(2020, 1, 2), date(2020, 1, 4), "h", "c", "v", 2).validate()


def test_inclusive_recent_window_boundaries() -> None:
    as_of = date(2020, 8, 11)
    assert inclusive_window_start(as_of, 7) == date(2020, 8, 5)
    assert inclusive_window_start(as_of, 28) == date(2020, 7, 15)
    con = duckdb.connect()
    con.execute("CREATE TABLE h(t_dat DATE)")
    con.execute("INSERT INTO h VALUES ('2020-07-14'),('2020-07-15'),('2020-08-05'),('2020-08-11'),('2020-08-12')")
    assert con.execute("SELECT count(*) FROM h WHERE t_dat BETWEEN DATE '2020-07-15' AND DATE '2020-08-11'").fetchone()[0] == 3
    assert con.execute("SELECT count(*) FROM h WHERE t_dat BETWEEN DATE '2020-08-05' AND DATE '2020-08-11'").fetchone()[0] == 2


def test_cutoff_plus_one_and_future_window_do_not_enter_dynamic_history() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE h(t_dat DATE, customer_key BIGINT, article_key BIGINT)")
    con.execute("INSERT INTO h VALUES ('2020-08-11',1,2),('2020-08-12',1,3),('2020-08-25',1,4)")
    assert con.execute("SELECT count(*) FROM h WHERE t_dat<=DATE '2020-08-11'").fetchone()[0] == 1


@pytest.mark.parametrize(
    ("count", "expected"), [(0, "cold"), (1, "short_history"), (4, "short_history"),
                            (5, "warm"), (20, "warm"), (21, "active")],
)
def test_user_segment_boundaries(count: int, expected: str) -> None:
    assert segment_from_count(count) == expected


def test_user_history_repeat_ratio_and_recency_semantics() -> None:
    assert repeat_ratio(5, 3) == pytest.approx(0.4)
    assert repeat_ratio(0, 0) is None
    assert recency_days(date(2020, 8, 11), date(2020, 8, 11)) == 0
    assert recency_days(date(2020, 8, 11), date(2020, 8, 1)) == 10
    assert recency_days(date(2020, 8, 11), None) is None


def test_item_and_user_item_zero_vs_null_reference_sql() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE h(customer_key BIGINT,article_key BIGINT,t_dat DATE)")
    con.execute("INSERT INTO h VALUES (1,2,'2020-08-11'),(1,2,'2020-08-10'),(2,2,'2020-08-05')")
    assert con.execute("SELECT count(*),count(DISTINCT customer_key),date_diff('day',max(t_dat),DATE '2020-08-11') FROM h WHERE article_key=2").fetchone() == (3, 2, 0)
    assert con.execute("SELECT count(*),date_diff('day',max(t_dat),DATE '2020-08-11') FROM h WHERE customer_key=1 AND article_key=2").fetchone() == (2, 0)
    unseen = con.execute("SELECT count(*),date_diff('day',max(t_dat),DATE '2020-08-11') FROM h WHERE customer_key=1 AND article_key=9").fetchone()
    assert unseen == (0, None)


def test_deterministic_category_encoding_missing_unknown_known_and_order() -> None:
    values = ["B", None, "A", "C"]
    expected = deterministic_category_vocabulary(values)
    random.Random(7).shuffle(values)
    assert deterministic_category_vocabulary(values) == expected == {"A": 2, "B": 3, "C": 4}
    assert category_code(None, expected) == 0
    assert category_code("future-unseen", expected) == 1
    assert category_code("A", expected) == 2
    assert "hash(" not in inspect.getsource(deterministic_category_vocabulary)


def test_category_affinity_cold_missing_and_known_zero() -> None:
    assert category_affinity(matching_interactions=3, history_interactions=6, candidate_category_code=2) == 0.5
    assert category_affinity(matching_interactions=None, history_interactions=6, candidate_category_code=2) == 0.0
    assert category_affinity(matching_interactions=0, history_interactions=0, candidate_category_code=2) is None
    assert category_affinity(matching_interactions=3, history_interactions=6, candidate_category_code=0) is None


def test_schema_contains_f0_f1_and_primary_source_is_not_eligible() -> None:
    rows = schema()
    by_name = {row["feature_name"]: row for row in rows}
    assert by_name["candidate_rank"]["model_input_eligible"] is True
    assert by_name["primary_source"]["model_input_eligible"] is False
    assert by_name["customer_key"]["model_input_eligible"] is False
    assert by_name["article_product_type_code"]["dynamic_or_static"] == "STATIC_ATTRIBUTE"
    assert by_name["user_product_type_interaction_share"]["dynamic_or_static"] == "DYNAMIC"


def test_schema_model_features_exclude_label_and_controls() -> None:
    rows = schema()
    assert_label_free_model_schema(rows, config().raw["feature_denylist"])
    assert not (set(model_feature_names(rows)) & set(config().raw["feature_denylist"]))
    assert LEAKAGE_DENYLIST_REQUIRED <= set(config().raw["feature_denylist"])


def test_leakage_denylist_regression_rejects_label() -> None:
    rows = schema()
    rows.append({"feature_name": "label", "model_input_eligible": True})
    with pytest.raises(ValueError):
        assert_label_free_model_schema(rows, config().raw["feature_denylist"])


def test_retrieval_schema_preserves_source_local_fields_without_aggregation() -> None:
    names = set(model_feature_names(schema()))
    for source in ("repurchase", "itemcf", "two_tower", "trend", "popularity"):
        assert {f"from_{source}", f"{source}_rank", f"{source}_score"} <= names
    assert not {"global_source_score", "source_score_sum", "source_score_mean"} & names


def test_projection_identity_detects_missing_extra_and_duplicate() -> None:
    rows = [("s", 1, 2), ("s", 1, 3)]
    assert_projection_identity(rows, list(reversed(rows)))
    with pytest.raises(ValueError):
        assert_projection_identity(rows, rows[:1])
    with pytest.raises(ValueError):
        assert_projection_identity(rows, [rows[0], rows[0]])


def _semantic_rows() -> list[dict[str, object]]:
    return [
        {"snapshot_id": "s", "customer_key": 1, "article_key": 10, "candidate_rank": 1,
         "ui_history_interaction_count": 0, "ui_recency_days": None, "score": 0.25},
        {"snapshot_id": "s", "customer_key": 1, "article_key": 11, "candidate_rank": 2,
         "ui_history_interaction_count": 2, "ui_recency_days": 0, "score": 0.5},
    ]


def test_feature_fingerprint_order_invariance_and_value_sensitivity() -> None:
    fields = [("snapshot_id", "string"), ("customer_key", "int64"), ("article_key", "int64"),
              ("ui_history_interaction_count", "int64"), ("ui_recency_days", "int32"), ("score", "float64")]
    rows = _semantic_rows()
    fp = reference_semantic_fingerprint(rows, fields, "a" * 64)
    assert fp == reference_semantic_fingerprint(list(reversed(rows)), fields, "a" * 64)
    changed = copy.deepcopy(rows)
    changed[0]["score"] = 0.25000000000000006
    assert fp != reference_semantic_fingerprint(changed, fields, "a" * 64)


def test_feature_fingerprint_null_differs_from_zero_and_schema_drift_changes_fp() -> None:
    fields = [("snapshot_id", "string"), ("customer_key", "int64"), ("article_key", "int64"),
              ("ui_recency_days", "int32")]
    rows = _semantic_rows()
    fp = reference_semantic_fingerprint(rows, fields, "a" * 64)
    changed = copy.deepcopy(rows)
    changed[0]["ui_recency_days"] = 0
    assert fp != reference_semantic_fingerprint(changed, fields, "a" * 64)
    assert fp != reference_semantic_fingerprint(rows, fields, "b" * 64)


def test_float_canonicalization_is_explicit_and_stable() -> None:
    assert canonical_value(0.1, "float64") == format(0.1, ".17g")
    assert canonical_value(None, "float64") == "<NULL>"
    with pytest.raises(ValueError):
        canonical_value(float("nan"), "float64")


def test_schema_fingerprint_order_invariant_and_semantic_sensitive() -> None:
    rows = schema()
    fp = schema_fingerprint(rows)
    assert fp == schema_fingerprint(list(reversed(rows)))
    changed = copy.deepcopy(rows)
    changed[0]["null_semantics"] = "changed"
    assert fp != schema_fingerprint(changed)


def test_query_digest_sql_is_partition_independent_and_excludes_partition_id() -> None:
    sql = query_digest_select(schema(), "a" * 64, source_glob="features.parquet")
    assert "partition_id" not in sql
    assert "ORDER BY candidate_rank" in sql
    assert "ranking-feature-row-left-v1" in sql
    assert "ranking-feature-row-right-v1" in sql
    assert "to_hex" in sql


def test_price_is_omitted_by_lineage_uncertainty() -> None:
    assert config().raw["price_features"]["status"] == "OMITTED_BY_LINEAGE_UNCERTAINTY"


def test_static_attribute_limitation_is_explicit() -> None:
    assert config().raw["static_attribute_policy"]["effective_timestamp_not_available"] is True


def test_private_paths_are_gitignored() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for suffix in ("private/", "tmp/", "checkpoints/"):
        assert f"artifacts/stage_4b3/{suffix}" in text


def test_config_public_paths_are_relative() -> None:
    value = config()
    assert all(not Path(path).is_absolute() for path in value.raw["paths"].values())
    assert all(not Path(path).is_absolute() for path in value.raw["upstream"].values())


def test_checkpoint_contract_requires_physical_fingerprints() -> None:
    checkpoint = config().raw["checkpoint"]
    assert checkpoint["require_input_physical_fingerprint"]
    assert checkpoint["require_output_physical_fingerprint"]


def test_failed_checkpoint_is_not_reused_and_success_can_recover() -> None:
    values = {
        "config_fingerprint": "c", "source_fingerprint": "s", "upstream_fingerprint": "u",
        "schema_fingerprint": "h", "checkpoint_version": CHECKPOINT_VERSION,
        "input_physical_fingerprint": "i", "output_physical_fingerprint": "o",
    }
    assert not checkpoint_is_reusable({"status": "FAILED", **values}, config_fingerprint="c",
                                      source_fingerprint="s", upstream_fingerprint="u",
                                      schema_fingerprint_value="h", input_fingerprint="i", output_fingerprint="o")
    assert checkpoint_is_reusable({"status": "SUCCESS", **values}, config_fingerprint="c",
                                  source_fingerprint="s", upstream_fingerprint="u",
                                  schema_fingerprint_value="h", input_fingerprint="i", output_fingerprint="o")


def test_checkpoint_fingerprint_mismatch_cannot_be_reused() -> None:
    previous = {
        "status": "SUCCESS", "config_fingerprint": "old", "source_fingerprint": "s",
        "upstream_fingerprint": "u", "schema_fingerprint": "h", "checkpoint_version": CHECKPOINT_VERSION,
        "input_physical_fingerprint": "i", "output_physical_fingerprint": "o",
    }
    assert not checkpoint_is_reusable(previous, config_fingerprint="new", source_fingerprint="s",
                                      upstream_fingerprint="u", schema_fingerprint_value="h",
                                      input_fingerprint="i", output_fingerprint="o")


def test_downstream_execution_remains_forbidden() -> None:
    assert all(value in {"NOT ENTERED", "NOT EXECUTED"}
               for value in config().raw["strictly_forbidden_downstream_execution"].values())


def test_current_source_reproduction_partition_is_in_frozen_replay_coverage() -> None:
    assert CURRENT_SOURCE_REPRODUCTION_PARTITION == 0
    assert CURRENT_SOURCE_REPRODUCTION_PARTITION in config().raw["execution"]["replay_partitions"]


def test_current_source_reproduction_audit_contract_is_safe_and_complete() -> None:
    required = {
        "producer_source_fingerprint", "current_final_source_fingerprint", "config_fingerprint",
        "upstream_fingerprint", "schema_fingerprint", "tested_partition", "formal_row_count",
        "reproduction_row_count", "row_count_mismatch", "row_key_mismatch", "schema_mismatch",
        "candidate_rank_mismatch", "candidate_provenance_mismatch", "model_feature_value_mismatch",
        "null_value_mismatch", "query_digest_mismatch", "formal_partition_semantic_fingerprint",
        "reproduction_partition_semantic_fingerprint", "partition_semantic_fingerprint_mismatch",
        "status",
    }
    assert set(CURRENT_SOURCE_REPRODUCTION_AUDIT_FIELDS) == required
    assert not ({"customer_key", "article_key", "local_path", "python_path"} & required)
