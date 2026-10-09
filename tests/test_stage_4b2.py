from __future__ import annotations

import hashlib
import inspect
import random
from pathlib import Path

import duckdb
import pytest

from ranking_dataset.config import Stage4B2Config, Stage4B2ConfigError
from ranking_dataset.fingerprint import semantic_fingerprint
from ranking_dataset.runner import optimizer_group_integrity
from ranking_dataset.semantics import (
    canonical_sampling_payload,
    classify_query,
    deterministic_sample_rows,
    label_candidates,
    sampling_digest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_4b2_ranking_dataset.yaml"


def row(article: int, rank: int, label: int, customer: int = 7) -> dict[str, object]:
    return {
        "snapshot_id": "ranking_train_14d_v1", "customer_key": customer,
        "article_key": article, "candidate_rank": rank, "label": label,
        "primary_source": "itemcf", "itemcf_rank": rank, "itemcf_score": float(article),
    }


def test_frozen_calendar_and_window() -> None:
    config = Stage4B2Config.load(CONFIG)
    assert config.raw["snapshot"] == {
        "snapshot_id": "ranking_train_14d_v1", "as_of_date": "2020-08-11",
        "request_date": "2020-08-12", "label_start": "2020-08-12",
        "label_end": "2020-08-25", "label_window_days": 14,
    }


def test_calendar_drift_is_rejected(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace('"label_end": "2020-08-25"', '"label_end": "2020-08-26"')
    path = tmp_path / "stage_4b2.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(Stage4B2ConfigError):
        Stage4B2Config.load(path)


def test_ground_truth_dedup_multiple_days_is_one_positive() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE f(t_dat DATE, customer_key BIGINT, article_key BIGINT)")
    con.execute("INSERT INTO f VALUES ('2020-08-12',1,2),('2020-08-13',1,2),('2020-08-13',1,3)")
    assert con.execute("SELECT count(*) FROM (SELECT customer_key,article_key FROM f GROUP BY ALL)").fetchone()[0] == 2


def test_label_join_annotates_without_candidate_drift() -> None:
    candidates = [row(10, 1, 0), row(11, 2, 0)]
    before = [{key: value for key, value in item.items() if key != "label"} for item in candidates]
    labeled = label_candidates(candidates, {("ranking_train_14d_v1", 7, 11)})
    after = [{key: value for key, value in item.items() if key != "label"} for item in labeled]
    assert before == after
    assert [item["label"] for item in labeled] == [0, 1]
    assert [item["candidate_rank"] for item in labeled] == [1, 2]
    assert labeled[1]["itemcf_score"] == 11.0


@pytest.mark.parametrize(
    ("history", "segment"), [(1, "short_history"), (5, "warm"), (21, "active")],
)
def test_main_population_includes_every_non_cold_segment(history: int, segment: str) -> None:
    del segment
    value = classify_query(history, 1, 1)
    assert value["is_historical_evaluable_warm"]
    assert value["optimizer_eligible"]


def test_cold_history_isolated_from_main_and_optimizer() -> None:
    value = classify_query(0, 1, 1)
    assert value["is_cold_history_label_user"]
    assert not value["is_historical_evaluable_warm"]
    assert not value["optimizer_eligible"]


def test_retrieval_miss_retained_full_but_excluded_optimizer() -> None:
    value = classify_query(3, 2, 0)
    assert value["is_retrieval_miss"] and value["is_all_zero_relevance_group"]
    assert value["full_main_evaluation_eligible"]
    assert not value["optimizer_eligible"]


def test_sampling_golden_vector_and_duckdb_identity() -> None:
    payload = canonical_sampling_payload("ranking_train_14d_v1", 7, 42, 20260811)
    assert payload == "ranking_train_14d_v1|7|42|20260811"
    expected = "793181309256e5678ed3118e3db96a83f5cf6f536583012aecb6049f95e488dd"
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == expected
    assert sampling_digest("ranking_train_14d_v1", 7, 42, 20260811) == expected
    assert duckdb.connect().execute("SELECT sha256(?)", [payload]).fetchone()[0] == expected


@pytest.mark.parametrize(("zero_count", "expected_zero"), [(17, 17), (100, 100), (130, 100)])
def test_sampling_zero_cap_all_positives_and_rank_preservation(zero_count: int, expected_zero: int) -> None:
    rows = [row(9000 + i, i + 1, 0) for i in range(zero_count)]
    rows += [row(100, zero_count + 1, 1), row(101, zero_count + 2, 1)]
    sampled = deterministic_sample_rows(rows)
    assert sum(item["label"] == 0 for item in sampled) == expected_zero
    assert {item["article_key"] for item in sampled if item["label"] == 1} == {100, 101}
    original_ranks = {item["article_key"]: item["candidate_rank"] for item in rows}
    assert all(item["candidate_rank"] == original_ranks[item["article_key"]] for item in sampled)


def test_sampling_order_and_partition_invariant_and_no_builtin_hash() -> None:
    rows = [row(i, i, int(i in {3, 9})) for i in range(1, 151)]
    expected = {(item["article_key"], item["candidate_rank"]) for item in deterministic_sample_rows(rows)}
    random.Random(123).shuffle(rows)
    actual = {(item["article_key"], item["candidate_rank"]) for item in deterministic_sample_rows(rows)}
    assert expected == actual
    assert "hash(" not in inspect.getsource(deterministic_sample_rows).replace("sampling_digest(", "")


def test_digest_sort_tie_breaks_by_article_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ranking_dataset.semantics.sampling_digest", lambda *args: "same")
    sampled = deterministic_sample_rows([row(3, 1, 0), row(1, 2, 0), row(2, 3, 0)], zero_cap=2)
    assert {item["article_key"] for item in sampled} == {1, 2}


def _write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)


def test_semantic_fingerprint_order_invariant_and_label_sensitive(tmp_path: Path) -> None:
    rows = [row(10, 1, 0), row(11, 2, 1)]
    left, right, changed = tmp_path / "left.parquet", tmp_path / "right.parquet", tmp_path / "changed.parquet"
    _write_parquet(left, rows)
    _write_parquet(right, list(reversed(rows)))
    altered = [dict(item) for item in rows]
    altered[0]["label"] = 1
    _write_parquet(changed, altered)
    assert semantic_fingerprint([left], optimizer=False) == semantic_fingerprint([right], optimizer=False)
    assert semantic_fingerprint([left], optimizer=False) != semantic_fingerprint([changed], optimizer=False)


def test_candidate_membership_or_rank_changes_projection_fingerprint(tmp_path: Path) -> None:
    from retrieval.stage3e.contract import semantic_fingerprint_from_partitions

    base = [row(10, 1, 0), row(11, 2, 0)]
    left, rank_changed, membership_changed = (tmp_path / name for name in ("a.parquet", "b.parquet", "c.parquet"))
    _write_parquet(left, base)
    rank_rows = [dict(item) for item in base]
    rank_rows[0]["candidate_rank"], rank_rows[1]["candidate_rank"] = 2, 1
    rank_rows.sort(key=lambda item: item["candidate_rank"])
    _write_parquet(rank_changed, rank_rows)
    member_rows = [dict(item) for item in base]
    member_rows[1]["article_key"] = 12
    _write_parquet(membership_changed, member_rows)
    fp = semantic_fingerprint_from_partitions([left])[0]
    assert semantic_fingerprint_from_partitions([rank_changed])[0] != fp
    assert semantic_fingerprint_from_partitions([membership_changed])[0] != fp


def test_private_paths_are_gitignored() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "artifacts/stage_4b2/private/" in text
    assert "artifacts/stage_4b2/tmp/" in text
    assert "artifacts/stage_4b2/checkpoints/" in text


def test_config_public_paths_are_relative_and_no_entity_values() -> None:
    config = Stage4B2Config.load(CONFIG)
    assert all(not Path(path).is_absolute() for path in config.raw["paths"].values())
    assert config.raw["keys"]["row"] == ["snapshot_id", "customer_key", "article_key"]


def test_materialized_public_artifacts_have_no_raw_ids_or_absolute_local_paths() -> None:
    public = ROOT / "artifacts/stage_4b2"
    if not public.exists():
        pytest.skip("formal public artifacts are not materialized in a clean unit-test checkout")
    for path in public.glob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        assert "customer_id" not in text
        assert "article_id" not in text
        assert "postal_code" not in text
        assert ":\\" not in text
        assert "D:/" not in text


def test_optimizer_group_integrity_uses_query_level_mean(tmp_path: Path) -> None:
    sampled = tmp_path / "sampled.parquet"
    groups = tmp_path / "groups.parquet"
    rows = [row(10, 1, 1, 7), row(11, 2, 0, 7), row(20, 1, 1, 8), row(21, 2, 1, 8), row(22, 3, 0, 8)]
    _write_parquet(sampled, rows)
    _write_parquet(groups, [
        {"snapshot_id": "ranking_train_14d_v1", "customer_key": 7, "group_size": 2,
         "positive_count": 1, "sampled_zero_count": 1},
        {"snapshot_id": "ranking_train_14d_v1", "customer_key": 8, "group_size": 3,
         "positive_count": 2, "sampled_zero_count": 1},
    ])
    con = duckdb.connect()
    sampled_summary, group_summary, mismatch = optimizer_group_integrity(
        con, sampled.as_posix(), groups.as_posix()
    )
    assert sampled_summary[:4] == (2, 5, 3, 2)
    assert sampled_summary[4:7] == (2, 3, 2.5)
    assert group_summary[:5] == (2, 5, 2, 3, 2.5)
    assert mismatch == (0, 0, 0, 0)
