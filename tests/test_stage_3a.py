from __future__ import annotations

import json
import hashlib
from datetime import date
from pathlib import Path

import pytest

from data_audit.stage1a import validate_public_artifacts
from retrieval.config import RetrievalConfig
from retrieval.contracts import (
    CandidateContext,
    CandidateProtocolError,
    validate_candidates,
)
from retrieval.evaluation import aggregate_segments, macro_average, user_metrics
from retrieval.locking import (
    LockValidationError,
    lock_fingerprint,
    select_trend_window,
    validate_test_lock,
)
from retrieval.popularity import popularity_candidates, popularity_ranking
from retrieval.repurchase import popularity_fallback, pure_repurchase
from retrieval.trend import trend_ranking


AS_OF = date(2020, 1, 28)


def interactions() -> list[tuple[date, int, int]]:
    return [
        (date(2020, 1, 1), 1, 10),
        (date(2020, 1, 1), 1, 10),  # 人工重复，B 口径只能计一次
        (date(2020, 1, 2), 1, 10),
        (date(2020, 1, 2), 2, 11),
        (date(2020, 1, 21), 2, 12),  # 7 天窗口下界，不包含
        (date(2020, 1, 22), 2, 12),
        (date(2020, 1, 28), 3, 11),  # as_of 当日包含
        (date(2020, 1, 29), 3, 13),  # cutoff 后不得进入
    ]


def test_popularity_uses_binary_history_cutoff_tie_break_and_seen_items() -> None:
    ranking = popularity_ranking(interactions(), AS_OF, {10, 11, 12})
    assert ranking == [(10, 2), (11, 2), (12, 2)]
    assert 13 not in dict(ranking)


def test_popularity_candidate_universe_determinism_and_seen_item_retention() -> None:
    ranking = popularity_ranking(interactions(), AS_OF, {10, 11, 12})
    universe = {1: {10, 12}, 2: {11}}
    first = popularity_candidates(ranking, [1, 2], universe, 500)
    second = popularity_candidates(ranking, [1, 2], universe, 500)
    assert first == second == {1: [(10, 2), (12, 2)], 2: [(11, 2)]}
    assert first[1][0][0] == 10  # 历史已购不被过滤


@pytest.mark.parametrize("window", [7, 14, 28])
def test_trend_boundaries_for_preregistered_windows(window: int) -> None:
    ranking = trend_ranking(interactions(), AS_OF, window, {10, 11, 12})
    lower = AS_OF.toordinal() - window
    expected = {
        item
        for day, _, item in set(interactions())
        if day.toordinal() > lower and day <= AS_OF and item in {10, 11, 12}
    }
    assert {row[0] for row in ranking} == expected


def test_trend_excludes_lower_boundary_and_future_and_uses_tie_breaks() -> None:
    ranking = trend_ranking(interactions(), AS_OF, 7, {10, 11, 12, 13})
    assert ranking == [(11, 1, 2), (12, 1, 2)]


def test_validation_and_test_history_are_not_implicitly_mixed() -> None:
    validation = trend_ranking(interactions(), AS_OF, 7, {10, 11, 12, 13})
    test = trend_ranking(interactions(), date(2020, 1, 29), 7, {10, 11, 12, 13})
    assert [row[0] for row in validation] == [11, 12]
    assert 13 in [row[0] for row in test]


def test_pure_repurchase_counts_days_deduplicates_and_sorts() -> None:
    result = pure_repurchase(
        interactions(), AS_OF, [1, 2, 99],
        {1: {10, 11}, 2: {11, 12}, 99: {10, 11, 12}}, 500,
    )
    assert result[1] == [(10, 2, date(2020, 1, 2))]
    assert result[2] == [
        (12, 2, date(2020, 1, 22)),
        (11, 1, date(2020, 1, 2)),
    ]
    assert result[99] == []


def test_pure_repurchase_does_not_invent_candidates_when_short() -> None:
    result = pure_repurchase(interactions(), AS_OF, [1], {1: {10}}, 500)
    assert len(result[1]) == 1


def test_repurchase_fallback_preserves_pure_deduplicates_and_filters_universe() -> None:
    pure = {
        1: [(10, 2, date(2020, 1, 2))],
        99: [],
    }
    result = popularity_fallback(
        pure,
        [(10, 100), (11, 90), (12, 80), (13, 70)],
        {1: {10, 11, 12}, 99: {11, 13}},
        3,
    )
    assert result[1] == [
        (10, "pure", 2),
        (11, "popularity_fallback", 90),
        (12, "popularity_fallback", 80),
    ]
    assert result[99] == [
        (11, "popularity_fallback", 90),
        (13, "popularity_fallback", 70),
    ]


def candidate_context() -> CandidateContext:
    return CandidateContext(
        snapshot_id="validation_14d",
        as_of_date=AS_OF,
        source="popularity",
        source_variant="global",
        feedback_version="feedback-v1",
        population_version="population-v1",
        dataset_fingerprint="d" * 64,
        config_fingerprint="c" * 64,
    )


def candidate(rank: int, item: int, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "snapshot_id": "validation_14d",
        "as_of_date": AS_OF,
        "customer_key": 1,
        "article_key": item,
        "source": "popularity",
        "source_variant": "global",
        "source_rank": rank,
        "source_score": 100 - rank,
        "candidate_universe_version": "history_seen_items_v1",
        "feedback_version": "feedback-v1",
        "population_version": "population-v1",
        "dataset_fingerprint": "d" * 64,
        "config_fingerprint": "c" * 64,
    }
    row.update(updates)
    return row


def test_candidate_protocol_accepts_valid_contiguous_rows() -> None:
    rows = [candidate(1, 10), candidate(2, 11)]
    assert validate_candidates(rows, candidate_context(), {1: {10, 11}}) == rows


@pytest.mark.parametrize(
    "rows",
    [
        [candidate(1, 10), candidate(2, 10)],
        [candidate(1, 10), candidate(1, 11)],
        [candidate(1, 10), candidate(3, 11)],
    ],
)
def test_candidate_protocol_rejects_duplicates_and_non_contiguous_rank(
    rows: list[dict[str, object]],
) -> None:
    with pytest.raises(CandidateProtocolError):
        validate_candidates(rows, candidate_context(), {1: {10, 11}})


def test_candidate_protocol_rejects_universe_fingerprint_snapshot_and_real_ids() -> None:
    with pytest.raises(CandidateProtocolError, match="universe"):
        validate_candidates([candidate(1, 12)], candidate_context(), {1: {10}})
    with pytest.raises(CandidateProtocolError, match="dataset_fingerprint"):
        validate_candidates(
            [candidate(1, 10, dataset_fingerprint="x" * 64)],
            candidate_context(), {1: {10}},
        )
    with pytest.raises(CandidateProtocolError, match="snapshot_id"):
        validate_candidates(
            [candidate(1, 10, snapshot_id="test_14d")],
            candidate_context(), {1: {10}},
        )
    with pytest.raises(CandidateProtocolError, match="禁止字段"):
        validate_candidates(
            [candidate(1, 10, customer_id="raw")],
            candidate_context(), {1: {10}},
        )


def test_evaluator_complete_partial_miss_and_multiple_ground_truth() -> None:
    complete = user_metrics([10, 11], {10, 11})
    partial = user_metrics([10, 12], {10, 11})
    miss = user_metrics([12], {10, 11})
    assert complete.recall[12] == 1.0 and complete.hit_rate[12] == 1.0
    assert partial.recall[12] == 0.5 and partial.precision_at_12 == 0.5
    assert miss.recall[12] == 0.0 and miss.hit_rate[12] == 0.0


def test_evaluator_short_candidates_recall_precision_ap_and_map() -> None:
    first = user_metrics([10, 99, 11], {10, 11, 12})
    second = user_metrics([12], {12})
    assert first.recall[12] == pytest.approx(2 / 3)
    assert first.precision_at_12 == pytest.approx(2 / 3)
    assert first.ap_at_12 == pytest.approx((1 + 2 / 3) / 3)
    aggregate = macro_average([first, second])
    assert aggregate["map_at_12"] == pytest.approx((first.ap_at_12 + 1) / 2)


def test_evaluator_rejects_duplicate_and_empty_ground_truth() -> None:
    with pytest.raises(ValueError, match="重复"):
        user_metrics([10, 10], {10})
    with pytest.raises(ValueError, match="空 ground truth"):
        user_metrics([10], set())


def test_segment_aggregation_excludes_cold_from_overall() -> None:
    metrics = {
        1: user_metrics([10], {10}),
        2: user_metrics([99], {10}),
        3: user_metrics([10], {10}),
        4: user_metrics([99], {10}),
    }
    segments = {1: "short_history", 2: "warm", 3: "active", 4: "cold"}
    result = aggregate_segments(metrics, segments, {1, 2, 3})
    assert result["evaluable_warm_users"]["recall_at_12"] == pytest.approx(2 / 3)
    assert result["cold_history_label_users"]["recall_at_12"] == 0.0


def trend_rows() -> list[dict[str, object]]:
    return [
        {"window_days": 7, "recall_at_500": 0.2, "map_at_12": 0.1, "candidate_user_coverage": 1.0},
        {"window_days": 14, "recall_at_500": 0.2, "map_at_12": 0.1, "candidate_user_coverage": 1.0},
        {"window_days": 28, "recall_at_500": 0.1, "map_at_12": 0.2, "candidate_user_coverage": 1.0},
    ]


def test_trend_selection_uses_registered_tie_break_and_shorter_window() -> None:
    assert select_trend_window(trend_rows()) == 7


def write_lock(path: Path, selection: Path) -> dict[str, object]:
    import hashlib

    payload: dict[str, object] = {
        "config_fingerprint": "c",
        "dataset_fingerprint": "d",
        "source_fingerprint": "s",
        "locked_trend_window": 7,
        "validation_selection_fingerprint": hashlib.sha256(
            selection.read_bytes()
        ).hexdigest(),
    }
    payload["lock_fingerprint"] = lock_fingerprint(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_test_gate_requires_lock_and_matching_fingerprints(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    selection = tmp_path / "selection.csv"
    selection.write_text("window\n7\n", encoding="utf-8")
    with pytest.raises(LockValidationError, match="不存在"):
        validate_test_lock(
            lock, config_fingerprint="c", dataset_fingerprint="d",
            source_fingerprint="s", allowed_windows=[7, 14, 28],
        )
    write_lock(lock, selection)
    value = validate_test_lock(
        lock, config_fingerprint="c", dataset_fingerprint="d",
        source_fingerprint="s", allowed_windows=[7, 14, 28],
        expected_selection_path=selection,
    )
    assert value["locked_trend_window"] == 7
    with pytest.raises(LockValidationError, match="config_fingerprint"):
        validate_test_lock(
            lock, config_fingerprint="changed", dataset_fingerprint="d",
            source_fingerprint="s", allowed_windows=[7, 14, 28],
        )


def test_test_gate_detects_selection_or_source_change(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    selection = tmp_path / "selection.csv"
    selection.write_text("window\n7\n", encoding="utf-8")
    write_lock(lock, selection)
    selection.write_text("window\n14\n", encoding="utf-8")
    with pytest.raises(LockValidationError, match="selection"):
        validate_test_lock(
            lock, config_fingerprint="c", dataset_fingerprint="d",
            source_fingerprint="s", allowed_windows=[7, 14, 28],
            expected_selection_path=selection,
        )
    with pytest.raises(LockValidationError, match="source_fingerprint"):
        validate_test_lock(
            lock, config_fingerprint="c", dataset_fingerprint="d",
            source_fingerprint="changed", allowed_windows=[7, 14, 28],
        )


def test_stage_3a_config_has_no_magic_parameter_drift() -> None:
    config = RetrievalConfig.load(Path("configs/stage_3a_recall.yaml"))
    assert config.raw["candidate_max_k"] == 500
    assert config.raw["evaluation_k"] == [12, 50, 100, 500]
    assert config.raw["test"]["locked_trend_window_source"] == "lock_artifact"


def test_stage_3a_private_and_tmp_are_git_ignored() -> None:
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "artifacts/stage_3a/private/" in ignore
    assert "artifacts/stage_3a/tmp/" in ignore


def test_public_schema_does_not_allow_entity_level_fields() -> None:
    public_fields = {
        "stage", "snapshot_id", "source", "source_variant", "metric",
        "metric_value", "segment", "population_denominator",
    }
    assert not ({"customer_key", "article_key", "customer_id", "article_id"} & public_fields)


def test_existing_stage_3a_public_artifacts_pass_formal_sensitive_scan() -> None:
    root = Path(".")
    public = [
        *(root / "artifacts" / "stage_3a").glob("*.csv"),
        *(root / "artifacts" / "stage_3a").glob("*.json"),
        root / "docs" / "stages" / "stage_3a_baseline_retrieval_and_evaluation.md",
        root / "README.md",
    ]
    validate_public_artifacts(public)


def test_stage_3a_population_denominators_match_stage_2b_manifest() -> None:
    import csv

    def read(path: str) -> list[dict[str, str]]:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    manifest = read("artifacts/stage_2b/evaluation_population_manifest.csv")
    counts = {
        (row["snapshot_id"], row["population"]): int(row["user_count"])
        for row in manifest
        if row["population"] in {
            "all_label_users",
            "evaluable_warm_users",
            "cold_history_label_users",
        }
    }
    assert counts == {
        ("validation_14d", "all_label_users"): 141_359,
        ("validation_14d", "evaluable_warm_users"): 129_705,
        ("validation_14d", "cold_history_label_users"): 11_654,
        ("test_14d", "all_label_users"): 128_333,
        ("test_14d", "evaluable_warm_users"): 117_366,
        ("test_14d", "cold_history_label_users"): 10_967,
    }
    assert counts[("validation_14d", "all_label_users")] == (
        counts[("validation_14d", "evaluable_warm_users")]
        + counts[("validation_14d", "cold_history_label_users")]
    )
    assert counts[("test_14d", "all_label_users")] == (
        counts[("test_14d", "evaluable_warm_users")]
        + counts[("test_14d", "cold_history_label_users")]
    )

    validation_overall = read("artifacts/stage_3a/validation_metrics_overall.csv")
    test_overall = read("artifacts/stage_3a/test_metrics_overall.csv")
    validation_cold = read("artifacts/stage_3a/validation_metrics_by_segment.csv")
    test_cold = read("artifacts/stage_3a/test_metrics_by_segment.csv")
    assert {int(row["population_denominator"]) for row in validation_overall} == {129_705}
    assert {int(row["population_denominator"]) for row in test_overall} == {117_366}
    assert {
        int(row["population_denominator"])
        for row in validation_cold
        if row["segment"] == "cold_history_label_users"
    } == {11_654}
    assert {
        int(row["population_denominator"])
        for row in test_cold
        if row["segment"] == "cold_history_label_users"
    } == {10_967}


def test_original_stage_3a_lock_and_test_metric_files_are_frozen() -> None:
    expected = {
        "artifacts/stage_3a/stage_3a_lock.json":
            "c8c8f233f05001a1d600433068c1874a875f9ec47aa7d7b8d612e51cfbe2d134",
        "artifacts/stage_3a/test_metrics_overall.csv":
            "ecfa40762661c7afe0f159ac9902e284936895e2e892009722c46cd7848a5b6e",
        "artifacts/stage_3a/test_metrics_by_segment.csv":
            "d36d8b3122df62bcd553140ea5cbe892ae40a46199eaf07dce7111ca3b17276b",
        "artifacts/stage_3a/test_candidate_coverage.csv":
            "d18ba14aa5c163ea979952b7c003cb6c09fdea2fa66bb2c2d6a519f9d4f064c8",
    }
    for raw_path, fingerprint in expected.items():
        assert hashlib.sha256(Path(raw_path).read_bytes()).hexdigest() == fingerprint
