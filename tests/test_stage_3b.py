from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import pytest
import duckdb

from retrieval.config import source_fingerprint as stage3a_source_fingerprint
from retrieval.stage3b.bootstrap import paired_user_bootstrap
from retrieval.stage3b.candidates import generate_itemcf_candidates, select_seed_items
from retrieval.stage3b.config import Stage3BConfig
from retrieval.stage3b.itemcf import (
    Neighbor,
    build_neighbors,
    build_user_item_sets,
    cooccurrence_pairs,
    item_support,
    select_variant,
)
from retrieval.stage3b.locking import (
    Stage3BLockError,
    lock_fingerprint,
    validate_test_lock,
)
from retrieval.stage3b.overlap import aggregate_change_classes, jaccard, user_overlap
from retrieval.stage3b.runner import (
    COMPATIBLE_CHECKPOINT_SOURCE_FINGERPRINTS,
    declared_stage3b_lock_fingerprints,
    public_error_message,
    query_neighbor_statistics,
)
from retrieval.stage3b.union import fixed_budget_union


AS_OF = date(2020, 1, 31)


def test_user_item_sets_deduplicate_cutoff_cap_and_stable_tie_break() -> None:
    rows = [(AS_OF - timedelta(days=i // 2), 1, 100 + i) for i in range(55)]
    rows += [(AS_OF, 1, 100), (AS_OF + timedelta(days=1), 1, 999)]
    result = build_user_item_sets(rows, AS_OF, 50)[1]
    assert len(result) == 50
    assert len({item for item, _ in result}) == 50
    assert 999 not in {item for item, _ in result}
    assert result[:2] == [(100, AS_OF), (101, AS_OF)]


def test_cosine_pairs_support_symmetry_self_exclusion_and_determinism() -> None:
    rows = [
        (date(2020, 1, 1), 1, 10),
        (date(2020, 1, 1), 1, 10),
        (date(2020, 1, 2), 1, 11),
        (date(2020, 1, 2), 2, 10),
        (date(2020, 1, 3), 2, 11),
        (date(2020, 2, 1), 3, 12),
    ]
    sets = build_user_item_sets(rows, AS_OF)
    pairs = cooccurrence_pairs(sets)
    support = item_support(sets)
    first = build_neighbors(pairs, support, "cosine", 200)
    second = build_neighbors(pairs, support, "cosine", 200)
    assert pairs[(10, 11)].raw_cooccurrence == 2
    assert support == {10: 2, 11: 2}
    assert first == second
    assert first[10][0].article_key == 11 and first[11][0].article_key == 10
    assert first[10][0].similarity == 1.0
    assert all(item != n.article_key for item, values in first.items() for n in values)


def test_iuf_uses_unique_truncated_set_length_and_downweights_active_user() -> None:
    short = [(date(2020, 1, 1), 1, 10), (date(2020, 1, 1), 1, 11)]
    active = [(date(2020, 1, 1), 2, item) for item in range(10, 20)]
    active.append((date(2020, 1, 1), 2, 10))
    pairs = cooccurrence_pairs(build_user_item_sets(short + active, AS_OF))
    expected = 1 / math.log(3) + 1 / math.log(11)
    assert pairs[(10, 11)].iuf_cooccurrence == pytest.approx(expected)
    assert 1 / math.log(11) < 1 / math.log(3)


def test_neighbor_sort_top_n_zero_support_guard() -> None:
    rows = [
        (date(2020, 1, 1), 1, 1),
        (date(2020, 1, 1), 1, 2),
        (date(2020, 1, 1), 1, 3),
    ]
    sets = build_user_item_sets(rows, AS_OF)
    neighbors = build_neighbors(cooccurrence_pairs(sets), item_support(sets), "iuf", 1)
    assert all(len(values) == 1 and values[0].neighbor_rank == 1 for values in neighbors.values())


def test_seed_selection_decay_candidate_aggregation_seen_universe_and_shortfall() -> None:
    history = [(AS_OF - timedelta(days=i // 2), 1, 100 + i) for i in range(25)]
    seeds = select_seed_items(history, AS_OF, 20)
    assert len(seeds[1]) == 20
    assert seeds[1][0][0] == 100 and seeds[1][1][0] == 101
    assert seeds[1][0][3] == 1.0
    neighbors = {
        100: [Neighbor(200, 0.5, 1.0, 1, 1), Neighbor(100, 0.9, 1.0, 1, 2)],
        101: [Neighbor(200, 0.25, 1.0, 1, 1), Neighbor(999, 1.0, 1.0, 1, 2)],
    }
    result = generate_itemcf_candidates(
        seeds, neighbors, {1: {100, 200}, 2: {200}}, {1: {100}, 2: set()}, 500
    )
    assert len(result[1]) == 1
    assert result[1][0]["article_key"] == 200
    assert result[1][0]["seed_hit_count"] == 2
    assert result[1][0]["source_rank"] == 1
    assert result[2] == []


def rows(items: list[int], score: float = 1.0) -> list[dict[str, object]]:
    return [
        {"article_key": item, "source_rank": rank, "source_score": score / rank}
        for rank, item in enumerate(items, 1)
    ]


def test_union_fixed_budget_deduplicates_preserves_provenance_and_does_not_sum_scores() -> None:
    union = fixed_budget_union(
        {
            "repurchase": rows([1, 2]),
            "itemcf": rows([2, 3, 4]),
            "trend": rows([4, 5]),
            "popularity_fallback": rows([1, 2, 3, 4, 5, 6, 7]),
        },
        {"repurchase": 2, "itemcf": 3, "trend": 2},
        6,
    )
    assert [row["article_key"] for row in union] == [1, 2, 3, 4, 5, 6]
    assert [row["source_rank"] for row in union] == list(range(1, 7))
    duplicate = union[1]
    assert duplicate["from_repurchase"] and duplicate["from_itemcf"]
    assert duplicate["repurchase_score"] != duplicate["itemcf_score"]
    assert duplicate["source_count"] >= 2


def test_base_and_itemcf_union_budget_and_cold_degradation() -> None:
    pop = rows(list(range(1000)))
    base = fixed_budget_union(
        {"repurchase": [], "itemcf": [], "trend": rows(list(range(100))), "popularity_fallback": pop},
        {"repurchase": 50, "itemcf": 0, "trend": 100},
        500,
    )
    itemcf = fixed_budget_union(
        {"repurchase": [], "itemcf": [], "trend": rows(list(range(100))), "popularity_fallback": pop},
        {"repurchase": 50, "itemcf": 300, "trend": 100},
        500,
    )
    assert len(base) == len(itemcf) == 500
    assert base == itemcf


def test_overlap_exclusive_shared_delta_change_and_cold_exclusion_aggregation() -> None:
    result = user_overlap(
        [3, 4], [1, 3], [2], [1, 5], [1, 2, 5], [1, 2, 3, 4], {2, 3, 4}
    )
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert result["itemcf_exclusive_hits"] == 1
    assert result["itemcf_shared_hits"] == 1
    assert result["delta_recall_at_500"] == pytest.approx(2 / 3)
    assert result["change_class"] == "improved"
    assert aggregate_change_classes([result, {"change_class": "declined"}]) == {
        "improved": 1, "unchanged": 0, "declined": 1
    }


def test_bootstrap_is_paired_reproducible_sign_correct_and_safe_empty() -> None:
    first = paired_user_bootstrap([0, 0, 1], [1, 0, 1], iterations=1000, seed=7)
    second = paired_user_bootstrap([0, 0, 1], [1, 0, 1], iterations=1000, seed=7)
    assert first == second
    assert first["delta_mean"] == pytest.approx(1 / 3)
    assert first["ci_lower"] <= first["delta_mean"] <= first["ci_upper"]
    empty = paired_user_bootstrap([], [], iterations=1000, seed=7)
    assert empty["user_count"] == 0 and math.isnan(float(empty["delta_mean"]))


def test_variant_selection_preregistered_order_and_cosine_exact_tie() -> None:
    values = [
        {"variant": "cosine", "recall_at_500": 0.1, "hit_rate_at_500": 0.2, "map_at_12": 0.01, "candidate_user_coverage": 1.0},
        {"variant": "iuf", "recall_at_500": 0.1, "hit_rate_at_500": 0.2, "map_at_12": 0.01, "candidate_user_coverage": 1.0},
    ]
    assert select_variant(values) == "cosine"
    values[1]["hit_rate_at_500"] = 0.3
    assert select_variant(values) == "iuf"


def test_stage3b_config_and_ignore_frozen_values() -> None:
    config = Stage3BConfig.load(Path("configs/stage_3b_itemcf.yaml"))
    assert config.raw["itemcf"]["top_neighbors_per_item"] == 200
    assert config.raw["union"]["itemcf_quotas"]["itemcf"] == 300
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "artifacts/stage_3b/private/" in ignore
    assert "artifacts/stage_3b/tmp/" in ignore


def test_adding_stage3b_subpackage_does_not_change_frozen_stage3a_source_fingerprint() -> None:
    assert stage3a_source_fingerprint(Path(".")) == (
        "86646f16fb94701b20fde34430f6d837a83d65b95de1688e7e4791f0bde3bcf8"
    )


def test_completed_test_source_is_compatible_only_for_finalize_fix() -> None:
    reason = COMPATIBLE_CHECKPOINT_SOURCE_FINGERPRINTS[
        "e5304699d8ee4d0bb25348400cfd5ba5973c477d58916a9dd3e35d3f26f8a84f"
    ]
    assert "finalize-only" in reason


def test_stage3b_lock_fingerprint_allowlist_is_field_aware_and_urls_are_sanitized() -> None:
    safe = "a" * 64
    unsafe = "b" * 64
    payload = {
        "validation_metrics_fingerprints": {"metrics.csv": safe},
        "customer_value": unsafe,
    }
    assert declared_stage3b_lock_fingerprints(payload) == {safe}
    assert "https://" not in public_error_message("failed; see https://example.test/path")


def test_neighbor_statistics_distinguishes_rows_from_items(tmp_path: Path) -> None:
    path = tmp_path / "neighbors.parquet"
    con = duckdb.connect()
    try:
        con.execute(
            f"""
            COPY (
              SELECT * FROM (VALUES
                (1::BIGINT, 2::BIGINT, 0.5::DOUBLE, 2::BIGINT),
                (1::BIGINT, 3::BIGINT, 0.4::DOUBLE, 1::BIGINT),
                (2::BIGINT, 1::BIGINT, 0.5::DOUBLE, 2::BIGINT)
              ) AS t(article_key,neighbor_article_key,similarity,raw_cooccurrence)
            ) TO '{path.as_posix()}' (FORMAT PARQUET)
            """
        )
        stats = query_neighbor_statistics(con, path)
    finally:
        con.close()
    assert stats[0] == 3
    assert stats[1] == 2
    assert stats[2] == pytest.approx(1.5)


def test_stage3b_test_lock_rejects_missing_fingerprint_variant_parameter_and_quota_changes(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.csv"
    selection.write_text("variant,selected\ncosine,true\n", encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    with pytest.raises(Stage3BLockError, match="不存在"):
        validate_test_lock(
            lock_path,
            config_fingerprint="c",
            source_fingerprint="s",
            stage_3a_lock_fingerprint="a",
            selection_path=selection,
        )
    parameters = {"max_seed_items": 20, "top_neighbors": 200}
    quotas = {"itemcf": 300}
    payload = {
        "config_fingerprint": "c",
        "source_fingerprint": "s",
        "stage_3a_lock_fingerprint": "a",
        "validation_selection_fingerprint": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "locked_itemcf_variant": "cosine",
        "itemcf_parameters": parameters,
        "union_quotas": quotas,
    }
    payload["lock_fingerprint"] = lock_fingerprint(payload)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    validate_test_lock(
        lock_path,
        config_fingerprint="c",
        source_fingerprint="s",
        stage_3a_lock_fingerprint="a",
        selection_path=selection,
        expected_variant="cosine",
        expected_parameters=parameters,
        expected_quotas=quotas,
    )
    for field, value, match in (
        ("config_fingerprint", "changed", "config_fingerprint"),
        ("source_fingerprint", "changed", "source_fingerprint"),
        ("stage_3a_lock_fingerprint", "changed", "stage_3a_lock_fingerprint"),
    ):
        kwargs = {
            "config_fingerprint": "c",
            "source_fingerprint": "s",
            "stage_3a_lock_fingerprint": "a",
        }
        kwargs[field] = value
        with pytest.raises(Stage3BLockError, match=match):
            validate_test_lock(lock_path, selection_path=selection, **kwargs)
    with pytest.raises(Stage3BLockError, match="variant"):
        validate_test_lock(
            lock_path, config_fingerprint="c", source_fingerprint="s",
            stage_3a_lock_fingerprint="a", selection_path=selection,
            expected_variant="iuf",
        )
    with pytest.raises(Stage3BLockError, match="参数"):
        validate_test_lock(
            lock_path, config_fingerprint="c", source_fingerprint="s",
            stage_3a_lock_fingerprint="a", selection_path=selection,
            expected_parameters={"max_seed_items": 19},
        )
    with pytest.raises(Stage3BLockError, match="quota"):
        validate_test_lock(
            lock_path, config_fingerprint="c", source_fingerprint="s",
            stage_3a_lock_fingerprint="a", selection_path=selection,
            expected_quotas={"itemcf": 301},
        )
