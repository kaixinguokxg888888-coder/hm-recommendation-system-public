from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from retrieval.stage3c.config import Stage3CConfig
from retrieval.stage3c.content import (
    CanonicalField,
    MISSING_VALUE,
    build_sparse_item_matrix,
    field_token,
    idf_value,
    paired_bootstrap,
    select_content_variant,
    top_k_cosine_block,
    validate_canonical_fields,
)
from retrieval.stage3c.locking import Stage3CLockError, lock_fingerprint, validate_test_lock
from retrieval.stage3c.runner import Stage3CRunner, validate_stage3c_public_artifacts
from retrieval.stage3c.union import select_quota_variant, source_rank_cap_union


FIELDS = [
    CanonicalField("taxonomy", "type", "type_code", "type_name"),
    CanonicalField("taxonomy", "group", "group_code", "group_name"),
    CanonicalField("appearance", "colour", "colour_code", "colour_name"),
]


def rows(items: list[int], source_ranks: list[int] | None = None) -> list[dict[str, object]]:
    ranks = source_ranks or list(range(1, len(items) + 1))
    return [
        {"article_key": item, "source_rank": rank, "source_score": 1.0 / rank}
        for item, rank in zip(items, ranks, strict=True)
    ]


def test_canonical_code_name_not_double_modeled_and_tokens_are_field_isolated() -> None:
    validate_canonical_fields(FIELDS)
    with pytest.raises(ValueError, match="不得重复"):
        validate_canonical_fields([FIELDS[0], CanonicalField("taxonomy", "type", "type_name", "type_name")])
    assert field_token("left", 1) != field_token("right", 1)
    assert field_token("left", None) == f"left={MISSING_VALUE}"


def test_snapshot_specific_df_idf_does_not_accept_external_labels() -> None:
    assert idf_value(3, 1) == pytest.approx(math.log(2) + 1)
    values_a = {"type_code": np.array([1, 1, 2], dtype=object)}
    values_b = {"type_code": np.array([1, 2, 3], dtype=object)}
    field = [FIELDS[0]]
    _, tokens_a = build_sparse_item_matrix(values_a, field, use_idf=True, group_balance=False)
    _, tokens_b = build_sparse_item_matrix(values_b, field, use_idf=True, group_balance=False)
    assert sorted(row["idf"] for row in tokens_a) != sorted(row["idf"] for row in tokens_b)
    assert all("label" not in row for row in tokens_a)


def test_group_balance_cosine_symmetry_self_similarity_and_missing_group_safety() -> None:
    values = {
        "type_code": np.array([1, 1, 2], dtype=object),
        "group_code": np.array([10, 10, 20], dtype=object),
        "colour_code": np.array([None, None, 8], dtype=object),
    }
    matrix, _ = build_sparse_item_matrix(values, FIELDS, use_idf=True, group_balance=True)
    similarity = (matrix @ matrix.T).toarray()
    assert np.diag(similarity) == pytest.approx(np.ones(3))
    assert similarity == pytest.approx(similarity.T)
    assert np.isfinite(similarity).all()


def test_group_balance_prevents_extra_same_group_field_from_changing_group_prior() -> None:
    values = {
        "type_code": np.array([1, 1], dtype=object),
        "group_code": np.array([10, 20], dtype=object),
        "colour_code": np.array([7, 7], dtype=object),
    }
    matrix, _ = build_sparse_item_matrix(values, FIELDS, use_idf=False, group_balance=True)
    # appearance 完全相同，固定贡献为 1/2；taxonomy 只有一半字段相同，贡献为 1/4。
    assert float((matrix @ matrix.T)[0, 1]) == pytest.approx(0.75)


def test_top_k_cosine_stable_tie_break_and_exact_scores() -> None:
    values = {"type_code": np.array([1, 1, 1], dtype=object)}
    matrix, _ = build_sparse_item_matrix(values, [FIELDS[0]], use_idf=False, group_balance=False)
    left, right, score, rank = top_k_cosine_block(matrix, np.array([30, 10, 20]), 0, 1, 2)
    assert left.tolist() == [30, 30]
    assert right.tolist() == [10, 20]
    assert score.tolist() == pytest.approx([1.0, 1.0])
    assert rank.tolist() == [1, 2]


def test_source_rank_cap_is_applied_before_dedup_and_does_not_scan_beyond_cap() -> None:
    result = source_rank_cap_union(
        {
            "repurchase": rows([1]),
            "itemcf": rows([1, 2]),
            "content": rows([]),
            "trend": rows([]),
            "popularity_fallback": rows([1, 2, 3, 4]),
        },
        {"repurchase": 1, "itemcf": 1, "content": 0, "trend": 0},
        3,
    )
    assert [row["article_key"] for row in result] == [1, 2, 3]
    assert result[0]["from_repurchase"] and result[0]["from_itemcf"]
    # ItemCF rank 2 不会因 rank 1 与 Repurchase 重复而被继续扫描；2 来自 Popularity。
    assert not result[1]["from_itemcf"] and result[1]["from_popularity"]


def test_union_merges_provenance_fills_popularity_and_never_exceeds_budget() -> None:
    result = source_rank_cap_union(
        {
            "repurchase": rows([1, 2]), "itemcf": rows([2, 3]),
            "content": rows([3, 4]), "trend": rows([4, 5]),
            "popularity_fallback": rows(list(range(1, 20))),
        },
        {"repurchase": 2, "itemcf": 2, "content": 2, "trend": 2}, 8,
    )
    assert len(result) == 8 == len({row["article_key"] for row in result})
    assert result[1]["source_count"] == 2
    assert any(row["from_popularity"] for row in result)
    assert [row["source_rank"] for row in result] == list(range(1, 9))


def test_q0_q1_q2_use_same_cap_semantics_and_frozen_caps() -> None:
    config = Stage3CConfig.load(Path("configs/stage_3c_content.yaml"))
    assert config.raw["union"]["quota_semantics"] == "source_rank_cap_before_cross_source_dedup"
    assert config.raw["union"]["variants"] == {
        "Q0": {"repurchase": 50, "itemcf": 300, "content": 0, "trend": 100},
        "Q1": {"repurchase": 50, "itemcf": 250, "content": 50, "trend": 100},
        "Q2": {"repurchase": 50, "itemcf": 200, "content": 100, "trend": 100},
    }


def test_q0_reuses_stage3b_locked_iuf_union_instead_of_pre_itemcf_base() -> None:
    runner = object.__new__(Stage3CRunner)
    runner.config = SimpleNamespace(
        raw={"upstream": {"stage_3b_private": "artifacts/stage_3b/private"}},
        resolve=lambda value: Path(value),
    )
    assert "itemcf_union_iuf" in runner._q0_glob("validation_14d")
    assert "base_union" not in runner._q0_glob("validation_14d")
    assert "itemcf_union/part_*.parquet" in runner._q0_glob("test_14d")
    assert runner.stage3b_q0_metric("validation_14d").name == (
        "user_metrics_validation_14d_union_itemcf_union_iuf.parquet"
    )
    assert runner.stage3b_q0_metric("test_14d").name == (
        "user_metrics_test_14d_union_itemcf_union.parquet"
    )


def test_content_selection_and_quota_near_tie_rules_are_preregistered() -> None:
    variants = [
        {"variant": name, "q1_recall_at_500": 0.1, "q1_hit_rate_at_500": 0.2,
         "content_exclusive_hits": 1, "declined_user_rate": 0.1, "field_count": index + 1}
        for index, name in enumerate(("C0", "C1", "C2", "C3"))
    ]
    assert select_content_variant(variants) == "C0"
    variants[2]["q1_recall_at_500"] = 0.11
    assert select_content_variant(variants) == "C2"
    quotas = [
        {"variant": "Q0", "recall_at_500": 0.1000},
        {"variant": "Q1", "recall_at_500": 0.1004},
        {"variant": "Q2", "recall_at_500": 0.1010},
    ]
    assert select_quota_variant(quotas, tolerance=0.0005) == "Q2"
    quotas[2]["recall_at_500"] = 0.1003
    assert select_quota_variant(quotas, tolerance=0.0005) == "Q0"


def test_bootstrap_is_paired_reproducible_and_reports_median() -> None:
    first = paired_bootstrap([0, 0, 1], [1, 0, 1], iterations=1000, seed=9)
    second = paired_bootstrap([0, 0, 1], [1, 0, 1], iterations=1000, seed=9)
    assert first == second
    assert first["delta_mean"] == pytest.approx(1 / 3)
    assert first["delta_median"] == 0


def test_lock_rejects_fingerprint_changes_and_validation_artifact_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "selection.csv"
    artifact.write_text("selected\nQ1\n", encoding="utf-8")
    lock_path = tmp_path / "artifacts" / "stage_3c" / "lock.json"
    lock_path.parent.mkdir(parents=True)
    relative = artifact.relative_to(tmp_path).as_posix()
    payload = {
        "config_fingerprint": "c", "source_fingerprint": "s",
        "stage_3a_lock_fingerprint": "a", "stage_3b_lock_fingerprint": "b",
        "validation_artifacts": {relative: hashlib.sha256(artifact.read_bytes()).hexdigest()},
    }
    payload["lock_fingerprint"] = lock_fingerprint(payload)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    # validate_test_lock resolves from repository-like root (lock parents[2]).
    validate_test_lock(lock_path, config_fingerprint="c", source_fingerprint="s",
                       stage_3a_lock_fingerprint="a", stage_3b_lock_fingerprint="b")
    with pytest.raises(Stage3CLockError, match="config_fingerprint"):
        validate_test_lock(lock_path, config_fingerprint="changed", source_fingerprint="s",
                           stage_3a_lock_fingerprint="a", stage_3b_lock_fingerprint="b")
    artifact.write_text("selected\nQ2\n", encoding="utf-8")
    with pytest.raises(Stage3CLockError, match="Validation"):
        validate_test_lock(lock_path, config_fingerprint="c", source_fingerprint="s",
                           stage_3a_lock_fingerprint="a", stage_3b_lock_fingerprint="b")


def test_stage3c_public_scan_allows_only_explicit_fingerprint_fields(tmp_path: Path) -> None:
    fingerprint = "a" * 64
    allowed = tmp_path / "allowed.csv"
    allowed.write_text(f"candidate_universe_fingerprint\n{fingerprint}\n", encoding="utf-8")
    validate_stage3c_public_artifacts([allowed])
    forbidden = tmp_path / "forbidden.csv"
    forbidden.write_text(f"notes\n{fingerprint}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="敏感信息"):
        validate_stage3c_public_artifacts([forbidden])


def test_private_and_tmp_are_ignored_and_public_contract_has_no_entity_headers() -> None:
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "artifacts/stage_3c/private/" in ignore
    assert "artifacts/stage_3c/tmp/" in ignore
    config = Stage3CConfig.load(Path("configs/stage_3c_content.yaml"))
    assert config.raw["candidate_universe_version"] == "history_seen_items_v1"
    assert config.raw["content"]["candidate_shortfall_fallback"] is False
    assert config.raw["content"]["seen_item_exclusion"] is True
