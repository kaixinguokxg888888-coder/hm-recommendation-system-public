from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import duckdb

from data_foundation.manifest import file_fingerprint
from retrieval.stage3c.union import source_rank_cap_union as stage3c_union
from retrieval.stage3d.ann import exact_top_k_blockwise, overlap_recall, rerank_ann_candidates
from retrieval.stage3d.config import Stage3DConfig
from retrieval.stage3d.data import (
    AttributeVocabulary,
    Interaction,
    build_training_examples,
    personalized_serviceable,
    recency_weight,
)
from retrieval.stage3d.locking import Stage3DLockError, lock_fingerprint, validate_test_lock
from retrieval.stage3d.model import TwoTowerModel, in_batch_negative_mask, in_batch_softmax_loss
from retrieval.stage3d.runner import checkpoint_reusable, validate_stage3d_public_artifacts
from retrieval.stage3d.runner import Stage3DRunner
from retrieval.stage3d.union import select_model, select_quota, source_rank_cap_union


def test_training_examples_enforce_first_acquisition_strict_prior_context_and_caps() -> None:
    rows = []
    for item in range(1, 27):
        rows.append(Interaction(7, item, date(2020, 1, min(item, 25))))
    rows.extend([
        Interaction(7, 90, date(2020, 1, 25)),
        Interaction(7, 91, date(2020, 1, 25)),  # 同日商品不能互作上下文
        Interaction(7, 90, date(2020, 1, 26)),  # 重复购买不是新的 acquisition target
    ])
    examples = build_training_examples(rows)
    assert len(examples) == 3
    assert all(len(row.history_article_keys) <= 20 for row in examples)
    assert all(row.target_article_key not in row.history_article_keys for row in examples)
    for row in examples:
        prior_dates = {event.article_key: event.interaction_date for event in rows
                       if event.article_key in row.history_article_keys}
        assert all(value < row.target_date for value in prior_dates.values())
    target_90 = next(row for row in examples if row.target_article_key == 90)
    assert 91 not in target_90.history_article_keys


def test_recency_weight_matches_frozen_itemcf_rule() -> None:
    assert recency_weight(1) == pytest.approx(1.0)
    assert recency_weight(2) == pytest.approx(1.0 / np.log2(3))


def _attribute_fixture() -> tuple[AttributeVocabulary, torch.Tensor, tuple[str, ...]]:
    fields = (
        {"source_column": "tax_a", "semantic_group": "taxonomy"},
        {"source_column": "tax_b", "semantic_group": "taxonomy"},
        {"source_column": "appearance", "semantic_group": "appearance"},
        {"source_column": "merch", "semantic_group": "merchandising_index"},
    )
    rows = [
        {"tax_a": "1", "tax_b": "x", "appearance": "red", "merch": None},
        {"tax_a": "2", "tax_b": "y", "appearance": "blue", "merch": "m"},
        {"tax_a": "3", "tax_b": "z", "appearance": "green", "merch": "n"},
    ]
    vocab = AttributeVocabulary.build(rows, fields)
    encoded = torch.tensor([vocab.encode_row(row) for row in rows], dtype=torch.long)
    return vocab, encoded, tuple(row["semantic_group"] for row in fields)


def test_attribute_vocabulary_is_field_isolated_and_rejects_duplicate_semantics() -> None:
    vocab, _, _ = _attribute_fixture()
    assert vocab.values[0]["__MISSING__"] == 0
    assert vocab.values[3]["__MISSING__"] == 0
    assert vocab.values[0] is not vocab.values[3]
    with pytest.raises(ValueError, match="重复"):
        AttributeVocabulary.build([{"a": 1}], [
            {"source_column": "a", "semantic_group": "taxonomy"},
            {"source_column": "a", "semantic_group": "appearance"},
        ])


@pytest.mark.parametrize("variant", ["TT0", "TT1"])
def test_two_tower_shapes_norm_and_group_balance(variant: str) -> None:
    vocab, encoded, groups = _attribute_fixture()
    model = TwoTowerModel(
        variant=variant, num_users=5, num_articles=3,
        attribute_vocab_sizes=vocab.sizes() if variant == "TT1" else (),
        attribute_groups=groups if variant == "TT1" else (),
    )
    if variant == "TT1":
        model.set_attribute_index(encoded)
        assert model.item_projection.in_features == 64 + 3 * 8
    users = torch.tensor([0, 1])
    history = torch.tensor([[0, 1, -1], [1, 2, 0]])
    weights = torch.tensor([[1.0, 0.6, 0.0], [1.0, 0.6, 0.5]])
    targets = torch.tensor([2, 0])
    user_out, item_out = model(users, history, weights, targets)
    assert user_out.shape == item_out.shape == (2, 64)
    assert torch.linalg.vector_norm(user_out, dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert torch.linalg.vector_norm(item_out, dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_in_batch_mask_removes_duplicate_targets_and_user_history_false_negatives() -> None:
    targets = torch.tensor([3, 3, 7, 9])
    history = torch.tensor([[1, 7], [2, -1], [9, -1], [4, -1]])
    mask = in_batch_negative_mask(targets, history)
    assert mask.diagonal().all()
    assert not mask[0, 1] and not mask[1, 0]
    assert not mask[0, 2]
    assert not mask[2, 3]
    embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    loss = in_batch_softmax_loss(embeddings, embeddings, targets, history)
    assert torch.isfinite(loss)


def test_cold_history_seen_exclusion_universe_filter_and_exact_rerank() -> None:
    assert not personalized_serviceable(0)
    assert personalized_serviceable(1)
    item_embeddings = np.asarray([[1, 0], [0.8, 0.2], [0, 1], [-1, 0]], dtype=np.float32)
    item_embeddings /= np.linalg.norm(item_embeddings, axis=1, keepdims=True)
    keys = np.asarray([10, 11, 12, 13], dtype=np.int64)
    user = np.asarray([1, 0], dtype=np.float32)
    reranked, scores = rerank_ann_candidates(user, [2, 1, 0], item_embeddings, keys,
                                              seen_items={10}, k=3)
    assert reranked.tolist() == [11, 12]
    assert scores[0] > scores[1]
    exact = exact_top_k_blockwise(user, item_embeddings, keys, seen_items={10}, k=2, block_items=2)
    assert exact.tolist() == [11, 12]
    assert overlap_recall(reranked, exact) == 1.0


def _rows(start: int, count: int) -> list[dict[str, float | int]]:
    return [{"article_key": start + i, "source_rank": i + 1, "source_score": 1.0 / (i + 1)}
            for i in range(count)]


def test_source_cap_precedes_dedup_duplicate_consumes_cap_provenance_and_popularity_fill() -> None:
    result = source_rank_cap_union({
        "repurchase": _rows(1, 2),
        "itemcf": [{"article_key": 1, "source_rank": 1, "source_score": 1.0},
                   {"article_key": 3, "source_rank": 2, "source_score": 0.5}],
        "two_tower": [{"article_key": 1, "source_rank": 1, "source_score": 1.0},
                      {"article_key": 4, "source_rank": 2, "source_score": 0.5}],
        "trend": [], "popularity_fallback": _rows(1, 600),
    }, {"repurchase": 2, "itemcf": 1, "two_tower": 1, "trend": 0}, budget=500)
    assert len(result) == 500
    first = result[0]
    assert first["from_repurchase"] and first["from_itemcf"] and first["from_two_tower"]
    assert first["first_source"] == "repurchase"  # provenance 存在不等于后序来源实际占位
    assert 3 not in {row["article_key"] for row in result[:2]}
    assert 4 not in {row["article_key"] for row in result[:2]}


def test_q0_is_exactly_compatible_with_stage3c_semantics() -> None:
    sources_3d = {"repurchase": _rows(1, 5), "itemcf": _rows(100, 10),
                  "two_tower": [], "trend": _rows(200, 5), "popularity_fallback": _rows(1, 600)}
    sources_3c = {"repurchase": sources_3d["repurchase"], "itemcf": sources_3d["itemcf"],
                  "content": [], "trend": sources_3d["trend"],
                  "popularity_fallback": sources_3d["popularity_fallback"]}
    left = source_rank_cap_union(sources_3d, {"repurchase": 5, "itemcf": 10, "two_tower": 0, "trend": 5})
    right = stage3c_union(sources_3c, {"repurchase": 5, "itemcf": 10, "content": 0, "trend": 5})
    assert [(r["article_key"], r["source_rank"]) for r in left] == [
        (r["article_key"], r["source_rank"]) for r in right
    ]


def test_model_selection_uses_t50_and_tt0_exact_tie() -> None:
    base = {"t50_recall_at_500": 0.2, "t50_hit_rate_at_500": 0.4,
            "two_tower_exclusive_hits": 10, "declined_user_rate": 0.1,
            "single_route_candidate_user_coverage": 1.0, "ann_query_ms_per_user": 2.0}
    rows = [{"model_variant": "TT1", **base}, {"model_variant": "TT0", **base}]
    assert select_model(rows) == "TT0"


def test_test_gate_requires_authorized_lock_matching_model_and_fingerprints(tmp_path: Path) -> None:
    root = tmp_path
    lock_path = root / "artifacts" / "stage_3d" / "stage_3d_lock.json"
    selection = root / "artifacts" / "stage_3d" / "selection.csv"
    selection.parent.mkdir(parents=True)
    selection.write_text("status\nSUCCESS\n", encoding="utf-8")
    payload = {
        "stage_status": "PASSED_VALUE_GATE", "test_authorized": True,
        "config_fingerprint": "c", "source_fingerprint": "s",
        "stage_3a_lock_fingerprint": "a", "stage_3b_lock_fingerprint": "b",
        "stage_3c_lock_fingerprint": "d", "locked_model": "TT0",
        "model_checkpoint_fingerprint": "f", "final_model_checkpoint_fingerprint": "f",
        "embedding_source_checkpoint_fingerprint": "f", "candidate_source_model_fingerprint": "f",
        "final_refit_reinitialized": True, "final_refit_internal_dev_examples": 0,
        "final_refit_training_examples": 2, "locked_epoch": 1,
        "validation_artifacts": {"artifacts/stage_3d/selection.csv": hashlib.sha256(selection.read_bytes()).hexdigest()},
    }
    payload["lock_fingerprint"] = lock_fingerprint(payload)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    validate_test_lock(lock_path, config_fingerprint="c", source_fingerprint="s",
                       upstream_lock_fingerprints={"stage_3a_lock_fingerprint": "a",
                                                   "stage_3b_lock_fingerprint": "b",
                                                   "stage_3c_lock_fingerprint": "d"},
                       requested_model="TT0")
    with pytest.raises(Stage3DLockError, match="非锁定模型"):
        validate_test_lock(lock_path, config_fingerprint="c", source_fingerprint="s",
                           upstream_lock_fingerprints={"stage_3a_lock_fingerprint": "a",
                                                       "stage_3b_lock_fingerprint": "b",
                                                       "stage_3c_lock_fingerprint": "d"},
                           requested_model="TT1")


def test_stage3d_config_and_ignore_contract() -> None:
    config = Stage3DConfig.load(Path("configs/stage_3d_two_tower.yaml"))
    assert config.raw["union"]["candidate_budget"] == 500
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    for relative in ("artifacts/stage_3d/private/", "artifacts/stage_3d/tmp/",
                     "artifacts/stage_3d/checkpoints/"):
        assert relative in ignore


def test_stage3d_report_records_final_refit_status_and_dev_id_limitation() -> None:
    report = Path("docs/stages/stage_3d_two_tower_retrieval_and_incremental_recall.md").read_text(
        encoding="utf-8"
    )
    for required in (
        "Stage 3D final status",
        "`PASSED_VALUE_GATE`",
        "Stage 3D Test status",
        "`SUCCESS`",
        "随机初始化值，未被梯度更新",
        "formal_full_refit",
        "0.012184852577349603",
        "locked TT1 Test final-refit",
    ):
        assert required in report


@pytest.mark.parametrize("quota,two_tower_cap", [("T50", 50), ("T100", 100)])
def test_t50_t100_keep_fixed_total_budget(quota: str, two_tower_cap: int) -> None:
    config = Stage3DConfig.load(Path("configs/stage_3d_two_tower.yaml"))
    caps = config.raw["union"]["variants"][quota]
    result = source_rank_cap_union({
        "repurchase": _rows(1, 100), "itemcf": _rows(1000, 400),
        "two_tower": _rows(2000, 200), "trend": _rows(3000, 200),
        "popularity_fallback": _rows(4000, 600),
    }, caps, budget=500)
    assert caps["two_tower"] == two_tower_cap
    assert len(result) == 500
    assert [row["source_rank"] for row in result] == list(range(1, 501))


def test_quota_selection_exact_tie_prefers_t50() -> None:
    base = {"recall_at_500": 0.2, "hit_rate_at_500": 0.4,
            "two_tower_exclusive_hits": 10, "declined_user_rate": 0.1}
    assert select_quota([{"union_variant": "T100", **base},
                         {"union_variant": "T50", **base}]) == "T50"


def test_test_gate_rejects_missing_lock(tmp_path: Path) -> None:
    with pytest.raises(Stage3DLockError):
        validate_test_lock(tmp_path / "artifacts/stage_3d/stage_3d_lock.json",
                           config_fingerprint="c", source_fingerprint="s",
                           upstream_lock_fingerprints={})


def test_test_gate_rejects_changed_validation_artifact(tmp_path: Path) -> None:
    lock_path = tmp_path / "artifacts/stage_3d/stage_3d_lock.json"
    selection = tmp_path / "artifacts/stage_3d/selection.csv"
    selection.parent.mkdir(parents=True)
    selection.write_text("status\nSUCCESS\n", encoding="utf-8")
    payload = {
        "stage_status": "PASSED_VALUE_GATE", "test_authorized": True,
        "config_fingerprint": "c", "source_fingerprint": "s", "locked_model": "TT0",
        "validation_artifacts": {"artifacts/stage_3d/selection.csv": file_fingerprint(selection)},
    }
    payload["lock_fingerprint"] = lock_fingerprint(payload)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    selection.write_text("status\nCHANGED\n", encoding="utf-8")
    with pytest.raises(Stage3DLockError):
        validate_test_lock(lock_path, config_fingerprint="c", source_fingerprint="s",
                           upstream_lock_fingerprints={})


def test_test_gate_rejects_selection_only_or_inconsistent_formal_lineage(tmp_path: Path) -> None:
    lock_path = tmp_path / "artifacts" / "stage_3d" / "stage_3d_lock.json"
    lock_path.parent.mkdir(parents=True)
    payload = {
        "stage_status": "PASSED_VALUE_GATE", "test_authorized": True,
        "config_fingerprint": "c", "source_fingerprint": "s", "locked_model": "TT1",
        "model_checkpoint_fingerprint": "selection",
        "final_model_checkpoint_fingerprint": "final",
        "embedding_source_checkpoint_fingerprint": "selection",
        "candidate_source_model_fingerprint": "selection",
        "final_refit_reinitialized": False, "final_refit_internal_dev_examples": 1,
        "final_refit_training_examples": 100, "locked_epoch": 1,
    }
    payload["lock_fingerprint"] = lock_fingerprint(payload)
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage3DLockError, match="full-refit lineage"):
        validate_test_lock(lock_path, config_fingerprint="c", source_fingerprint="s",
                           upstream_lock_fingerprints={})


def test_checkpoint_reuse_rejects_output_fingerprint_change(tmp_path: Path) -> None:
    output = tmp_path / "output.bin"
    output.write_bytes(b"first")
    payload = {
        "status": "SUCCESS", "run_fingerprint": "run",
        "output_fingerprints": {"output.bin": file_fingerprint(output)},
    }
    assert checkpoint_reusable(payload, run_fingerprint="run", outputs=[output], root=tmp_path)
    output.write_bytes(b"changed")
    assert not checkpoint_reusable(payload, run_fingerprint="run", outputs=[output], root=tmp_path)


def test_formal_checkpoint_validation_rejects_selection_only_checkpoint(tmp_path: Path) -> None:
    runner = Stage3DRunner.__new__(Stage3DRunner)
    runner.config = SimpleNamespace(private_dir=tmp_path)
    checkpoint = runner.model_checkpoint("validation_14d", "TT1")
    history = runner.final_refit_history_path("validation_14d", "TT1")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"selection-only")
    history.write_text(json.dumps({
        "checkpoint_role": "model_selection_only",
        "uses_all_eligible_samples": False,
        "reinitialized_after_epoch_selection": False,
        "internal_dev_examples": 1,
        "final_checkpoint_fingerprint": file_fingerprint(checkpoint),
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="selection-only"):
        runner.validate_formal_checkpoint("validation_14d", "TT1")


def test_stage3d_privacy_scanner_allows_declared_hash_and_rejects_entity_value(tmp_path: Path) -> None:
    safe = tmp_path / "safe.json"
    safe.write_text(json.dumps({"config_fingerprint": "a" * 64}), encoding="utf-8")
    validate_stage3d_public_artifacts([safe])
    unsafe = tmp_path / "unsafe.csv"
    unsafe.write_text("metric,value\narticle,0123456789\n", encoding="utf-8")
    with pytest.raises(ValueError, match="敏感信息"):
        validate_stage3d_public_artifacts([unsafe])


def test_synthetic_training_builder_has_no_filesystem_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_open(*args: object, **kwargs: object) -> None:
        raise AssertionError("synthetic builder must not read raw data")
    monkeypatch.setattr(Path, "open", reject_open)
    rows = [Interaction(1, 10, date(2020, 1, 1)), Interaction(1, 11, date(2020, 1, 2))]
    examples = build_training_examples(rows)
    assert len(examples) == 1 and examples[0].target_article_key == 11


def test_leakage_audit_unpivot_syntax_keeps_non_null_history_items() -> None:
    con = duckdb.connect(":memory:")
    try:
        values = con.execute("""WITH samples AS (
          SELECT 1 sample_id,10 customer_key,20 target_article_key,DATE '2020-01-02' target_date,
                 11::BIGINT history_01,NULL::BIGINT history_02),
        contexts AS (UNPIVOT samples ON history_01,history_02
          INTO NAME history_rank VALUE article_key)
        SELECT sample_id,history_rank,article_key FROM contexts""").fetchall()
    finally:
        con.close()
    assert values == [(1, "history_01", 11)]
