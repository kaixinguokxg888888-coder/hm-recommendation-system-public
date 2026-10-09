from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest

from data_foundation.manifest import file_fingerprint
from retrieval.stage3d.config import source_fingerprint as stage3d_source_fingerprint
from retrieval.stage3d.data import Interaction, build_training_examples
from retrieval.stage3e.config import source_fingerprint as stage3e_source_fingerprint
from retrieval.stage4b1.config import Stage4B1Config, Stage4B1ConfigError
from retrieval.stage4b1.contract import (
    HistoricalReplayError,
    assert_history_cutoff,
    assert_protocol_identity,
    directory_fingerprint,
    forbidden_query_scope_fields,
    fuse_candidate_sources,
    privacy_findings,
    protocol_identity_rows,
    semantic_fingerprint,
    semantic_replay_audit,
    validate_candidate_rows,
    validate_formal_historical_lineage,
    validate_registered_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_4b1_historical_replay.yaml"


def _sources() -> dict[str, list[dict[str, object]]]:
    return {
        "repurchase": [
            {"article_key": 10, "source_rank": 1, "source_score": 3.0},
            {"article_key": 11, "source_rank": 2, "source_score": 2.0},
        ],
        "itemcf": [
            {"article_key": 10, "source_rank": 1, "source_score": 0.9},
            {"article_key": 12, "source_rank": 2, "source_score": 0.8},
        ],
        "two_tower": [{"article_key": 13, "source_rank": 1, "source_score": 0.7}],
        "trend": [{"article_key": 14, "source_rank": 1, "source_score": 7.0}],
        "popularity": [
            {"article_key": key, "source_rank": rank, "source_score": float(100 - rank)}
            for rank, key in enumerate(range(10, 30), 1)
        ],
    }


def _fused(budget: int = 8) -> list[dict[str, object]]:
    return fuse_candidate_sources(
        snapshot_id="ranking_train_14d_v1", customer_key=9, sources=_sources(),
        caps={"repurchase": 2, "itemcf": 2, "two_tower": 1, "trend": 1},
        candidate_budget=budget,
    )


def test_historical_snapshot_registration() -> None:
    config = Stage4B1Config.load(CONFIG)
    assert config.snapshot.snapshot_id == "ranking_train_14d_v1"
    validate_registered_snapshot(config.raw["snapshot"])


def test_historical_snapshot_rejects_wrong_as_of() -> None:
    config = Stage4B1Config.load(CONFIG)
    raw = copy.deepcopy(config.raw)
    raw["snapshot"]["as_of_date"] = "2020-08-12"
    with pytest.raises(Stage4B1ConfigError):
        Stage4B1Config(config.root, config.path, raw).validate()


def test_as_of_cutoff_enforcement() -> None:
    assert_history_cutoff(["2020-08-10", "2020-08-11"])
    with pytest.raises(HistoricalReplayError):
        assert_history_cutoff(["2020-08-12"])


def test_label_agnostic_candidate_membership() -> None:
    before = _sources()
    after = copy.deepcopy(before)
    for rows in after.values():
        for row in rows:
            row["future_label"] = int(row["article_key"]) % 2
    left = _fused()
    right = fuse_candidate_sources(
        snapshot_id="ranking_train_14d_v1", customer_key=9, sources=after,
        caps={"repurchase": 2, "itemcf": 2, "two_tower": 1, "trend": 1},
        candidate_budget=8,
    )
    assert [(r["article_key"], r["candidate_rank"]) for r in left] == [
        (r["article_key"], r["candidate_rank"]) for r in right
    ]


def test_request_scope_rejects_item_level_supervision_fields() -> None:
    assert forbidden_query_scope_fields(["user_key", "history_interaction_count"]) == []
    assert forbidden_query_scope_fields(["user_key", "article_key", "future_label"]) == [
        "article_key", "future_label",
    ]


def test_stage_report_discloses_upstream_maintenance_retry_and_warning() -> None:
    report = (ROOT / "docs/stages/stage_4b1_historical_candidate_replay.md").read_text(
        encoding="utf-8"
    )
    assert "stage_3b_source_current" in report
    assert "stage_3b_source_at_lock" in report
    assert "不作 source hash equality 声明" in report
    assert "private directory 尚未初始化" in report
    assert "只读 NumPy view 转 tensor warning" in report


def test_tt1_historical_causal_boundaries() -> None:
    interactions = [
        Interaction(1, 10, date(2020, 8, 1)),
        Interaction(1, 11, date(2020, 8, 2)),
        Interaction(1, 12, date(2020, 8, 2)),
        Interaction(1, 13, date(2020, 8, 11)),
    ]
    examples = build_training_examples(
        interactions, max_targets_per_user=3, max_history_items=20,
    )
    assert examples
    for example in examples:
        assert all(value != example.target_article_key for value in example.history_article_keys)
        assert example.target_date <= date(2020, 8, 11)


def test_formal_full_refit_lineage_enforced() -> None:
    lineage = {
        "model_role": "formal_historical_full_refit", "snapshot_id": "ranking_train_14d_v1",
        "as_of_date": "2020-08-11", "uses_all_eligible_samples": True,
        "internal_dev_examples": 0, "locked_epoch": 1,
    }
    validate_formal_historical_lineage(lineage)


@pytest.mark.parametrize("field,value", [
    ("model_role", "model_selection_only"),
    ("model_role", "formal_full_refit"),
    ("snapshot_id", "validation_14d"),
    ("as_of_date", "2020-08-25"),
    ("uses_all_eligible_samples", False),
    ("internal_dev_examples", 1),
])
def test_formal_lineage_rejects_selection_wrong_snapshot_and_partial_refit(field: str, value: object) -> None:
    lineage = {
        "model_role": "formal_historical_full_refit", "snapshot_id": "ranking_train_14d_v1",
        "as_of_date": "2020-08-11", "uses_all_eligible_samples": True,
        "internal_dev_examples": 0, "locked_epoch": 1,
    }
    lineage[field] = value
    with pytest.raises(HistoricalReplayError):
        validate_formal_historical_lineage(lineage)


def test_source_cap_applies_before_cross_source_dedup() -> None:
    rows = _fused()
    articles = [row["article_key"] for row in rows]
    assert 10 in articles
    assert 12 in articles
    assert rows[0]["itemcf_rank"] == 1
    assert rows[0]["source_count"] == 2


def test_duplicate_provenance_merge_does_not_sum_scores() -> None:
    row = _fused()[0]
    assert row["from_repurchase"] and row["from_itemcf"]
    assert row["repurchase_score"] == 3.0
    assert row["itemcf_score"] == 0.9
    assert "source_score" not in row


def test_popularity_unique_fill_and_budget_rank_invariants() -> None:
    rows = _fused()
    validate_candidate_rows(rows, budget=8)
    assert len(rows) == 8
    assert len({row["article_key"] for row in rows}) == 8
    assert [row["candidate_rank"] for row in rows] == list(range(1, 9))
    assert rows[-1]["from_popularity"]


def test_candidate_uniqueness_rejected() -> None:
    rows = _fused()
    rows[-1]["article_key"] = rows[0]["article_key"]
    with pytest.raises(HistoricalReplayError):
        validate_candidate_rows(rows, budget=8)


def test_candidate_rank_discontinuity_rejected() -> None:
    rows = _fused()
    rows[-1]["candidate_rank"] = 9
    with pytest.raises(HistoricalReplayError):
        validate_candidate_rows(rows, budget=8)


def test_protocol_identity() -> None:
    actual = {"model": "TT1", "quota": "T100", "candidate_budget": 500}
    rows = protocol_identity_rows(actual, dict(actual), list(actual))
    assert_protocol_identity(rows)
    rows[0]["status"] = "FAIL"
    with pytest.raises(HistoricalReplayError):
        assert_protocol_identity(rows)


def test_semantic_fingerprint_scheduling_invariance() -> None:
    rows = _fused()
    assert semantic_fingerprint(rows) == semantic_fingerprint(list(reversed(rows)))
    audit = semantic_replay_audit(rows, list(reversed(rows)))
    assert audit["status"] == "PASS"


def test_semantic_fingerprint_membership_sensitivity() -> None:
    rows = _fused()
    changed = copy.deepcopy(rows)
    changed[-1]["article_key"] = 999
    assert semantic_fingerprint(rows) != semantic_fingerprint(changed)


def test_semantic_fingerprint_rank_sensitivity() -> None:
    rows = _fused()
    changed = copy.deepcopy(rows)
    changed[0]["candidate_rank"], changed[1]["candidate_rank"] = 2, 1
    assert semantic_fingerprint(rows) != semantic_fingerprint(changed)


def test_directory_immutability_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "guarded"
    path.mkdir()
    (path / "a.txt").write_text("first", encoding="utf-8")
    before = directory_fingerprint(path)
    (path / "a.txt").write_text("second", encoding="utf-8")
    after = directory_fingerprint(path)
    assert before[0] != after[0]


def test_stage3d_stage3e_formal_source_fingerprints_unchanged() -> None:
    lock3d = json.loads((ROOT / "artifacts/stage_3d/stage_3d_lock.json").read_text(encoding="utf-8"))
    manifest3e = ROOT / "artifacts/stage_3e/upstream_fingerprint_manifest.csv"
    assert stage3d_source_fingerprint(ROOT) == lock3d["source_fingerprint"]
    assert stage3e_source_fingerprint(ROOT)
    assert file_fingerprint(manifest3e) == "2bc75375af8b460346bf43da21b32bfe242137ef70a4373b0f8248f3eb0383f3"


def test_privacy_audit_detects_forbidden_header_and_absolute_path(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("customer_id,status\nabc,C:\\Users\\name\n", encoding="utf-8")
    findings = privacy_findings([bad], local_username="name")
    assert any("forbidden_field_customer_id" in value for value in findings)
    assert any("absolute_path" in value for value in findings)
    assert any("local_username" in value for value in findings)


def test_stage4b2_remains_unimplemented_in_config() -> None:
    raw = Stage4B1Config.load(CONFIG).raw
    serialized = json.dumps(raw, sort_keys=True).lower()
    assert "lightgbm" not in serialized
    assert "negative_sampling" not in serialized
    assert "ranking_feature" not in serialized
