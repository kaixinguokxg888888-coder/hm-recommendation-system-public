from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retrieval.evaluation import user_metrics
from retrieval.stage3d.locking import lock_fingerprint
from retrieval.stage3d.union import source_rank_cap_union
from retrieval.stage3e.config import Stage3EConfig, source_fingerprint
from retrieval.stage3e.contract import (
    CONTRACT_COLUMNS,
    CONTRACT_VERSION,
    PROTOCOL_VERSION,
    CandidateContractError,
    migrate_stage3d_row,
    schema_manifest_rows,
    semantic_candidate_fingerprint,
    semantic_fingerprint_from_partitions,
    validate_contract_rows,
    zero_drift_counts,
)
from retrieval.stage3e.locking import Stage3ELineageError, validate_stage3d_identity
from retrieval.stage3e.runner import Stage3ERunner, validate_public_artifacts
from scripts.finalize_stage_3e_public import (
    build_trusted_public_hash_allowlist,
    observed_memory_limit_range,
    public_finalization_source_fingerprint,
    validate_public_artifacts_field_aware,
)


def _stage3d_row(
    *, snapshot: str = "validation_14d", customer: int = 7, article: int = 11,
    rank: int = 1,
) -> dict[str, object]:
    return {
        "snapshot_id": snapshot, "as_of_date": date(2020, 8, 25),
        "customer_key": customer, "article_key": article,
        "source": "union", "source_variant": "TT1_T100", "source_rank": rank,
        "source_score": 1.0 / rank, "candidate_universe_version": "history_seen_items_v1",
        "feedback_version": "B-primary_C-log1p-p99-v1", "population_version": "1.0.0",
        "dataset_fingerprint": "d" * 64, "config_fingerprint": "c" * 64,
        "first_source": "repurchase" if rank == 1 else "itemcf", "source_count": 1,
        "from_repurchase": rank == 1, "from_itemcf": rank != 1,
        "from_two_tower": False, "from_trend": False, "from_popularity": False,
        "repurchase_rank": 1 if rank == 1 else None,
        "repurchase_score": 1.0 if rank == 1 else None,
        "itemcf_rank": rank if rank != 1 else None,
        "itemcf_score": 1.0 / rank if rank != 1 else None,
        "two_tower_rank": None, "two_tower_score": None,
        "trend_rank": None, "trend_score": None,
        "popularity_rank": None, "popularity_score": None,
        "source_score_kind": "source_rank_cap_pre_dedup_then_fixed_insertion",
    }


def _contract_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    upstream = [_stage3d_row(article=11, rank=1), _stage3d_row(article=12, rank=2)]
    contract = [migrate_stage3d_row(row, stage3e_config_fingerprint="e" * 64) for row in upstream]
    return upstream, contract


def test_config_freezes_t100_contract_and_private_paths_are_ignored() -> None:
    config = Stage3EConfig.load(Path("configs/stage_3e_candidate_contract.yaml"))
    assert config.raw["frozen_retrieval"]["source_caps"] == {
        "repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100,
    }
    assert config.raw["frozen_retrieval"]["candidate_budget"] == 500
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    for relative in ("artifacts/stage_3e/private/", "artifacts/stage_3e/tmp/",
                     "artifacts/stage_3e/checkpoints/"):
        assert relative in ignore


def test_schema_has_version_snapshot_rank_and_source_local_evidence() -> None:
    rows = {row["column"]: row for row in schema_manifest_rows()}
    assert tuple(rows) == CONTRACT_COLUMNS
    assert rows["candidate_rank"]["membership_critical"]
    assert "only formal candidate order" in rows["candidate_rank"]["semantic_meaning"]
    assert rows["itemcf_score"]["nullable"]
    assert "never summed cross-source" in rows["itemcf_score"]["semantic_meaning"]


def test_contract_migration_preserves_rank_provenance_and_snapshot_isolation() -> None:
    upstream, contract = _contract_rows()
    validate_contract_rows(contract, expected_snapshot="validation_14d", expected_budget=2)
    assert zero_drift_counts(upstream, contract) == {
        "stage3d_minus_stage3e": 0, "stage3e_minus_stage3d": 0,
        "rank_mismatch": 0, "provenance_mismatch": 0, "user_count_mismatch": 0,
    }
    assert contract[0]["candidate_contract_version"] == CONTRACT_VERSION
    assert contract[0]["candidate_protocol_version"] == PROTOCOL_VERSION
    with pytest.raises(CandidateContractError, match="snapshot isolation"):
        validate_contract_rows(contract, expected_snapshot="test_14d", expected_budget=2)


def test_contract_rejects_duplicate_membership_rank_gap_wrong_primary_and_cap() -> None:
    _, rows = _contract_rows()
    duplicate = [dict(rows[0]), dict(rows[0])]
    duplicate[1]["candidate_rank"] = 2
    with pytest.raises(CandidateContractError, match="membership 重复"):
        validate_contract_rows(duplicate, expected_budget=2)
    gap = [dict(rows[0]), dict(rows[1])]
    gap[1]["candidate_rank"] = 3
    with pytest.raises(CandidateContractError, match="连续"):
        validate_contract_rows(gap, expected_budget=3)
    wrong_primary = [dict(rows[0])]
    wrong_primary[0]["primary_source"] = "itemcf"
    with pytest.raises(CandidateContractError, match="insertion order"):
        validate_contract_rows(wrong_primary)
    over_cap = [dict(rows[1])]
    over_cap[0]["candidate_rank"] = 1
    over_cap[0]["itemcf_rank"] = 201
    with pytest.raises(CandidateContractError, match="source-rank cap"):
        validate_contract_rows(over_cap)


def _source_rows(start: int, count: int) -> list[dict[str, object]]:
    return [{"article_key": start + index, "source_rank": index + 1,
             "source_score": 1.0 / (index + 1)} for index in range(count)]


def test_frozen_fusion_cap_precedes_dedup_merges_provenance_and_never_sums_scores() -> None:
    result = source_rank_cap_union({
        "repurchase": _source_rows(1, 2),
        "itemcf": [{"article_key": 1, "source_rank": 1, "source_score": 4.0},
                   {"article_key": 3, "source_rank": 2, "source_score": 3.0}],
        "two_tower": [{"article_key": 1, "source_rank": 1, "source_score": 9.0}],
        "trend": [], "popularity_fallback": _source_rows(1, 600),
    }, {"repurchase": 2, "itemcf": 1, "two_tower": 1, "trend": 0}, budget=500)
    assert len(result) == 500
    assert result[0]["first_source"] == "repurchase"
    assert result[0]["source_count"] == 3
    assert result[0]["repurchase_score"] == 1.0
    assert result[0]["itemcf_score"] == 4.0
    assert result[0]["two_tower_score"] == 9.0
    assert "final_score" not in result[0]
    assert 3 not in {row["article_key"] for row in result[:2]}


def test_semantic_fingerprint_is_order_invariant_and_membership_rank_sensitive() -> None:
    _, rows = _contract_rows()
    original = semantic_candidate_fingerprint(rows)
    assert semantic_candidate_fingerprint(reversed(rows)) == original
    rank_changed = [dict(row) for row in rows]
    rank_changed[0]["candidate_rank"], rank_changed[1]["candidate_rank"] = 2, 1
    assert semantic_candidate_fingerprint(rank_changed) != original
    membership_changed = [dict(row) for row in rows]
    membership_changed[1]["article_key"] = 99
    assert semantic_candidate_fingerprint(membership_changed) != original


def _write_membership_parquet(path: Path, rows: list[dict[str, object]], compression: str) -> None:
    table = pa.table({
        "snapshot_id": [row["snapshot_id"] for row in rows],
        "customer_key": pa.array([row["customer_key"] for row in rows], type=pa.int64()),
        "candidate_rank": pa.array([row["candidate_rank"] for row in rows], type=pa.int32()),
        "article_key": pa.array([row["article_key"] for row in rows], type=pa.int64()),
    })
    pq.write_table(table, path, compression=compression, row_group_size=1)


def test_parquet_semantic_fingerprint_ignores_filename_compression_and_partition_order(tmp_path: Path) -> None:
    _, rows = _contract_rows()
    part_a = tmp_path / "renamed_a.parquet"
    part_b = tmp_path / "different_name.parquet"
    _write_membership_parquet(part_a, rows, "snappy")
    _write_membership_parquet(part_b, rows, "zstd")
    left, _ = semantic_fingerprint_from_partitions([part_a])
    right, _ = semantic_fingerprint_from_partitions([part_b])
    assert left == right == semantic_candidate_fingerprint(rows)
    customer_two = [dict(row, customer_key=8) for row in rows]
    part_c = tmp_path / "second_user.parquet"
    _write_membership_parquet(part_c, customer_two, "zstd")
    forward, _ = semantic_fingerprint_from_partitions([part_a, part_c])
    reverse, _ = semantic_fingerprint_from_partitions([part_c, part_a])
    assert forward == reverse


def _valid_lock_and_lineages() -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    lock: dict[str, object] = {
        "stage_status": "PASSED_VALUE_GATE", "recommended_stage_status": "PASSED_VALUE_GATE",
        "test_status": "SUCCESS", "locked_model": "TT1", "locked_quota": "T100",
        "candidate_protocol_version": PROTOCOL_VERSION,
        "locked_quota_caps": {"repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100},
        "config_fingerprint": "config", "source_fingerprint": "source",
        "stage_3a_lock_fingerprint": "a", "stage_3b_lock_fingerprint": "b",
        "stage_3c_lock_fingerprint": "c", "ann_parameters": {"algorithm": "hnsw", "ef_search": 200},
        "final_model_checkpoint_fingerprint": "validation-final",
        "test_final_model_checkpoint_fingerprint": "test-final",
    }
    lock["lock_fingerprint"] = lock_fingerprint(lock)
    lineages = {
        snapshot: {
            "snapshot_id": snapshot, "model_variant": "TT1", "candidate_role": "formal_full_refit",
            "lineage_passed": True, "candidate_partitions": 32,
            "source_model_checkpoint_fingerprint": model,
        } for snapshot, model in (
            ("validation_14d", "validation-final"), ("test_14d", "test-final")
        )
    }
    return lock, lineages


@pytest.mark.parametrize("mutation", ["lock", "config", "checkpoint", "quota", "protocol"])
def test_lineage_gate_rejects_wrong_lock_config_checkpoint_t100_or_protocol(mutation: str) -> None:
    lock, lineages = _valid_lock_and_lineages()
    config = "config"
    if mutation == "lock":
        lock["lock_fingerprint"] = "wrong"
    elif mutation == "config":
        config = "wrong"
    elif mutation == "checkpoint":
        lineages["test_14d"]["source_model_checkpoint_fingerprint"] = "selection-only"
    elif mutation == "quota":
        lock["locked_quota"] = "T50"
        lock["lock_fingerprint"] = lock_fingerprint(lock)
    else:
        lock["candidate_protocol_version"] = "wrong"
        lock["lock_fingerprint"] = lock_fingerprint(lock)
    with pytest.raises(Stage3ELineageError):
        validate_stage3d_identity(
            lock, actual_config_fingerprint=config, actual_source_fingerprint="source",
            upstream_lock_fingerprints={"stage_3a_lock_fingerprint": "a",
                                        "stage_3b_lock_fingerprint": "b",
                                        "stage_3c_lock_fingerprint": "c"},
            candidate_lineages=lineages,
        )


def test_checkpoint_resume_rejects_config_upstream_source_input_and_output_drift(tmp_path: Path) -> None:
    runner = Stage3ERunner.__new__(Stage3ERunner)
    runner.config = type("Config", (), {"fingerprint": "config"})()
    runner.source_fingerprint = "source"
    runner.upstream_fingerprint = "upstream"
    runner.run_fingerprint = "run"
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"source")
    target.write_bytes(b"target")
    checkpoint = {
        "status": "SUCCESS", "config_fingerprint": "config", "source_fingerprint": "source",
        "upstream_fingerprint": "upstream", "run_fingerprint": "run",
        "snapshot_id": "validation_14d", "partition_id": 0,
        "candidate_contract_version": CONTRACT_VERSION,
        "input_physical_fingerprint": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "output_physical_fingerprint": __import__("hashlib").sha256(target.read_bytes()).hexdigest(),
    }
    assert runner._checkpoint_reusable(checkpoint, snapshot="validation_14d", part=0,
                                       source=source, target=target)
    for field in ("config_fingerprint", "source_fingerprint", "upstream_fingerprint"):
        changed = dict(checkpoint, **{field: "wrong"})
        assert not runner._checkpoint_reusable(changed, snapshot="validation_14d", part=0,
                                               source=source, target=target)
    target.write_bytes(b"changed")
    assert not runner._checkpoint_reusable(checkpoint, snapshot="validation_14d", part=0,
                                           source=source, target=target)


def test_zero_drift_candidates_produce_identical_offline_metrics() -> None:
    upstream, contract = _contract_rows()
    truth = {11, 99}
    left = user_metrics([int(row["article_key"]) for row in sorted(upstream, key=lambda r: r["source_rank"])], truth)
    right = user_metrics([int(row["article_key"]) for row in sorted(contract, key=lambda r: r["candidate_rank"])], truth)
    assert left == right


def test_privacy_scan_allows_declared_hash_and_rejects_entity_or_machine_values(tmp_path: Path) -> None:
    declared = "a" * 64
    safe = tmp_path / "safe.csv"
    safe.write_text(f"semantic_candidate_fingerprint\n{declared}\n", encoding="utf-8")
    validate_public_artifacts([safe], allowed_hashes={declared})
    raw_article = "0" + "123456789"
    unsafe = tmp_path / "unsafe.csv"
    unsafe.write_text("value\n" + raw_article + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="privacy scan"):
        validate_public_artifacts([unsafe], allowed_hashes=set())


def test_field_aware_privacy_scan_scrubs_declared_hash_before_article_pattern(tmp_path: Path) -> None:
    declared = "a" * 20 + "0" + "123456789" + "b" * 34
    artifact = tmp_path / "declared.csv"
    artifact.write_text("physical_file_fingerprint\n" + declared + "\n", encoding="utf-8")
    validate_public_artifacts_field_aware([artifact], allowed_hashes={declared})


def test_public_finalization_fingerprint_is_deterministic_and_byte_sensitive(tmp_path: Path) -> None:
    finalizer = tmp_path / "scripts" / "finalize_stage_3e_public.py"
    finalizer.parent.mkdir(parents=True)
    finalizer.write_bytes(b"first-version")
    first = public_finalization_source_fingerprint(tmp_path)
    assert public_finalization_source_fingerprint(tmp_path) == first
    finalizer.write_bytes(b"second-version")
    assert public_finalization_source_fingerprint(tmp_path) != first


def test_finalizer_bytes_do_not_change_candidate_source_fingerprint(tmp_path: Path) -> None:
    module_dir = tmp_path / "src" / "retrieval" / "stage3e"
    scripts_dir = tmp_path / "scripts"
    module_dir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    (module_dir / "contract.py").write_bytes(b"candidate-contract")
    (scripts_dir / "run_stage_3e_candidate_contract.py").write_bytes(b"candidate-runner")
    finalizer = scripts_dir / "finalize_stage_3e_public.py"
    finalizer.write_bytes(b"public-v1")
    candidate_before = source_fingerprint(tmp_path)
    public_before = public_finalization_source_fingerprint(tmp_path)
    finalizer.write_bytes(b"public-v2")
    assert source_fingerprint(tmp_path) == candidate_before
    assert public_finalization_source_fingerprint(tmp_path) != public_before


def test_field_aware_privacy_scan_rejects_unknown_64hex(tmp_path: Path) -> None:
    unknown = "9" * 64
    artifact = tmp_path / "unknown.csv"
    artifact.write_text("value\n" + unknown + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared_64hex_value"):
        validate_public_artifacts_field_aware([artifact], allowed_hashes=set())


def test_scan_target_cannot_self_authorize_unknown_64hex(tmp_path: Path) -> None:
    unknown = "8" * 64
    artifact = tmp_path / "self_declared.csv"
    artifact.write_text("value\n" + unknown + "\n", encoding="utf-8")
    allowed = build_trusted_public_hash_allowlist(
        structured_sources=(), artifact_paths=[artifact],
    )
    assert unknown not in allowed
    with pytest.raises(ValueError, match="undeclared_64hex_value"):
        validate_public_artifacts_field_aware([artifact], allowed_hashes=allowed)


def test_public_finalization_audit_manifest_and_resource_range_use_formal_inputs() -> None:
    root = Path.cwd()
    config = Stage3EConfig.load(root / "configs" / "stage_3e_candidate_contract.yaml")
    candidate_source = source_fingerprint(root)
    public_source = public_finalization_source_fingerprint(root)
    assert candidate_source == "b1913743005b5ca04a482af76be49f8c0b0552d1952bda4d396722dc9f870187"

    audit = list(__import__("csv").DictReader(
        (root / "artifacts" / "stage_3e" / "public_finalization_audit.csv").open(
            encoding="utf-8-sig", newline="",
        )
    ))
    assert len(audit) == 1
    assert audit[0]["public_finalization_source_fingerprint"] == public_source
    assert audit[0]["candidate_source_fingerprint"] == candidate_source
    manifest = list(__import__("csv").DictReader(
        (root / "artifacts" / "stage_3e" / "stage_3e_manifest.csv").open(
            encoding="utf-8-sig", newline="",
        )
    ))
    assert manifest
    assert {row["source_fingerprint"] for row in manifest} == {candidate_source}
    assert {row["public_finalization_source_fingerprint"] for row in manifest} == {public_source}

    resources = list(__import__("csv").DictReader(
        (root / "artifacts" / "stage_3e" / "resource_summary.csv").open(
            encoding="utf-8-sig", newline="",
        )
    ))
    observed_min, observed_max = observed_memory_limit_range(resources)
    expected_min = min(int(row["duckdb_memory_limit_mb"]) for row in resources)
    expected_max = max(int(row["duckdb_memory_limit_mb"]) for row in resources)
    assert (observed_min, observed_max) == (expected_min, expected_max)
    duckdb = config.raw["duckdb"]
    report = config.report_path.read_text(encoding="utf-8")
    assert (
        f"bounded to {duckdb['memory_min_mb']}–{duckdb['memory_max_mb']} MiB"
        in report
    )
    assert f"range was {observed_min}–{observed_max} MiB" in report
