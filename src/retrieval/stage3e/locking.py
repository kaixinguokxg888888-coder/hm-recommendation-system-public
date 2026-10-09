from __future__ import annotations

from typing import Any, Mapping

from retrieval.stage3d.locking import lock_fingerprint

from .config import EXPECTED_CAPS


class Stage3ELineageError(ValueError):
    """Frozen Stage 3D identity or final-refit lineage does not match Stage 3E."""


def validate_stage3d_identity(
    lock: Mapping[str, Any], *, actual_config_fingerprint: str,
    actual_source_fingerprint: str, upstream_lock_fingerprints: Mapping[str, str],
    candidate_lineages: Mapping[str, Mapping[str, Any]],
) -> None:
    checks = {
        "stage_status": "PASSED_VALUE_GATE",
        "recommended_stage_status": "PASSED_VALUE_GATE",
        "test_status": "SUCCESS",
        "locked_model": "TT1",
        "locked_quota": "T100",
        "candidate_protocol_version": "source_rank_cap_pre_dedup_v1",
        "locked_quota_caps": EXPECTED_CAPS,
        "config_fingerprint": actual_config_fingerprint,
        "source_fingerprint": actual_source_fingerprint,
    }
    for field, expected in checks.items():
        if lock.get(field) != expected:
            raise Stage3ELineageError(f"Stage 3D lock identity mismatch: {field}")
    ann = lock.get("ann_parameters", {})
    if ann.get("algorithm") != "hnsw" or ann.get("ef_search") != 200:
        raise Stage3ELineageError("Stage 3D locked ANN identity mismatch")
    if lock.get("lock_fingerprint") != lock_fingerprint(dict(lock)):
        raise Stage3ELineageError("Stage 3D lock self fingerprint mismatch")
    for field, expected in upstream_lock_fingerprints.items():
        if lock.get(field) != expected:
            raise Stage3ELineageError(f"Stage 3D upstream lock mismatch: {field}")
    expected_models = {
        "validation_14d": lock.get("final_model_checkpoint_fingerprint"),
        "test_14d": lock.get("test_final_model_checkpoint_fingerprint"),
    }
    for snapshot, expected_model in expected_models.items():
        lineage = candidate_lineages.get(snapshot, {})
        if (
            lineage.get("snapshot_id") != snapshot
            or lineage.get("model_variant") != "TT1"
            or lineage.get("candidate_role") != "formal_full_refit"
            or lineage.get("lineage_passed") is not True
            or lineage.get("candidate_partitions") != 32
            or lineage.get("source_model_checkpoint_fingerprint") != expected_model
        ):
            raise Stage3ELineageError(f"Stage 3D {snapshot} candidate checkpoint lineage mismatch")
