from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from .contracts import RETRIEVAL_FEATURES


SCHEMA_FINGERPRINT_VERSION = "ranking_feature_schema_sha256_v1"


def _row(
    name: str, feature_set: str, family: str, dtype: str, nullable: bool,
    eligible: bool, dynamic: str, lineage: str, cutoff: str,
    zero: str, null: str, default: str, transformation: str,
) -> dict[str, Any]:
    return {
        "feature_name": name, "dtype": dtype, "nullable": nullable,
        "feature_family": family, "feature_set": feature_set,
        "model_input_eligible": eligible, "dynamic_or_static": dynamic,
        "source_lineage": lineage, "cutoff_rule": cutoff,
        "zero_semantics": zero, "null_semantics": null,
        "default_semantics": default, "transformation_version": transformation,
    }


def feature_schema_rows(canonical_fields: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    rows = [
        _row("snapshot_id", "IDENTITY", "identity", "string", False, False, "METADATA",
             "snapshot_spec", "explicit snapshot key", "not applicable", "not allowed", "none", "identity_v1"),
        _row("as_of_date", "IDENTITY", "identity", "date32", False, False, "METADATA",
             "snapshot_spec", "feature cutoff", "not applicable", "not allowed", "none", "identity_v1"),
        _row("customer_key", "IDENTITY", "identity", "int64", False, False, "METADATA",
             "internal_id_policy", "join only", "valid key value", "not allowed", "none", "identity_v1"),
        _row("article_key", "IDENTITY", "identity", "int64", False, False, "METADATA",
             "internal_id_policy", "join only", "valid key value", "not allowed", "none", "identity_v1"),
        _row("candidate_contract_version", "METADATA", "lineage", "string", False, False, "METADATA",
             "stage_4b1_candidate_contract", "frozen upstream", "not applicable", "not allowed", "none", "lineage_v1"),
        _row("candidate_protocol_version", "METADATA", "lineage", "string", False, False, "METADATA",
             "stage_4b1_candidate_contract", "frozen upstream", "not applicable", "not allowed", "none", "lineage_v1"),
        _row("primary_source", "METADATA", "candidate_audit", "string", False, False, "STATIC",
             "stage_4b1_candidate_contract", "frozen upstream", "not applicable", "not allowed", "none", "frozen_raw_v1"),
        _row("feature_pipeline_version", "METADATA", "lineage", "string", False, False, "METADATA",
             "stage_4b3_config", "frozen builder version", "not applicable", "not allowed", "none", "lineage_v1"),
        _row("feature_schema_fingerprint", "METADATA", "lineage", "string", False, False, "METADATA",
             "stage_4b3_schema", "canonical schema identity", "not applicable", "not allowed", "none", "lineage_v1"),
    ]
    user_specs = [
        ("user_history_interaction_count", "int64", False, "0 means no B interaction before cutoff"),
        ("user_history_distinct_item_count", "int64", False, "0 means no distinct historical item"),
        ("user_recency_days", "int32", True, "0 means latest interaction occurred on as-of"),
        ("user_recent_interaction_count_28d", "int64", False, "0 means no B interaction in inclusive 28d window"),
        ("user_repeat_ratio", "float64", True, "0 means every historical interaction is on a distinct item"),
        ("user_segment", "string", False, "not numeric; frozen cold/short_history/warm/active category"),
    ]
    for name, dtype, nullable, zero in user_specs:
        rows.append(_row(name, "F0,F1", "user", dtype, nullable, True, "DYNAMIC",
                         "B_feedback_history", "t_dat <= snapshot as-of",
                         zero, "no history/undefined denominator" if nullable else "not allowed",
                         "0 only for count; cold for segment" if not nullable else "none",
                         "user_snapshot_aggregate_v1"))
    item_specs = [
        ("item_history_interaction_count", "int64", False, "0 means item unseen in cutoff history"),
        ("item_history_distinct_buyer_count", "int64", False, "0 means no historical buyer"),
        ("item_recent_interaction_count_7d", "int64", False, "0 means no B interaction in inclusive 7d window"),
        ("item_recency_days", "int32", True, "0 means latest interaction occurred on as-of"),
    ]
    for name, dtype, nullable, zero in item_specs:
        rows.append(_row(name, "F0,F1", "item", dtype, nullable, True, "DYNAMIC",
                         "B_feedback_history", "t_dat <= snapshot as-of",
                         zero, "item unseen in cutoff history" if nullable else "not allowed",
                         "0 only for count" if not nullable else "none", "item_snapshot_aggregate_v1"))
    ui_specs = [
        ("ui_history_interaction_count", "int64", False, "0 means user-item unseen before cutoff"),
        ("ui_seen_before", "bool", False, "false means user-item unseen before cutoff"),
        ("ui_recency_days", "int32", True, "0 means latest user-item interaction occurred on as-of"),
    ]
    for name, dtype, nullable, zero in ui_specs:
        rows.append(_row(name, "F0,F1", "user_item", dtype, nullable, True, "DYNAMIC",
                         "B_feedback_history", "t_dat <= snapshot as-of",
                         zero, "user-item unseen before cutoff" if nullable else "not allowed",
                         "0/false for unseen counts/flag" if not nullable else "none",
                         "user_item_snapshot_aggregate_v1"))
    for name in RETRIEVAL_FEATURES:
        if name.startswith("from_"):
            dtype, nullable, zero, null = "bool", False, "false means source did not contribute", "not allowed"
        elif name.endswith("_score"):
            dtype, nullable, zero, null = "float64", True, "source-local raw zero if upstream emitted it", "source did not contribute"
        else:
            dtype, nullable = "int32", name.endswith("_rank")
            zero = "not a valid rank" if "rank" in name else "0 means one contributing source is impossible under contract"
            null = "source did not contribute" if nullable else "not allowed"
        rows.append(_row(name, "F0,F1", "retrieval", dtype, nullable, True, "STATIC",
                         "stage_4b1_frozen_candidate", "candidate-before-label frozen value",
                         zero, null, "none", "frozen_source_local_raw_v1"))
    for field in canonical_fields:
        semantic = field["semantic_field"]
        source = field["source_column"]
        rows.append(_row(
            f"article_{semantic}_code", "F1", "static_article", "int32", False, True, "STATIC_ATTRIBUTE",
            f"stage_3c_canonical:{source}", "effective_timestamp_not_available",
            "0=MISSING", "not allowed", "1=UNKNOWN; 2+=known canonical values",
            "canonical_category_missing0_unknown1_value_asc2plus_v1",
        ))
        rows.append(_row(
            f"user_{semantic}_interaction_share", "F1", "user_category_affinity", "float64", True, True, "DYNAMIC",
            f"B_feedback_history+stage_3c_canonical:{source}", "t_dat <= snapshot as-of",
            "0 means known candidate category with no matching historical interaction",
            "cold user or missing candidate category", "none", "user_category_share_v1",
        ))
    return rows


def schema_fingerprint(rows: Iterable[dict[str, Any]]) -> str:
    canonical = sorted((dict(row) for row in rows), key=lambda row: row["feature_name"])
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((SCHEMA_FINGERPRINT_VERSION + "\0" + payload).encode("utf-8")).hexdigest()


def model_feature_names(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [row["feature_name"] for row in rows if bool(row["model_input_eligible"])]
