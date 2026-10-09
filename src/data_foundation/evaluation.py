from __future__ import annotations

from typing import Any


CANDIDATE_DOMAINS = {
    "main_warm_evaluation": "history_seen_items_only",
    "cold_item_evaluation": "full_article_dimension_with_visibility_limitation",
}


def evaluation_object(
    user_key: int,
    history_interaction_count: int,
    history_distinct_item_count: int,
    positive_item_count: int,
    warm_positive_count: int,
    cold_positive_count: int,
) -> dict[str, Any]:
    if warm_positive_count + cold_positive_count != positive_item_count:
        raise ValueError("warm/cold positive 与 positive_item_count 不一致")
    from .populations import user_segment

    return {
        "user_key": user_key,
        "user_segment": user_segment(history_interaction_count),
        "history_interaction_count": history_interaction_count,
        "history_distinct_item_count": history_distinct_item_count,
        "positive_item_count": positive_item_count,
        "warm_positive_count": warm_positive_count,
        "cold_positive_count": cold_positive_count,
        "has_cold_positive": cold_positive_count > 0,
        "is_main_evaluable": history_interaction_count > 0,
    }


def checkpoint_matches(
    checkpoint: dict[str, Any] | None,
    dataset_fingerprint: str,
    config_fingerprint: str,
    output_exists: bool,
    force: bool = False,
) -> bool:
    return bool(
        checkpoint
        and checkpoint.get("status") == "SUCCESS"
        and checkpoint.get("dataset_fingerprint") == dataset_fingerprint
        and checkpoint.get("config_fingerprint") == config_fingerprint
        and output_exists
        and not force
    )
