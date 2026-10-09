from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Mapping, Sequence


IDENTITY_COLUMNS = ("snapshot_id", "as_of_date", "customer_key", "article_key")
CANDIDATE_METADATA_COLUMNS = (
    "candidate_contract_version", "candidate_protocol_version", "primary_source",
)
USER_FEATURES = (
    "user_history_interaction_count", "user_history_distinct_item_count",
    "user_recency_days", "user_recent_interaction_count_28d", "user_repeat_ratio", "user_segment",
)
ITEM_FEATURES = (
    "item_history_interaction_count", "item_history_distinct_buyer_count",
    "item_recent_interaction_count_7d", "item_recency_days",
)
UI_FEATURES = ("ui_history_interaction_count", "ui_seen_before", "ui_recency_days")
RETRIEVAL_FEATURES = (
    "candidate_rank", "source_count",
    "from_repurchase", "from_itemcf", "from_two_tower", "from_trend", "from_popularity",
    "repurchase_rank", "repurchase_score", "itemcf_rank", "itemcf_score",
    "two_tower_rank", "two_tower_score", "trend_rank", "trend_score",
    "popularity_rank", "popularity_score",
)
LEAKAGE_DENYLIST_REQUIRED = frozenset({
    "label", "future_positive_count", "candidate_positive_count", "is_retrieval_miss",
    "optimizer_eligible", "sampling_digest", "zero_sampling_rank", "query_order",
})


@dataclass(frozen=True)
class SnapshotSpec:
    snapshot_id: str
    as_of_date: date
    request_date: date
    history_source: str
    candidate_source: str
    feature_version: str
    partitions: int

    def validate(self) -> None:
        if self.request_date != self.as_of_date + timedelta(days=1):
            raise ValueError("request date 必须紧随 snapshot as-of")
        if self.partitions <= 0:
            raise ValueError("partitions 必须为正数")


def segment_from_count(count: int) -> str:
    if count < 0:
        raise ValueError("history count 不得为负")
    if count == 0:
        return "cold"
    if count <= 4:
        return "short_history"
    if count <= 20:
        return "warm"
    return "active"


def inclusive_window_start(as_of: date, days: int) -> date:
    if days <= 0:
        raise ValueError("window days 必须为正数")
    return as_of - timedelta(days=days - 1)


def recency_days(as_of: date, latest: date | None) -> int | None:
    if latest is None:
        return None
    value = (as_of - latest).days
    if value < 0:
        raise ValueError("latest event 越过 snapshot cutoff")
    return value


def repeat_ratio(interactions: int, distinct_items: int) -> float | None:
    if interactions == 0:
        return None
    if not 0 <= distinct_items <= interactions:
        raise ValueError("distinct item count 与 interaction count 不一致")
    return 1.0 - distinct_items / interactions


def deterministic_category_vocabulary(values: Iterable[Any]) -> dict[Any, int]:
    """Reserve 0/1 for MISSING/UNKNOWN and deterministically encode known values."""
    known = {value for value in values if value is not None}
    ordered = sorted(known, key=lambda value: (type(value).__name__, str(value)))
    return {value: index + 2 for index, value in enumerate(ordered)}


def category_code(value: Any, vocabulary: Mapping[Any, int], *, known_domain: bool = True) -> int:
    if value is None:
        return 0
    if value in vocabulary:
        return int(vocabulary[value])
    return 1 if known_domain else 0


def category_affinity(
    *, matching_interactions: int | None, history_interactions: int,
    candidate_category_code: int,
) -> float | None:
    if history_interactions == 0 or candidate_category_code == 0:
        return None
    return float(matching_interactions or 0) / history_interactions


def assert_label_free_model_schema(schema_rows: Sequence[Mapping[str, Any]], denylist: Iterable[str]) -> None:
    denied = set(denylist)
    violations = [
        str(row["feature_name"]) for row in schema_rows
        if bool(row["model_input_eligible"]) and str(row["feature_name"]) in denied
    ]
    if violations:
        raise ValueError(f"model-input schema 命中 leakage denylist: {sorted(violations)}")


def assert_projection_identity(
    upstream: Iterable[tuple[str, int, int]], projected: Iterable[tuple[str, int, int]],
) -> None:
    upstream_rows, projected_rows = list(upstream), list(projected)
    left, right = set(upstream_rows), set(projected_rows)
    if left != right or len(left) != len(upstream_rows) or len(right) != len(projected_rows):
        raise ValueError("feature projection membership drift")
