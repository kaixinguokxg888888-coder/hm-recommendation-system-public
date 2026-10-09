from __future__ import annotations

import math
from typing import Iterable


def nearest_rank(values: Iterable[int], quantile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("需要至少一个 seen item")
    return ordered[max(0, math.ceil(len(ordered) * quantile) - 1)]


def popularity_thresholds(history_counts: Iterable[int]) -> tuple[int, int]:
    values = list(history_counts)
    return nearest_rank(values, 0.5), nearest_rank(values, 0.9)


def popularity_bucket(history_count: int, p50: int, p90: int) -> str:
    if history_count <= 0:
        return "zero_history"
    if history_count >= p90:
        return "head"
    if history_count >= p50:
        return "mid"
    return "tail"


def item_state(history_count: int, label_count: int) -> dict[str, bool]:
    return {
        "seen_item": history_count > 0,
        "cold_item": history_count == 0 and label_count > 0,
        "zero_history_dimension_item": history_count == 0,
    }
