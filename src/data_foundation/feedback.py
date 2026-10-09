from __future__ import annotations

import math
from typing import Iterable


def binary_interactions(rows: Iterable[tuple[str, str, str]]) -> set[tuple[str, str, str]]:
    return set(rows)


def daily_counts(
    rows: Iterable[tuple[str, str, str]],
) -> dict[tuple[str, str, str], int]:
    result: dict[tuple[str, str, str], int] = {}
    for row in rows:
        result[row] = result.get(row, 0) + 1
    return result


def nearest_rank_cap(values: Iterable[int], quantile: float = 0.99) -> int:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("计算 cap 需要非空 history")
    if not 0 < quantile <= 1:
        raise ValueError("quantile 必须位于 (0, 1]")
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return int(ordered[index])


def capped_strength(count: int, cap: int) -> tuple[int, float]:
    if count < 1 or cap < 1:
        raise ValueError("count 和 cap 必须为正")
    capped = min(count, cap)
    return capped, 1.0 + math.log1p(capped - 1)
