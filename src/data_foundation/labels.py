from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable


def deduplicate_ground_truth(
    rows: Iterable[tuple[int, int, date, int]],
) -> dict[tuple[int, int], dict[str, object]]:
    grouped: dict[tuple[int, int], list[tuple[date, int]]] = defaultdict(list)
    for user_key, article_key, label_date, strength in rows:
        grouped[(user_key, article_key)].append((label_date, strength))
    return {
        key: {
            "label": 1,
            "label_date_min": min(day for day, _ in values),
            "label_date_max": max(day for day, _ in values),
            "user_item_label_days": len({day for day, _ in values}),
            "label_daily_strength_sum": sum(value for _, value in values),
        }
        for key, values in grouped.items()
    }
