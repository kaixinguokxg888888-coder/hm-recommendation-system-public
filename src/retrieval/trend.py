from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Iterable

from .popularity import Interaction, popularity_ranking


def trend_ranking(
    interactions: Iterable[Interaction],
    as_of_date: date,
    window_days: int,
    legal_items: set[int],
) -> list[tuple[int, int, int]]:
    if window_days <= 0:
        raise ValueError("Trend window_days 必须为正")
    rows = set(interactions)
    lower_exclusive = as_of_date - timedelta(days=window_days)
    recent = Counter(
        article
        for day, customer, article in rows
        if lower_exclusive < day <= as_of_date and article in legal_items
    )
    full = dict(popularity_ranking(rows, as_of_date, legal_items))
    return sorted(
        ((article, score, full[article]) for article, score in recent.items()),
        key=lambda row: (-row[1], -row[2], row[0]),
    )
