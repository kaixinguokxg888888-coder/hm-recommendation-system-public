from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Iterable


Interaction = tuple[date, int, int]


def popularity_ranking(
    interactions: Iterable[Interaction],
    as_of_date: date,
    legal_items: set[int],
) -> list[tuple[int, int]]:
    # B 口径按 user-item-day 唯一；此处再次 set 只用于让公共函数对人工输入安全确定。
    counts = Counter(
        article
        for day, customer, article in set(interactions)
        if day <= as_of_date and article in legal_items
    )
    return sorted(counts.items(), key=lambda row: (-row[1], row[0]))


def popularity_candidates(
    ranking: Iterable[tuple[int, int]],
    users: Iterable[int],
    legal_universe: dict[int, set[int]],
    max_k: int,
) -> dict[int, list[tuple[int, int]]]:
    ordered = list(ranking)
    return {
        user: [
            row for row in ordered if row[0] in legal_universe[user]
        ][:max_k]
        for user in users
    }
