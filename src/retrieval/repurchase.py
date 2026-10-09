from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Iterable

from .popularity import Interaction


def pure_repurchase(
    interactions: Iterable[Interaction],
    as_of_date: date,
    users: Iterable[int],
    legal_universe: dict[int, set[int]],
    max_k: int,
) -> dict[int, list[tuple[int, int, date]]]:
    grouped: dict[tuple[int, int], set[date]] = defaultdict(set)
    requested = set(users)
    for day, customer, article in set(interactions):
        if (
            day <= as_of_date
            and customer in requested
            and article in legal_universe[customer]
        ):
            grouped[(customer, article)].add(day)
    result: dict[int, list[tuple[int, int, date]]] = {user: [] for user in requested}
    for (customer, article), days in grouped.items():
        result[customer].append((article, len(days), max(days)))
    for user in result:
        result[user] = sorted(
            result[user], key=lambda row: (-row[1], -row[2].toordinal(), row[0])
        )[:max_k]
    return result


def popularity_fallback(
    pure: dict[int, list[tuple[int, int, date]]],
    popularity: Iterable[tuple[int, int]],
    legal_universe: dict[int, set[int]],
    max_k: int,
) -> dict[int, list[tuple[int, str, int | None]]]:
    global_rows = list(popularity)
    result: dict[int, list[tuple[int, str, int | None]]] = {}
    for user, pure_rows in pure.items():
        output = [(item, "pure", count) for item, count, _ in pure_rows]
        existing = {row[0] for row in output}
        for item, score in global_rows:
            if len(output) >= max_k:
                break
            if item in legal_universe[user] and item not in existing:
                output.append((item, "popularity_fallback", score))
                existing.add(item)
        result[user] = output
    return result
