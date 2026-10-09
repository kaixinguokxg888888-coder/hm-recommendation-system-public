from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Iterable, Mapping, Sequence

from .itemcf import Interaction, Neighbor


def select_seed_items(
    interactions: Iterable[Interaction],
    as_of_date: date,
    max_seeds: int = 20,
) -> dict[int, list[tuple[int, date, int, float]]]:
    latest: dict[tuple[int, int], date] = {}
    for day, user, item in interactions:
        if day <= as_of_date and (user, item) not in latest or (
            day <= as_of_date and day > latest[(user, item)]
        ):
            latest[(user, item)] = day
    grouped: dict[int, list[tuple[int, date]]] = defaultdict(list)
    for (user, item), day in latest.items():
        grouped[user].append((item, day))
    result: dict[int, list[tuple[int, date, int, float]]] = {}
    for user, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: (-row[1].toordinal(), row[0]))[:max_seeds]
        result[user] = [
            (item, day, rank, 1.0 / math.log2(rank + 1))
            for rank, (item, day) in enumerate(ordered, 1)
        ]
    return result


def generate_itemcf_candidates(
    seeds: Mapping[int, Sequence[tuple[int, date, int, float]]],
    neighbors: Mapping[int, Sequence[Neighbor]],
    legal_universe: Mapping[int, set[int]],
    seen_items: Mapping[int, set[int]],
    max_k: int = 500,
) -> dict[int, list[dict[str, float | int]]]:
    output: dict[int, list[dict[str, float | int]]] = {}
    for user in sorted(legal_universe):
        scores: dict[int, list[float]] = {}
        seed_hits: dict[int, set[int]] = defaultdict(set)
        max_similarity: dict[int, float] = defaultdict(float)
        for seed_item, _, _, seed_weight in seeds.get(user, ()):
            for neighbor in neighbors.get(seed_item, ()):
                item = neighbor.article_key
                if item in seen_items.get(user, set()) or item not in legal_universe[user]:
                    continue
                scores.setdefault(item, []).append(seed_weight * neighbor.similarity)
                seed_hits[item].add(seed_item)
                max_similarity[item] = max(max_similarity[item], neighbor.similarity)
        ordered = sorted(
            (
                (item, sum(parts), len(seed_hits[item]), max_similarity[item])
                for item, parts in scores.items()
            ),
            key=lambda row: (-row[1], -row[2], -row[3], row[0]),
        )[:max_k]
        output[user] = [
            {
                "article_key": item,
                "source_score": score,
                "seed_hit_count": hit_count,
                "max_similarity": max_sim,
                "source_rank": rank,
            }
            for rank, (item, score, hit_count, max_sim) in enumerate(ordered, 1)
        ]
    return output
