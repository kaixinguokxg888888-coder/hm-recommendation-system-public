from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence


Interaction = tuple[date, int, int]


@dataclass(frozen=True)
class PairStats:
    raw_cooccurrence: int
    iuf_cooccurrence: float


@dataclass(frozen=True)
class Neighbor:
    article_key: int
    similarity: float
    weighted_cooccurrence: float
    raw_cooccurrence: int
    neighbor_rank: int


def build_user_item_sets(
    interactions: Iterable[Interaction],
    as_of_date: date,
    max_items: int = 50,
) -> dict[int, list[tuple[int, date]]]:
    if max_items <= 0:
        raise ValueError("max_items 必须为正")
    last_dates: dict[tuple[int, int], date] = {}
    for day, user, item in interactions:
        if day > as_of_date:
            continue
        key = (user, item)
        if key not in last_dates or day > last_dates[key]:
            last_dates[key] = day
    grouped: dict[int, list[tuple[int, date]]] = defaultdict(list)
    for (user, item), last_day in last_dates.items():
        grouped[user].append((item, last_day))
    # 同日按匿名 key 固定顺序，保证截断边界和重复运行都稳定。
    return {
        user: sorted(rows, key=lambda row: (-row[1].toordinal(), row[0]))[:max_items]
        for user, rows in grouped.items()
    }


def item_support(user_items: Mapping[int, Sequence[tuple[int, date]]]) -> dict[int, int]:
    support: dict[int, int] = defaultdict(int)
    for rows in user_items.values():
        for item in {item for item, _ in rows}:
            support[item] += 1
    return dict(support)


def cooccurrence_pairs(
    user_items: Mapping[int, Sequence[tuple[int, date]]],
) -> dict[tuple[int, int], PairStats]:
    raw: dict[tuple[int, int], int] = defaultdict(int)
    iuf: dict[tuple[int, int], float] = defaultdict(float)
    for rows in user_items.values():
        items = sorted({item for item, _ in rows})
        if len(items) < 2:
            continue
        weight = 1.0 / math.log1p(len(items))
        for left_index, left in enumerate(items[:-1]):
            for right in items[left_index + 1 :]:
                pair = (left, right)
                raw[pair] += 1
                iuf[pair] += weight
    return {
        pair: PairStats(count, iuf[pair])
        for pair, count in raw.items()
    }


def build_neighbors(
    pairs: Mapping[tuple[int, int], PairStats],
    support: Mapping[int, int],
    variant: str,
    top_n: int = 200,
) -> dict[int, list[Neighbor]]:
    if variant not in {"cosine", "iuf"}:
        raise ValueError("ItemCF variant 只允许 cosine/iuf")
    if top_n <= 0:
        raise ValueError("top_n 必须为正")
    directed: dict[int, list[tuple[int, float, float, int]]] = defaultdict(list)
    for (left, right), pair in pairs.items():
        denominator = math.sqrt(support.get(left, 0) * support.get(right, 0))
        if denominator <= 0:
            continue
        weighted = (
            float(pair.raw_cooccurrence)
            if variant == "cosine"
            else pair.iuf_cooccurrence
        )
        similarity = weighted / denominator
        if not math.isfinite(similarity):
            continue
        directed[left].append((right, similarity, weighted, pair.raw_cooccurrence))
        directed[right].append((left, similarity, weighted, pair.raw_cooccurrence))
    output: dict[int, list[Neighbor]] = {}
    for item, rows in directed.items():
        ordered = sorted(rows, key=lambda row: (-row[1], -row[2], -row[3], row[0]))[:top_n]
        output[item] = [
            Neighbor(neighbor, similarity, weighted, raw_count, rank)
            for rank, (neighbor, similarity, weighted, raw_count) in enumerate(ordered, 1)
        ]
    return output


def select_variant(rows: Iterable[Mapping[str, object]]) -> str:
    values = list(rows)
    if {str(row["variant"]) for row in values} != {"cosine", "iuf"}:
        raise ValueError("Validation 必须完整包含 cosine/iuf")
    return str(
        sorted(
            values,
            key=lambda row: (
                -float(row["recall_at_500"]),
                -float(row["hit_rate_at_500"]),
                -float(row["map_at_12"]),
                -float(row["candidate_user_coverage"]),
                0 if str(row["variant"]) == "cosine" else 1,
            ),
        )[0]["variant"]
    )
