from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def user_overlap(
    itemcf: Sequence[int],
    trend: Sequence[int],
    repurchase: Sequence[int],
    popularity: Sequence[int],
    base_union: Sequence[int],
    itemcf_union: Sequence[int],
    ground_truth: set[int],
) -> dict[str, float | int | str]:
    sets = {
        "itemcf": set(itemcf),
        "trend": set(trend),
        "repurchase": set(repurchase),
        "popularity": set(popularity),
        "base_union": set(base_union),
        "itemcf_union": set(itemcf_union),
    }
    base_hits = sets["base_union"] & ground_truth
    union_hits = sets["itemcf_union"] & ground_truth
    itemcf_hits = sets["itemcf"] & ground_truth
    prior_hits = (sets["trend"] | sets["repurchase"] | sets["popularity"]) & ground_truth
    denominator = len(ground_truth)
    delta = (len(union_hits) - len(base_hits)) / denominator
    return {
        "itemcf_trend_jaccard": jaccard(sets["itemcf"], sets["trend"]),
        "itemcf_repurchase_jaccard": jaccard(sets["itemcf"], sets["repurchase"]),
        "itemcf_base_overlap_rate": (
            len(sets["itemcf"] & sets["base_union"]) / len(sets["itemcf"])
            if sets["itemcf"] else 0.0
        ),
        "itemcf_exclusive_hits": len(itemcf_hits - prior_hits),
        "itemcf_shared_hits": len(itemcf_hits & prior_hits),
        "base_recall_at_500": len(base_hits) / denominator,
        "itemcf_union_recall_at_500": len(union_hits) / denominator,
        "delta_recall_at_500": delta,
        "change_class": "improved" if delta > 0 else "declined" if delta < 0 else "unchanged",
    }


def aggregate_change_classes(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["change_class"]) for row in rows)
    return {name: counts.get(name, 0) for name in ("improved", "unchanged", "declined")}
