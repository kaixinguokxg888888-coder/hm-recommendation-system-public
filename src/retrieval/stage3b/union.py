from __future__ import annotations

from typing import Iterable, Mapping, Sequence


SOURCE_FIELDS = {
    "repurchase": "repurchase",
    "itemcf": "itemcf",
    "trend": "trend",
    "popularity_fallback": "popularity",
}


def fixed_budget_union(
    sources: Mapping[str, Sequence[Mapping[str, object]]],
    quotas: Mapping[str, int],
    budget: int = 500,
) -> list[dict[str, object]]:
    if budget <= 0:
        raise ValueError("union budget 必须为正")
    order = ("repurchase", "itemcf", "trend", "popularity_fallback")
    records: dict[int, dict[str, object]] = {}
    ordered_items: list[int] = []
    for source in order:
        rows = list(sources.get(source, ()))
        limit = budget if source == "popularity_fallback" else int(quotas.get(source, 0))
        for row in rows[:limit]:
            item = int(row["article_key"])
            if item not in records:
                if len(ordered_items) >= budget:
                    continue
                records[item] = {
                    "article_key": item,
                    "first_source": source,
                    "from_repurchase": False,
                    "from_itemcf": False,
                    "from_trend": False,
                    "from_popularity": False,
                    "repurchase_rank": None,
                    "repurchase_score": None,
                    "itemcf_rank": None,
                    "itemcf_score": None,
                    "trend_rank": None,
                    "trend_score": None,
                    "popularity_rank": None,
                    "popularity_score": None,
                }
                ordered_items.append(item)
            record = records[item]
            suffix = SOURCE_FIELDS[source]
            record[f"from_{suffix}"] = True
            record[f"{suffix}_rank"] = int(row["source_rank"])
            record[f"{suffix}_score"] = float(row["source_score"])
    result = []
    for rank, item in enumerate(ordered_items, 1):
        row = records[item]
        row["source_rank"] = rank
        row["source_count"] = sum(
            bool(row[name])
            for name in ("from_repurchase", "from_itemcf", "from_trend", "from_popularity")
        )
        result.append(row)
    return result
