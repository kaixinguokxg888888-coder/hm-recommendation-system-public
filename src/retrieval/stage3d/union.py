from __future__ import annotations

from typing import Mapping, Sequence


ORDER = ("repurchase", "itemcf", "two_tower", "trend", "popularity_fallback")
PROVENANCE = {
    "repurchase": "repurchase", "itemcf": "itemcf", "two_tower": "two_tower",
    "trend": "trend", "popularity_fallback": "popularity",
}


def source_rank_cap_union(
    sources: Mapping[str, Sequence[Mapping[str, object]]],
    source_rank_caps: Mapping[str, int],
    budget: int = 500,
) -> list[dict[str, object]]:
    if budget <= 0:
        raise ValueError("union budget 必须为正")
    records: dict[int, dict[str, object]] = {}
    ordered: list[int] = []
    for source in ORDER:
        cap = budget if source == "popularity_fallback" else int(source_rank_caps.get(source, 0))
        for row in sources.get(source, ()):
            rank = int(row["source_rank"])
            if source != "popularity_fallback" and rank > cap:
                continue
            item = int(row["article_key"])
            if source == "popularity_fallback" and item in records:
                continue
            if item not in records:
                if len(ordered) >= budget:
                    continue
                records[item] = {
                    "article_key": item, "first_source": source,
                    **{f"from_{name}": False for name in PROVENANCE.values()},
                    **{f"{name}_rank": None for name in PROVENANCE.values()},
                    **{f"{name}_score": None for name in PROVENANCE.values()},
                }
                ordered.append(item)
            suffix = PROVENANCE[source]
            records[item][f"from_{suffix}"] = True
            records[item][f"{suffix}_rank"] = rank
            records[item][f"{suffix}_score"] = float(row["source_score"])
            if source == "popularity_fallback" and len(ordered) >= budget:
                break
    output = []
    for rank, item in enumerate(ordered[:budget], 1):
        row = records[item]
        row["source_rank"] = rank
        row["source_count"] = sum(bool(row[f"from_{name}"]) for name in PROVENANCE.values())
        output.append(row)
    return output


def select_model(rows: Sequence[Mapping[str, object]]) -> str:
    if {str(row["model_variant"]) for row in rows} != {"TT0", "TT1"}:
        raise ValueError("模型选择必须完整包含 TT0/TT1")
    order = {"TT0": 0, "TT1": 1}
    return str(sorted(rows, key=lambda row: (
        -float(row["t50_recall_at_500"]), -float(row["t50_hit_rate_at_500"]),
        -int(row["two_tower_exclusive_hits"]), float(row["declined_user_rate"]),
        -float(row["single_route_candidate_user_coverage"]), float(row["ann_query_ms_per_user"]),
        order[str(row["model_variant"])],
    ))[0]["model_variant"])


def select_quota(rows: Sequence[Mapping[str, object]]) -> str:
    if {str(row["union_variant"]) for row in rows} != {"T50", "T100"}:
        raise ValueError("锁定配额比较必须完整包含 T50/T100")
    order = {"T50": 0, "T100": 1}
    return str(sorted(rows, key=lambda row: (
        -float(row["recall_at_500"]), -float(row["hit_rate_at_500"]),
        -int(row["two_tower_exclusive_hits"]), float(row["declined_user_rate"]),
        order[str(row["union_variant"])],
    ))[0]["union_variant"])
