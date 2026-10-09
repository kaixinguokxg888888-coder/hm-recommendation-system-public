from __future__ import annotations

from typing import Mapping, Sequence


ORDER = ("repurchase", "itemcf", "content", "trend", "popularity_fallback")
PROVENANCE = {
    "repurchase": "repurchase",
    "itemcf": "itemcf",
    "content": "content",
    "trend": "trend",
    "popularity_fallback": "popularity",
}


def source_rank_cap_union(
    sources: Mapping[str, Sequence[Mapping[str, object]]],
    source_rank_caps: Mapping[str, int],
    budget: int = 500,
) -> list[dict[str, object]]:
    """Stage 3B-compatible: source-rank cap first, cross-source dedup second."""
    if budget <= 0:
        raise ValueError("union budget 必须为正")
    records: dict[int, dict[str, object]] = {}
    ordered: list[int] = []
    for source in ORDER:
        rows = sources.get(source, ())
        cap = budget if source == "popularity_fallback" else int(source_rank_caps.get(source, 0))
        for row in rows:
            source_rank = int(row["source_rank"])
            if source != "popularity_fallback" and source_rank > cap:
                continue
            item = int(row["article_key"])
            # Popularity 是对 primary union 的 anti-join 补齐，不作为已经入选
            # primary 候选的额外 provenance；这与 Stage 3B 冻结 Q0 完全一致。
            if source == "popularity_fallback" and item in records:
                continue
            if item not in records:
                if len(ordered) >= budget:
                    continue
                records[item] = {
                    "article_key": item,
                    "first_source": source,
                    **{f"from_{name}": False for name in PROVENANCE.values()},
                    **{f"{name}_rank": None for name in PROVENANCE.values()},
                    **{f"{name}_score": None for name in PROVENANCE.values()},
                }
                ordered.append(item)
            record = records[item]
            suffix = PROVENANCE[source]
            record[f"from_{suffix}"] = True
            record[f"{suffix}_rank"] = source_rank
            record[f"{suffix}_score"] = float(row["source_score"])
            if len(ordered) >= budget and source == "popularity_fallback":
                break
    result: list[dict[str, object]] = []
    for rank, item in enumerate(ordered[:budget], 1):
        record = records[item]
        record["source_rank"] = rank
        record["source_count"] = sum(bool(record[f"from_{name}"]) for name in PROVENANCE.values())
        result.append(record)
    return result


def select_quota_variant(
    rows: Sequence[Mapping[str, object]],
    *,
    tolerance: float = 0.0005,
) -> str:
    if {str(row["variant"]) for row in rows} != {"Q0", "Q1", "Q2"}:
        raise ValueError("Quota Validation 必须完整包含 Q0/Q1/Q2")
    best_recall = max(float(row["recall_at_500"]) for row in rows)
    near = [row for row in rows if best_recall - float(row["recall_at_500"]) < tolerance]
    content_cap = {"Q0": 0, "Q1": 50, "Q2": 100}
    order = {"Q0": 0, "Q1": 1, "Q2": 2}
    selected = sorted(
        near,
        key=lambda row: (
            content_cap[str(row["variant"])],
            float(row.get("declined_user_rate", 0.0)),
            float(row.get("resource_seconds", 0.0)),
            order[str(row["variant"])],
        ),
    )[0]
    return str(selected["variant"])
