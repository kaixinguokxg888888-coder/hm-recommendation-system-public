from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class UserMetrics:
    recall: dict[int, float]
    hit_rate: dict[int, float]
    precision_at_12: float
    ap_at_12: float
    candidate_count: int


def user_metrics(
    ranked_items: Sequence[int],
    ground_truth: set[int],
    ks: Sequence[int] = (12, 50, 100, 500),
) -> UserMetrics:
    if not ground_truth:
        raise ValueError("空 ground truth 用户不得进入评估分母")
    if len(ranked_items) != len(set(ranked_items)):
        raise ValueError("候选存在重复 article")
    recall: dict[int, float] = {}
    hit_rate: dict[int, float] = {}
    for k in ks:
        hits = len(set(ranked_items[:k]) & ground_truth)
        recall[k] = hits / len(ground_truth)
        hit_rate[k] = float(hits > 0)
    top12 = ranked_items[:12]
    hits12 = sum(item in ground_truth for item in top12)
    precision = hits12 / len(top12) if top12 else 0.0
    running_hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(top12, start=1):
        if item in ground_truth:
            running_hits += 1
            precision_sum += running_hits / rank
    ap = precision_sum / min(len(ground_truth), 12)
    return UserMetrics(recall, hit_rate, precision, ap, len(ranked_items))


def macro_average(values: Iterable[UserMetrics]) -> dict[str, float]:
    rows = list(values)
    if not rows:
        return {}
    result = {
        "map_at_12": mean(row.ap_at_12 for row in rows),
        "precision_at_12": mean(row.precision_at_12 for row in rows),
    }
    for k in sorted(rows[0].recall):
        result[f"recall_at_{k}"] = mean(row.recall[k] for row in rows)
        result[f"hit_rate_at_{k}"] = mean(row.hit_rate[k] for row in rows)
    return result


def aggregate_segments(
    metrics: Mapping[int, UserMetrics],
    segments: Mapping[int, str],
    main_evaluable: set[int],
) -> dict[str, dict[str, float]]:
    output = {
        "evaluable_warm_users": macro_average(
            metrics[user] for user in sorted(main_evaluable) if user in metrics
        )
    }
    for segment in ("short_history", "warm", "active", "cold"):
        output[
            "cold_history_label_users" if segment == "cold" else segment
        ] = macro_average(
            metrics[user]
            for user in sorted(metrics)
            if segments[user] == segment
        )
    return output
