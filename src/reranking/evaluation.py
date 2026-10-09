from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Sequence

from ranking_model.validation_selection import METRICS, binary_top12_metrics

from .contracts import Candidate
from .greedy import counts


def metrics(selected: Sequence[tuple[Candidate, str, int]], positives: set[int], positive_count: int) -> dict[str, float]:
    if positive_count != len(positives) or positive_count <= 0:
        raise ValueError("future-positive denominator accounting mismatch")
    return asdict(binary_top12_metrics([r.article in positives for r, _, _ in selected], positive_count))


def list_stats(rows: Sequence[Candidate]) -> dict[str, float | int]:
    products, garments = counts(rows, "product"), counts(rows, "garment")
    return {"unique_article": len({r.article for r in rows}), "unique_product": len(products),
            "unique_garment": len(garments), "product_excess": sum(max(0, n - 2) for n in products.values()),
            "garment_excess": sum(max(0, n - 4) for n in garments.values()),
            "product_violation": int(any(n > 2 for n in products.values())),
            "garment_violation": int(any(n > 4 for n in garments.values())),
            "tail_count": sum(r.tail for r in rows)}


def diagnostics(ordered: Sequence[Candidate], selected: Sequence[tuple[Candidate, str, int]], window: int | None) -> dict:
    baseline = ordered[:12]
    result = [r for r, _, _ in selected]
    before, after = list_stats(baseline), list_stats(result)
    level = max((level for _, _, level in selected), default=0)
    changed = sum(a.article != b.article for a, b in zip(baseline, result)) + abs(len(baseline) - len(result))
    explored = [r for r, reason, _ in selected if reason == "tail_exploration"]
    eligible = int(window is not None and any(r.tail and r.rank <= window for r in ordered))
    ranks = [r.rank for r in result]
    return {**{"before_" + key: value for key, value in before.items()},
            **{"after_" + key: value for key, value in after.items()},
            "changed": int(changed > 0), "changed_positions": changed,
            "mean_original_rank": sum(ranks) / len(ranks) if ranks else 0.0,
            "max_original_rank": max(ranks, default=0),
            "mean_absolute_displacement": sum(abs(r.rank - pos) for pos, r in enumerate(result, 1)) / len(result) if result else 0.0,
            "max_relaxation_level": level, "relaxed_garment_count": after["garment_excess"],
            "relaxed_product_count": after["product_excess"], "fill_failure": int(len(result) < 12),
            "insufficient_unique_candidates": int(len({r.article for r in ordered}) < 12),
            "duplicate_removed_count": len(ordered) - len({r.article for r in ordered}),
            "exploration_eligible": eligible, "exploration_received": len(explored),
            "explored_original_rank": explored[0].rank if explored else 0,
            **{"level_" + str(n): int(level == n) for n in range(4)}}


def relevance_guard(comparisons: dict[str, dict], baseline_metrics: dict[str, float], relative: float) -> bool:
    return all(comparisons[metric]["delta_mean"] >= -relative * baseline_metrics[metric]
               and comparisons[metric]["ci_lower"] >= -relative * baseline_metrics[metric]
               for metric in ("map_at_12", "ndcg_at_12"))


def select_config(summary: dict[str, dict], comparisons: dict[tuple[str, str], dict], selection: dict) -> tuple[str, list[dict]]:
    chosen = "R1"
    checks = []
    baseline = summary["R0"]
    guard = lambda version: relevance_guard(comparisons[(version, "R0")], baseline, selection["r0_loss_budget_relative"])
    r2 = guard("R2") and summary["R2"]["after_product_violation"] < baseline["after_product_violation"] and summary["R2"]["after_unique_product"] > baseline["after_unique_product"]
    checks.append({"variant": "R2", "relevance_guard": guard("R2"), "eligible": r2})
    if r2:
        chosen = "R2"
    r3 = r2 and guard("R3") and summary["R3"]["after_unique_garment"] - summary["R2"]["after_unique_garment"] >= selection["r3_min_unique_garment_gain_vs_r2"] and summary["R3"]["after_garment_violation"] < summary["R2"]["after_garment_violation"]
    checks.append({"variant": "R3", "relevance_guard": guard("R3"), "eligible": r3})
    if r3:
        chosen = "R3"
    for version in ("R4_N20", "R4_N30", "R4_N50"):
        extra = relevance_guard(comparisons[(version, "R3")], summary["R3"], selection["r4_extra_loss_budget_relative_vs_r3"])
        eligible = r3 and guard(version) and extra and summary[version]["after_tail_count"] - summary["R3"]["after_tail_count"] >= selection["r4_min_tail_count_gain_vs_r3"]
        checks.append({"variant": version, "relevance_guard": guard(version), "extra_exploration_guard": extra, "eligible": eligible})
        if eligible and chosen == "R3":
            chosen = version
    return chosen, checks
