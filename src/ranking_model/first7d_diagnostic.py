from __future__ import annotations

from dataclasses import dataclass


FIRST7D_VERSIONS = ("B0", "L1_600")
FIRST7D_METRICS = ("map_at_12", "ndcg_at_12", "recall_at_12", "hit_rate_at_12")


@dataclass(frozen=True)
class RobustnessDecision:
    direction_status: str
    confidence_status: str


def classify_robustness(delta: float, ci_lower: float, ci_upper: float) -> RobustnessDecision:
    """Apply the preregistered First7d rules without changing the locked winner."""
    direction = "ROBUSTNESS_DIRECTION_PASS" if delta >= 0.0 else "VALIDATION_ROBUSTNESS_NOT_MET"
    if ci_lower > 0.0:
        confidence = "POSITIVE_AND_SIGNIFICANT"
    elif ci_upper < 0.0:
        confidence = "SIGNIFICANT_NEGATIVE"
    else:
        confidence = "NONNEGATIVE_BUT_INCONCLUSIVE" if delta >= 0.0 else "NEGATIVE_BUT_INCONCLUSIVE"
    return RobustnessDecision(direction, confidence)


def segment_review_required(ci_uppers: list[float]) -> bool:
    return any(value < 0.0 for value in ci_uppers)
