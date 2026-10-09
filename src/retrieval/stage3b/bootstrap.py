from __future__ import annotations

import math
import random
from statistics import mean
from typing import Sequence


def paired_user_bootstrap(
    base: Sequence[float],
    challenger: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int = 20260803,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    if len(base) != len(challenger):
        raise ValueError("paired bootstrap 两侧用户数必须相同")
    if not base:
        return {
            "user_count": 0,
            "delta_mean": math.nan,
            "ci_lower": math.nan,
            "ci_upper": math.nan,
            "positive_fraction": math.nan,
        }
    if iterations <= 0:
        raise ValueError("bootstrap iterations 必须为正")
    deltas = [right - left for left, right in zip(base, challenger, strict=True)]
    rng = random.Random(seed)
    n = len(deltas)
    estimates = [mean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(iterations)]
    estimates.sort()
    alpha = (1.0 - confidence_level) / 2.0
    lower_index = max(0, min(iterations - 1, int(math.floor(alpha * iterations))))
    upper_index = max(0, min(iterations - 1, int(math.ceil((1 - alpha) * iterations)) - 1))
    return {
        "user_count": n,
        "delta_mean": mean(deltas),
        "ci_lower": estimates[lower_index],
        "ci_upper": estimates[upper_index],
        "positive_fraction": sum(value > 0 for value in estimates) / iterations,
    }
