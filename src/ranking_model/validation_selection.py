from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

import numpy as np


VERSIONS = ("B0", "L0_100", "L0_300", "L0_600", "L1_100", "L1_300", "L1_600")
METRICS = ("map_at_12", "ndcg_at_12", "recall_at_12", "hit_rate_at_12")


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_fingerprint(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def query_metric_digest(
    version: str,
    snapshot_id: str,
    customer_key: int,
    metrics: Mapping[str, float | int | bool | str],
) -> str:
    digest = hashlib.sha256(b"stage-4b4-validation-query-metrics-v1\0")
    digest.update(version.encode("ascii") + b"\0" + snapshot_id.encode("ascii") + b"\0")
    digest.update(struct.pack("<q", int(customer_key)))
    for name in sorted(metrics):
        value = metrics[name]
        digest.update(name.encode("ascii") + b"=")
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("query metric contains NaN/Inf")
            encoded = value.hex()
        else:
            encoded = str(value)
        digest.update(encoded.encode("utf-8") + b"\0")
    return digest.hexdigest()


def query_merkle(rows: Iterable[tuple[str, int, str]], *, binding: str) -> str:
    digest = hashlib.sha256(b"stage-4b4-validation-query-merkle-v1\0" + binding.encode("ascii") + b"\0")
    for snapshot_id, customer_key, value in sorted(rows, key=lambda row: (row[0], row[1])):
        digest.update(snapshot_id.encode("ascii") + b"\0")
        digest.update(struct.pack("<q", int(customer_key)))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def paired_bootstrap(
    baseline: Sequence[float], challenger: Sequence[float], *, iterations: int, seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    if len(baseline) != len(challenger):
        raise ValueError("paired bootstrap requires identical query membership")
    if len(baseline) == 0:
        raise ValueError("paired bootstrap population is empty")
    deltas = np.asarray(challenger, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if not np.isfinite(deltas).all():
        raise ValueError("paired bootstrap contains NaN/Inf")
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        estimates[index] = float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))]))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "query_count": int(len(deltas)),
        "delta_mean": mean(float(value) for value in deltas),
        "delta_median": median(float(value) for value in deltas),
        "ci_lower": float(np.quantile(estimates, alpha)),
        "ci_upper": float(np.quantile(estimates, 1.0 - alpha)),
        "positive_fraction": float(np.mean(estimates > 0)),
    }


def select_winner(aggregate: Mapping[str, Mapping[str, float]]) -> str:
    """Frozen lexicographic rule; final key prefers the lower-complexity order."""
    missing = set(VERSIONS) - set(aggregate)
    if missing:
        raise ValueError(f"missing versions: {sorted(missing)}")
    complexity = {version: index for index, version in enumerate(VERSIONS)}
    return max(
        VERSIONS,
        key=lambda version: (
            *(float(aggregate[version][metric]) for metric in METRICS),
            -complexity[version],
        ),
    )


def value_gate(winner: str, map_delta: float, ci_lower: float) -> str:
    if winner == "B0" or map_delta <= 0.0:
        return "FAILED_VALUE_GATE"
    if ci_lower > 0.0:
        return "PASSED_VALUE_GATE"
    return "INCONCLUSIVE_VALUE_GATE"


@dataclass(frozen=True)
class BinaryTopKMetrics:
    map_at_12: float
    ndcg_at_12: float
    recall_at_12: float
    hit_rate_at_12: float


def binary_top12_metrics(ranked_hits: Sequence[bool], positive_count: int) -> BinaryTopKMetrics:
    if positive_count <= 0:
        raise ValueError("positive_count must be positive")
    hits = list(ranked_hits[:12])
    running = 0
    precision_sum = 0.0
    dcg = 0.0
    for rank, hit in enumerate(hits, 1):
        if hit:
            running += 1
            precision_sum += running / rank
            dcg += 1.0 / math.log2(rank + 1)
    ideal = min(positive_count, 12)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return BinaryTopKMetrics(
        precision_sum / ideal,
        dcg / idcg,
        running / positive_count,
        float(running > 0),
    )
