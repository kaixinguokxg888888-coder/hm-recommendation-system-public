from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Sequence


def canonical_score(value: float) -> str:
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("formal prediction contains NaN/Inf")
    return float(value).hex()


def prediction_query_digest(
    model_id: str, snapshot_id: str, customer_key: int,
    rows: Sequence[tuple[int, int, float]],
) -> tuple[str, str]:
    raw = hashlib.sha256(b"ranking-prediction-raw-query-v1\0")
    ranked = hashlib.sha256(b"ranking-prediction-ranked-query-v1\0")
    raw.update(model_id.encode("utf-8") + b"\0" + snapshot_id.encode("utf-8") + b"\0")
    ranked.update(model_id.encode("utf-8") + b"\0" + snapshot_id.encode("utf-8") + b"\0")
    for article, candidate_rank, score in sorted(rows, key=lambda value: value[1]):
        raw.update(struct.pack("<qI", int(article), int(candidate_rank)))
        raw.update(canonical_score(score).encode("ascii") + b"\0")
    ordered = sorted(rows, key=lambda value: (-value[2], value[1], value[0]))
    for predicted_rank, (article, candidate_rank, _score) in enumerate(ordered, 1):
        ranked.update(struct.pack("<qII", int(article), int(candidate_rank), predicted_rank))
    return raw.hexdigest(), ranked.hexdigest()


def merkle_fingerprint(rows: Iterable[tuple[str, int, str]], domain: str, binding: str) -> str:
    values = sorted(rows, key=lambda value: (value[0], value[1]))
    digest = hashlib.sha256(domain.encode("ascii") + b"\0" + binding.encode("utf-8") + b"\0")
    for snapshot, customer, value in values:
        digest.update(snapshot.encode("utf-8") + b"\0")
        digest.update(struct.pack("<q", int(customer)))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()
