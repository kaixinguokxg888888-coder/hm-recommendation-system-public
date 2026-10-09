from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


ORDERING_VERSION = "stage_4b4_training_global_query_order_v2"


class TrainingOrderError(RuntimeError):
    pass


def canonical_payload_fingerprint(payload: Any) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def query_boundary_digest(
    snapshot_id: str,
    customer_key: int,
    query_order: int,
    start_offset: int,
    article_keys: np.ndarray,
    candidate_ranks: np.ndarray,
    labels: np.ndarray,
) -> str:
    digest = hashlib.sha256(b"stage-4b4-query-boundary-v2\0")
    digest.update(snapshot_id.encode("utf-8") + b"\0")
    digest.update(struct.pack("<qqq", int(customer_key), int(query_order), int(start_offset)))
    rows = np.empty(len(article_keys), dtype=[("article", "<i8"), ("rank", "<i4"), ("label", "u1")])
    rows["article"] = article_keys
    rows["rank"] = candidate_ranks
    rows["label"] = labels
    digest.update(rows.tobytes())
    return digest.hexdigest()


def matrix_order_fingerprint(boundaries: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256(ORDERING_VERSION.encode("ascii") + b"\0")
    for row in boundaries:
        digest.update(str(row["snapshot_id"]).encode("utf-8") + b"\0")
        digest.update(struct.pack(
            "<qqqqqq",
            int(row["customer_key"]), int(row["query_order"]), int(row["start_offset"]),
            int(row["end_offset"]), int(row["group_size"]), int(row["positive_count"]),
        ))
        digest.update(bytes.fromhex(str(row["ordered_row_digest"])))
    return digest.hexdigest()


def audit_ordered_arrays(
    *,
    snapshot_id: str,
    customer_keys: np.ndarray,
    query_orders: np.ndarray,
    article_keys: np.ndarray,
    candidate_ranks: np.ndarray,
    labels: np.ndarray,
    expected_queries: Sequence[tuple[str, int, int, int, int, int]],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    rows = len(labels)
    if not all(len(value) == rows for value in (customer_keys, query_orders, article_keys, candidate_ranks)):
        raise TrainingOrderError("ordered sidecar lengths differ")
    boundaries: list[dict[str, Any]] = []
    audit = {
        "query_count": len(expected_queries), "row_count": rows, "query_position_mismatch": 0,
        "group_size_position_mismatch": 0, "positive_count_position_mismatch": 0,
        "sampled_zero_count_position_mismatch": 0, "row_key_mismatch": 0,
        "candidate_rank_order_violations": 0, "all_zero_groups": 0,
        "cross_query_boundary_groups": 0, "duplicate_row_keys": 0,
    }
    offset = 0
    seen_queries: set[tuple[str, int]] = set()
    for expected_snapshot, expected_customer, expected_order, expected_size, expected_positive, expected_zero in expected_queries:
        end = offset + int(expected_size)
        if end > rows:
            audit["group_size_position_mismatch"] += 1
            break
        customers = customer_keys[offset:end]
        orders = query_orders[offset:end]
        articles = article_keys[offset:end]
        ranks = candidate_ranks[offset:end]
        group_labels = labels[offset:end]
        actual_customer = int(customers[0]) if len(customers) else -1
        actual_order = int(orders[0]) if len(orders) else -1
        if actual_customer != int(expected_customer) or actual_order != int(expected_order):
            audit["query_position_mismatch"] += 1
        if len(customers) != expected_size:
            audit["group_size_position_mismatch"] += 1
        if len(set(map(int, customers))) != 1 or len(set(map(int, orders))) != 1:
            audit["cross_query_boundary_groups"] += 1
        positive = int(group_labels.sum())
        zero = int(len(group_labels) - positive)
        if positive != int(expected_positive):
            audit["positive_count_position_mismatch"] += 1
        if zero != int(expected_zero):
            audit["sampled_zero_count_position_mismatch"] += 1
        if positive == 0:
            audit["all_zero_groups"] += 1
        if len(ranks) > 1:
            invalid = (ranks[1:] < ranks[:-1]) | ((ranks[1:] == ranks[:-1]) & (articles[1:] < articles[:-1]))
            audit["candidate_rank_order_violations"] += int(invalid.sum())
        audit["duplicate_row_keys"] += int(len(articles) - len(set(map(int, articles))))
        query_key = (expected_snapshot, int(expected_customer))
        if query_key in seen_queries:
            audit["duplicate_row_keys"] += len(articles)
        seen_queries.add(query_key)
        digest = query_boundary_digest(
            snapshot_id, actual_customer, actual_order, offset, articles, ranks, group_labels
        )
        boundaries.append({
            "snapshot_id": snapshot_id, "customer_key": actual_customer, "query_order": actual_order,
            "start_offset": offset, "end_offset": end, "group_size": len(group_labels),
            "positive_count": positive, "sampled_zero_count": zero, "ordered_row_digest": digest,
        })
        offset = end
    if offset != rows:
        audit["group_size_position_mismatch"] += 1
    violations = sum(value for key, value in audit.items() if key not in {"query_count", "row_count"})
    if violations:
        raise TrainingOrderError(json.dumps(audit, sort_keys=True))
    return audit, boundaries


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)
