from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any, Iterable, Mapping


def canonical_sampling_payload(snapshot_id: str, customer_key: int, article_key: int, seed: int) -> str:
    """Build the exact UTF-8 payload frozen by Stage 4A."""
    return f"{snapshot_id}|{int(customer_key)}|{int(article_key)}|{int(seed)}"


def sampling_digest(snapshot_id: str, customer_key: int, article_key: int, seed: int) -> str:
    return hashlib.sha256(
        canonical_sampling_payload(snapshot_id, customer_key, article_key, seed).encode("utf-8")
    ).hexdigest()


def classify_query(history_count: int, future_count: int, candidate_positive_count: int) -> dict[str, bool]:
    main = future_count >= 1 and history_count >= 1
    cold = future_count >= 1 and history_count == 0
    miss = main and candidate_positive_count == 0
    return {
        "is_historical_evaluable_warm": main,
        "is_cold_history_label_user": cold,
        "is_retrieval_miss": miss,
        "is_all_zero_relevance_group": miss,
        # Retrieval miss 对端到端评估是真实失败，因此保留完整 500；但全零 query 无排序 pair，不能进 optimizer。
        "full_main_evaluation_eligible": main,
        "cold_evaluation_eligible": cold,
        "optimizer_eligible": main and candidate_positive_count >= 1,
    }


def deterministic_sample_rows(
    rows: Iterable[Mapping[str, Any]], *, seed: int = 20260811, zero_cap: int = 100,
) -> list[dict[str, Any]]:
    """Small-table reference implementation used by unit tests and smoke checks."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for value in rows:
        row = dict(value)
        groups[(str(row["snapshot_id"]), int(row["customer_key"]))].append(row)
    output: list[dict[str, Any]] = []
    for (snapshot, customer), group in groups.items():
        positives = [row for row in group if int(row["label"]) == 1]
        zeros = [row for row in group if int(row["label"]) == 0]
        zeros.sort(key=lambda row: (
            sampling_digest(snapshot, customer, int(row["article_key"]), seed),
            int(row["article_key"]),
        ))
        # candidate_rank 是 frozen retrieval prior；采样只选 membership，绝不重新编号。
        output.extend(positives)
        output.extend(zeros[:zero_cap])
    return sorted(output, key=lambda row: (
        str(row["snapshot_id"]), int(row["customer_key"]), int(row["candidate_rank"])
    ))


def label_candidates(
    candidates: Iterable[Mapping[str, Any]], positives: set[tuple[str, int, int]],
) -> list[dict[str, Any]]:
    """Annotate a frozen candidate universe without changing membership, rank, or provenance."""
    output = []
    for source in candidates:
        row = dict(source)
        key = (str(row["snapshot_id"]), int(row["customer_key"]), int(row["article_key"]))
        # label=0 只表示未来窗未观察到购买，不能解释为 dislike 或真实负反馈。
        row["label"] = int(key in positives)
        output.append(row)
    return output
