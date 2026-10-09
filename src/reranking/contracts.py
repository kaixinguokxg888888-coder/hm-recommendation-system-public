from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa

INPUT_SCHEMA = pa.schema([
    ("snapshot_id", pa.string()), ("customer_key", pa.int64()), ("article_key", pa.int64()),
    ("candidate_rank", pa.int32()), ("raw_score", pa.float64()),
    ("predicted_rank", pa.int32()), ("partition_id", pa.int32()),
])
OUTPUT_SCHEMA = pa.schema([
    ("snapshot_id", pa.string()), ("customer_key", pa.int64()), ("article_key", pa.int64()),
    ("partition_id", pa.int32()), ("rerank_position", pa.int32()),
    ("original_predicted_rank", pa.int32()), ("raw_score", pa.float64()),
    ("candidate_rank", pa.int32()), ("product_code", pa.int64()),
    ("article_garment_group_code", pa.int32()), ("is_tail_item", pa.bool_()),
    ("selection_reason", pa.string()), ("relaxation_level", pa.int32()),
])
REASONS = {"base", "constraint_fill", "tail_exploration", "relax_garment", "relax_product", "final_fill"}


@dataclass(frozen=True, slots=True)
class Candidate:
    article: int
    rank: int
    score: float
    candidate_rank: int
    product: int | None
    garment: int | None
    tail: bool = False


def normalize(table: pa.Table, partition: int, *, snapshot: str = "validation_14d") -> pa.Table:
    columns: dict[str, Any] = {}
    for field in INPUT_SCHEMA:
        if field.name == "partition_id" and field.name not in table.column_names:
            columns[field.name] = pa.array([partition] * len(table), type=field.type)
        else:
            if field.name not in table.column_names:
                raise ValueError("missing normalized input field: " + field.name)
            column = table[field.name]
            if column.null_count:
                raise ValueError("null normalized input field: " + field.name)
            columns[field.name] = column.cast(field.type, safe=True)
    result = pa.table(columns, schema=INPUT_SCHEMA)
    if set(result["snapshot_id"].to_pylist()) != {snapshot}:
        raise ValueError("snapshot mismatch")
    parts = result["partition_id"].to_numpy()
    customers = result["customer_key"].to_numpy()
    if np.any(parts != partition) or np.any(customers % 32 != partition):
        raise ValueError("partition identity mismatch")
    if not np.isfinite(result["raw_score"].to_numpy()).all():
        raise ValueError("non-finite score")
    return result


def order_query(rows: Sequence[Candidate], *, formal: bool = False) -> list[Candidate]:
    if formal:
        if len(rows) != 500 or len({r.article for r in rows}) != 500:
            raise ValueError("formal full500/uniqueness violation")
        if {r.candidate_rank for r in rows} != set(range(1, 501)):
            raise ValueError("candidate rank integrity violation")
    ranks = [r.rank for r in rows]
    if len(set(ranks)) != len(rows) or any(r < 1 or r > len(rows) for r in ranks):
        raise ValueError("predicted rank integrity violation")
    # 正式排名已经冻结，用固定长度索引恢复顺序，避免每 query 再做全排序。
    ordered: list[Any] = [None] * len(rows)
    for row in rows:
        if not np.isfinite(row.score):
            raise ValueError("non-finite score")
        ordered[row.rank - 1] = row
    if formal and any(
        (-a.score, a.candidate_rank, a.article) > (-b.score, b.candidate_rank, b.article)
        for a, b in zip(ordered, ordered[1:])
    ):
        raise ValueError("frozen ranking tie-break violation")
    return ordered


def candidates_from_columns(columns: Mapping[str, list], metadata: Mapping[int, tuple], start: int, end: int) -> list[Candidate]:
    result = []
    for i in range(start, end):
        article = columns["article_key"][i]
        if article not in metadata:
            raise ValueError("metadata join lost candidate")
        product, garment, support, tail = metadata[article]
        if support <= 0 or product is None or garment is None or garment == 0:
            raise ValueError("formal candidate missing history-safe metadata")
        result.append(Candidate(article, columns["predicted_rank"][i], columns["raw_score"][i],
                                columns["candidate_rank"][i], product, garment, tail))
    return result


def output_rows(customer: int, partition: int, selected: Sequence[tuple[Candidate, str, int]]) -> list[dict]:
    return [{"snapshot_id": "validation_14d", "customer_key": customer, "article_key": row.article,
             "partition_id": partition, "rerank_position": pos, "original_predicted_rank": row.rank,
             "raw_score": row.score, "candidate_rank": row.candidate_rank, "product_code": row.product,
             "article_garment_group_code": row.garment, "is_tail_item": row.tail,
             "selection_reason": reason, "relaxation_level": level}
            for pos, (row, reason, level) in enumerate(selected, 1)]


def validate_selection(selected: Sequence[tuple[Candidate, str, int]], input_rows: Sequence[Candidate], k: int = 12) -> None:
    articles = [r.article for r, _, _ in selected]
    if len(articles) != len(set(articles)) or len(articles) != min(k, len({r.article for r in input_rows})):
        raise ValueError("output uniqueness/fill violation")
    if not set(articles) <= {r.article for r in input_rows}:
        raise ValueError("output item outside original candidates")
    if any(reason not in REASONS or level not in range(4) for _, reason, level in selected):
        raise ValueError("selection audit enum violation")
    if [r.rank for r, _, _ in selected] != sorted(r.rank for r, _, _ in selected):
        raise ValueError("output original order changed")
