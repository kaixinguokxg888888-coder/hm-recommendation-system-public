from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse


MISSING_VALUE = "__MISSING__"


@dataclass(frozen=True)
class CanonicalField:
    semantic_group: str
    semantic_field: str
    source_column: str
    display_column: str


def field_token(field_name: str, value: object) -> str:
    normalized = MISSING_VALUE if value is None or str(value) == "" else str(value)
    return f"{field_name}={normalized}"


def validate_canonical_fields(fields: Sequence[CanonicalField]) -> None:
    semantic = [field.semantic_field for field in fields]
    sources = [field.source_column for field in fields]
    if len(semantic) != len(set(semantic)) or len(sources) != len(set(sources)):
        raise ValueError("canonical semantic/source field 不得重复建模")
    for field in fields:
        if not field.semantic_group or not field.semantic_field or not field.source_column:
            raise ValueError("canonical field 定义不完整")


def idf_value(n_items: int, document_frequency: int) -> float:
    if n_items <= 0 or document_frequency < 0 or document_frequency > n_items:
        raise ValueError("IDF N/df 非法")
    return math.log((n_items + 1) / (document_frequency + 1)) + 1.0


def build_sparse_item_matrix(
    field_values: Mapping[str, np.ndarray],
    fields: Sequence[CanonicalField],
    *,
    use_idf: bool,
    group_balance: bool,
) -> tuple[sparse.csr_matrix, list[dict[str, Any]]]:
    validate_canonical_fields(fields)
    if not fields:
        raise ValueError("Content 至少需要一个 canonical field")
    n_items = len(next(iter(field_values.values())))
    if any(len(field_values[field.source_column]) != n_items for field in fields):
        raise ValueError("属性列长度不一致")
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    values: list[np.ndarray] = []
    token_rows: list[dict[str, Any]] = []
    column_offset = 0
    field_weight_arrays: dict[str, np.ndarray] = {}
    field_column_arrays: dict[str, np.ndarray] = {}
    for field in fields:
        raw = np.asarray(field_values[field.source_column], dtype=object)
        normalized = np.asarray(
            [MISSING_VALUE if value is None or str(value) == "" else str(value) for value in raw],
            dtype=object,
        )
        uniques, inverse, counts = np.unique(normalized, return_inverse=True, return_counts=True)
        weights_by_value = np.asarray(
            [idf_value(n_items, int(count)) if use_idf else 1.0 for count in counts],
            dtype=np.float64,
        )
        item_weights = weights_by_value[inverse]
        item_columns = inverse.astype(np.int64) + column_offset
        field_weight_arrays[field.source_column] = item_weights
        field_column_arrays[field.source_column] = item_columns
        for local, (token_value, count, weight) in enumerate(zip(uniques, counts, weights_by_value, strict=True)):
            token_rows.append(
                {
                    "semantic_group": field.semantic_group,
                    "semantic_field": field.semantic_field,
                    "source_column": field.source_column,
                    "token": field_token(field.source_column, token_value),
                    "document_frequency": int(count),
                    "document_frequency_rate": float(count) / n_items,
                    "idf": float(weight),
                    "matrix_column": column_offset + local,
                }
            )
        column_offset += len(uniques)

    groups = sorted({field.semantic_group for field in fields})
    normalized_weights: dict[str, np.ndarray] = {}
    if group_balance:
        # 先组内归一化，再给每个非空组相同先验；missing 也是显式 token，
        # 因而每个已注册组始终非空且不会静默丢失属性语义。
        group_prior = 1.0 / math.sqrt(len(groups))
        for group in groups:
            members = [field for field in fields if field.semantic_group == group]
            denominator = np.sqrt(
                sum(field_weight_arrays[field.source_column] ** 2 for field in members)
            )
            denominator[denominator == 0] = 1.0
            for field in members:
                normalized_weights[field.source_column] = (
                    field_weight_arrays[field.source_column] / denominator * group_prior
                )
    else:
        denominator = np.sqrt(
            sum(field_weight_arrays[field.source_column] ** 2 for field in fields)
        )
        denominator[denominator == 0] = 1.0
        for field in fields:
            normalized_weights[field.source_column] = (
                field_weight_arrays[field.source_column] / denominator
            )

    item_rows = np.arange(n_items, dtype=np.int64)
    for field in fields:
        rows.append(item_rows)
        cols.append(field_column_arrays[field.source_column])
        values.append(normalized_weights[field.source_column])
    matrix = sparse.coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, column_offset),
        dtype=np.float64,
    ).tocsr()
    return matrix, token_rows


def top_k_cosine_block(
    matrix: sparse.csr_matrix,
    article_keys: np.ndarray,
    start: int,
    stop: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if top_k <= 0:
        raise ValueError("top_k 必须为正")
    scores = (matrix[start:stop] @ matrix.T).tocsr()
    left_values: list[int] = []
    right_values: list[int] = []
    similarities: list[float] = []
    ranks: list[int] = []
    for local, row_index in enumerate(range(start, stop)):
        begin, end = scores.indptr[local], scores.indptr[local + 1]
        indices = scores.indices[begin:end]
        data = scores.data[begin:end]
        keep = (indices != row_index) & np.isfinite(data) & (data > 0)
        indices, data = indices[keep], data[keep]
        if len(indices) > top_k:
            chosen = np.argpartition(-data, top_k - 1)[:top_k]
            indices, data = indices[chosen], data[chosen]
        order = np.lexsort((article_keys[indices], -data))
        indices, data = indices[order][:top_k], data[order][:top_k]
        count = len(indices)
        left_values.extend([int(article_keys[row_index])] * count)
        right_values.extend(int(value) for value in article_keys[indices])
        similarities.extend(float(value) for value in data)
        ranks.extend(range(1, count + 1))
    return (
        np.asarray(left_values, dtype=np.int64),
        np.asarray(right_values, dtype=np.int64),
        np.asarray(similarities, dtype=np.float64),
        np.asarray(ranks, dtype=np.int32),
    )


def select_content_variant(rows: Sequence[Mapping[str, object]]) -> str:
    if {str(row["variant"]) for row in rows} != {"C0", "C1", "C2", "C3"}:
        raise ValueError("Validation 必须完整包含 C0-C3")
    order = {name: index for index, name in enumerate(("C0", "C1", "C2", "C3"))}
    return str(
        sorted(
            rows,
            key=lambda row: (
                -float(row["q1_recall_at_500"]),
                -float(row["q1_hit_rate_at_500"]),
                -int(row["content_exclusive_hits"]),
                float(row["declined_user_rate"]),
                int(row["field_count"]),
                order[str(row["variant"])],
            ),
        )[0]["variant"]
    )


def paired_bootstrap(
    base: Sequence[float],
    challenger: Sequence[float],
    *,
    iterations: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    if len(base) != len(challenger):
        raise ValueError("paired bootstrap 两侧用户数必须相同")
    if not base:
        return {"user_count": 0, "delta_mean": math.nan, "delta_median": math.nan,
                "ci_lower": math.nan, "ci_upper": math.nan, "positive_fraction": math.nan}
    deltas = np.asarray(challenger, dtype=np.float64) - np.asarray(base, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    # 分批抽样避免一次创建 iterations×users 的大矩阵。
    for index in range(iterations):
        estimates[index] = float(np.mean(deltas[rng.integers(0, len(deltas), len(deltas))]))
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "user_count": len(deltas),
        "delta_mean": mean(float(value) for value in deltas),
        "delta_median": median(float(value) for value in deltas),
        "ci_lower": float(np.quantile(estimates, alpha)),
        "ci_upper": float(np.quantile(estimates, 1.0 - alpha)),
        "positive_fraction": float(np.mean(estimates > 0)),
    }
