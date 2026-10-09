from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np


MISSING_TOKEN = "__MISSING__"


def personalized_serviceable(history_length: int) -> bool:
    return history_length > 0


def recency_weight(rank: int) -> float:
    if rank < 1:
        raise ValueError("recency rank 必须从 1 开始")
    return 1.0 / math.log2(rank + 1.0)


def stable_dev_assignment(user_key: int, seed: int = 20260805, modulus: int = 20) -> bool:
    """只依赖匿名内部 key 的稳定整数混合，与全量 DuckDB SQL 完全一致。"""
    mixed = (int(user_key) * 6364136223846793005 + seed) % 9223372036854775783
    return mixed % modulus == 0


@dataclass(frozen=True)
class Interaction:
    user_key: int
    article_key: int
    interaction_date: date


@dataclass(frozen=True)
class TrainingExample:
    user_key: int
    target_article_key: int
    target_date: date
    history_article_keys: tuple[int, ...]
    history_weights: tuple[float, ...]
    is_dev: bool


def build_training_examples(
    interactions: Sequence[Interaction],
    *,
    max_targets_per_user: int = 3,
    max_history_items: int = 20,
    seed: int = 20260805,
) -> list[TrainingExample]:
    """合成/单元测试版本；全量数据由 runner 使用同一 SQL 语义分区构建。"""
    grouped: dict[int, list[Interaction]] = {}
    for row in interactions:
        grouped.setdefault(row.user_key, []).append(row)
    result: list[TrainingExample] = []
    for user_key, rows in grouped.items():
        first_date: dict[int, date] = {}
        last_by_day_item: dict[tuple[date, int], Interaction] = {}
        for row in rows:
            first_date[row.article_key] = min(first_date.get(row.article_key, row.interaction_date), row.interaction_date)
            last_by_day_item[(row.interaction_date, row.article_key)] = row
        targets: list[tuple[date, int, list[tuple[date, int]]]] = []
        for article_key, target_date in first_date.items():
            # 数据没有日内时间和 order_id，同日其他商品不能虚构为 target 前上下文。
            prior: dict[int, date] = {}
            for row in last_by_day_item.values():
                if row.interaction_date < target_date and row.article_key != article_key:
                    prior[row.article_key] = max(prior.get(row.article_key, row.interaction_date), row.interaction_date)
            if prior:
                ordered = sorted(((d, item) for item, d in prior.items()), key=lambda x: (-x[0].toordinal(), x[1]))
                targets.append((target_date, article_key, ordered[:max_history_items]))
        targets.sort(key=lambda x: (-x[0].toordinal(), x[1]))
        for target_date, article_key, history in targets[:max_targets_per_user]:
            keys = tuple(item for _, item in history)
            result.append(TrainingExample(
                user_key=user_key,
                target_article_key=article_key,
                target_date=target_date,
                history_article_keys=keys,
                history_weights=tuple(recency_weight(i) for i in range(1, len(keys) + 1)),
                is_dev=stable_dev_assignment(user_key, seed),
            ))
    return sorted(result, key=lambda row: (row.user_key, -row.target_date.toordinal(), row.target_article_key))


@dataclass(frozen=True)
class AttributeVocabulary:
    fields: tuple[str, ...]
    groups: tuple[str, ...]
    values: tuple[dict[str, int], ...]

    @classmethod
    def build(
        cls,
        rows: Sequence[Mapping[str, Any]],
        fields: Sequence[Mapping[str, str]],
    ) -> "AttributeVocabulary":
        names = tuple(row["source_column"] for row in fields)
        if len(names) != len(set(names)):
            raise ValueError("canonical code/name 不得重复建模")
        groups = tuple(row["semantic_group"] for row in fields)
        mappings: list[dict[str, int]] = []
        for name in names:
            tokens = sorted({MISSING_TOKEN if row.get(name) in (None, "") else str(row[name]) for row in rows})
            # 每个字段拥有独立 vocabulary，missing token 不跨字段共享 embedding table。
            ordered = [MISSING_TOKEN, *(token for token in tokens if token != MISSING_TOKEN)]
            mappings.append({token: index for index, token in enumerate(ordered)})
        return cls(names, groups, tuple(mappings))

    def encode_row(self, row: Mapping[str, Any]) -> tuple[int, ...]:
        encoded = []
        for name, mapping in zip(self.fields, self.values, strict=True):
            token = MISSING_TOKEN if row.get(name) in (None, "") else str(row[name])
            encoded.append(mapping.get(token, mapping[MISSING_TOKEN]))
        return tuple(encoded)

    def sizes(self) -> tuple[int, ...]:
        return tuple(len(mapping) for mapping in self.values)

    def to_json(self) -> dict[str, Any]:
        return {"fields": self.fields, "groups": self.groups, "values": self.values,
                "missing_token": MISSING_TOKEN}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "AttributeVocabulary":
        return cls(
            tuple(payload["fields"]), tuple(payload["groups"]),
            tuple({str(k): int(v) for k, v in mapping.items()} for mapping in payload["values"]),
        )


def padded_history(
    histories: Sequence[Sequence[int]],
    *,
    max_items: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    items = np.full((len(histories), max_items), -1, dtype=np.int64)
    weights = np.zeros((len(histories), max_items), dtype=np.float32)
    for row_index, history in enumerate(histories):
        for rank, item in enumerate(history[:max_items], 1):
            items[row_index, rank - 1] = int(item)
            weights[row_index, rank - 1] = recency_weight(rank)
    return items, weights
