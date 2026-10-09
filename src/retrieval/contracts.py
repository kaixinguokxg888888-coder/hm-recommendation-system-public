from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping


CANDIDATE_FIELDS = (
    "snapshot_id",
    "as_of_date",
    "customer_key",
    "article_key",
    "source",
    "source_variant",
    "source_rank",
    "source_score",
    "candidate_universe_version",
    "feedback_version",
    "population_version",
    "dataset_fingerprint",
    "config_fingerprint",
)
FORBIDDEN_ID_FIELDS = {"customer_id", "article_id", "postal_code", "user_key"}


class CandidateProtocolError(ValueError):
    """候选违反统一协议。"""


@dataclass(frozen=True)
class CandidateContext:
    snapshot_id: str
    as_of_date: date
    source: str
    source_variant: str
    feedback_version: str
    population_version: str
    dataset_fingerprint: str
    config_fingerprint: str
    max_candidates: int = 500


def validate_candidates(
    candidates: Iterable[Mapping[str, Any]],
    context: CandidateContext,
    legal_universe: Mapping[int, set[int]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in candidates]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        missing = set(CANDIDATE_FIELDS) - set(row)
        if missing:
            raise CandidateProtocolError(f"候选缺少字段: {sorted(missing)}")
        forbidden = FORBIDDEN_ID_FIELDS & set(row)
        if forbidden:
            raise CandidateProtocolError(f"候选包含禁止字段: {sorted(forbidden)}")
        if any(row[field] is None for field in CANDIDATE_FIELDS):
            raise CandidateProtocolError("候选协议字段不得为空")
        expected = {
            "snapshot_id": context.snapshot_id,
            "as_of_date": context.as_of_date,
            "source": context.source,
            "source_variant": context.source_variant,
            "feedback_version": context.feedback_version,
            "population_version": context.population_version,
            "dataset_fingerprint": context.dataset_fingerprint,
            "config_fingerprint": context.config_fingerprint,
        }
        for field, value in expected.items():
            if row[field] != value:
                raise CandidateProtocolError(f"{field} 与评估上下文不匹配")
        user = int(row["customer_key"])
        item = int(row["article_key"])
        if user not in legal_universe or item not in legal_universe[user]:
            raise CandidateProtocolError("候选商品不属于用户合法 candidate universe")
        grouped.setdefault(user, []).append(row)

    for user, values in grouped.items():
        if len(values) > context.max_candidates:
            raise CandidateProtocolError(f"用户 {user} 候选超过声明上限")
        items = [int(row["article_key"]) for row in values]
        ranks = [int(row["source_rank"]) for row in values]
        if len(items) != len(set(items)):
            raise CandidateProtocolError("同一用户候选 article_key 重复")
        if len(ranks) != len(set(ranks)):
            raise CandidateProtocolError("同一用户 source_rank 重复")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise CandidateProtocolError("source_rank 必须从 1 开始且连续")
    return rows
