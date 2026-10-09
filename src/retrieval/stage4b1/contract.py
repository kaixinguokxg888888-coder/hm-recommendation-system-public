from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data_foundation.manifest import file_fingerprint
from retrieval.stage3e.contract import semantic_candidate_fingerprint


SOURCES = ("repurchase", "itemcf", "two_tower", "trend", "popularity")
FORBIDDEN_PUBLIC_FIELDS = {"customer_id", "article_id", "postal_code"}
FORBIDDEN_QUERY_SCOPE_FIELDS = {
    "article_id", "article_key", "customer_id", "future_label", "label",
    "positive_count", "relevance",
}
WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:\\|[a-z]:/|\\users\\|/users/)")


class HistoricalReplayError(ValueError):
    """Raised when a mandatory Stage 4B-1 contract is violated."""


def validate_registered_snapshot(snapshot: Mapping[str, Any]) -> None:
    expected = {
        "snapshot_id": "ranking_train_14d_v1",
        "as_of_date": "2020-08-11",
        "request_date": "2020-08-12",
        "history_cutoff": "2020-08-11",
    }
    for key, value in expected.items():
        if snapshot.get(key) != value:
            raise HistoricalReplayError(f"historical snapshot mismatch: {key}")


def assert_history_cutoff(event_dates: Iterable[str], cutoff: str = "2020-08-11") -> None:
    if any(str(value) > cutoff for value in event_dates):
        raise HistoricalReplayError("historical snapshot reads future history")


def forbidden_query_scope_fields(columns: Iterable[str]) -> list[str]:
    """Future labels may enumerate users, but request scope cannot carry item supervision."""
    return sorted({str(column).lower() for column in columns} & FORBIDDEN_QUERY_SCOPE_FIELDS)


def validate_formal_historical_lineage(
    lineage: Mapping[str, Any], *, snapshot_id: str = "ranking_train_14d_v1", as_of_date: str = "2020-08-11",
) -> None:
    if lineage.get("model_role") != "formal_historical_full_refit":
        raise HistoricalReplayError("formal historical lineage rejects selection-only checkpoint")
    if lineage.get("snapshot_id") != snapshot_id or lineage.get("as_of_date") != as_of_date:
        raise HistoricalReplayError("formal historical lineage snapshot/as-of mismatch")
    if lineage.get("uses_all_eligible_samples") is not True:
        raise HistoricalReplayError("formal historical lineage is not full refit")
    if int(lineage.get("internal_dev_examples", -1)) != 0:
        raise HistoricalReplayError("formal historical full refit must use internal dev=0")
    if int(lineage.get("locked_epoch", -1)) != 1:
        raise HistoricalReplayError("formal historical full refit epoch mismatch")


def protocol_identity_rows(
    actual: Mapping[str, Any], frozen: Mapping[str, Any], fields: Sequence[str],
) -> list[dict[str, Any]]:
    rows = []
    for field in fields:
        left = actual.get(field)
        right = frozen.get(field)
        rows.append({
            "protocol_field": field,
            "historical_value": json.dumps(left, sort_keys=True, separators=(",", ":")),
            "frozen_value": json.dumps(right, sort_keys=True, separators=(",", ":")),
            "status": "PASS" if left == right else "FAIL",
        })
    return rows


def assert_protocol_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    failed = [row["protocol_field"] for row in rows if row.get("status") != "PASS"]
    if failed:
        raise HistoricalReplayError(f"HISTORICAL_PROTOCOL_IDENTITY failed: {failed}")


def fuse_candidate_sources(
    *,
    snapshot_id: str,
    customer_key: int,
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
    caps: Mapping[str, int],
    candidate_budget: int,
) -> list[dict[str, Any]]:
    """Reference implementation of the frozen pre-dedup source-cap union.

    This small in-memory helper is for unit/synthetic checks. Formal data-scale
    materialization uses the same ordering semantics through bounded DuckDB jobs.
    """
    order = ("repurchase", "itemcf", "two_tower", "trend")
    merged: dict[int, dict[str, Any]] = {}
    insertion: list[int] = []
    for source in order:
        for row in list(sources.get(source, ()))[: int(caps[source])]:
            article = int(row["article_key"])
            if article not in merged:
                merged[article] = {"snapshot_id": snapshot_id, "customer_key": customer_key, "article_key": article}
                insertion.append(article)
            target = merged[article]
            target[f"from_{source}"] = True
            target[f"{source}_rank"] = int(row["source_rank"])
            target[f"{source}_score"] = row.get("source_score")
    for article in insertion:
        row = merged[article]
        for source in SOURCES:
            row.setdefault(f"from_{source}", False)
            row.setdefault(f"{source}_rank", None)
            row.setdefault(f"{source}_score", None)
    selected = insertion[:candidate_budget]
    for row in sources.get("popularity", ()):
        if len(selected) >= candidate_budget:
            break
        article = int(row["article_key"])
        if article in merged:
            continue
        merged[article] = {
            "snapshot_id": snapshot_id, "customer_key": customer_key, "article_key": article,
            **{f"from_{source}": source == "popularity" for source in SOURCES},
            **{f"{source}_rank": (int(row["source_rank"]) if source == "popularity" else None) for source in SOURCES},
            **{f"{source}_score": (row.get("source_score") if source == "popularity" else None) for source in SOURCES},
        }
        selected.append(article)
    if len(selected) != candidate_budget:
        raise HistoricalReplayError("popularity unique fill did not reach candidate budget")
    result = []
    for rank, article in enumerate(selected, 1):
        row = dict(merged[article])
        row["candidate_rank"] = rank
        contributors = [source for source in SOURCES if row[f"from_{source}"]]
        row["source_count"] = len(contributors)
        row["primary_source"] = contributors[0] if contributors != ["popularity"] else "popularity_fallback"
        result.append(row)
    return result


def validate_candidate_rows(rows: Sequence[Mapping[str, Any]], *, budget: int = 500) -> None:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["snapshot_id"]), int(row["customer_key"]))].append(row)
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda row: int(row["candidate_rank"]))
        if len(ordered) != budget:
            raise HistoricalReplayError(f"candidate budget mismatch: {key}")
        articles = [int(row["article_key"]) for row in ordered]
        ranks = [int(row["candidate_rank"]) for row in ordered]
        if len(set(articles)) != budget:
            raise HistoricalReplayError(f"duplicate candidate: {key}")
        if ranks != list(range(1, budget + 1)):
            raise HistoricalReplayError(f"candidate rank discontinuity: {key}")
        for row in ordered:
            count = sum(bool(row.get(f"from_{source}")) for source in SOURCES)
            if count != int(row.get("source_count", -1)) or count < 1:
                raise HistoricalReplayError(f"provenance incomplete: {key}")


def semantic_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    return semantic_candidate_fingerprint(rows)


def semantic_replay_audit(
    formal: Sequence[Mapping[str, Any]], replay: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    formal_key = {(str(r["snapshot_id"]), int(r["customer_key"]), int(r["article_key"])): int(r["candidate_rank"]) for r in formal}
    replay_key = {(str(r["snapshot_id"]), int(r["customer_key"]), int(r["article_key"])): int(r["candidate_rank"]) for r in replay}
    membership = len(set(formal_key) ^ set(replay_key))
    rank = sum(formal_key[key] != replay_key[key] for key in set(formal_key) & set(replay_key))
    formal_fp = semantic_fingerprint(formal)
    replay_fp = semantic_fingerprint(replay)
    return {
        "formal_semantic_fingerprint": formal_fp,
        "replay_semantic_fingerprint": replay_fp,
        "membership_mismatch": membership,
        "rank_mismatch": rank,
        "status": "PASS" if formal_fp == replay_fp and membership == 0 and rank == 0 else "FAIL",
    }


def directory_fingerprint(root: Path) -> tuple[str, int, int]:
    records = []
    total_size = 0
    for path in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        fingerprint = file_fingerprint(path)
        size = path.stat().st_size
        total_size += size
        records.append((relative, size, fingerprint))
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(records), total_size


def privacy_findings(paths: Iterable[Path], *, local_username: str | None = None) -> list[str]:
    findings: list[str] = []
    username = (local_username or "").lower()
    for path in paths:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        lowered = text.lower()
        if WINDOWS_ABSOLUTE_PATH.search(text):
            findings.append(f"absolute_path:{path.name}")
        if username and username in lowered:
            findings.append(f"local_username:{path.name}")
        header = text.splitlines()[0].lower().split(",") if text.splitlines() else []
        for field in FORBIDDEN_PUBLIC_FIELDS:
            if field in header:
                findings.append(f"forbidden_field_{field}:{path.name}")
    return sorted(set(findings))
