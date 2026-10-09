from __future__ import annotations

import hashlib
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


CONTRACT_VERSION = "stage_3e_candidate_contract_v1"
PROTOCOL_VERSION = "source_rank_cap_pre_dedup_v1"
SEMANTIC_FINGERPRINT_VERSION = "candidate_semantic_sha256_user_merkle_v1"
SOURCES = ("repurchase", "itemcf", "two_tower", "trend", "popularity")
PRIMARY_SOURCES = ("repurchase", "itemcf", "two_tower", "trend", "popularity_fallback")
SOURCE_CAPS = {"repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100}

CONTRACT_COLUMNS = (
    "snapshot_id",
    "as_of_date",
    "customer_key",
    "article_key",
    "candidate_rank",
    "candidate_contract_version",
    "candidate_protocol_version",
    "primary_source",
    "source_count",
    "from_repurchase",
    "from_itemcf",
    "from_two_tower",
    "from_trend",
    "from_popularity",
    "repurchase_rank",
    "repurchase_score",
    "itemcf_rank",
    "itemcf_score",
    "two_tower_rank",
    "two_tower_score",
    "trend_rank",
    "trend_score",
    "popularity_rank",
    "popularity_score",
    "candidate_universe_version",
    "feedback_version",
    "population_version",
    "dataset_fingerprint",
    "stage_3d_config_fingerprint",
    "stage_3e_config_fingerprint",
    "upstream_ordering_kind",
)

NULLABLE_COLUMNS = frozenset(
    f"{source}_{suffix}" for source in SOURCES for suffix in ("rank", "score")
)


class CandidateContractError(ValueError):
    """Candidate rows or semantic fingerprint input violate the frozen contract."""


def schema_manifest_rows() -> list[dict[str, Any]]:
    dtypes = {
        "snapshot_id": "string", "as_of_date": "date32", "customer_key": "int64",
        "article_key": "int64", "candidate_rank": "int32",
        "candidate_contract_version": "string", "candidate_protocol_version": "string",
        "primary_source": "string", "source_count": "int32",
        "candidate_universe_version": "string", "feedback_version": "string",
        "population_version": "string", "dataset_fingerprint": "string",
        "stage_3d_config_fingerprint": "string", "stage_3e_config_fingerprint": "string",
        "upstream_ordering_kind": "string",
    }
    for source in SOURCES:
        dtypes[f"from_{source}"] = "bool"
        dtypes[f"{source}_rank"] = "int32"
        dtypes[f"{source}_score"] = "float64"
    membership = {"snapshot_id", "customer_key", "article_key", "candidate_rank"}
    ranking_feature = {
        "candidate_rank", "source_count", *(f"from_{source}" for source in SOURCES),
        *(f"{source}_rank" for source in SOURCES), *(f"{source}_score" for source in SOURCES),
    }
    meanings = {
        "candidate_rank": "1-based continuous final retrieval ordering; the only formal candidate order",
        "primary_source": "first contributing source under the frozen insertion order",
        "source_count": "number of formal capped union sources contributing provenance",
        "stage_3d_config_fingerprint": "frozen Stage 3D producer configuration lineage",
        "stage_3e_config_fingerprint": "Stage 3E contract materialization configuration lineage",
        "upstream_ordering_kind": "rank-preserving migration from locked Stage 3D TT1-T100",
    }
    rows = []
    for column in CONTRACT_COLUMNS:
        source = next((name for name in SOURCES if name in column), "candidate_contract")
        meaning = meanings.get(column, (
            "formal capped-union provenance flag" if column.startswith("from_") else
            "source-local rank; null when the source did not contribute" if column.endswith("_rank") else
            "source-local raw evidence; null when unavailable and never summed cross-source"
            if column.endswith("_score") else column.replace("_", " ")
        ))
        rows.append({
            "column": column,
            "dtype": dtypes[column],
            "nullable": column in NULLABLE_COLUMNS,
            "semantic_meaning": meaning,
            "membership_critical": column in membership,
            "ranking_feature_eligible": column in ranking_feature,
            "source_lineage": source,
            "snapshot_semantics": "explicit_validation_or_test_key; never inferred from date",
        })
    return rows


def migrate_stage3d_row(
    row: Mapping[str, Any], *, stage3e_config_fingerprint: str,
) -> dict[str, Any]:
    migrated = {
        "snapshot_id": row["snapshot_id"],
        "as_of_date": row["as_of_date"],
        "customer_key": int(row["customer_key"]),
        "article_key": int(row["article_key"]),
        "candidate_rank": int(row["source_rank"]),
        "candidate_contract_version": CONTRACT_VERSION,
        "candidate_protocol_version": PROTOCOL_VERSION,
        "primary_source": row["first_source"],
        "source_count": int(row["source_count"]),
        "candidate_universe_version": row["candidate_universe_version"],
        "feedback_version": row["feedback_version"],
        "population_version": row["population_version"],
        "dataset_fingerprint": row["dataset_fingerprint"],
        "stage_3d_config_fingerprint": row["config_fingerprint"],
        "stage_3e_config_fingerprint": stage3e_config_fingerprint,
        "upstream_ordering_kind": "stage3d_locked_t100_source_rank",
    }
    for source in SOURCES:
        migrated[f"from_{source}"] = bool(row[f"from_{source}"])
        migrated[f"{source}_rank"] = row[f"{source}_rank"]
        migrated[f"{source}_score"] = row[f"{source}_score"]
    return migrated


def validate_contract_rows(
    rows: Iterable[Mapping[str, Any]], *, expected_snapshot: str | None = None,
    expected_budget: int = 500,
) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in values:
        missing = set(CONTRACT_COLUMNS) - set(row)
        if missing:
            raise CandidateContractError(f"candidate contract 缺少字段: {sorted(missing)}")
        if expected_snapshot is not None and row["snapshot_id"] != expected_snapshot:
            raise CandidateContractError("snapshot isolation 失败")
        if row["candidate_contract_version"] != CONTRACT_VERSION:
            raise CandidateContractError("candidate contract version 不匹配")
        if row["candidate_protocol_version"] != PROTOCOL_VERSION:
            raise CandidateContractError("candidate protocol version 不匹配")
        if row["primary_source"] not in PRIMARY_SOURCES:
            raise CandidateContractError("primary_source 不属于冻结 insertion order")
        flags = [bool(row[f"from_{source}"]) for source in SOURCES]
        if int(row["source_count"]) != sum(flags) or not any(flags):
            raise CandidateContractError("source provenance 不完整")
        expected_primary = next(
            source for source, flag in zip(PRIMARY_SOURCES, flags) if flag
        )
        if row["primary_source"] != expected_primary:
            raise CandidateContractError("primary_source 未按冻结 insertion order 定义")
        for source in SOURCES:
            rank = row[f"{source}_rank"]
            score = row[f"{source}_score"]
            flag = bool(row[f"from_{source}"])
            if flag != (rank is not None) or flag != (score is not None):
                raise CandidateContractError("source flag/rank/score nullability 不一致")
            if source in SOURCE_CAPS and rank is not None and int(rank) > SOURCE_CAPS[source]:
                raise CandidateContractError("正式 provenance 超出 source-rank cap")
        grouped[(str(row["snapshot_id"]), int(row["customer_key"]))].append(row)
    for (_, _), user_rows in grouped.items():
        items = [int(row["article_key"]) for row in user_rows]
        ranks = [int(row["candidate_rank"]) for row in user_rows]
        if len(items) != len(set(items)):
            raise CandidateContractError("同一 snapshot/user 的 candidate membership 重复")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise CandidateContractError("candidate_rank 必须 1-based、唯一且连续")
        if len(user_rows) > expected_budget:
            raise CandidateContractError("candidate count 超过冻结预算")
    return values


def _user_digest(snapshot_id: str, customer_key: int, records: Sequence[tuple[int, int]]) -> bytes:
    ordered = sorted((int(rank), int(article)) for rank, article in records)
    ranks = np.asarray([rank for rank, _ in ordered], dtype=np.int64)
    articles = np.asarray([article for _, article in ordered], dtype=np.int64)
    return _user_digest_arrays(snapshot_id, customer_key, ranks, articles)


def _user_digest_arrays(
    snapshot_id: str, customer_key: int, ranks: np.ndarray, articles: np.ndarray,
) -> bytes:
    expected = np.arange(1, len(ranks) + 1, dtype=np.int64)
    if not np.array_equal(ranks, expected):
        raise CandidateContractError("semantic fingerprint 输入 rank 不连续")
    if len(np.unique(articles)) != len(articles):
        raise CandidateContractError("semantic fingerprint 输入 membership 重复")
    digest = hashlib.sha256()
    digest.update(b"stage3e-candidate-user-v1\0")
    encoded = snapshot_id.encode("utf-8")
    digest.update(struct.pack(">I", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack(">qI", customer_key, len(ranks)))
    records = np.empty(len(ranks), dtype=np.dtype([("rank", ">u4"), ("article", ">i8")]))
    records["rank"] = ranks
    records["article"] = articles
    digest.update(records.tobytes())
    return digest.digest()


def fingerprint_from_user_digests(
    user_digests: Iterable[tuple[str, int, int, bytes]],
) -> str:
    values = sorted(user_digests, key=lambda row: (row[0], row[1]))
    if len({(row[0], row[1]) for row in values}) != len(values):
        raise CandidateContractError("semantic fingerprint user 跨 partition 重复")
    digest = hashlib.sha256()
    digest.update(b"candidate-semantic-sha256-user-merkle-v1\0")
    for snapshot_id, customer_key, count, user_digest in values:
        encoded = snapshot_id.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">qI", int(customer_key), int(count)))
        digest.update(user_digest)
    return digest.hexdigest()


def semantic_candidate_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        key = (str(row["snapshot_id"]), int(row["customer_key"]))
        grouped[key].append((int(row["candidate_rank"]), int(row["article_key"])))
    user_digests = [
        (snapshot, customer, len(records), _user_digest(snapshot, customer, records))
        for (snapshot, customer), records in grouped.items()
    ]
    return fingerprint_from_user_digests(user_digests)


def zero_drift_counts(
    stage3d_rows: Iterable[Mapping[str, Any]],
    stage3e_rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    upstream = {
        (str(row["snapshot_id"]), int(row["customer_key"]), int(row["article_key"])): row
        for row in stage3d_rows
    }
    contract = {
        (str(row["snapshot_id"]), int(row["customer_key"]), int(row["article_key"])): row
        for row in stage3e_rows
    }
    shared = upstream.keys() & contract.keys()
    provenance_mismatch = 0
    for key in shared:
        left, right = upstream[key], contract[key]
        fields_equal = (
            left["first_source"] == right["primary_source"]
            and int(left["source_count"]) == int(right["source_count"])
        )
        for source in SOURCES:
            fields_equal = fields_equal and all(
                left[field] == right[field]
                for field in (f"from_{source}", f"{source}_rank", f"{source}_score")
            )
        provenance_mismatch += int(not fields_equal)
    upstream_users = {(key[0], key[1]) for key in upstream}
    contract_users = {(key[0], key[1]) for key in contract}
    return {
        "stage3d_minus_stage3e": len(upstream.keys() - contract.keys()),
        "stage3e_minus_stage3d": len(contract.keys() - upstream.keys()),
        "rank_mismatch": sum(
            int(upstream[key]["source_rank"]) != int(contract[key]["candidate_rank"])
            for key in shared
        ),
        "provenance_mismatch": provenance_mismatch,
        "user_count_mismatch": len(upstream_users ^ contract_users),
    }


def parquet_partition_user_digests(
    path: Path, *, batch_rows: int = 65_536,
) -> list[tuple[str, int, int, bytes]]:
    """Hash a sorted contract partition without loading millions of rows into Python heap."""
    del batch_rows  # 当前正式分区的 membership 列小于资源上限；一次 Arrow 列读取避免逐行 Python heap。
    table = pq.read_table(path, columns=["snapshot_id", "customer_key", "candidate_rank", "article_key"])
    snapshots = table.column("snapshot_id").combine_chunks()
    unique_snapshots = snapshots.unique().to_pylist()
    if len(unique_snapshots) != 1:
        raise CandidateContractError("contract partition 混入多个 snapshot")
    snapshot = str(unique_snapshots[0])
    customers = table.column("customer_key").combine_chunks().to_numpy(zero_copy_only=False)
    ranks = table.column("candidate_rank").combine_chunks().to_numpy(zero_copy_only=False)
    articles = table.column("article_key").combine_chunks().to_numpy(zero_copy_only=False)
    if len(customers) == 0:
        return []
    if np.any(customers[1:] < customers[:-1]):
        raise CandidateContractError("contract partition customer key 未排序")
    boundaries = np.concatenate((
        np.asarray([0]), np.flatnonzero(customers[1:] != customers[:-1]) + 1,
        np.asarray([len(customers)]),
    ))
    output: list[tuple[str, int, int, bytes]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        user = int(customers[start])
        user_ranks = ranks[start:end].astype(np.int64, copy=False)
        user_articles = articles[start:end].astype(np.int64, copy=False)
        output.append((snapshot, user, int(end - start),
                       _user_digest_arrays(snapshot, user, user_ranks, user_articles)))
    return output


def semantic_fingerprint_from_partitions(
    paths: Iterable[Path], *, batch_rows: int = 65_536,
) -> tuple[str, list[tuple[str, int, int, bytes]]]:
    user_digests = [
        row for path in paths for row in parquet_partition_user_digests(path, batch_rows=batch_rows)
    ]
    return fingerprint_from_user_digests(user_digests), user_digests
