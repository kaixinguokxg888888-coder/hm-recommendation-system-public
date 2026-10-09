from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq


SEMANTIC_FINGERPRINT_VERSION = "ranking_feature_semantic_sha256_query_merkle_v1"
NULL_TOKEN = "<NULL>"


def canonical_value(value: Any, dtype: str) -> str:
    if value is None:
        return NULL_TOKEN
    if dtype == "bool":
        return "1" if bool(value) else "0"
    if dtype == "float64":
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("formal float feature 不允许 NaN/Inf")
        return format(number, ".17g")
    if dtype in {"int32", "int64"}:
        return str(int(value))
    if dtype == "date32":
        return value.isoformat()
    return str(value)


def canonical_record(
    row: Mapping[str, Any], fields: Sequence[tuple[str, str]],
) -> bytes:
    values = []
    for name, dtype in fields:
        value = canonical_value(row.get(name), dtype)
        encoded = value.encode("utf-8")
        values.append(name.encode("utf-8") + b"=" + str(len(encoded)).encode("ascii") + b":" + encoded)
    return b"\x1f".join(values)


def reference_semantic_fingerprint(
    rows: Iterable[Mapping[str, Any]], fields: Sequence[tuple[str, str]], schema_fingerprint: str,
) -> str:
    """Small-table reference implementation; formal build uses equivalent query Merkle SQL."""
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["snapshot_id"]), int(row["customer_key"])), []).append(row)
    query_values: list[tuple[str, int, int, bytes]] = []
    for (snapshot, customer), values in groups.items():
        ordered = sorted(values, key=lambda row: (int(row["candidate_rank"]), int(row["article_key"])))
        digest = hashlib.sha256()
        digest.update(b"ranking-feature-query-v1\0")
        digest.update(schema_fingerprint.encode("ascii") + b"\0")
        for row in ordered:
            digest.update(hashlib.sha256(canonical_record(row, fields)).hexdigest().encode("ascii"))
        query_values.append((snapshot, customer, len(ordered), digest.digest()))
    return fingerprint_from_query_digests(query_values, schema_fingerprint)


def fingerprint_from_query_digests(
    rows: Iterable[tuple[str, int, int, bytes]], schema_fingerprint: str,
) -> str:
    values = sorted(rows, key=lambda row: (row[0], row[1]))
    if len({(row[0], row[1]) for row in values}) != len(values):
        raise ValueError("feature query digest 跨 partition 重复")
    digest = hashlib.sha256()
    digest.update((SEMANTIC_FINGERPRINT_VERSION + "\0").encode("ascii"))
    digest.update(schema_fingerprint.encode("ascii") + b"\0")
    for snapshot, customer, count, query_digest in values:
        encoded = snapshot.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">qI", int(customer), int(count)))
        digest.update(query_digest)
    return digest.hexdigest()


def read_query_digest_rows(paths: Iterable[Path]) -> list[tuple[str, int, int, bytes]]:
    output: list[tuple[str, int, int, bytes]] = []
    for path in paths:
        table = pq.read_table(path, columns=["snapshot_id", "customer_key", "row_count", "query_digest"])
        for snapshot, customer, count, value in zip(
            table["snapshot_id"].to_pylist(), table["customer_key"].to_pylist(),
            table["row_count"].to_pylist(), table["query_digest"].to_pylist(),
        ):
            output.append((str(snapshot), int(customer), int(count), bytes.fromhex(str(value))))
    return output


def feature_semantic_fingerprint(paths: Iterable[Path], schema_fingerprint: str) -> str:
    return fingerprint_from_query_digests(read_query_digest_rows(paths), schema_fingerprint)


def statistics_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    canonical = sorted((dict(row) for row in rows), key=lambda row: row["feature_name"])
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(b"ranking-feature-statistics-v1\0" + payload.encode("utf-8")).hexdigest()
