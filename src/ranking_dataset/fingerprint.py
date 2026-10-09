from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


DATASET_FINGERPRINT_VERSION = "ranking_dataset_semantic_sha256_query_merkle_v1"
OPTIMIZER_FINGERPRINT_VERSION = "ranking_optimizer_semantic_sha256_query_merkle_v1"


def _query_digest(
    snapshot_id: str, customer_key: int, ranks: np.ndarray, articles: np.ndarray, labels: np.ndarray,
    *, domain: bytes,
) -> bytes:
    order = np.argsort(ranks, kind="stable")
    ranks, articles, labels = ranks[order], articles[order], labels[order]
    if len(np.unique(ranks)) != len(ranks) or len(np.unique(articles)) != len(articles):
        raise ValueError("ranking semantic fingerprint 输入 query key/rank 不唯一")
    digest = hashlib.sha256()
    digest.update(domain + b"\0")
    encoded = snapshot_id.encode("utf-8")
    digest.update(struct.pack(">I", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack(">qI", int(customer_key), len(ranks)))
    records = np.empty(
        len(ranks), dtype=np.dtype([("rank", ">u4"), ("article", ">i8"), ("label", "u1")])
    )
    records["rank"], records["article"], records["label"] = ranks, articles, labels
    digest.update(records.tobytes())
    return digest.digest()


def partition_query_digests(path: Path, *, optimizer: bool) -> list[tuple[str, int, int, bytes]]:
    table = pq.read_table(path, columns=[
        "snapshot_id", "customer_key", "candidate_rank", "article_key", "label",
    ])
    snapshots = table.column("snapshot_id").combine_chunks().unique().to_pylist()
    if len(snapshots) != 1:
        raise ValueError("ranking partition 必须只含一个 snapshot")
    snapshot = str(snapshots[0])
    customers = table.column("customer_key").combine_chunks().to_numpy(zero_copy_only=False)
    ranks = table.column("candidate_rank").combine_chunks().to_numpy(zero_copy_only=False)
    articles = table.column("article_key").combine_chunks().to_numpy(zero_copy_only=False)
    labels = table.column("label").combine_chunks().to_numpy(zero_copy_only=False)
    if len(customers) == 0:
        return []
    if np.any(customers[1:] < customers[:-1]):
        raise ValueError("ranking partition customer_key 未排序")
    boundaries = np.concatenate((
        np.asarray([0]), np.flatnonzero(customers[1:] != customers[:-1]) + 1,
        np.asarray([len(customers)]),
    ))
    domain = b"ranking-optimizer-query-v1" if optimizer else b"ranking-dataset-query-v1"
    return [
        (
            snapshot, int(customers[start]), int(end - start),
            _query_digest(
                snapshot, int(customers[start]), ranks[start:end].astype(np.int64, copy=False),
                articles[start:end].astype(np.int64, copy=False),
                labels[start:end].astype(np.uint8, copy=False), domain=domain,
            ),
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]


def fingerprint_from_query_digests(
    values: Iterable[tuple[str, int, int, bytes]], *, optimizer: bool,
) -> str:
    rows = sorted(values, key=lambda row: (row[0], row[1]))
    if len({(row[0], row[1]) for row in rows}) != len(rows):
        raise ValueError("ranking query 跨 partition 重复")
    digest = hashlib.sha256()
    digest.update(
        (b"ranking-optimizer-semantic-sha256-query-merkle-v1\0" if optimizer else
         b"ranking-dataset-semantic-sha256-query-merkle-v1\0")
    )
    for snapshot, customer, count, query_digest in rows:
        encoded = snapshot.encode("utf-8")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">qI", customer, count))
        digest.update(query_digest)
    return digest.hexdigest()


def semantic_fingerprint(paths: Iterable[Path], *, optimizer: bool) -> str:
    digests = [row for path in paths for row in partition_query_digests(path, optimizer=optimizer)]
    return fingerprint_from_query_digests(digests, optimizer=optimizer)
