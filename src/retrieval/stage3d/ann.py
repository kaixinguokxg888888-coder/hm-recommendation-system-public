from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import hnswlib
import numpy as np


def stable_top_k(scores: np.ndarray, article_keys: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k 必须为正")
    if len(scores) <= k:
        chosen = np.arange(len(scores))
    else:
        chosen = np.argpartition(-scores, k - 1)[:k]
    order = np.lexsort((article_keys[chosen], -scores[chosen]))
    return chosen[order][:k]


def exact_top_k_blockwise(
    user_embedding: np.ndarray,
    item_embeddings: np.ndarray,
    article_keys: np.ndarray,
    *,
    seen_items: set[int],
    k: int = 500,
    block_items: int = 8192,
) -> np.ndarray:
    best_scores = np.empty(0, dtype=np.float32)
    best_keys = np.empty(0, dtype=np.int64)
    for start in range(0, len(article_keys), block_items):
        stop = min(start + block_items, len(article_keys))
        keys = article_keys[start:stop]
        scores = item_embeddings[start:stop] @ user_embedding
        keep = np.fromiter((int(key) not in seen_items for key in keys), dtype=bool, count=len(keys))
        keys, scores = keys[keep], scores[keep]
        best_keys = np.concatenate([best_keys, keys])
        best_scores = np.concatenate([best_scores, scores.astype(np.float32, copy=False)])
        chosen = stable_top_k(best_scores, best_keys, min(k, len(best_keys)))
        best_keys, best_scores = best_keys[chosen], best_scores[chosen]
    return best_keys[stable_top_k(best_scores, best_keys, min(k, len(best_keys)))]


def rerank_ann_candidates(
    user_embedding: np.ndarray,
    labels: Sequence[int],
    item_embeddings: np.ndarray,
    article_keys: np.ndarray,
    *,
    seen_items: set[int],
    k: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(labels, dtype=np.int64)
    keys = article_keys[positions]
    keep = np.fromiter((int(key) not in seen_items for key in keys), dtype=bool, count=len(keys))
    positions, keys = positions[keep], keys[keep]
    scores = item_embeddings[positions] @ user_embedding
    chosen = stable_top_k(scores, keys, min(k, len(keys)))
    return keys[chosen], scores[chosen].astype(np.float32, copy=False)


def build_hnsw_index(
    item_embeddings: np.ndarray,
    target: Path,
    *,
    m: int = 32,
    ef_construction: int = 200,
    seed: int = 20260805,
) -> tuple[hnswlib.Index, float]:
    started = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=item_embeddings.shape[1])
    index.init_index(max_elements=len(item_embeddings), ef_construction=ef_construction, M=m, random_seed=seed)
    index.add_items(item_embeddings, np.arange(len(item_embeddings), dtype=np.int64))
    target.parent.mkdir(parents=True, exist_ok=True)
    index.save_index(str(target))
    return index, time.perf_counter() - started


def load_hnsw_index(path: Path, *, dimension: int, max_elements: int, ef_search: int) -> hnswlib.Index:
    index = hnswlib.Index(space="cosine", dim=dimension)
    index.load_index(str(path), max_elements=max_elements)
    index.set_ef(ef_search)
    return index


def overlap_recall(approximate: Iterable[int], exact: Iterable[int]) -> float:
    exact_set = set(int(value) for value in exact)
    return len(set(int(value) for value in approximate) & exact_set) / len(exact_set) if exact_set else 0.0
