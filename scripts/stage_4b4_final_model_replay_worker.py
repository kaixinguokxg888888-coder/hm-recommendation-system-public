from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


def fingerprint(payload: object) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def score(model: lgb.Booster, features: np.ndarray, articles: np.ndarray, ranks: np.ndarray) -> dict[str, object]:
    values = np.asarray(model.predict(features, num_iteration=600), dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("formal model replay has non-finite scores")
    raw = hashlib.sha256(b"stage-4b4-final-refit-raw-score-replay-v1\0")
    for value in values:
        raw.update(float(value).hex().encode("ascii") + b"\0")
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), int(ranks[index]), int(articles[index])))
    ranked = hashlib.sha256(b"stage-4b4-final-refit-rank-replay-v1\0")
    for predicted_rank, index in enumerate(order, 1):
        ranked.update(struct.pack("<qii", int(articles[index]), int(ranks[index]), predicted_rank))
    return {
        "audit_rows": len(values), "raw_score_fingerprint": raw.hexdigest(),
        "rank_tiebreak_fingerprint": ranked.hexdigest(), "non_finite_scores": 0,
    }


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-free formal final-refit model replay worker")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--article-keys", type=Path, required=True)
    parser.add_argument("--candidate-ranks", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--total-rows", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    forbidden = {"duckdb", "pyarrow", "ranking_features"}
    if any(name.split(".", 1)[0] in forbidden for name in sys.modules):
        raise RuntimeError("model replay worker imported forbidden pipeline module")
    first_group = np.fromfile(args.groups, dtype=np.int32, count=1)
    if len(first_group) != 1 or int(first_group[0]) <= 0:
        raise RuntimeError("label-free replay query boundary unavailable")
    audit_rows = int(first_group[0])
    matrix = np.memmap(args.feature_matrix, dtype=np.float32, mode="r", shape=(args.total_rows, 52))
    article_values = np.memmap(args.article_keys, dtype=np.int64, mode="r", shape=(args.total_rows,))
    rank_values = np.memmap(args.candidate_ranks, dtype=np.int32, mode="r", shape=(args.total_rows,))
    features = np.array(matrix[:audit_rows], dtype=np.float32, copy=True)
    articles = np.array(article_values[:audit_rows], dtype=np.int64, copy=True)
    ranks = np.array(rank_values[:audit_rows], dtype=np.int32, copy=True)
    first = lgb.Booster(model_file=str(args.model))
    second = lgb.Booster(model_file=str(args.model))
    if first.current_iteration() != 600 or first.num_trees() != 600:
        raise RuntimeError("formal replay model round/tree mismatch")
    first_result = score(first, features, articles, ranks)
    second_result = score(second, features, articles, ranks)
    if first_result != second_result:
        raise RuntimeError("formal model independent reload replay mismatch")
    payload = {
        "status": "PASS", "audit_slice": "first_complete_combined_query_label_free_v1",
        "first_reload": first_result, "second_reload": second_result,
        "raw_score_exact_match": True, "rank_tiebreak_exact_match": True,
        "label_vector_opened": False, "validation_metric_computed": False,
        "first7d_labels_read": False, "test_access": False,
        "wall_time_seconds": time.perf_counter() - started,
    }
    payload["replay_fingerprint"] = fingerprint(payload)
    write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()
