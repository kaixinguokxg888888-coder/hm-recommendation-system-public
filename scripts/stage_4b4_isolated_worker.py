from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from data_foundation.manifest import file_fingerprint
from ranking_model import Stage4B4Config
from ranking_model.ordering import ORDERING_VERSION, matrix_order_fingerprint, query_boundary_digest


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 isolated LambdaRank worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-set", choices=("F0", "F1"), required=True)
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--customer-keys", type=Path, required=True)
    parser.add_argument("--article-keys", type=Path, required=True)
    parser.add_argument("--candidate-ranks", type=Path, required=True)
    parser.add_argument("--query-orders", type=Path, required=True)
    parser.add_argument("--matrix-metadata", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--temporary-model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--worker-source-fingerprint", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    config = Stage4B4Config.load(args.config)
    raw = config.raw
    matrix_metadata = json.loads(args.matrix_metadata.read_text(encoding="utf-8"))
    if matrix_metadata.get("ordering_version") != ORDERING_VERSION:
        raise RuntimeError("worker rejects non-v2 matrix ordering")
    adapter = json.loads(args.adapter_manifest.read_text(encoding="utf-8"))
    feature_names = list(adapter["ordered_feature_names"])
    categorical_names = list(adapter["categorical_features"])
    expected_width = 30 if args.feature_set == "F0" else 52
    expected_categories = ["user_segment"] if args.feature_set == "F0" else [
        "user_segment",
        "article_product_type_code",
        "article_product_group_code",
        "article_garment_group_code",
        "article_section_code",
        "article_graphical_appearance_code",
        "article_colour_group_code",
        "article_perceived_colour_value_code",
        "article_perceived_colour_master_code",
        "article_department_code",
        "article_index_code",
        "article_index_group_code",
    ]
    if len(feature_names) != expected_width or categorical_names != expected_categories:
        raise RuntimeError("categorical or ordered-feature contract mismatch")

    rows = int(raw["expected"]["train_rows"])
    queries = int(raw["expected"]["train_queries"])
    expected_feature_bytes = rows * expected_width * np.dtype(np.float32).itemsize
    if args.feature_matrix.stat().st_size != expected_feature_bytes:
        raise RuntimeError("feature matrix byte-size mismatch")
    if args.labels.stat().st_size != rows * np.dtype(np.uint8).itemsize:
        raise RuntimeError("label array byte-size mismatch")

    matrix = np.memmap(args.feature_matrix, dtype=np.float32, mode="r", shape=(rows, expected_width))
    labels = np.memmap(args.labels, dtype=np.uint8, mode="r", shape=(rows,))
    groups = np.fromfile(args.groups, dtype=np.int32)
    customers = np.memmap(args.customer_keys, dtype=np.int64, mode="r", shape=(rows,))
    articles = np.memmap(args.article_keys, dtype=np.int64, mode="r", shape=(rows,))
    ranks = np.memmap(args.candidate_ranks, dtype=np.int32, mode="r", shape=(rows,))
    orders = np.memmap(args.query_orders, dtype=np.int64, mode="r", shape=(rows,))
    if len(groups) != queries or int(groups.sum()) != rows:
        raise RuntimeError("query/group reconciliation mismatch")
    if np.any((labels != 0) & (labels != 1)):
        raise RuntimeError("labels must remain binary")
    starts = np.r_[0, np.cumsum(groups[:-1], dtype=np.int64)]
    positives = np.add.reduceat(labels, starts)
    if np.any(positives < 1):
        raise RuntimeError("all-zero optimizer group detected")
    boundaries: list[dict[str, object]] = []
    previous_order: int | None = None
    for index, (start, size, positive) in enumerate(zip(starts, groups, positives)):
        end = int(start + size)
        group_customers = customers[start:end]
        group_orders = orders[start:end]
        group_articles = articles[start:end]
        group_ranks = ranks[start:end]
        if len(set(map(int, group_customers))) != 1 or len(set(map(int, group_orders))) != 1:
            raise RuntimeError("cross-query boundary detected")
        current_order = int(group_orders[0])
        if previous_order is not None and current_order <= previous_order:
            raise RuntimeError("query_order is not globally ascending")
        previous_order = current_order
        invalid = (group_ranks[1:] < group_ranks[:-1]) | (
            (group_ranks[1:] == group_ranks[:-1]) & (group_articles[1:] < group_articles[:-1])
        )
        if np.any(invalid):
            raise RuntimeError("within-query candidate ordering drift")
        boundaries.append({
            "snapshot_id": raw["snapshots"]["train"]["snapshot_id"],
            "customer_key": int(group_customers[0]), "query_order": current_order,
            "start_offset": int(start), "end_offset": end, "group_size": int(size),
            "positive_count": int(positive), "sampled_zero_count": int(size - positive),
            "ordered_row_digest": query_boundary_digest(
                raw["snapshots"]["train"]["snapshot_id"], int(group_customers[0]), current_order,
                int(start), group_articles, group_ranks, labels[start:end],
            ),
        })
    actual_order_fingerprint = matrix_order_fingerprint(boundaries)
    if actual_order_fingerprint != matrix_metadata.get("matrix_order_fingerprint"):
        raise RuntimeError("matrix-order fingerprint mismatch")
    required_physical = {
        "labels": args.labels, "groups": args.groups, "customers": args.customer_keys,
        "articles": args.article_keys, "ranks": args.candidate_ranks, "orders": args.query_orders,
        args.feature_set.lower(): args.feature_matrix,
    }
    for role, path in required_physical.items():
        if file_fingerprint(path) != matrix_metadata["physical_fingerprints"][role]:
            raise RuntimeError(f"matrix physical fingerprint mismatch: {role}")

    feature_inf_violations = 0
    for offset in range(0, rows, 65_536):
        feature_inf_violations += int(np.isinf(matrix[offset:offset + 65_536]).sum())
    if feature_inf_violations:
        raise RuntimeError("training features contain infinity")

    categorical_indices = [feature_names.index(name) for name in categorical_names]
    params = dict(raw["lightgbm"]["params"])
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        group=groups,
        feature_name=feature_names,
        categorical_feature=categorical_indices,
        # Dataset 构造与训练绑定完全相同的正式参数，避免 construct 后 seed 等 binning 参数漂移。
        params=params,
        free_raw_data=True,
    )
    dataset.construct()
    if dataset.num_data() != rows or dataset.num_feature() != expected_width:
        raise RuntimeError("constructed LightGBM Dataset dimensions mismatch")
    registered = [feature_names[index] if isinstance(index, int) else str(index)
                  for index in dataset.categorical_feature]
    if registered != categorical_names:
        raise RuntimeError("constructed LightGBM categorical metadata mismatch")

    # Dataset 已完成 binning 后释放 raw memmap 引用，避免训练期间同时常驻两份 dense matrix。
    del matrix, labels, groups, customers, articles, ranks, orders, boundaries
    gc.collect()
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=int(raw["lightgbm"]["num_boost_round"]),
        keep_training_booster=True,
    )
    if booster.current_iteration() != 600:
        raise RuntimeError("formal booster did not complete 600 rounds")

    args.temporary_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.temporary_model, num_iteration=600)
    reloaded = lgb.Booster(model_file=str(args.temporary_model))
    if reloaded.current_iteration() != 600 or reloaded.feature_name() != feature_names:
        raise RuntimeError("temporary model reload audit failed")
    model_string = reloaded.model_to_string(num_iteration=600).replace("\r\n", "\n")
    payload: dict[str, object] = {
        "status": "WORKER_SUCCESS_AWAITING_EXTERNAL_RESOURCE_GATE",
        "model_id": "L0" if args.feature_set == "F0" else "L1",
        "feature_set": args.feature_set,
        "rows": rows,
        "queries": queries,
        "groups": queries,
        "features": expected_width,
        "rounds": reloaded.current_iteration(),
        "categorical_features": categorical_names,
        "categorical_metadata_fingerprint": sha256_json(categorical_names),
        "ordered_feature_fingerprint": sha256_json(feature_names),
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "model_physical_fingerprint": file_fingerprint(args.temporary_model),
        "model_semantic_fingerprint": hashlib.sha256(model_string.encode("utf-8")).hexdigest(),
        "feature_matrix_fingerprint": file_fingerprint(args.feature_matrix),
        "label_fingerprint": file_fingerprint(args.labels),
        "group_fingerprint": file_fingerprint(args.groups),
        "train_optimizer_fingerprint": raw["expected"]["train_optimizer_fingerprint"],
        "matrix_ordering_version": ORDERING_VERSION,
        "matrix_order_fingerprint": actual_order_fingerprint,
        "query_boundary_fingerprint": matrix_metadata["query_boundary_fingerprint"],
        "config_fingerprint": config.fingerprint,
        "worker_source_fingerprint": args.worker_source_fingerprint,
        "lightgbm_version": lgb.__version__,
        "params_fingerprint": sha256_json(params),
        "seeds": {name: params[name] for name in (
            "seed", "data_random_seed", "feature_fraction_seed", "bagging_seed", "drop_seed"
        )},
        "feature_inf_violations": feature_inf_violations,
        "validation_label_read_count": 0,
        "first7d_read_count": 0,
        "test_read_count": 0,
        "wall_time_seconds": time.perf_counter() - started,
    }
    write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()
