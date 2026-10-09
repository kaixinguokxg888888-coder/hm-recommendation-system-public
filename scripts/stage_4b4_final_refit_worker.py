from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


ORDERING_VERSION = "stage_4b4_final_refit_snapshot_query_order_v1"
BOUNDARY_DOMAIN = b"stage-4b4-final-refit-query-boundary-v1\0"


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def file_fingerprint(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def boundary_digest(
    snapshot: str, customer: int, query_order: int, start: int,
    articles: np.ndarray, ranks: np.ndarray, labels: np.ndarray,
) -> str:
    digest = hashlib.sha256(BOUNDARY_DOMAIN)
    digest.update(snapshot.encode("utf-8") + b"\0")
    digest.update(struct.pack("<qqq", customer, query_order, start))
    rows = np.empty(len(articles), dtype=[("article", "<i8"), ("rank", "<i4"), ("label", "u1")])
    rows["article"], rows["rank"], rows["label"] = articles, ranks, labels
    digest.update(rows.tobytes())
    return digest.hexdigest()


def score_replay(booster: lgb.Booster, matrix: np.ndarray, articles: np.ndarray, ranks: np.ndarray) -> dict[str, object]:
    scores = np.asarray(booster.predict(matrix, num_iteration=600), dtype=np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError("prediction replay contains NaN/Inf")
    raw = hashlib.sha256(b"stage-4b4-final-refit-raw-score-replay-v1\0")
    for score in scores:
        raw.update(float(score).hex().encode("ascii") + b"\0")
    ordered = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), int(ranks[index]), int(articles[index])))
    rank_digest = hashlib.sha256(b"stage-4b4-final-refit-rank-replay-v1\0")
    for predicted_rank, index in enumerate(ordered, 1):
        rank_digest.update(struct.pack("<qii", int(articles[index]), int(ranks[index]), predicted_rank))
    return {
        "audit_rows": len(scores),
        "raw_score_fingerprint": raw.hexdigest(),
        "rank_tiebreak_fingerprint": rank_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Stage 4B-4 locked final-refit worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--matrix-directory", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--resource-preregistration", type=Path, required=True)
    parser.add_argument("--absolute-rss-cap-bytes", type=int, required=True)
    parser.add_argument("--temporary-model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--worker-source-fingerprint", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    forbidden = {"duckdb", "pyarrow", "ranking_features"}
    imported_forbidden = sorted(name for name in sys.modules if name.split(".", 1)[0] in forbidden)
    if imported_forbidden:
        raise RuntimeError(f"isolated worker imported forbidden modules: {imported_forbidden}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    phase1_config = json.loads((args.config.parent / "stage_4b4_lambdarank.yaml").read_text(encoding="utf-8"))
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    resource = json.loads(args.resource_preregistration.read_text(encoding="utf-8"))
    metadata = json.loads((args.matrix_directory / "matrix_metadata.json").read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    if resource["frozen_absolute_rss_cap_bytes"] != args.absolute_rss_cap_bytes:
        raise RuntimeError("worker absolute RSS cap argument drift")
    if resource["rss_hard_fraction_of_available"] != 0.70:
        raise RuntimeError("worker rejects non-70% resource contract")
    if config["lightgbm"]["params"] != phase1_config["lightgbm"]["params"]:
        raise RuntimeError("LightGBM parameter drift from formal Phase 1 L1 config")
    if config["lightgbm"]["weights"] != phase1_config["lightgbm"]["weights"]:
        raise RuntimeError("LightGBM weight contract drift")
    if metadata["ordering_version"] != ORDERING_VERSION:
        raise RuntimeError("combined matrix ordering version drift")
    if prereg["model_contract"]["initialization"] != "fresh" or prereg["model_contract"]["warm_start"]:
        raise RuntimeError("fresh initialization contract drift")

    feature_names = list(adapter["ordered_feature_names"])
    categorical_names = list(adapter["categorical_features"])
    if len(feature_names) != 52 or categorical_names != config["categorical_features"]:
        raise RuntimeError("F1 feature/categorical contract drift")
    rows, queries = int(metadata["row_count"]), int(metadata["query_count"])
    paths = {
        "features": args.matrix_directory / "features_f1.float32",
        "labels": args.matrix_directory / "labels.uint8",
        "groups": args.matrix_directory / "groups.int32",
        "snapshot_codes": args.matrix_directory / "snapshot_codes.uint8",
        "customers": args.matrix_directory / "customer_keys.int64",
        "articles": args.matrix_directory / "article_keys.int64",
        "ranks": args.matrix_directory / "candidate_ranks.int32",
        "orders": args.matrix_directory / "query_orders.int64",
    }
    expected_sizes = {
        "features": rows * 52 * 4, "labels": rows, "groups": queries * 4,
        "snapshot_codes": rows, "customers": rows * 8, "articles": rows * 8,
        "ranks": rows * 4, "orders": rows * 8,
    }
    for role, path in paths.items():
        if path.stat().st_size != expected_sizes[role]:
            raise RuntimeError(f"combined {role} byte-size mismatch")
        if file_fingerprint(path) != metadata["physical_fingerprints"][role]:
            raise RuntimeError(f"combined {role} physical fingerprint mismatch")

    matrix = np.memmap(paths["features"], dtype=np.float32, mode="r", shape=(rows, 52))
    labels = np.memmap(paths["labels"], dtype=np.uint8, mode="r", shape=(rows,))
    groups = np.fromfile(paths["groups"], dtype=np.int32)
    snapshots = np.memmap(paths["snapshot_codes"], dtype=np.uint8, mode="r", shape=(rows,))
    customers = np.memmap(paths["customers"], dtype=np.int64, mode="r", shape=(rows,))
    articles = np.memmap(paths["articles"], dtype=np.int64, mode="r", shape=(rows,))
    ranks = np.memmap(paths["ranks"], dtype=np.int32, mode="r", shape=(rows,))
    orders = np.memmap(paths["orders"], dtype=np.int64, mode="r", shape=(rows,))
    if len(groups) != queries or int(groups.sum()) != rows:
        raise RuntimeError("combined query/group reconciliation mismatch")
    if np.any((labels != 0) & (labels != 1)):
        raise RuntimeError("combined labels are not binary")
    starts = np.r_[0, np.cumsum(groups[:-1], dtype=np.int64)]
    positives = np.add.reduceat(labels, starts)
    if np.any(positives < 1):
        raise RuntimeError("combined all-zero optimizer group detected")

    order_digest = hashlib.sha256(ORDERING_VERSION.encode("ascii") + b"\0")
    previous_snapshot = -1
    previous_order = -1
    for start, size, positive in zip(starts, groups, positives):
        end = int(start + size)
        group_snapshots, group_customers = snapshots[start:end], customers[start:end]
        group_articles, group_ranks = articles[start:end], ranks[start:end]
        group_orders, group_labels = orders[start:end], labels[start:end]
        if len(set(map(int, group_snapshots))) != 1 or len(set(map(int, group_customers))) != 1 or len(set(map(int, group_orders))) != 1:
            raise RuntimeError("combined cross-query boundary")
        snapshot_code, customer, query_order = int(group_snapshots[0]), int(group_customers[0]), int(group_orders[0])
        if snapshot_code < previous_snapshot or (snapshot_code == previous_snapshot and query_order <= previous_order):
            raise RuntimeError("combined snapshot/query order drift")
        if snapshot_code != previous_snapshot:
            previous_order = -1
        previous_snapshot, previous_order = snapshot_code, query_order
        invalid = (group_ranks[1:] < group_ranks[:-1]) | ((group_ranks[1:] == group_ranks[:-1]) & (group_articles[1:] < group_articles[:-1]))
        if np.any(invalid) or len(set(map(int, group_articles))) != len(group_articles):
            raise RuntimeError("combined candidate order/uniqueness drift")
        snapshot = "ranking_train_14d_v1" if snapshot_code == 0 else "validation_14d"
        row_digest = boundary_digest(snapshot, customer, query_order, int(start), group_articles, group_ranks, group_labels)
        order_digest.update(snapshot.encode("utf-8") + b"\0")
        order_digest.update(struct.pack("<qqqqqq", customer, query_order, int(start), end, int(size), int(positive)))
        order_digest.update(bytes.fromhex(row_digest))
    actual_order_fingerprint = order_digest.hexdigest()
    if actual_order_fingerprint != metadata["combined_matrix_order_fingerprint"]:
        raise RuntimeError("combined matrix-order fingerprint mismatch")
    for offset in range(0, rows, 65_536):
        if np.isinf(matrix[offset:offset + 65_536]).any():
            raise RuntimeError("combined feature matrix contains infinity")

    audit_rows = min(int(groups[0]), 512)
    audit_features = np.array(matrix[:audit_rows], dtype=np.float32, copy=True)
    audit_articles = np.array(articles[:audit_rows], dtype=np.int64, copy=True)
    audit_ranks = np.array(ranks[:audit_rows], dtype=np.int32, copy=True)
    categorical_indices = [feature_names.index(name) for name in categorical_names]
    params = dict(config["lightgbm"]["params"])
    dataset = lgb.Dataset(
        matrix, label=labels, group=groups, feature_name=feature_names,
        categorical_feature=categorical_indices, params=params, free_raw_data=True,
    )
    dataset.construct()
    if dataset.num_data() != rows or dataset.num_feature() != 52:
        raise RuntimeError("constructed final-refit Dataset dimension mismatch")
    registered = [feature_names[index] if isinstance(index, int) else str(index) for index in dataset.categorical_feature]
    if registered != categorical_names:
        raise RuntimeError("constructed final-refit categorical metadata mismatch")
    del matrix, labels, groups, snapshots, customers, articles, ranks, orders, starts, positives
    gc.collect()

    booster = lgb.train(params, dataset, num_boost_round=600, keep_training_booster=True)
    if booster.current_iteration() != 600:
        raise RuntimeError("formal final-refit booster did not complete exactly 600 rounds")
    pre_save_replay = score_replay(booster, audit_features, audit_articles, audit_ranks)
    args.temporary_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.temporary_model, num_iteration=600)
    reloaded = lgb.Booster(model_file=str(args.temporary_model))
    if reloaded.current_iteration() != 600 or reloaded.feature_name() != feature_names:
        raise RuntimeError("temporary final-refit model reload integrity failure")
    post_reload_replay = score_replay(reloaded, audit_features, audit_articles, audit_ranks)
    if pre_save_replay != post_reload_replay:
        raise RuntimeError("temporary model prediction replay mismatch")
    model_string = reloaded.model_to_string(num_iteration=600).replace("\r\n", "\n")
    dump_fingerprint = canonical_fingerprint(reloaded.dump_model(num_iteration=600))
    payload = {
        "status": "WORKER_SUCCESS_AWAITING_EXTERNAL_RESOURCE_GATE",
        "model_id": "L1_FINAL_REFIT_600", "feature_set": "F1", "features": 52,
        "rows": rows, "queries": queries, "rounds": 600,
        "fresh_initialization": True, "warm_start": False, "init_model_used": False,
        "internal_validation": False, "valid_sets_used": False, "early_stopping": False,
        "model_physical_fingerprint": file_fingerprint(args.temporary_model),
        "model_semantic_fingerprint": hashlib.sha256(model_string.encode("utf-8")).hexdigest(),
        "lightgbm_dump_semantic_fingerprint": dump_fingerprint,
        "feature_name_order_fingerprint": canonical_fingerprint(feature_names),
        "categorical_metadata_fingerprint": canonical_fingerprint(categorical_names),
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "parameters_fingerprint": canonical_fingerprint(params),
        "weights_fingerprint": canonical_fingerprint(config["lightgbm"]["weights"]),
        "seeds": {name: params[name] for name in ("seed", "data_random_seed", "feature_fraction_seed", "bagging_seed", "drop_seed")},
        "lightgbm_version": lgb.__version__,
        "combined_matrix_manifest_fingerprint": metadata["manifest_fingerprint"],
        "combined_matrix_order_fingerprint": actual_order_fingerprint,
        "resource_preregistration_fingerprint": resource["resource_preregistration_fingerprint"],
        "frozen_absolute_rss_cap_bytes_consumed": args.absolute_rss_cap_bytes,
        "worker_recalculated_available_memory": False,
        "worker_source_fingerprint": args.worker_source_fingerprint,
        "prediction_replay": post_reload_replay,
        "validation_metric_read_count": 0, "first7d_label_read_count": 0,
        "test_candidate_read_count": 0, "test_feature_read_count": 0,
        "test_label_read_count": 0, "test_prediction_write_count": 0, "test_metric_read_count": 0,
        "wall_time_seconds": time.perf_counter() - started,
    }
    write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()
