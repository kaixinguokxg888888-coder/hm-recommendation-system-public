from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


def canonical(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh binary-only final-refit training worker")
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--phase1-config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--binary-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--resource-preregistration", type=Path, required=True)
    parser.add_argument("--absolute-rss-cap-bytes", type=int, required=True)
    parser.add_argument("--temporary-model", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--worker-source-fingerprint", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    forbidden = {"duckdb", "pyarrow", "ranking_features"}
    if any(name.split(".", 1)[0] in forbidden for name in sys.modules):
        raise RuntimeError("binary-only training worker imported forbidden pipeline module")
    recovery = json.loads(args.recovery_config.read_text(encoding="utf-8"))
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    phase1 = json.loads(args.phase1_config.read_text(encoding="utf-8"))
    binary_manifest = json.loads(args.binary_manifest.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    resource = json.loads(args.resource_preregistration.read_text(encoding="utf-8"))
    params = dict(recovery["lightgbm"]["params"])
    if params != base["lightgbm"]["params"] or params != phase1["lightgbm"]["params"]:
        raise RuntimeError("FINAL_REFIT_SEMANTIC_DRIFT: training params")
    if recovery["lightgbm"]["weights"] != phase1["lightgbm"]["weights"]:
        raise RuntimeError("FINAL_REFIT_SEMANTIC_DRIFT: training weights")
    if resource["frozen_absolute_training_cap_bytes"] != args.absolute_rss_cap_bytes:
        raise RuntimeError("training absolute cap argument drift")
    if resource["rss_fraction"] != 0.70:
        raise RuntimeError("training worker rejects non-70% cap")
    if file_fingerprint(args.binary) != binary_manifest["binary_physical_fingerprint"]:
        raise RuntimeError("formal binary physical fingerprint drift")
    names = list(adapter["ordered_feature_names"])
    categorical = list(adapter["categorical_features"])
    dataset = lgb.Dataset(str(args.binary), params=params)
    dataset.construct()
    labels = np.asarray(dataset.get_label(), dtype=np.uint8)
    groups = np.asarray(dataset.get_group(), dtype=np.int32)
    expected = recovery["expected"]
    if dataset.num_data() != expected["combined_rows"] or dataset.num_feature() != 52:
        raise RuntimeError("formal binary training dimensions mismatch")
    if dataset.feature_name != names or len(groups) != expected["combined_queries"]:
        raise RuntimeError("formal binary training feature/group mismatch")
    if int(groups.sum()) != expected["combined_rows"] or int(labels.sum()) != expected["combined_positive_rows"]:
        raise RuntimeError("formal binary training label/group count mismatch")
    if hashlib.sha256(labels.tobytes()).hexdigest() != expected["label_physical_fingerprint"]:
        raise RuntimeError("formal binary training label identity drift")
    if hashlib.sha256(groups.tobytes()).hexdigest() != expected["group_physical_fingerprint"]:
        raise RuntimeError("formal binary training group identity drift")
    del labels, groups
    gc.collect()
    booster = lgb.train(params, dataset, num_boost_round=600, keep_training_booster=True)
    if booster.current_iteration() != 600 or booster.num_trees() != 600:
        raise RuntimeError("formal binary-only booster did not complete exactly 600 rounds/trees")
    args.temporary_model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.temporary_model, num_iteration=600)
    reloaded = lgb.Booster(model_file=str(args.temporary_model))
    if reloaded.current_iteration() != 600 or reloaded.num_trees() != 600 or reloaded.feature_name() != names:
        raise RuntimeError("temporary binary-only model reload integrity failure")
    dump = reloaded.dump_model(num_iteration=600)
    if len(dump["tree_info"]) != 600:
        raise RuntimeError("LightGBM dump does not contain exactly 600 trees")
    model_string = reloaded.model_to_string(num_iteration=600).replace("\r\n", "\n")
    payload = {
        "status": "BINARY_TRAINING_WORKER_SUCCESS_AWAITING_RESOURCE_GATE",
        "model_id": "L1_FINAL_REFIT_600", "feature_set": "F1", "features": 52,
        "rows": dataset.num_data(), "queries": expected["combined_queries"],
        "rounds": reloaded.current_iteration(), "trees": reloaded.num_trees(),
        "fresh_initialization": True, "warm_start": False,
        "continued_training_from_L1_600": False, "init_model_used": False,
        "internal_validation": False, "valid_sets_used": False, "early_stopping": False,
        "loaded_formal_binary_only": True, "loaded_raw_combined_memmap": False,
        "model_physical_fingerprint": file_fingerprint(args.temporary_model),
        "model_semantic_fingerprint": hashlib.sha256(model_string.encode("utf-8")).hexdigest(),
        "lightgbm_dump_semantic_fingerprint": fingerprint(dump),
        "feature_name_order_fingerprint": fingerprint(names),
        "categorical_metadata_fingerprint": fingerprint(categorical),
        "parameter_fingerprint": fingerprint(params),
        "weights_fingerprint": fingerprint(recovery["lightgbm"]["weights"]),
        "seeds": {name: params[name] for name in (
            "seed", "data_random_seed", "feature_fraction_seed", "bagging_seed", "drop_seed"
        )},
        "binary_physical_fingerprint": binary_manifest["binary_physical_fingerprint"],
        "binary_semantic_fingerprint": binary_manifest["binary_dataset_semantic_fingerprint"],
        "binary_manifest_fingerprint": binary_manifest["binary_manifest_fingerprint"],
        "resource_preregistration_fingerprint": resource["resource_preregistration_fingerprint"],
        "frozen_absolute_training_cap_bytes_consumed": args.absolute_rss_cap_bytes,
        "runtime_cap_recomputed": False, "worker_source_fingerprint": args.worker_source_fingerprint,
        "lightgbm_version": lgb.__version__, "wall_time_seconds": time.perf_counter() - started,
        "validation_metric_computed": False, "first7d_labels_read": False, "test_access": False,
    }
    payload["training_worker_result_fingerprint"] = fingerprint(payload)
    write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()
