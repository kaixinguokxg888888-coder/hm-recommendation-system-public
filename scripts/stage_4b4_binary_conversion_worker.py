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


def canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


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
    parser = argparse.ArgumentParser(description="Isolated LightGBM binary Dataset conversion worker")
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--combined-directory", type=Path, required=True)
    parser.add_argument("--combined-manifest", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--resource-preregistration", type=Path, required=True)
    parser.add_argument("--absolute-rss-cap-bytes", type=int, required=True)
    parser.add_argument("--temporary-binary", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--worker-source-fingerprint", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    forbidden = {"duckdb", "pyarrow", "ranking_features"}
    imported = sorted(name for name in sys.modules if name.split(".", 1)[0] in forbidden)
    if imported:
        raise RuntimeError(f"conversion worker imported forbidden modules: {imported}")
    recovery = json.loads(args.recovery_config.read_text(encoding="utf-8"))
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.combined_manifest.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    resource = json.loads(args.resource_preregistration.read_text(encoding="utf-8"))
    if resource["frozen_absolute_conversion_cap_bytes"] != args.absolute_rss_cap_bytes:
        raise RuntimeError("conversion absolute cap argument drift")
    if resource["rss_fraction"] != 0.70:
        raise RuntimeError("conversion worker rejects non-70% cap")
    if recovery["lightgbm"]["params"] != base["lightgbm"]["params"]:
        raise RuntimeError("conversion Dataset parameter drift")
    params = dict(recovery["lightgbm"]["params"])
    names = list(adapter["ordered_feature_names"])
    categorical = list(adapter["categorical_features"])
    if len(names) != 52 or categorical != base["categorical_features"]:
        raise RuntimeError("conversion feature/categorical contract drift")
    rows = int(manifest["row_count"])
    queries = int(manifest["query_count"])
    feature_path = args.combined_directory / "features_f1.float32"
    label_path = args.combined_directory / "labels.uint8"
    group_path = args.combined_directory / "groups.int32"
    expected_sizes = {"features": rows * 52 * 4, "labels": rows, "groups": queries * 4}
    actual_paths = {"features": feature_path, "labels": label_path, "groups": group_path}
    for role, path in actual_paths.items():
        if path.stat().st_size != expected_sizes[role]:
            raise RuntimeError(f"conversion {role} byte-size mismatch")
        if file_fingerprint(path) != manifest["physical_fingerprints"][role]:
            raise RuntimeError(f"conversion {role} fingerprint drift")
    matrix = np.memmap(feature_path, dtype=np.float32, mode="r", shape=(rows, 52))
    labels = np.memmap(label_path, dtype=np.uint8, mode="r", shape=(rows,))
    groups = np.fromfile(group_path, dtype=np.int32)
    if len(groups) != queries or int(groups.sum()) != rows:
        raise RuntimeError("conversion group reconciliation mismatch")
    if np.any((labels != 0) & (labels != 1)):
        raise RuntimeError("conversion labels are not binary")
    if int(labels.sum()) != manifest["positive_rows"]:
        raise RuntimeError("conversion positive count mismatch")
    infinity_violations = 0
    for offset in range(0, rows, 65_536):
        infinity_violations += int(np.isinf(matrix[offset:offset + 65_536]).sum())
    if infinity_violations:
        raise RuntimeError("conversion feature matrix contains infinity")
    categorical_indices = [names.index(name) for name in categorical]
    dataset = lgb.Dataset(
        matrix, label=labels, group=groups, feature_name=names,
        categorical_feature=categorical_indices, params=params, free_raw_data=True,
    )
    dataset.construct()
    if dataset.num_data() != rows or dataset.num_feature() != 52 or dataset.feature_name != names:
        raise RuntimeError("constructed binary Dataset dimensions/features mismatch")
    registered = [names[index] if isinstance(index, int) else str(index) for index in dataset.categorical_feature]
    if registered != categorical:
        raise RuntimeError("constructed binary Dataset categorical metadata mismatch")
    args.temporary_binary.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_binary(args.temporary_binary)
    del matrix, labels, groups, dataset
    gc.collect()
    loaded = lgb.Dataset(str(args.temporary_binary), params=params)
    loaded.construct()
    loaded_labels = np.asarray(loaded.get_label(), dtype=np.uint8)
    loaded_groups = np.asarray(loaded.get_group(), dtype=np.int32)
    if loaded.num_data() != rows or loaded.num_feature() != 52 or loaded.feature_name != names:
        raise RuntimeError("temporary binary reload dimensions/features mismatch")
    if hashlib.sha256(loaded_labels.tobytes()).hexdigest() != manifest["label_physical_fingerprint"]:
        raise RuntimeError("temporary binary label reorder/drift")
    if hashlib.sha256(loaded_groups.tobytes()).hexdigest() != manifest["group_physical_fingerprint"]:
        raise RuntimeError("temporary binary group reorder/drift")
    binning = {
        "max_bin": params["max_bin"], "data_random_seed": params["data_random_seed"],
        "categorical_features": categorical, "categorical_indices": categorical_indices,
        "feature_names": names, "missing_semantics": recovery["lightgbm"]["missing_semantics"],
        "zero_as_missing": recovery["lightgbm"]["zero_as_missing"],
    }
    payload = {
        "status": "CONVERSION_WORKER_SUCCESS_AWAITING_VERIFICATION_AND_RESOURCE_GATE",
        "rows": rows, "queries": queries, "features": 52,
        "positive_rows": int(loaded_labels.sum()), "zero_rows": int(rows - loaded_labels.sum()),
        "group_count": len(loaded_groups), "group_sum": int(loaded_groups.sum()),
        "binary_physical_fingerprint": file_fingerprint(args.temporary_binary),
        "binary_size_bytes": args.temporary_binary.stat().st_size,
        "label_semantic_fingerprint": hashlib.sha256(loaded_labels.tobytes()).hexdigest(),
        "group_semantic_fingerprint": hashlib.sha256(loaded_groups.tobytes()).hexdigest(),
        "feature_name_order_fingerprint": fingerprint(names),
        "categorical_metadata_fingerprint": fingerprint(categorical),
        "dataset_binning_fingerprint": fingerprint(binning),
        "parameter_fingerprint": fingerprint(params),
        "adapter_fingerprint": adapter["adapter_fingerprint"],
        "combined_matrix_manifest_fingerprint": manifest["manifest_fingerprint"],
        "combined_feature_matrix_fingerprint": manifest["feature_matrix_physical_fingerprint"],
        "combined_matrix_order_fingerprint": manifest["combined_matrix_order_fingerprint"],
        "infinity_violations": infinity_violations,
        "native_nan_missing_allowed": True, "zero_as_missing": False,
        "model_trained": False, "business_metric_computed": False,
        "source_label_files_opened": False, "runtime_cap_recomputed": False,
        "frozen_absolute_conversion_cap_bytes_consumed": args.absolute_rss_cap_bytes,
        "resource_preregistration_fingerprint": resource["resource_preregistration_fingerprint"],
        "worker_source_fingerprint": args.worker_source_fingerprint,
        "lightgbm_version": lgb.__version__, "wall_time_seconds": time.perf_counter() - started,
        "validation_labels_reopened": False, "first7d_labels_read": False, "test_access": False,
    }
    payload["conversion_result_fingerprint"] = fingerprint(payload)
    write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()
