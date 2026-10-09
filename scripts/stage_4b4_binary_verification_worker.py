from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Independent LightGBM binary Dataset verifier")
    parser.add_argument("--recovery-config", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--conversion-result", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    forbidden = {"duckdb", "pyarrow", "ranking_features"}
    if any(name.split(".", 1)[0] in forbidden for name in sys.modules):
        raise RuntimeError("binary verifier imported forbidden pipeline module")
    recovery = json.loads(args.recovery_config.read_text(encoding="utf-8"))
    conversion = json.loads(args.conversion_result.read_text(encoding="utf-8"))
    adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
    params = dict(recovery["lightgbm"]["params"])
    expected = recovery["expected"]
    names = list(adapter["ordered_feature_names"])
    categorical = list(adapter["categorical_features"])
    physical = file_fingerprint(args.binary)
    if physical != conversion["binary_physical_fingerprint"]:
        raise RuntimeError("BINARY_DATASET_SEMANTIC_MISMATCH: physical")
    dataset = lgb.Dataset(str(args.binary), params=params)
    dataset.construct()
    labels = np.asarray(dataset.get_label(), dtype=np.uint8)
    groups = np.asarray(dataset.get_group(), dtype=np.int32)
    checks = {
        "row_count": dataset.num_data() == expected["combined_rows"],
        "query_count": len(groups) == expected["combined_queries"],
        "feature_count": dataset.num_feature() == expected["combined_feature_count"],
        "group_sum": int(groups.sum()) == expected["combined_rows"],
        "label_count": len(labels) == expected["combined_rows"],
        "positive_count": int(labels.sum()) == expected["combined_positive_rows"],
        "zero_count": int(len(labels) - labels.sum()) == expected["combined_zero_rows"],
        "feature_names": dataset.feature_name == names,
        "label_identity": hashlib.sha256(labels.tobytes()).hexdigest() == expected["label_physical_fingerprint"],
        "group_identity": hashlib.sha256(groups.tobytes()).hexdigest() == expected["group_physical_fingerprint"],
        "feature_name_identity": fingerprint(names) == expected["feature_name_order_fingerprint"],
        "categorical_metadata_identity": fingerprint(categorical) == expected["categorical_metadata_fingerprint"],
        "binning_identity": conversion["dataset_binning_fingerprint"] == fingerprint({
            "max_bin": params["max_bin"], "data_random_seed": params["data_random_seed"],
            "categorical_features": categorical,
            "categorical_indices": [names.index(name) for name in categorical],
            "feature_names": names, "missing_semantics": recovery["lightgbm"]["missing_semantics"],
            "zero_as_missing": recovery["lightgbm"]["zero_as_missing"],
        }),
        "no_infinity_violation": conversion["infinity_violations"] == 0,
        "no_group_label_reorder": (
            conversion["label_semantic_fingerprint"] == expected["label_physical_fingerprint"]
            and conversion["group_semantic_fingerprint"] == expected["group_physical_fingerprint"]
        ),
        "no_training_or_metrics": not conversion["model_trained"] and not conversion["business_metric_computed"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"BINARY_DATASET_SEMANTIC_MISMATCH: {[k for k,v in checks.items() if not v]}")
    semantic_binding = {
        "binary_physical_fingerprint": physical,
        "combined_matrix_manifest_fingerprint": conversion["combined_matrix_manifest_fingerprint"],
        "combined_feature_matrix_fingerprint": conversion["combined_feature_matrix_fingerprint"],
        "combined_matrix_order_fingerprint": conversion["combined_matrix_order_fingerprint"],
        "label_fingerprint": conversion["label_semantic_fingerprint"],
        "group_fingerprint": conversion["group_semantic_fingerprint"],
        "feature_name_order_fingerprint": conversion["feature_name_order_fingerprint"],
        "categorical_metadata_fingerprint": conversion["categorical_metadata_fingerprint"],
        "dataset_binning_fingerprint": conversion["dataset_binning_fingerprint"],
        "parameter_fingerprint": conversion["parameter_fingerprint"],
        "lightgbm_version": conversion["lightgbm_version"],
        "rows": dataset.num_data(), "queries": len(groups), "features": dataset.num_feature(),
    }
    payload = {
        "status": "BINARY_DATASET_VERIFIED_AWAITING_ATOMIC_PROMOTION",
        "checks": checks, **semantic_binding,
        "binary_dataset_semantic_fingerprint": fingerprint(semantic_binding),
        "dataset_metadata_fingerprint": fingerprint({
            "rows": dataset.num_data(), "queries": len(groups), "features": dataset.num_feature(),
            "positive_rows": int(labels.sum()), "zero_rows": int(len(labels) - labels.sum()),
            "feature_names": names, "categorical_features": categorical,
        }),
        "verification_process_loaded_binary_only": True,
        "validation_labels_reopened": False, "first7d_labels_read": False, "test_access": False,
        "wall_time_seconds": time.perf_counter() - started,
    }
    payload["verification_audit_fingerprint"] = fingerprint(payload)
    write_json_atomic(args.result, payload)


if __name__ == "__main__":
    main()
