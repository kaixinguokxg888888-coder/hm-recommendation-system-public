from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import psutil

from data_foundation.manifest import file_fingerprint
from ranking_model import Stage4B4Config, Stage4B4Runner
from ranking_model.ordering import ORDERING_VERSION


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = [root / "scripts/stage_4b4_isolated_worker.py", root / "scripts/stage_4b4_recovery.py"]
    paths.extend(sorted((root / "src/ranking_model").glob("*.py")))
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def process_tree_rss(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        members = [process, *process.children(recursive=True)]
        return sum(member.memory_info().rss for member in members if member.is_running())
    except (psutil.Error, ProcessLookupError, ValueError):
        return 0


def terminate_tree(pid: int) -> None:
    try:
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        for member in reversed(children):
            member.kill()
        process.kill()
        psutil.wait_procs([*children, process], timeout=10)
    except (psutil.Error, ProcessLookupError):
        return


def train_isolated(runner: Stage4B4Runner, feature_set: str, matrix: Path, worker_fp: str) -> dict[str, Any]:
    model_id = "L0" if feature_set == "F0" else "L1"
    root = runner.config.root
    attempt_root = runner.config.private_dir / "models" / "attempts"
    prior = sorted(attempt_root.glob(f"{model_id}_isolated_*"))
    attempt_number = max([int(path.name.rsplit("_", 1)[1]) for path in prior] or [0]) + 1
    attempt_dir = attempt_root / f"{model_id}_isolated_{attempt_number:03d}"
    temporary_model = attempt_dir / f"{model_id}_600.pending.txt"
    worker_result = attempt_dir / "worker_result.json"
    external_result = attempt_dir / "external_resource_gate.json"
    formal_model = runner.model_path(model_id)
    formal_metadata = runner.config.private_dir / "models" / f"{model_id}_metadata.json"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    available = psutil.virtual_memory().available
    cap = int(available * float(runner.raw["execution"]["peak_rss_fraction_of_available"]))
    parent_rss = psutil.Process().memory_info().rss
    training = runner.config.private_dir / "training_query_order_v2"
    matrix_metadata = training / "matrix_metadata.json"
    ordering = json.loads(matrix_metadata.read_text(encoding="utf-8"))
    if ordering.get("ordering_version") != ORDERING_VERSION:
        raise RuntimeError("coordinator rejects non-v2 training matrix")
    command = [
        sys.executable,
        str(root / "scripts/stage_4b4_isolated_worker.py"),
        "--config", str(runner.config.path),
        "--feature-set", feature_set,
        "--feature-matrix", str(matrix),
        "--labels", str(training / "labels.uint8"),
        "--groups", str(training / "groups.int32"),
        "--customer-keys", str(training / "customer_keys.int64"),
        "--article-keys", str(training / "article_keys.int64"),
        "--candidate-ranks", str(training / "candidate_ranks.int32"),
        "--query-orders", str(training / "query_orders.int64"),
        "--matrix-metadata", str(matrix_metadata),
        "--adapter-manifest", str(runner.config.artifact_dir / f"model_input_adapter_{feature_set.lower()}.json"),
        "--temporary-model", str(temporary_model),
        "--result", str(worker_result),
        "--worker-source-fingerprint", worker_fp,
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    peak = 0
    exceeded = False
    while process.poll() is None:
        peak = max(peak, process_tree_rss(process.pid))
        if peak > cap:
            exceeded = True
            terminate_tree(process.pid)
            break
        time.sleep(0.05)
    peak = max(peak, process_tree_rss(process.pid))
    _stdout, stderr = process.communicate()
    result: dict[str, Any] = {
        "attempt": attempt_number,
        "model_id": model_id,
        "feature_set": feature_set,
        "child_pid": process.pid,
        "available_memory_at_start_bytes": available,
        "rss_hard_cap_bytes": cap,
        "parent_coordinator_rss_bytes": parent_rss,
        "child_process_tree_peak_rss_bytes": peak,
        "wall_time_seconds": time.perf_counter() - started,
        "exit_code": process.returncode,
        "rss_gate_exceeded": exceeded,
        "worker_source_fingerprint": worker_fp,
        "validation_label_read_count": 0,
        "first7d_read_count": 0,
        "test_read_count": 0,
        "error": stderr[-2000:].replace(str(root), ".") if stderr else "",
    }
    if exceeded or process.returncode != 0 or not worker_result.exists() or not temporary_model.exists():
        result["status"] = "FAILED"
        write_json_atomic(external_result, result)
        raise RuntimeError(f"{model_id} isolated worker failed resource/process gate")

    worker = json.loads(worker_result.read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(temporary_model))
    if (worker["rounds"] != 600 or booster.current_iteration() != 600
            or worker["model_physical_fingerprint"] != file_fingerprint(temporary_model)):
        result["status"] = "FAILED"
        result["error"] = "temporary model acceptance audit failed"
        write_json_atomic(external_result, result)
        raise RuntimeError(result["error"])
    result["status"] = "PASS"
    result["worker_result"] = worker
    write_json_atomic(external_result, result)
    formal_model.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_model, formal_model)
    metadata = {
        **worker,
        "status": "ACCEPTED_AFTER_ISOLATED_RESOURCE_GATE",
        "model_physical_fingerprint": file_fingerprint(formal_model),
        "external_resource_gate": {key: value for key, value in result.items() if key != "worker_result"},
    }
    write_json_atomic(formal_metadata, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 narrow resource recovery")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_lambdarank.yaml"))
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    config = Stage4B4Config.load(args.config)
    runner = Stage4B4Runner(config)
    preflight = runner.preflight()
    worker_fp = source_fingerprint(config.root)
    training = config.private_dir / "training_query_order_v2"
    metadata_path = training / "matrix_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("query-order v2 matrix must be built and audited first")
    ordering = json.loads(metadata_path.read_text(encoding="utf-8"))
    if ordering.get("ordering_version") != ORDERING_VERSION or any(ordering["audit"][key] for key in (
        "query_position_mismatch", "group_size_position_mismatch", "positive_count_position_mismatch",
        "sampled_zero_count_position_mismatch", "candidate_rank_order_violations", "all_zero_groups",
        "cross_query_boundary_groups", "duplicate_row_keys",
    )):
        raise RuntimeError("query-order v2 matrix audit failed")
    model_rows = []
    for feature_set, matrix in (("F0", training / "features_f0.float32"),
                                ("F1", training / "features_f1.float32")):
        model_id = "L0" if feature_set == "F0" else "L1"
        metadata = config.private_dir / "models" / f"{model_id}_metadata.json"
        if metadata.exists() and runner.model_path(model_id).exists():
            previous = json.loads(metadata.read_text(encoding="utf-8"))
            if (previous.get("worker_source_fingerprint") == worker_fp
                    and previous.get("model_physical_fingerprint") == file_fingerprint(runner.model_path(model_id))
                    and previous.get("rounds") == 600):
                model_rows.append(previous)
                continue
        model_rows.append(train_isolated(runner, feature_set, matrix, worker_fp))
    result: dict[str, Any] = {"preflight": preflight, "matrix_ordering": ordering, "models": model_rows}
    if not args.train_only:
        validation = json.loads((config.artifact_dir / "validation_feature_fingerprint_manifest.csv").read_text(
            encoding="utf-8").splitlines()[1]) if False else {
                "candidate_projection_fingerprint": raw_value(config.artifact_dir / "validation_feature_fingerprint_manifest.csv", "candidate_projection_fingerprint"),
                "feature_semantic_fingerprint": raw_value(config.artifact_dir / "validation_feature_fingerprint_manifest.csv", "feature_semantic_fingerprint"),
            }
        predictions = runner.freeze_predictions(validation, model_rows)
        result["predictions"] = predictions
    write_json_atomic(config.private_dir / "recovery_result.json", result)
    print(json.dumps({"status": "RECOVERY_TRAINING_COMPLETE" if args.train_only else "RECOVERY_PREDICTIONS_COMPLETE",
                      "models": [row["model_id"] for row in model_rows]}, indent=2))


def raw_value(path: Path, field: str) -> str:
    import csv
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle))[field]


if __name__ == "__main__":
    main()
