from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


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


def process_tree_rss(process: psutil.Process) -> int:
    values = [process]
    try:
        values.extend(process.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    total = 0
    for value in values:
        try:
            total += int(value.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def terminate_process_tree(process: psutil.Process) -> None:
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    try:
        process.terminate()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    _, alive = psutil.wait_procs(children + [process], timeout=5)
    for value in alive:
        try:
            value.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def replay_formal_model(model_path: Path, matrix_dir: Path, rows: int) -> dict[str, object]:
    import lightgbm as lgb
    import numpy as np

    metadata = json.loads((matrix_dir / "matrix_metadata.json").read_text(encoding="utf-8"))
    total_rows = int(metadata["row_count"])
    groups = np.fromfile(matrix_dir / "groups.int32", dtype=np.int32)
    audit_rows = min(int(groups[0]), rows)
    matrix = np.memmap(matrix_dir / "features_f1.float32", dtype=np.float32, mode="r", shape=(total_rows, 52))
    articles = np.memmap(matrix_dir / "article_keys.int64", dtype=np.int64, mode="r", shape=(total_rows,))
    ranks = np.memmap(matrix_dir / "candidate_ranks.int32", dtype=np.int32, mode="r", shape=(total_rows,))
    features = np.array(matrix[:audit_rows], dtype=np.float32, copy=True)
    audit_articles = np.array(articles[:audit_rows], dtype=np.int64, copy=True)
    audit_ranks = np.array(ranks[:audit_rows], dtype=np.int32, copy=True)
    booster = lgb.Booster(model_file=str(model_path))
    if booster.current_iteration() != 600:
        raise RuntimeError("formal promoted model reload round mismatch")
    scores = np.asarray(booster.predict(features, num_iteration=600), dtype=np.float64)
    raw = hashlib.sha256(b"stage-4b4-final-refit-raw-score-replay-v1\0")
    for score in scores:
        raw.update(float(score).hex().encode("ascii") + b"\0")
    ordered = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), int(audit_ranks[index]), int(audit_articles[index])))
    rank_digest = hashlib.sha256(b"stage-4b4-final-refit-rank-replay-v1\0")
    import struct
    for predicted_rank, index in enumerate(ordered, 1):
        rank_digest.update(struct.pack("<qii", int(audit_articles[index]), int(audit_ranks[index]), predicted_rank))
    return {
        "audit_rows": audit_rows,
        "raw_score_fingerprint": raw.hexdigest(),
        "rank_tiebreak_fingerprint": rank_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="External resource-gated Stage 4B-4 final-refit launcher")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_final_refit.yaml"))
    args = parser.parse_args()
    root = args.config.resolve().parent.parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    paths = {name: root / value for name, value in config["paths"].items()}
    prereg = json.loads(paths["preregistration"].read_text(encoding="utf-8"))
    combined = json.loads(paths["combined_matrix_manifest"].read_text(encoding="utf-8"))
    lineage = json.loads(paths["lineage_manifest"].read_text(encoding="utf-8"))
    matrix_dir = paths["private_directory"] / "combined_matrix"
    model_dir = paths["private_directory"] / "models"
    temp_dir = paths["temp_directory"] / "training_worker"
    checkpoint_dir = paths["checkpoint_directory"]
    for path in (model_dir, temp_dir, checkpoint_dir):
        path.mkdir(parents=True, exist_ok=True)
    worker_path = root / "scripts/stage_4b4_final_refit_worker.py"
    worker_source_fingerprint = file_fingerprint(worker_path)
    resource_path = paths["resource_preregistration"]
    if resource_path.exists():
        resource_prereg = json.loads(resource_path.read_text(encoding="utf-8"))
        if resource_prereg["worker_source_fingerprint"] != worker_source_fingerprint:
            raise RuntimeError("existing resource preregistration worker source drift")
        if resource_prereg["combined_matrix_manifest_fingerprint"] != combined["manifest_fingerprint"]:
            raise RuntimeError("existing resource preregistration matrix drift")
    else:
        memory = psutil.virtual_memory()
        available = int(memory.available)
        cap = int(available * 0.70)
        resource_prereg = {
            "status": "FINAL_REFIT_RESOURCE_PREREGISTERED_BEFORE_WORKER",
            "available_memory_before_worker_bytes": available,
            "rss_hard_fraction_of_available": 0.70,
            "frozen_absolute_rss_cap_bytes": cap,
            "worker_consumes_frozen_absolute_cap_only": True,
            "external_process_tree_monitor": True,
            "runtime_available_memory_is_observation_only": True,
            "combined_matrix_manifest_fingerprint": combined["manifest_fingerprint"],
            "phase4_preregistration_fingerprint": prereg["preregistration_fingerprint"],
            "worker_source_fingerprint": worker_source_fingerprint,
        }
        resource_prereg["resource_preregistration_fingerprint"] = canonical_fingerprint(resource_prereg)
        write_json_atomic(resource_path, resource_prereg)
    cap = int(resource_prereg["frozen_absolute_rss_cap_bytes"])
    temporary_model = temp_dir / "L1_FINAL_REFIT_600.txt.tmp"
    worker_result_path = temp_dir / "worker_result.json"
    stdout_path = temp_dir / "worker.stdout.log"
    stderr_path = temp_dir / "worker.stderr.log"
    formal_model = model_dir / "L1_FINAL_REFIT_600.txt"
    if formal_model.exists():
        raise RuntimeError("formal final-refit model already exists; refusing unregistered retraining")
    command = [
        sys.executable, str(worker_path), "--config", str(args.config),
        "--matrix-directory", str(matrix_dir),
        "--adapter", str(root / config["upstream"]["phase1_adapter"]),
        "--preregistration", str(paths["preregistration"]),
        "--resource-preregistration", str(resource_path),
        "--absolute-rss-cap-bytes", str(cap),
        "--temporary-model", str(temporary_model),
        "--result", str(worker_result_path),
        "--worker-source-fingerprint", worker_source_fingerprint,
    ]
    started = time.perf_counter()
    peak_rss = 0
    available_min = int(psutil.virtual_memory().available)
    available_max = available_min
    cap_exceeded = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            rss = process_tree_rss(process)
            peak_rss = max(peak_rss, rss)
            observed = int(psutil.virtual_memory().available)
            available_min, available_max = min(available_min, observed), max(available_max, observed)
            if rss > cap:
                cap_exceeded = True
                terminate_process_tree(process)
                break
            time.sleep(0.25)
        exit_code = child.wait()
    resource_audit = {
        "status": "BLOCKED_BY_RESOURCE" if cap_exceeded else ("PASS" if exit_code == 0 else "POST_FINAL_REFIT_IMPLEMENTATION_FAILURE"),
        "worker_pid_recorded": True,
        "worker_exit_code": exit_code,
        "rss_hard_fraction_of_available": 0.70,
        "frozen_absolute_rss_cap_bytes": cap,
        "child_process_tree_peak_rss_bytes": peak_rss,
        "rss_gate_exceeded": cap_exceeded,
        "runtime_available_memory_min_bytes_observation_only": available_min,
        "runtime_available_memory_max_bytes_observation_only": available_max,
        "worker_recalculated_available_memory": False,
        "spill_directory": "artifacts/stage_4b4/tmp/final_refit/training_worker",
        "oom_observed": False,
        "wall_time_seconds": time.perf_counter() - started,
        "resource_preregistration_fingerprint": resource_prereg["resource_preregistration_fingerprint"],
    }
    resource_audit["resource_audit_fingerprint"] = canonical_fingerprint(resource_audit)
    write_json_atomic(paths["resource_audit"], resource_audit)
    if cap_exceeded:
        raise RuntimeError("STOP BLOCKED_BY_RESOURCE")
    if exit_code != 0 or not worker_result_path.exists() or not temporary_model.exists():
        raise RuntimeError("STOP POST_FINAL_REFIT_IMPLEMENTATION_FAILURE")
    worker_result = json.loads(worker_result_path.read_text(encoding="utf-8"))
    if worker_result["model_physical_fingerprint"] != file_fingerprint(temporary_model):
        raise RuntimeError("temporary model integrity mismatch before promotion")
    if worker_result["frozen_absolute_rss_cap_bytes_consumed"] != cap:
        raise RuntimeError("worker did not consume frozen absolute cap")
    os.replace(temporary_model, formal_model)
    if file_fingerprint(formal_model) != worker_result["model_physical_fingerprint"]:
        raise RuntimeError("atomic model promotion fingerprint mismatch")
    formal_replay = replay_formal_model(formal_model, matrix_dir, rows=512)
    if formal_replay != worker_result["prediction_replay"]:
        raise RuntimeError("formal promoted model reload/prediction replay mismatch")
    replay_audit = {
        "status": "PASS", "model_id": "L1_FINAL_REFIT_600",
        "audit_slice": "first_complete_combined_query_label_free_v1",
        "labels_read_for_replay": False,
        "temporary_reload_replay": worker_result["prediction_replay"],
        "formal_reload_replay": formal_replay,
        "raw_score_exact_match": True, "rank_tiebreak_exact_match": True,
        "validation_metric_recalculated": False, "first7d_label_read_count": 0,
        "test_label_read_count": 0,
    }
    replay_audit["replay_fingerprint"] = canonical_fingerprint(replay_audit)
    write_json_atomic(paths["replay_audit"], replay_audit)
    checkpoint_payload = {
        "status": "FORMAL_MODEL_ATOMICALLY_PROMOTED_AFTER_RESOURCE_GATE",
        "model_physical_fingerprint": worker_result["model_physical_fingerprint"],
        "resource_audit_fingerprint": resource_audit["resource_audit_fingerprint"],
        "replay_fingerprint": replay_audit["replay_fingerprint"],
    }
    checkpoint_payload["checkpoint_fingerprint"] = canonical_fingerprint(checkpoint_payload)
    write_json_atomic(checkpoint_dir / "formal_model_promotion.json", checkpoint_payload)
    model_manifest = dict(worker_result)
    model_manifest.update({
        "status": "FINAL_REFIT_MODEL_FROZEN",
        "formal_model_relative_path": "artifacts/stage_4b4/private/final_refit/models/L1_FINAL_REFIT_600.txt",
        "formal_model_size_bytes": formal_model.stat().st_size,
        "resource_audit_fingerprint": resource_audit["resource_audit_fingerprint"],
        "checkpoint_fingerprint": checkpoint_payload["checkpoint_fingerprint"],
        "training_lineage_fingerprint": lineage["training_lineage_fingerprint"],
        "formal_replay_fingerprint": replay_audit["replay_fingerprint"],
        "atomic_promotion": True,
    })
    model_manifest["model_manifest_fingerprint"] = canonical_fingerprint(model_manifest)
    write_json_atomic(paths["model_manifest"], model_manifest)
    print(json.dumps({
        "status": model_manifest["status"],
        "model_physical_fingerprint": model_manifest["model_physical_fingerprint"],
        "model_semantic_fingerprint": model_manifest["model_semantic_fingerprint"],
        "peak_rss_bytes": peak_rss, "frozen_absolute_rss_cap_bytes": cap,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
