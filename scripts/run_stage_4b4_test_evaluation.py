from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

from ranking_model.test_evaluation import (
    PostLabelFailure,
    PreLabelConflict,
    TestEvaluationConfig,
    finalize_prelabel_gate,
    finalize_test_evaluation,
    preregister,
    read_json,
    record_resource_audit,
    run_prelabel_engineering_audit,
    self_fingerprint,
)


def _run_monitored(config: TestEvaluationConfig, phase: str) -> dict[str, Any]:
    prereg = read_json(config.public_path("preregistration"))
    cap = int(prereg["resource_contract"]["absolute_rss_cap_bytes"])
    log_dir = config.private_dir / "worker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{phase}.stdout.log"
    stderr_path = log_dir / f"{phase}.stderr.log"
    command = [
        sys.executable, str(config.root / "scripts/stage_4b4_test_evaluation_worker.py"),
        "--config", str(config.path), "--phase", phase,
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.perf_counter()
    peak = 0
    exceeded = False
    runtime_available_min: int | None = None
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, cwd=config.root, stdout=stdout, stderr=stderr, creationflags=creationflags
        )
        root = psutil.Process(process.pid)
        while process.poll() is None:
            rss = 0
            try:
                for member in (root, *root.children(recursive=True)):
                    try:
                        rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            peak = max(peak, rss)
            observed = int(psutil.virtual_memory().available)
            runtime_available_min = observed if runtime_available_min is None else min(runtime_available_min, observed)
            if rss > cap:
                exceeded = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            time.sleep(0.2)
        return_code = process.wait()
    payload = {
        "status": "PASS" if return_code == 0 and not exceeded else "FAIL",
        "frozen_absolute_rss_cap_bytes": cap,
        "peak_process_tree_rss_bytes": peak,
        "cap_exceeded": exceeded,
        "oom_observed": return_code in (-9, -1073741571, 137),
        "exit_code": return_code,
        "wall_time_seconds": time.perf_counter() - started,
        "runtime_available_memory_observation_bytes": runtime_available_min,
    }
    record_resource_audit(config, phase, payload)
    if payload["status"] != "PASS":
        raise PostLabelFailure(
            "BLOCKED_BY_RESOURCE" if exceeded or payload["oom_observed"] else "POST_TEST_LABEL_IMPLEMENTATION_FAILURE"
        )
    return payload


def _execute_once(config: TestEvaluationConfig) -> dict[str, Any]:
    for phase in ("ground_truth", "evaluation", "bootstrap"):
        _run_monitored(config, phase)
    result = finalize_test_evaluation(config)
    lock = read_json(config.public_path("lock"))
    if self_fingerprint(lock, "lock_fingerprint") != lock["lock_fingerprint"]:
        raise PostLabelFailure("independent final lock self-fingerprint verification failed")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_test_evaluation.yaml"))
    parser.add_argument(
        "--phase", required=True,
        choices=("preregister", "prelabel-audit", "prelabel-finalize", "execute"),
    )
    args = parser.parse_args()
    config = TestEvaluationConfig.load(args.config)
    try:
        if args.phase == "preregister":
            result = preregister(config)
        elif args.phase == "prelabel-audit":
            result = run_prelabel_engineering_audit(config)
        elif args.phase == "prelabel-finalize":
            result = {"gate": finalize_prelabel_gate(config)}
        else:
            result = _execute_once(config)
    except PreLabelConflict as exc:
        print(json.dumps({
            "status": "STOP", "reason": "REPORT_CONFLICT",
            "raw_test_label_source_read_count": 0, "detail": str(exc),
        }, ensure_ascii=False))
        return 2
    except PostLabelFailure as exc:
        state_path = config.private_dir / "access_state.json"
        state = read_json(state_path) if state_path.exists() else {}
        print(json.dumps({
            "status": "STOP", "reason": "POST_TEST_LABEL_IMPLEMENTATION_FAILURE",
            "raw_test_label_source_read_count": state.get("raw_test_label_source_read_count", 0),
            "canonical_ground_truth_frozen": state.get("canonical_ground_truth_frozen", False),
            "completed_partitions": state.get("completed_partitions", []),
            "formal_metrics_generated": state.get("formal_metrics_generated", False),
            "test_labels_reopened": state.get("test_labels_reopened", False),
            "detail": str(exc),
        }, ensure_ascii=False))
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
