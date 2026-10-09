from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from ranking_model.test_prediction_freeze import (
    FreezeConfig,
    finalize,
    preregister,
    record_resource_audit,
)


def run_monitored(config_path: Path, phase: str) -> None:
    config = FreezeConfig.load(config_path)
    cap = int(config.raw["resource"]["absolute_rss_cap_bytes"])
    log_dir = config.private_dir / "resource_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{phase}.stdout.log"
    stderr_path = log_dir / f"{phase}.stderr.log"
    command = [
        sys.executable,
        str(config.root / "scripts/stage_4b4_test_prediction_worker.py"),
        "--config", str(config_path), "--phase", phase,
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    started = time.perf_counter()
    peak = 0
    exceeded = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, cwd=config.root, stdout=stdout, stderr=stderr, creationflags=creationflags
        )
        root = psutil.Process(process.pid)
        while process.poll() is None:
            rss = 0
            try:
                members = [root, *root.children(recursive=True)]
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            peak = max(peak, rss)
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
    }
    record_resource_audit(config, phase, payload)
    if payload["status"] != "PASS":
        raise RuntimeError(
            "BLOCKED_BY_RESOURCE" if exceeded or payload["oom_observed"] else "PRE_LABEL_IMPLEMENTATION_FAILURE"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_test_prediction_freeze.yaml"))
    parser.add_argument(
        "--phase",
        choices=("preregister", "features", "predictions_formal", "predictions_replay", "finalize", "all"),
        required=True,
    )
    args = parser.parse_args()
    if args.phase in {"preregister", "all"}:
        preregister(args.config)
    if args.phase in {"features", "all"}:
        run_monitored(args.config, "features")
    if args.phase in {"predictions_formal", "all"}:
        run_monitored(args.config, "predictions_formal")
    if args.phase in {"predictions_replay", "all"}:
        run_monitored(args.config, "predictions_replay")
    if args.phase in {"finalize", "all"}:
        result = finalize(args.config)
        print(json.dumps(result["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
