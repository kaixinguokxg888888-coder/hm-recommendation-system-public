from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import psutil

from data_foundation.manifest import write_csv
from ranking_model.resource_contract import consume_frozen_rss_contract
from ranking_model.validation_runner import json_atomic
from reranking.config import Config
from reranking.runner import Runner, read_json, rows_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_5_reranking.yaml"))
    parser.add_argument("--phase", choices=("metadata", "tests", "sanity", "validation", "evaluate", "aggregate", "freeze"), required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    config = Config.load(args.config)
    if args.worker:
        runner = Runner(config)
        result = runner.audit_tests(full=False) if args.phase == "tests" else getattr(runner, args.phase)()
        print(json.dumps({"phase": args.phase, "status": "PASS", "selected_variant": result.get("selected_variant")}), flush=True)
        return 0
    registration = read_json(config.public / "preregistration.json")
    contract = consume_frozen_rss_contract(registration, runtime_available_memory_bytes=psutil.virtual_memory().available)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen([sys.executable, "-B", str(Path(__file__)), "--config", str(args.config), "--phase", args.phase, "--worker"], creationflags=flags)
    started = time.perf_counter(); peak = 0; exceeded = False; temp_peak = 0
    while process.poll() is None:
        try:
            root = psutil.Process(process.pid)
            members = [root, *root.children(recursive=True)]
            rss = sum(p.memory_info().rss for p in members if p.is_running())
            peak = max(peak, rss)
            if rss > contract.frozen_absolute_rss_cap_bytes:
                exceeded = True
                for p in reversed(members):
                    try:
                        p.terminate()
                    except psutil.NoSuchProcess:
                        pass
                break
            temp_peak = max(temp_peak, sum(p.stat().st_size for p in config.path("tmp").rglob("*") if p.is_file()))
        except (psutil.NoSuchProcess, psutil.AccessDenied, FileNotFoundError):
            pass
        time.sleep(0.2)
    code = process.wait()
    record = {"phase": args.phase, "status": "PASS" if code == 0 and not exceeded else "FAIL",
              "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": peak,
              "frozen_cap_bytes": contract.frozen_absolute_rss_cap_bytes, "cap_exceeded": exceeded,
              "temp_peak_bytes": temp_peak, "exit_code": code}
    if args.phase == "freeze":
        # lock 创建后不改其已绑定的 resource_audit；最终 worker 单独留证据。
        json_atomic(config.public / "freeze_resource_audit.json", record)
    else:
        path = config.public / "resource_audit.csv"
        previous = rows_csv(path) if path.exists() else []
        write_csv(path, [*previous, record], record.keys())
    print(json.dumps(record), flush=True)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "STOP", "error_category": type(exc).__name__, "detail": str(exc) if isinstance(exc, ValueError) else "inspect private execution evidence"}), flush=True)
        raise SystemExit(1)
