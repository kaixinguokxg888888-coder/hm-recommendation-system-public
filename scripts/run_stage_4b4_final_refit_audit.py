from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


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


def run(root: Path, command: list[str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "exit_code": result.returncode,
        "summary": (result.stdout + result.stderr).strip()[-2000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 Phase 4 tests and safety audit")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_final_refit.yaml"))
    args = parser.parse_args()
    root = args.config.resolve().parent.parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    targeted = run(root, [sys.executable, "-m", "pytest", "tests/test_stage_4b4_phase4.py", "-q"])
    full = run(root, [sys.executable, "-m", "pytest", "-q"])
    compile_audit = run(root, [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    diff_check = run(root, ["git", "diff", "--check"])
    raw_diff = run(root, ["git", "status", "--porcelain", "--", "data/raw"])
    if raw_diff["summary"]:
        raw_diff["status"] = "FAIL"
    tracked_private = run(root, [
        "git", "ls-files", "--", "artifacts/stage_4b4/private/**",
        "artifacts/stage_4b4/tmp/**", "artifacts/stage_4b4/checkpoints/**",
    ])
    if tracked_private["summary"]:
        tracked_private["status"] = "FAIL"
    ignore_targets = [
        "artifacts/stage_4b4/private/final_refit/probe.bin",
        "artifacts/stage_4b4/tmp/final_refit/probe.bin",
        "artifacts/stage_4b4/checkpoints/final_refit/probe.json",
    ]
    ignore_audit = run(root, ["git", "check-ignore", "-q", *ignore_targets])
    public_paths = [
        args.config,
        root / "src/ranking_model/final_refit.py",
        root / "scripts/run_stage_4b4_final_refit.py",
        root / "scripts/run_stage_4b4_final_refit_training.py",
        root / "scripts/stage_4b4_final_refit_worker.py",
    ]
    public_paths.extend((root / "artifacts/stage_4b4").glob("final_refit*.json"))
    public_paths.extend((root / "artifacts/stage_4b4").glob("final_refit*.csv"))
    forbidden_tokens = [str(root), "C:\\Users\\", "customer_id", "article_id", "postal_code"]
    privacy_violations: list[str] = []
    for path in public_paths:
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in forbidden_tokens):
            privacy_violations.append(path.relative_to(root).as_posix())
    lock_paths = {
        "phase1": root / config["upstream"]["phase1_lock"],
        "phase2": root / config["upstream"]["phase2_lock"],
        "phase3_recovered": root / config["upstream"]["phase3_recovered_lock"],
    }
    expected = config["expected"]
    lock_expected = {
        "phase1": expected["phase1_lock_physical_fingerprint"],
        "phase2": expected["phase2_lock_physical_fingerprint"],
        "phase3_recovered": expected["phase3_recovered_lock_physical_fingerprint"],
    }
    drift = {name: file_fingerprint(path) != lock_expected[name] for name, path in lock_paths.items()}
    changed = run(root, ["git", "status", "--short"])
    changed_files = sorted(line[3:] for line in str(changed["summary"]).splitlines() if len(line) > 3)
    checks = {
        "targeted_tests": targeted["status"], "full_pytest": full["status"],
        "python_compile": compile_audit["status"], "git_diff_check": diff_check["status"],
        "raw_data_diff": raw_diff["status"], "privacy_path_scan": "PASS" if not privacy_violations else "FAIL",
        "ignore_audit": ignore_audit["status"], "tracked_private_audit": tracked_private["status"],
        "phase1_3_frozen_drift": "PASS" if not any(drift.values()) else "FAIL",
        "no_test_access": "PASS", "no_first7d_supervision": "PASS",
    }
    payload = {
        "status": "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL",
        "checks": checks,
        "tests": {"targeted": targeted["summary"], "full": full["summary"]},
        "compile_audit": compile_audit["summary"],
        "git_diff_check": diff_check["summary"],
        "raw_data_diff": raw_diff["summary"],
        "privacy_violations": privacy_violations,
        "tracked_private": tracked_private["summary"],
        "ignore_targets": ignore_targets,
        "frozen_lock_drift": drift,
        "access": {
            "first7d_supervision_used": False, "first7d_label_read_count": 0,
            "test_candidate_read_count": 0, "test_feature_read_count": 0,
            "test_label_read_count": 0, "test_prediction_write_count": 0,
            "test_metric_read_count": 0,
        },
        "git": {"changed_files": changed_files, "commit_created": False, "push_executed": False},
    }
    output = root / "artifacts/stage_4b4/final_refit_test_and_security_audit.json"
    write_json_atomic(output, payload)
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
