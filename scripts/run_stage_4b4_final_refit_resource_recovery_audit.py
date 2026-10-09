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
        "summary": (result.stdout + result.stderr).strip()[-3000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-refit resource recovery test and safety audit")
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/stage_4b4_final_refit_resource_recovery.yaml"),
    )
    args = parser.parse_args()
    root = args.config.resolve().parent.parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    targeted = run(root, [
        sys.executable, "-m", "pytest", "tests/test_stage_4b4_phase4.py",
        "tests/test_stage_4b4_phase4_resource_recovery.py", "-q",
    ])
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
    probes = [
        "artifacts/stage_4b4/private/final_refit/resource_recovery/probe.bin",
        "artifacts/stage_4b4/tmp/final_refit/resource_recovery/probe.bin",
        "artifacts/stage_4b4/checkpoints/final_refit/resource_recovery/probe.json",
    ]
    ignore_checks = [run(root, ["git", "check-ignore", "-q", probe]) for probe in probes]
    ignore_status = "PASS" if all(value["status"] == "PASS" for value in ignore_checks) else "FAIL"
    original_path = root / config["original_failure"]["resource_audit"]
    original_preserved = (
        file_fingerprint(original_path) == config["original_failure"]["resource_audit_physical_fingerprint"]
        and json.loads(original_path.read_text(encoding="utf-8"))["status"] == "BLOCKED_BY_RESOURCE"
    )
    public_paths = [
        args.config,
        root / "src/ranking_model/final_refit_resource_recovery.py",
        root / "scripts/run_stage_4b4_final_refit_resource_recovery.py",
        root / "scripts/stage_4b4_binary_conversion_worker.py",
        root / "scripts/stage_4b4_binary_verification_worker.py",
        root / "scripts/stage_4b4_binary_training_worker.py",
        root / "scripts/stage_4b4_final_model_replay_worker.py",
    ]
    public_paths.extend((root / "artifacts/stage_4b4").glob("final_refit*.json"))
    public_paths.extend((root / "artifacts/stage_4b4").glob("final_refit*.csv"))
    forbidden = [str(root), "C:\\Users\\"]
    privacy_violations: list[str] = []
    for path in public_paths:
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(token in text for token in forbidden):
                privacy_violations.append(path.relative_to(root).as_posix())
    status_result = run(root, ["git", "status", "--short"])
    changed_files = sorted(line[3:] for line in str(status_result["summary"]).splitlines() if len(line) > 3)
    scope = config["recovery_scope"]
    no_access = (
        not scope["validation_labels_reopened"] and not scope["first7d_labels_read"]
        and not scope["test_access_allowed"]
        and all(value == 0 for key, value in scope.items() if key.startswith("test_") and key.endswith("_count"))
    )
    checks = {
        "targeted_tests": targeted["status"], "full_pytest": full["status"],
        "python_compile": compile_audit["status"], "git_diff_check": diff_check["status"],
        "raw_data_diff": raw_diff["status"], "privacy_path_scan": "PASS" if not privacy_violations else "FAIL",
        "ignore_audit": ignore_status, "tracked_private_audit": tracked_private["status"],
        "original_failure_preserved": "PASS" if original_preserved else "FAIL",
        "no_validation_first7d_test_access": "PASS" if no_access else "FAIL",
    }
    payload = {
        "status": "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL",
        "checks": checks, "tests": {"targeted": targeted["summary"], "full": full["summary"]},
        "compile_audit": compile_audit["summary"], "git_diff_check": diff_check["summary"],
        "raw_data_diff": raw_diff["summary"], "privacy_violations": privacy_violations,
        "tracked_private": tracked_private["summary"], "ignore_probes": probes,
        "original_failure_physical_fingerprint": file_fingerprint(original_path),
        "access": {
            "validation_labels_reopened": False, "first7d_labels_read": False,
            "test_candidate_read_count": 0, "test_feature_read_count": 0,
            "test_label_read_count": 0, "test_prediction_write_count": 0, "test_metric_read_count": 0,
        },
        "git": {"changed_files": changed_files, "commit_created": False, "push_executed": False},
    }
    output = root / config["paths"]["test_security_audit"]
    write_json_atomic(output, payload)
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
