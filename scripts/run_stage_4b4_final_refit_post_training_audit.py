from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ranking_model.final_refit_post_training_recovery import (
    PostTrainingConfig,
    canonical_fingerprint,
    evaluate_raw_data_git_results,
    file_fingerprint,
    file_set_fingerprint,
    post_training_source_fingerprint,
    read_json,
    verify_frozen_post_training_artifacts,
    warning_categories,
    write_json_atomic,
)


def run(root: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=root, text=True, capture_output=True)


def safe_summary(result: subprocess.CompletedProcess[str]) -> str:
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else ("PASS" if result.returncode == 0 else "NO_STDOUT")
    summary = re.sub(r"(?i)\b[a-z]:[\\/][^\s]+", "<absolute-path>", summary)
    summary = re.sub(r"(?i)(?:users|home)[\\/][^\\/\s]+", "user/<redacted>", summary)
    return summary[:500]


def command_audit(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "return_code": result.returncode,
        "summary": safe_summary(result),
        "warning_categories": warning_categories(result.stderr),
    }


def privacy_audit(config: PostTrainingConfig) -> dict[str, Any]:
    root = config.root
    public_artifacts = [
        path
        for path in (root / "artifacts/stage_4b4").glob("final_refit*")
        if path.is_file()
    ]
    code_and_docs = [
        config.path,
        root / "src/ranking_model/final_refit_post_training_recovery.py",
        root / "scripts/run_stage_4b4_final_refit_post_training_recovery.py",
        root / "scripts/run_stage_4b4_final_refit_post_training_audit.py",
        root / "tests/test_stage_4b4_phase4_post_training_recovery.py",
        root / "docs/stages/stage_4b4_locked_final_refit.md",
    ]
    violations: set[str] = set()
    drive_pattern = re.compile(r"(?i)\b[a-z]:[\\/](?![\\/])")
    root_text = str(root)
    user_name = Path.home().name
    for path in public_artifacts:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if drive_pattern.search(text) or root_text in text or (user_name and user_name in text):
            violations.add(path.relative_to(root).as_posix())
    for path in code_and_docs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if root_text in text or (user_name and user_name in text):
            violations.add(path.relative_to(root).as_posix())
    payload: dict[str, Any] = {
        "status": "PASS" if not violations else "FAIL",
        "scanned_public_file_count": len(public_artifacts) + sum(path.exists() for path in code_and_docs),
        "violation_count": len(violations),
        "violation_files": sorted(violations),
        "absolute_paths_recorded": False,
        "usernames_recorded": False,
    }
    payload["audit_fingerprint"] = canonical_fingerprint(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 post-training tests and safety audit")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_4b4_final_refit_post_training_recovery.yaml"),
    )
    args = parser.parse_args()
    config = PostTrainingConfig.load(args.config)
    root = config.root
    prereg = read_json(config.path_for("preregistration"))
    identity = verify_frozen_post_training_artifacts(config, allow_corrected_manifest=True)

    targeted_result = run(root, [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_stage_4b4_phase4.py",
        "tests/test_stage_4b4_phase4_resource_recovery.py",
        "tests/test_stage_4b4_phase4_post_training_recovery.py",
        "-q",
    ])
    full_result = run(root, [sys.executable, "-m", "pytest", "-q"])
    compile_result = run(root, [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    diff_result = run(root, ["git", "diff", "--check"])

    raw_working = run(root, ["git", "diff", "--name-only", "--", "data/raw"])
    raw_staged = run(root, ["git", "diff", "--cached", "--name-only", "--", "data/raw"])
    raw_audit = evaluate_raw_data_git_results(raw_working, raw_staged)
    raw_audit["audit_fingerprint"] = canonical_fingerprint(raw_audit)

    tracked_private_result = run(root, [
        "git",
        "ls-files",
        "--",
        "artifacts/stage_4b4/private/**",
        "artifacts/stage_4b4/tmp/**",
        "artifacts/stage_4b4/checkpoints/**",
    ])
    tracked_private_count = len(
        [line for line in tracked_private_result.stdout.splitlines() if line.strip()]
    )
    probes = [
        "artifacts/stage_4b4/private/final_refit/resource_recovery/failure_evidence/probe.json",
        "artifacts/stage_4b4/tmp/final_refit/resource_recovery/probe.bin",
        "artifacts/stage_4b4/checkpoints/final_refit/resource_recovery/probe.json",
    ]
    ignore_results = [run(root, ["git", "check-ignore", "-q", probe]) for probe in probes]
    private_evidence_paths = [
        config.path_for("original_manifest_private"),
        config.path_for("failed_test_audit_private"),
    ]
    private_evidence_ignored = all(
        run(root, ["git", "check-ignore", "-q", path.relative_to(root).as_posix()]).returncode == 0
        for path in private_evidence_paths
    )

    original_checkpoint_names = [
        "binary_conversion.json",
        "binary_training_worker.json",
        "binary_verification.json",
        "model_verification.json",
        "prediction_replay.json",
    ]
    original_checkpoints = [
        config.path_for("checkpoint_directory") / name for name in original_checkpoint_names
    ]
    preexisting_checkpoints_unchanged = (
        len(original_checkpoints) == prereg["preexisting_checkpoint_count"]
        and file_set_fingerprint(original_checkpoints, root)
        == prereg["preexisting_checkpoint_set_fingerprint"]
    )
    correction = read_json(config.path_for("manifest_correction_audit"))
    sanitized_failure = read_json(config.path_for("sanitized_failure_audit"))
    manifest = read_json(config.path_for("model_manifest"))
    manifest_schema_complete = (
        manifest.get("prediction_replay", {}).get("raw_score_fingerprint")
        == config.raw["expected"]["raw_score_fingerprint"]
        and manifest["prediction_replay"].get("tie_break_fingerprint")
        == config.raw["expected"]["tie_break_fingerprint"]
        and correction.get("status") == "PASS"
    )
    public_privacy = privacy_audit(config)

    tests = {
        "targeted": command_audit(targeted_result),
        "full": command_audit(full_result),
    }
    engineering = {
        "compile": command_audit(compile_result),
        "git_diff_check": command_audit(diff_result),
        "tracked_private": {
            "status": "PASS" if tracked_private_result.returncode == 0 and tracked_private_count == 0 else "FAIL",
            "return_code": tracked_private_result.returncode,
            "tracked_file_count": tracked_private_count,
            "warning_categories": warning_categories(tracked_private_result.stderr),
        },
        "ignore": {
            "status": "PASS" if all(result.returncode == 0 for result in ignore_results) and private_evidence_ignored else "FAIL",
            "probe_count": len(probes),
            "private_failure_evidence_ignored": private_evidence_ignored,
        },
    }
    checks = {
        "manifest_schema_complete": "PASS" if manifest_schema_complete else "FAIL",
        "model_unchanged": "PASS" if identity["model_physical_fingerprint"] == prereg["model_physical_fingerprint"] else "FAIL",
        "binary_unchanged": "PASS" if identity["binary_physical_fingerprint"] == prereg["binary_physical_fingerprint"] else "FAIL",
        "replay_unchanged": "PASS" if identity["replay_audit_physical_fingerprint"] == prereg["replay_audit_physical_fingerprint"] else "FAIL",
        "no_retraining": "PASS" if preexisting_checkpoints_unchanged and not correction["model_retrained_during_fix"] else "FAIL",
        "no_reprediction": "PASS" if not correction["prediction_recomputed_during_fix"] else "FAIL",
        "no_label_or_test_access": "PASS" if not config.raw["scope"]["label_access"] and all(
            value == 0 for key, value in config.raw["scope"].items()
            if (key.startswith("test_") or key.startswith("first7d_")) and key.endswith("_count")
        ) else "FAIL",
        "targeted_tests": tests["targeted"]["status"],
        "full_pytest": tests["full"]["status"],
        "python_compile": engineering["compile"]["status"],
        "git_diff_check": engineering["git_diff_check"]["status"],
        "raw_data_diff": raw_audit["status"],
        "public_privacy": public_privacy["status"],
        "ignore_audit": engineering["ignore"]["status"],
        "tracked_private_audit": engineering["tracked_private"]["status"],
        "original_failure_sanitized_and_preserved": "PASS" if (
            sanitized_failure.get("status") == "SANITIZED_FAILURE_EVIDENCE_PRESERVED"
            and file_fingerprint(config.path_for("failed_test_audit_private"))
            == config.raw["expected"]["failed_test_audit_physical_fingerprint"]
        ) else "FAIL",
        "post_training_source_identity": "PASS" if post_training_source_fingerprint(config) == prereg["source_fingerprint"] else "FAIL",
    }
    payload: dict[str, Any] = {
        "status": "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL",
        "phase": config.raw["phase"],
        "checks": checks,
        "tests": tests,
        "engineering": engineering,
        "raw_data_audit": raw_audit,
        "privacy_audit": public_privacy,
        "artifact_identity": {
            "model_physical_fingerprint": identity["model_physical_fingerprint"],
            "model_semantic_fingerprint": identity["model_semantic_fingerprint"],
            "binary_physical_fingerprint": identity["binary_physical_fingerprint"],
            "binary_semantic_fingerprint": identity["binary_semantic_fingerprint"],
            "replay_audit_physical_fingerprint": identity["replay_audit_physical_fingerprint"],
            "model_retrained_during_fix": False,
            "prediction_recomputed_during_fix": False,
        },
        "access": {
            "validation_labels_reopened": False,
            "first7d_read_count": 0,
            "test_candidate_read_count": 0,
            "test_feature_read_count": 0,
            "test_label_read_count": 0,
            "test_prediction_write_count": 0,
            "test_metric_read_count": 0,
        },
        "git": {
            "commit_created": False,
            "push_executed": False,
        },
    }
    payload["test_security_audit_fingerprint"] = canonical_fingerprint(payload)
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if re.search(r"(?i)\b[a-z]:[\\/]", serialized) or str(root) in serialized or Path.home().name in serialized:
        raise SystemExit("PUBLIC_PRIVACY_GATE_FAILED")
    write_json_atomic(config.path_for("test_security_audit"), payload)
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2, sort_keys=True))
    if raw_audit["status"] == "UNKNOWN":
        raise SystemExit("RAW_DATA_AUDIT_UNRESOLVED")
    if payload["status"] != "PASS":
        raise SystemExit("POST_TRAINING_ENGINEERING_GATE_FAILED")


if __name__ == "__main__":
    main()
