from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from data_foundation.manifest import write_csv
from ranking_model.first7d_diagnostic import FIRST7D_VERSIONS
from ranking_model.first7d_runner import First7dDiagnosticRunner
from ranking_model.validation_selection import canonical_fingerprint, file_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_4b4_first7d_diagnostic.yaml")
    parser.add_argument("--targeted-result", required=True)
    parser.add_argument("--full-result", required=True)
    parser.add_argument("--compile-result", required=True)
    parser.add_argument("--diff-check-result", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = First7dDiagnosticRunner(root, root / args.config)
    public = runner.public
    lock_path = runner.paths["first7d_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    prereg = json.loads((public / "first7d_diagnostic_preregistration.json").read_text(encoding="utf-8"))
    access = json.loads((public / "first7d_diagnostic_access_audit.json").read_text(encoding="utf-8"))

    def run(*command: str) -> str:
        return subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    checks: list[tuple[str, bool, str]] = []
    checks.append(("A01_LOCK_SELF_IDENTITY",
                   runner._self_fingerprint(lock, "lock_fingerprint") == lock.get("lock_fingerprint"),
                   lock.get("lock_fingerprint", "")))
    checks.append(("A02_PREREG_SELF_IDENTITY",
                   runner._self_fingerprint(prereg, "preregistration_fingerprint") == prereg.get("preregistration_fingerprint"),
                   prereg.get("preregistration_fingerprint", "")))
    checks.append(("A03_SOURCE_FROZEN", runner._source_fingerprint() == prereg.get("source_fingerprint"),
                   prereg.get("source_fingerprint", "")))
    gates = list(__import__("csv").DictReader(
        (public / "first7d_diagnostic_contract_gate.csv").open(encoding="utf-8-sig", newline="")
    ))
    checks.append(("A04_P3_GATES", len(gates) == 13 and all(row["status"] == "PASS" for row in gates),
                   f"P3G1-P3G13={sum(row['status'] == 'PASS' for row in gates)}/13"))
    checkpoints = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in runner.checkpoints.glob("*.json")
        if path.name not in {"preregistration.json", "first7d_materialization.json"}
    ]
    checks.append(("A05_CHECKPOINT_COMPLETENESS",
                   len(checkpoints) == 160 and all(row.get("status") == "SUCCESS" for row in checkpoints),
                   f"{sum(row.get('status') == 'SUCCESS' for row in checkpoints)}/160"))
    checks.append(("A06_ONLY_TWO_VERSIONS", set(lock["metric_fingerprints"]) == set(FIRST7D_VERSIONS),
                   ",".join(sorted(lock["metric_fingerprints"]))))
    checks.append(("A07_EXACT_WINDOW",
                   access.get("label_window") == "2020-08-26..2020-09-01"
                   and lock["first7d_ground_truth"]["label_window"] == "2020-08-26..2020-09-01",
                   access.get("label_window", "")))
    checks.append(("A08_ACCESS_COUNTS",
                   access.get("first7d_read_count") == 1
                   and lock.get("first7d_read_count") == 1
                   and all(lock.get(key) == 0 for key in (
                       "test_candidate_read_count", "test_feature_read_count",
                       "test_label_read_count", "test_metric_read_count")),
                   "first7d=1;test=0"))
    checks.append(("A09_WINNER_IMMUTABLE",
                   lock.get("winner") == "L1_600" and lock.get("winner_feature_set") == "F1"
                   and lock.get("winner_iteration") == 600 and lock.get("winner_changed") is False,
                   "L1_600/F1/600;changed=false"))
    checks.append(("A10_B0_L1_PREDICTION_IDENTITY",
                   lock.get("prediction_global_merkle_fingerprints") == runner.raw["expected"]["prediction_global_merkle_fingerprints"],
                   "B0,L1_600 Merkle exact"))
    replay = list(__import__("csv").DictReader(
        (public / "first7d_diagnostic_replay_audit.csv").open(encoding="utf-8-sig", newline="")
    ))
    checks.append(("A11_METRIC_REPLAY", len(replay) == 2 and all(row["status"] == "PASS" for row in replay),
                   f"{sum(row['status'] == 'PASS' for row in replay)}/2"))
    boot_replay = list(__import__("csv").DictReader(
        (public / "first7d_diagnostic_bootstrap_replay.csv").open(encoding="utf-8-sig", newline="")
    ))
    checks.append(("A12_BOOTSTRAP_REPLAY", bool(boot_replay) and all(row["status"] == "PASS" for row in boot_replay),
                   f"{sum(row['status'] == 'PASS' for row in boot_replay)}/{len(boot_replay)}"))
    resource = json.loads((public / "first7d_diagnostic_resource_audit.json").read_text(encoding="utf-8"))
    checks.append(("A13_RESOURCE_BOUND",
                   resource["threads"] == 1 and resource["simultaneous_prediction_versions"] == 1
                   and resource["peak_rss_bytes"] <= resource["rss_hard_limit_bytes"],
                   f"peak={resource['peak_rss_bytes']};limit={resource['rss_hard_limit_bytes']}"))
    public_files = [path for path in public.rglob("*") if path.is_file()
                    and "private" not in path.parts and "tmp" not in path.parts and "checkpoints" not in path.parts]
    unsafe = []
    patterns = [re.compile(r"[A-Za-z]:\\"), re.compile(r"/Users/|/home/", re.IGNORECASE),
                re.compile(r"LXR15892896597B0216", re.IGNORECASE)]
    for path in public_files:
        if path.suffix.lower() not in {".json", ".csv", ".md", ".yaml", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        if any(pattern.search(text) for pattern in patterns):
            unsafe.append(path.relative_to(root).as_posix())
    checks.append(("A14_PRIVACY_PATH_SCAN", not unsafe, f"hits={len(unsafe)}"))
    ignored = []
    for path in (runner.private, runner.temp, runner.checkpoints):
        completed = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=root)
        ignored.append(completed.returncode == 0)
    checks.append(("A15_IGNORE_AUDIT", all(ignored), "private/tmp/checkpoints ignored"))
    tracked = run("git", "ls-files", "artifacts/stage_4b4/private", "artifacts/stage_4b4/tmp", "artifacts/stage_4b4/checkpoints")
    checks.append(("A16_TRACKED_PRIVATE", not tracked, "tracked=0" if not tracked else "tracked>0"))
    raw_diff = run("git", "diff", "--", "data/raw")
    checks.append(("A17_RAW_DATA_DIFF", not raw_diff, "empty" if not raw_diff else "nonempty"))
    checks.append(("A18_NO_DOWNSTREAM_EXECUTION",
                   all(lock.get(key) in (0, False) for key in runner.raw["forbidden"]), "all_zero_or_false"))
    checks.append(("A19_DIRECTION_RULE_BOUNDARY",
                   lock["direction_status"] in {"ROBUSTNESS_DIRECTION_PASS", "VALIDATION_ROBUSTNESS_NOT_MET"},
                   lock["direction_status"]))
    checks.append(("A20_CONFIDENCE_RULE_BOUNDARY",
                   lock["confidence_status"] in {"POSITIVE_AND_SIGNIFICANT", "NONNEGATIVE_BUT_INCONCLUSIVE",
                                                  "SIGNIFICANT_NEGATIVE", "NEGATIVE_BUT_INCONCLUSIVE"},
                   lock["confidence_status"]))
    checks.append(("A21_TARGETED_TESTS", "passed" in args.targeted_result.lower(), args.targeted_result))
    checks.append(("A22_FULL_TESTS", "passed" in args.full_result.lower(), args.full_result))
    checks.append(("A23_COMPILE_AUDIT", args.compile_result == "PASS", args.compile_result))
    checks.append(("A24_GIT_DIFF_CHECK", args.diff_check_result == "PASS", args.diff_check_result))

    rows = [{"check_id": check, "status": "PASS" if passed else "FAIL", "observed": observed}
            for check, passed, observed in checks]
    write_csv(public / "first7d_diagnostic_final_audit.csv", rows, list(rows[0]))
    if any(row["status"] != "PASS" for row in rows):
        return 1
    payload = {
        "stage": "4B-4", "phase": runner.raw["phase"], "status": "FINAL_AUDIT_PASS",
        "lock_fingerprint": lock["lock_fingerprint"], "audit_pass_count": len(rows),
        "audit_semantic_fingerprint": canonical_fingerprint(rows),
        "audit_physical_fingerprint": file_fingerprint(public / "first7d_diagnostic_final_audit.csv"),
    }
    (public / "first7d_diagnostic_final_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
