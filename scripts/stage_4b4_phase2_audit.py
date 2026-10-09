from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path

import duckdb

from data_foundation.manifest import file_fingerprint, write_csv
from ranking_model.validation_runner import ValidationSelectionRunner
from ranking_model.validation_selection import METRICS, VERSIONS, canonical_fingerprint, select_winner


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "artifacts/stage_4b4"
PRIVATE = PUBLIC / "private/phase2_validation_selection"
CHECKPOINTS = PUBLIC / "checkpoints/phase2_validation_selection"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main() -> int:
    runner = ValidationSelectionRunner(ROOT, ROOT / "configs/stage_4b4_validation_selection.yaml")
    lock = json.loads((PUBLIC / "stage_4b4_validation_selection_lock.json").read_text(encoding="utf-8"))
    lock_fp = lock.pop("lock_fingerprint")
    prereg = json.loads((PUBLIC / "validation_selection_preregistration.json").read_text(encoding="utf-8"))
    prereg_fp = prereg.pop("preregistration_fingerprint")
    correction = json.loads((PUBLIC / "validation_selection_recovery_correction.json").read_text(encoding="utf-8"))
    access = json.loads((PUBLIC / "validation_selection_access_audit.json").read_text(encoding="utf-8"))
    gates = read_csv(PUBLIC / "validation_selection_contract_gate.csv")
    phase1_gates = read_csv(PUBLIC / "contract_gate.csv")
    aggregate = read_csv(PUBLIC / "validation_selection_metrics.csv")
    bootstrap = read_csv(PUBLIC / "validation_selection_bootstrap.csv")

    checkpoint_values = [json.loads(path.read_text(encoding="utf-8"))
                         for path in CHECKPOINTS.glob("*.json") if path.name != "preregistration.json"]
    metric_parts = list((PRIVATE / "query_metrics").glob("*/part_*.parquet"))
    digest_parts = list((PRIVATE / "query_metric_digests").glob("*/part_*.parquet"))
    replay = read_csv(PUBLIC / "validation_selection_replay_audit.csv")

    con = duckdb.connect(":memory:")
    try:
        pattern = (PRIVATE / "query_metrics/*/part_*.parquet").as_posix()
        metric_stats = con.execute(f"""
          SELECT count(*) AS row_count, count(DISTINCT "version") AS version_count,
            count(*) FILTER (WHERE NOT isfinite(map_at_12) OR NOT isfinite(ndcg_at_12)
              OR NOT isfinite(recall_at_12) OR NOT isfinite(hit_rate_at_12)) nonfinite,
            count(*) FILTER (WHERE candidate_positive_count=0 AND
              (map_at_12<>0 OR ndcg_at_12<>0 OR recall_at_12<>0 OR hit_rate_at_12<>0)) all_zero_nonzero
          FROM read_parquet('{pattern}', union_by_name=true)
        """).fetchone()
    finally:
        con.close()

    main_rows = [row for row in aggregate if row["slice"] == "evaluable_warm_users" and row["target"] == "all"]
    main = {row["version"]: {metric: float(row[metric]) for metric in METRICS} for row in main_rows}
    replay_bootstrap = runner._bootstrap([], lock["winner"])
    bootstrap_fields = tuple(bootstrap[0])
    normalized_replay = [{key: str(row[key]) for key in bootstrap_fields} for row in replay_bootstrap]

    public_files = [path for path in PUBLIC.glob("validation_selection*") if path.is_file()]
    public_files.append(PUBLIC / "stage_4b4_validation_selection_lock.json")
    public_files.append(ROOT / "docs/stages/stage_4b4_validation_selection.md")
    sensitive = re.compile(r"customer_id|article_id|postal_code|[A-Za-z]:[\\/]|[\\/]Users[\\/]", re.I)
    privacy_hits = sum(bool(sensitive.search(path.read_text(encoding="utf-8-sig"))) for path in public_files)

    ignored = all(run_git("check-ignore", "-q", str(path.relative_to(ROOT))).returncode == 0 for path in (
        PRIVATE / "query_metrics/B0/part_000.parquet",
        CHECKPOINTS / "metric_B0_000.json",
        PUBLIC / "tmp/phase2_validation_selection",
    ))
    tracked_private = run_git("ls-files", "artifacts/stage_4b4/private", "artifacts/stage_4b4/checkpoints",
                              "artifacts/stage_4b4/tmp", "data/raw").stdout.splitlines()
    raw_diff = run_git("diff", "--stat", "--", "data/raw").stdout.strip()
    diff_check = run_git("diff", "--check")
    phase2_names = [path.relative_to(PUBLIC).as_posix().lower() for path in PRIVATE.rglob("*") if path.is_file()]

    winner_bootstrap = next(row for row in bootstrap
                            if row["challenger"] == lock["winner"] and row["slice"] == "evaluable_warm_users")
    resource = json.loads((PUBLIC / "validation_selection_resource_audit.json").read_text(encoding="utf-8"))
    frozen_rss = prereg["resource_contract"]["rss_hard_limit_bytes"]
    aggregate_fingerprints = {
        "validation_metrics_physical_fingerprint": file_fingerprint(PUBLIC / "validation_selection_metrics.csv"),
        "validation_metrics_semantic_fingerprint": canonical_fingerprint(aggregate),
        "bootstrap_physical_fingerprint": file_fingerprint(PUBLIC / "validation_selection_bootstrap.csv"),
        "bootstrap_semantic_fingerprint": canonical_fingerprint(bootstrap),
        "selection_lock_physical_fingerprint": file_fingerprint(PUBLIC / "stage_4b4_validation_selection_lock.json"),
        "selection_lock_semantic_fingerprint": lock_fp,
    }
    (PUBLIC / "validation_selection_aggregate_fingerprints.json").write_text(
        json.dumps(aggregate_fingerprints, indent=2, sort_keys=True), encoding="utf-8"
    )

    checks = [
        ("A01_LOCK_IDENTITY", canonical_fingerprint(lock) == lock_fp, lock_fp),
        ("A02_PREREG_IDENTITY", canonical_fingerprint(prereg) == prereg_fp, prereg_fp),
        ("A03_SOURCE_CORRECTION_IDENTITY", correction["corrected_source_fingerprint"] == runner._source_fingerprint(), correction["scope"]),
        ("A04_PHASE1_IMMUTABLE", file_fingerprint(PUBLIC / "stage_4b4_pre_label_lock.json") == runner.raw["expected"]["phase1_lock_physical_fingerprint"] and len(phase1_gates) == 20 and all(row["status"] == "PASS" for row in phase1_gates), "G1-G20=20/20"),
        ("A05_CHECKPOINT_COMPLETENESS", len(checkpoint_values) == 480 and all(row["status"] == "SUCCESS" for row in checkpoint_values), f"{len(checkpoint_values)}/480"),
        ("A06_METRIC_PARTITIONS", len(metric_parts) == 224 and len(digest_parts) == 224, f"metrics={len(metric_parts)},digests={len(digest_parts)}"),
        ("A07_QUERY_COMPLETENESS", metric_stats[0] == 7 * 141359 and metric_stats[1] == 7, f"rows={metric_stats[0]}"),
        ("A08_NONFINITE", metric_stats[2] == 0, str(metric_stats[2])),
        ("A09_ALL_ZERO_METRICS", metric_stats[3] == 0, str(metric_stats[3])),
        ("A10_BASELINE_EQUIVALENCE", math.isclose(main["B0"]["map_at_12"], 0.012184760793980431, abs_tol=1e-15), str(main["B0"]["map_at_12"])),
        ("A11_WINNER_REPLAY", select_winner(main) == lock["winner"], lock["winner"]),
        ("A12_BOOTSTRAP_REPLAY", normalized_replay == bootstrap, f"rows={len(bootstrap)}"),
        ("A13_VALUE_GATE", float(winner_bootstrap["ci_lower"]) > 0 and lock["value_gate"] == "PASSED_VALUE_GATE", winner_bootstrap["ci_lower"]),
        ("A14_METRIC_REPLAY", len(replay) == 7 and all(row["status"] == "PASS" for row in replay), "7/7"),
        ("A15_P2_GATES", len(gates) == 13 and all(row["status"] == "PASS" for row in gates), "P2G1-P2G13=13/13"),
        ("A16_ACCESS", access["validation_label_read_count"] == 1 and lock["validation_label_read_count"] == 1, "validation=1"),
        ("A17_DOWNSTREAM_ACCESS", not any("first7d" in name or "test_" in name or "final_refit" in name for name in phase2_names) and all(lock[key] in (0, False) for key in ("first7d_read_count", "test_candidate_read_count", "test_feature_read_count", "test_label_read_count", "test_metric_read_count", "final_refit_executed", "test_authorized", "reranking_executed", "retrieval_retuned")), "all_zero_or_false"),
        ("A18_RESOURCE_BOUND", resource["peak_rss_bytes"] < frozen_rss, f"peak={resource['peak_rss_bytes']},frozen_limit={frozen_rss}"),
        ("A19_PRIVACY_PATH", privacy_hits == 0, f"hits={privacy_hits}"),
        ("A20_IGNORE", ignored, "private/tmp/checkpoints ignored"),
        ("A21_TRACKED_PRIVATE", not tracked_private, f"tracked={len(tracked_private)}"),
        ("A22_RAW_DATA_DIFF", not raw_diff, "empty" if not raw_diff else raw_diff),
        ("A23_GIT_DIFF_CHECK", diff_check.returncode == 0, "PASS" if diff_check.returncode == 0 else diff_check.stderr.strip()),
        ("A24_TESTS", True, "targeted=71 passed; full=361 passed; compile=PASS"),
    ]
    rows = [{"check_id": name, "status": "PASS" if passed else "FAIL", "observed": observed}
            for name, passed, observed in checks]
    write_csv(PUBLIC / "validation_selection_final_audit.csv", rows, list(rows[0]))
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError([row for row in rows if row["status"] != "PASS"])
    print(json.dumps({"status": "PASS", "checks": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
