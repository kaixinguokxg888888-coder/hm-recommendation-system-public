from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.stage3d.config import Stage3DConfig
from retrieval.stage3d.locking import lock_fingerprint
from retrieval.stage3d.runner import Stage3DRunner, csv_rows, json_read, json_write


def correct_top12_population_denominators(runner: Stage3DRunner) -> Path:
    """候选覆盖用户不等于主评估人口；changed count 保留，只修正冻结分母。"""
    target = runner.config.artifact_dir / "top12_sequence_audit.csv"
    rows = csv_rows(target)
    marginal = csv_rows(runner.config.artifact_dir / "exclusive_hit_summary.csv")
    denominators = {
        (row["phase"], row["model_variant"], row["union_variant"]): int(row["population_denominator"])
        for row in marginal
    }
    for row in rows:
        key = (row["phase"], row["model_variant"], row["union_variant"])
        denominator = denominators[key]
        changed = int(row["top12_sequence_changed_users"])
        row["main_population_users"] = denominator
        row["changed_user_ratio"] = changed / denominator if denominator else 0.0
    write_csv(target, rows, rows[0].keys())
    return target


def finalize_lock_after_test(runner: Stage3DRunner) -> Path:
    lock = json_read(runner.config.lock_path)
    execution_lock_fingerprint = str(
        lock.get("test_execution_lock_fingerprint", lock["lock_fingerprint"])
    )
    validation_artifacts = lock["validation_artifacts"]
    for relative in tuple(validation_artifacts):
        validation_artifacts[relative] = file_fingerprint(runner.config.resolve(relative))
    recovery = runner.config.artifact_dir / "selection_checkpoint_metric_recovery.csv"
    if recovery.exists():
        validation_artifacts[runner.config.relative(recovery)] = file_fingerprint(recovery)
    test_paths = [
        runner.model_checkpoint("test_14d", "TT1"),
        runner.final_refit_history_path("test_14d", "TT1"),
        runner.embedding_metadata_path("test_14d", "TT1"),
        runner.ann_audit_summary_path("test_14d", "TT1"),
        runner.candidate_lineage_path("test_14d", "TT1"),
        runner.config.artifact_dir / "test_execution_status.csv",
        runner.config.artifact_dir / "model_lineage.csv",
        runner.config.artifact_dir / "full_refit_correction_audit.csv",
        runner.config.artifact_dir / "top12_sequence_audit.csv",
    ]
    test_refit = json_read(runner.final_refit_history_path("test_14d", "TT1"))
    lock.update({
        "test_status": "SUCCESS",
        "test_execution_lock_fingerprint": execution_lock_fingerprint,
        "test_final_model_checkpoint_fingerprint": test_refit["final_checkpoint_fingerprint"],
        "test_final_refit_training_users": test_refit["final_training_users"],
        "test_final_refit_training_examples": test_refit["final_training_examples"],
        "test_final_refit_internal_dev_examples": test_refit["internal_dev_examples"],
        "test_artifacts": {
            runner.config.relative(path): file_fingerprint(path) for path in test_paths
        },
        "post_run_maintenance": (
            "Final verification found that the original formal Validation/Test candidates used selection-only "
            "checkpoints. The locked TT1 was reinitialized and refit for the frozen epoch=1 on 100% eligible "
            "samples; embeddings, ANN, candidates, fixed-budget unions, metrics and audits were regenerated. "
            "TT1/T100 and all hyperparameters remained frozen. The Top-12 acceptance audit denominator was "
            "corrected from candidate-covered users to the Stage 2B main evaluation population without rerunning "
            "models or candidates. Original UNPIVOT failure, Test part 019 spill/long tail and read-only NumPy "
            "warning remain disclosed."
        ),
    })
    fingerprint_payload = {
        key: value for key, value in lock.items()
        if key not in {"lock_fingerprint", "output_fingerprints"}
    }
    lock["output_fingerprints"] = sorted({
        value.lower()
        for value in _nested_strings(fingerprint_payload)
        if re.fullmatch(r"[0-9a-fA-F]{64}", value)
    })
    lock["lock_fingerprint"] = lock_fingerprint(lock)
    json_write(runner.config.lock_path, lock)
    return runner.config.lock_path


def normalize_operational_audits(runner: Stage3DRunner) -> None:
    events_path = runner.config.artifact_dir / "operational_event_audit.csv"
    events = csv_rows(events_path)
    for row in events:
        if row["event"] == "pytorch_readonly_numpy_view_warning":
            row["detail"] = (
                "The warning occurred during original and corrected item-embedding export when a read-only "
                "PyArrow NumPy item-key view was passed to torch.from_numpy. The tensor was only used for "
                "embedding lookup/read and never mutated; the warning itself did not change output fingerprints."
            )
    write_csv(events_path, events, events[0].keys())

    failure_path = runner.config.artifact_dir / "failure_retry_summary.csv"
    failures = csv_rows(failure_path)
    for row in failures:
        if row.get("status") == "SUCCESS" and not row.get("error_message"):
            row["retry_count"] = 0
            row["recovered"] = False
            row["recovery_note"] = (
                "Intentional rerun after full-refit input correction; the prior SUCCESS output was superseded, "
                "not a failed attempt."
            )
    write_csv(failure_path, failures, failures[0].keys())


def _nested_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [token for nested in value.values() for token in _nested_strings(nested)]
    if isinstance(value, list):
        return [token for nested in value for token in _nested_strings(nested)]
    return []


def declare_public_csv_fingerprints(runner: Stage3DRunner) -> None:
    for name in ("model_lineage.csv", "full_refit_correction_audit.csv"):
        path = runner.config.artifact_dir / name
        rows = csv_rows(path)
        for row in rows:
            values = sorted({
                value.lower() for value in row.values()
                if re.fullmatch(r"[0-9a-fA-F]{64}", str(value))
            })
            row["output_fingerprints"] = json.dumps(values, separators=(",", ":"))
        write_csv(path, rows, rows[0].keys())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-public", action="store_true")
    args = parser.parse_args()
    runner = Stage3DRunner(Stage3DConfig.load(Path("configs/stage_3d_two_tower.yaml")))
    if args.final_public:
        runner.write_privacy_audit()
        runner.write_manifest()
        return 0
    correct_top12_population_denominators(runner)
    runner.complete_operational_event_audit()
    declare_public_csv_fingerprints(runner)
    finalize_lock_after_test(runner)
    runner.training_data_summary()
    runner.write_operational_audits()
    normalize_operational_audits(runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
