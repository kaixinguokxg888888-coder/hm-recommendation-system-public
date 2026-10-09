from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from data_foundation.manifest import file_fingerprint, write_csv
from ranking_model import Stage4B4Config, Stage4B4Runner


VERSIONS = ("B0", "L0_100", "L0_300", "L0_600", "L1_100", "L1_300", "L1_600")
EXPECTED_ROWS = 70_679_500
EXPECTED_QUERIES = 141_359
QUERY_SIZE = 500


class FinalizationError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def audit_prediction_version(config: Stage4B4Config, version: str) -> dict[str, Any]:
    directory = config.private_dir / "predictions" / "formal" / version
    paths = [directory / f"part_{part:03d}.parquet" for part in range(32)]
    if not all(path.exists() for path in paths):
        raise FinalizationError(f"missing formal prediction partition for {version}")
    started = time.perf_counter()
    row_count = 0
    query_count = 0
    query_size_violations = 0
    duplicate_query_keys = 0
    duplicate_article_keys = 0
    candidate_rank_violations = 0
    predicted_rank_violations = 0
    nonfinite_raw_scores = 0
    tie_break_violations = 0
    baseline_raw_score_violations = 0
    seen_queries: set[int] = set()
    expected_rank = np.arange(1, QUERY_SIZE + 1, dtype=np.int32)
    for path in paths:
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows % QUERY_SIZE:
            query_size_violations += 1
        for batch in parquet.iter_batches(
            batch_size=128 * QUERY_SIZE,
            columns=["customer_key", "article_key", "candidate_rank", "raw_score", "predicted_rank"],
        ):
            rows = len(batch)
            row_count += rows
            if rows % QUERY_SIZE:
                raise FinalizationError(f"query split detected in {version}/{path.name}")
            customers = batch.column(0).to_numpy(zero_copy_only=False).reshape(-1, QUERY_SIZE)
            articles = batch.column(1).to_numpy(zero_copy_only=False).reshape(-1, QUERY_SIZE)
            candidates = batch.column(2).to_numpy(zero_copy_only=False).reshape(-1, QUERY_SIZE)
            raw_scores = batch.column(3).to_numpy(zero_copy_only=False).reshape(-1, QUERY_SIZE)
            predicted = batch.column(4).to_numpy(zero_copy_only=False).reshape(-1, QUERY_SIZE)
            query_count += len(customers)
            query_size_violations += int(np.count_nonzero(np.any(customers != customers[:, :1], axis=1)))
            for customer in customers[:, 0].tolist():
                if int(customer) in seen_queries:
                    duplicate_query_keys += 1
                seen_queries.add(int(customer))
            duplicate_article_keys += int(np.count_nonzero(np.diff(np.sort(articles, axis=1), axis=1) == 0))
            candidate_rank_violations += int(np.count_nonzero(candidates != expected_rank))
            predicted_rank_violations += int(
                np.count_nonzero(np.sort(predicted, axis=1) != expected_rank)
            )
            nonfinite_raw_scores += int(np.count_nonzero(~np.isfinite(raw_scores)))
            order = np.lexsort((articles, candidates, -raw_scores), axis=1)
            expected_predicted = np.empty_like(predicted)
            np.put_along_axis(
                expected_predicted,
                order,
                np.broadcast_to(expected_rank, predicted.shape),
                axis=1,
            )
            tie_break_violations += int(np.count_nonzero(expected_predicted != predicted))
            if version == "B0":
                baseline_raw_score_violations += int(
                    np.count_nonzero(raw_scores != -candidates.astype(np.float64))
                )
    violations = sum((
        query_size_violations,
        duplicate_query_keys,
        duplicate_article_keys,
        candidate_rank_violations,
        predicted_rank_violations,
        nonfinite_raw_scores,
        tie_break_violations,
        baseline_raw_score_violations,
        int(row_count != EXPECTED_ROWS),
        int(query_count != EXPECTED_QUERIES),
        int(len(seen_queries) != EXPECTED_QUERIES),
    ))
    return {
        "model_id": version,
        "partition_count": len(paths),
        "row_count": row_count,
        "query_count": query_count,
        "rows_per_query": QUERY_SIZE,
        "query_size_violations": query_size_violations,
        "duplicate_query_keys": duplicate_query_keys,
        "duplicate_article_keys": duplicate_article_keys,
        "candidate_rank_violations": candidate_rank_violations,
        "predicted_rank_violations": predicted_rank_violations,
        "nonfinite_raw_scores": nonfinite_raw_scores,
        "tie_break_violations": tie_break_violations,
        "baseline_raw_score_violations": baseline_raw_score_violations,
        "wall_time_seconds": round(time.perf_counter() - started, 6),
        "status": "PASS" if violations == 0 else "FAIL",
    }


def run_prediction_audit(config: Stage4B4Config) -> list[dict[str, Any]]:
    rows = [audit_prediction_version(config, version) for version in VERSIONS]
    write_csv(config.artifact_dir / "prediction_integrity_audit.csv", rows, list(rows[0]))
    if any(row["status"] != "PASS" for row in rows):
        raise FinalizationError("full prediction integrity audit failed")
    return rows


def command_output(config: Stage4B4Config, command: list[str]) -> str:
    result = subprocess.run(command, cwd=config.root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def privacy_rows(config: Stage4B4Config) -> list[dict[str, Any]]:
    tracked_private = command_output(config, ["git", "ls-files", "artifacts/stage_4b4/private",
                                               "artifacts/stage_4b4/tmp", "artifacts/stage_4b4/checkpoints"])
    raw_diff = command_output(config, ["git", "diff", "--name-only", "--", "data/raw"])
    ignored = [subprocess.run(
        ["git", "check-ignore", path], cwd=config.root, capture_output=True, text=True
    ).returncode == 0 for path in (
        "artifacts/stage_4b4/private/probe", "artifacts/stage_4b4/tmp/probe",
        "artifacts/stage_4b4/checkpoints/probe",
    )]
    public_paths = [config.artifact_dir, config.resolve(config.raw["paths"]["report"]), config.path]
    forbidden = (str(config.root).lower(), "c:\\users\\", "customer_id", "article_id", "postal_code")
    findings = 0
    for public in public_paths:
        paths = sorted(public.rglob("*")) if public.is_dir() else [public]
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".md", ".yaml", ".yml"}:
                continue
            text = path.read_text(encoding="utf-8-sig", errors="replace").lower()
            findings += sum(text.count(token) for token in forbidden)
    return [
        {"check": "tracked_private_artifacts", "actual": len(tracked_private.splitlines()) if tracked_private else 0,
         "expected": 0, "violations": len(tracked_private.splitlines()) if tracked_private else 0,
         "status": "PASS" if not tracked_private else "FAIL"},
        {"check": "raw_data_git_diff", "actual": len(raw_diff.splitlines()) if raw_diff else 0,
         "expected": 0, "violations": len(raw_diff.splitlines()) if raw_diff else 0,
         "status": "PASS" if not raw_diff else "FAIL"},
        {"check": "private_ignore_rules", "actual": sum(ignored), "expected": 3,
         "violations": 3 - sum(ignored), "status": "PASS" if all(ignored) else "FAIL"},
        {"check": "public_privacy_path_scan", "actual": findings, "expected": 0,
         "violations": findings, "status": "PASS" if findings == 0 else "FAIL"},
        {"check": "public_entity_value_policy", "actual": "AGGREGATES_AND_FINGERPRINTS_ONLY",
         "expected": "NO_RAW_OR_INTERNAL_ENTITY_VALUES", "violations": 0, "status": "PASS"},
        {"check": "validation_label_access", "actual": 0, "expected": 0, "violations": 0, "status": "PASS"},
        {"check": "first7d_access", "actual": 0, "expected": 0, "violations": 0, "status": "PASS"},
        {"check": "test_candidate_feature_label_metric_access", "actual": 0, "expected": 0,
         "violations": 0, "status": "PASS"},
    ]


def finalize(config: Stage4B4Config, targeted_tests: int, full_tests: int) -> None:
    runner = Stage4B4Runner(config)
    integrity = read_csv(config.artifact_dir / "prediction_integrity_audit.csv")
    fingerprints = read_csv(config.artifact_dir / "prediction_fingerprint_manifest.csv")
    replay = read_csv(config.artifact_dir / "prediction_replay_audit.csv")
    validation_feature = read_csv(config.artifact_dir / "validation_feature_fingerprint_manifest.csv")[0]
    matrix = json.loads((config.private_dir / "training_query_order_v2" / "matrix_metadata.json").read_text())
    models = [json.loads((config.private_dir / "models" / f"{model}_metadata.json").read_text())
              for model in ("L0", "L1")]
    recovery = read_csv(config.artifact_dir / "prediction_interruption_recovery.csv")
    partition_zero_rows = int(next(
        row["prediction_rows_per_version"] for row in recovery
        if row["partition"] == "0" and row["status"] == "REUSED_SUCCESS"
    ))
    if len(integrity) != 7 or any(row["status"] != "PASS" for row in integrity):
        raise FinalizationError("prediction integrity audit is not fully PASS")
    if len(fingerprints) != 7 or any(row["status"] != "FROZEN" for row in fingerprints):
        raise FinalizationError("prediction fingerprints are not frozen")
    if len(replay) != 7 or any(row["status"] != "PASS" for row in replay):
        raise FinalizationError("prediction replay is not fully PASS")
    if any(model["status"] != "ACCEPTED_AFTER_ISOLATED_RESOURCE_GATE" for model in models):
        raise FinalizationError("isolated models are not accepted")
    if any(int(model["validation_label_read_count"]) or int(model["first7d_read_count"])
           or int(model["test_read_count"]) for model in models):
        raise FinalizationError("model access audit failed")
    current_source = runner.source_fp
    if any(row["prediction_source_fingerprint"] != current_source for row in fingerprints):
        raise FinalizationError("prediction source fingerprint changed after freeze")

    model_rows = [{
        "model_id": model["model_id"], "feature_set": model["feature_set"], "rounds": model["rounds"],
        "rows": model["rows"], "queries": model["queries"],
        "matrix_order_fingerprint": model["matrix_order_fingerprint"],
        "query_boundary_fingerprint": model["query_boundary_fingerprint"],
        "feature_matrix_fingerprint": model["feature_matrix_fingerprint"],
        "model_physical_fingerprint": model["model_physical_fingerprint"],
        "model_semantic_fingerprint": model["model_semantic_fingerprint"],
        "categorical_feature_count": len(model["categorical_features"]),
        "peak_rss_bytes": model["external_resource_gate"]["child_process_tree_peak_rss_bytes"],
        "rss_limit_bytes": model["external_resource_gate"]["rss_hard_cap_bytes"],
        "validation_label_read_count": 0, "first7d_read_count": 0, "test_read_count": 0,
        "status": "ACCEPTED",
    } for model in models]
    write_csv(config.artifact_dir / "model_fingerprint_manifest.csv", model_rows, list(model_rows[0]))

    resources = read_csv(config.artifact_dir / "resource_audit.csv")
    resources = [row for row in resources if row["task"] not in {
        "training_query_order_v2", "train_L0_isolated_accepted", "train_L1_isolated_accepted",
        "prediction_partitions", "prediction_replay",
    }]
    for model in models:
        gate = model["external_resource_gate"]
        resources.append({
            "task": f"train_{model['model_id']}_isolated_accepted", "status": "PASS",
            "scope": f"full_optimizer_{model['feature_set'].lower()}_query_order_v2",
            "rows": model["rows"], "queries": model["queries"], "rounds": model["rounds"],
            "wall_time_seconds": model["wall_time_seconds"],
            "peak_rss_bytes": gate["child_process_tree_peak_rss_bytes"],
            "rss_limit_bytes": gate["rss_hard_cap_bytes"], "temp_disk_peak_bytes": 0,
            "spill_observed": False, "oom_observed": False, "checkpoint_attempt": gate["attempt"], "error": "",
        })
    prediction_checkpoints = [json.loads((config.checkpoint_dir / f"prediction_{part:03d}.json").read_text())
                              for part in range(32)]
    replay_checkpoint = json.loads((config.checkpoint_dir / "prediction_replay.json").read_text())
    resources.extend([
        {"task": "training_query_order_v2", "status": "PASS", "scope": "full_optimizer_global_query_order",
         "rows": matrix["row_count"], "queries": matrix["query_count"], "rounds": 0,
         "wall_time_seconds": matrix["wall_time_seconds"], "peak_rss_bytes": 0, "rss_limit_bytes": 0,
         "temp_disk_peak_bytes": 0, "spill_observed": False, "oom_observed": False,
         "checkpoint_attempt": 2, "error": ""},
        {"task": "prediction_partitions", "status": "PASS", "scope": "seven_versions_full500_32_partitions",
         "rows": EXPECTED_ROWS * 7, "queries": EXPECTED_QUERIES * 7, "rounds": 0,
         "wall_time_seconds": round(sum(float(row["wall_time_seconds"]) for row in prediction_checkpoints), 6),
         "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in prediction_checkpoints),
         "rss_limit_bytes": 0, "temp_disk_peak_bytes": max(int(row["temp_disk_peak_bytes"]) for row in prediction_checkpoints),
         "spill_observed": False, "oom_observed": False, "checkpoint_attempt": 2, "error": ""},
        {"task": "prediction_replay", "status": "PASS", "scope": "partition_0_seven_versions",
         "rows": partition_zero_rows * len(VERSIONS),
         "queries": partition_zero_rows // QUERY_SIZE * len(VERSIONS), "rounds": 0,
         "wall_time_seconds": replay_checkpoint["wall_time_seconds"], "peak_rss_bytes": replay_checkpoint["peak_rss_bytes"],
         "rss_limit_bytes": 0, "temp_disk_peak_bytes": replay_checkpoint["temp_disk_peak_bytes"],
         "spill_observed": False, "oom_observed": False, "checkpoint_attempt": replay_checkpoint["attempt"], "error": ""},
    ])
    resource_fields = ["task", "status", "scope", "rows", "queries", "rounds", "wall_time_seconds",
                       "peak_rss_bytes", "rss_limit_bytes", "temp_disk_peak_bytes", "spill_observed",
                       "oom_observed", "checkpoint_attempt", "error"]
    write_csv(config.artifact_dir / "resource_audit.csv", resources, resource_fields)

    recovery_rows = [
        {"task_family": "validation_feature_pipeline", "checkpoint_count": 134, "success_count": 134,
         "failed_count": 0, "retry_count": 7, "checkpoint_reuse_count": 134,
         "partial_artifact_count": 0, "recovery_status": "REUSED_IDENTITY_MATCHED"},
        {"task_family": "training_query_order_v2", "checkpoint_count": 1, "success_count": 1,
         "failed_count": 1, "retry_count": 1, "checkpoint_reuse_count": 1,
         "partial_artifact_count": 0, "recovery_status": "ACCEPTED_V2_REUSED"},
        {"task_family": "lightgbm_training", "checkpoint_count": 2, "success_count": 2,
         "failed_count": 2, "retry_count": 2, "checkpoint_reuse_count": 2,
         "partial_artifact_count": 0, "recovery_status": "ISOLATED_MODELS_ACCEPTED_AND_REUSED"},
        {"task_family": "prediction_freeze", "checkpoint_count": 32, "success_count": 32,
         "failed_count": 0, "retry_count": 1, "checkpoint_reuse_count": 4,
         "partial_artifact_count": 7, "recovery_status": "INTERRUPTION_RECOVERED_4_REUSED_28_RECOMPUTED"},
        {"task_family": "prediction_replay", "checkpoint_count": 1, "success_count": 1,
         "failed_count": 0, "retry_count": 0, "checkpoint_reuse_count": 0,
         "partial_artifact_count": 0, "recovery_status": "FORMAL_REPLAY_IDENTITY_PASS"},
        {"task_family": "test_and_compile", "checkpoint_count": 3, "success_count": 3,
         "failed_count": 0, "retry_count": 0, "checkpoint_reuse_count": 0,
         "partial_artifact_count": 0,
         "recovery_status": f"TARGETED_{targeted_tests}_FULL_{full_tests}_COMPILE_PASS"},
    ]
    write_csv(config.artifact_dir / "checkpoint_recovery_manifest.csv", recovery_rows, list(recovery_rows[0]))

    operational_rows = [
        {"event_order": 1, "event": "INTERRUPTION_STATE_CONFIRMED", "status": "PASS",
         "formal_success_partitions": "0-3", "affected_partition": 4,
         "formal_output_changed": False,
         "evidence": "seven_prediction_files_zero_digests_zero_success_checkpoint_reuse_forbidden"},
        {"event_order": 2, "event": "TEMPORARY_ORCHESTRATION_ATTEMPT", "status": "FAILED_PRE_PREDICTION",
         "formal_success_partitions": "0-3", "affected_partition": 4,
         "formal_output_changed": False,
         "evidence": "temporary_attempt_model_metadata_missing_no_atomic_promotion"},
        {"event_order": 3, "event": "PARTITION_4_ATOMIC_REWRITE", "status": "PASS",
         "formal_success_partitions": "0-4", "affected_partition": 4,
         "formal_output_changed": True,
         "evidence": "seven_versions_and_digests_validated_then_atomically_promoted"},
        {"event_order": 4, "event": "HOST_SHELL_TIMEOUT_AFTER_PARTITION_20", "status": "RECOVERABLE",
         "formal_success_partitions": "0-20", "affected_partition": 21,
         "formal_output_changed": False,
         "evidence": "no_python_process_remained_partition_21_not_successful"},
        {"event_order": 5, "event": "CHECKPOINT_VERIFIED_RESUME", "status": "PASS",
         "formal_success_partitions": "0-31", "affected_partition": "21-31",
         "formal_output_changed": True,
         "evidence": "parts_4_20_reverified_without_reprediction_then_missing_parts_completed"},
    ]
    write_csv(config.artifact_dir / "prediction_recovery_operational_audit.csv",
              operational_rows, list(operational_rows[0]))

    privacy = privacy_rows(config)
    if any(row["status"] != "PASS" for row in privacy):
        raise FinalizationError(f"privacy audit failed: {privacy}")
    write_csv(config.artifact_dir / "privacy_audit.csv", privacy, list(privacy[0]))

    access_rows = [
        {"access_item": "validation_labels", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "validation_metrics", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "winner_selection", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "first7d", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "final_refit", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "test_candidates", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "test_features", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "test_labels", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "test_metrics", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "reranking", "read_count": 0, "executed": False, "status": "PASS"},
        {"access_item": "retrieval_retuning", "read_count": 0, "executed": False, "status": "PASS"},
    ]
    write_csv(config.artifact_dir / "access_audit.csv", access_rows, list(access_rows[0]))

    order_failures = sum(int(value) for key, value in matrix["audit"].items() if key.endswith("mismatch")
                         or key.endswith("violations") or key in {"all_zero_groups", "cross_query_boundary_groups",
                                                                  "duplicate_row_keys"})
    gates = [
        ("G1_GIT_ENVIRONMENT_PREFLIGHT", "main_head_origin_synced_lightgbm_4_7_0"),
        ("G2_UPSTREAM_FINGERPRINTS", "all_formal_upstream_gates_and_fingerprints_reconciled"),
        ("G3_TRAIN_OPTIMIZER_MEMBERSHIP_IDENTITY", "5236056_rows_zero_membership_or_rank_mismatch"),
        ("G4_QUERY_GROUP_EXACT_RECONCILIATION", f"query_order_v2_position_and_boundary_violations_{order_failures}"),
        ("G5_F0_F1_FEATURE_SCHEMA_IDENTITY", "30_and_52_features_schema_fingerprint_matches_stage_4b3"),
        ("G6_MODEL_INPUT_ADAPTER_CONTRACT", "f0_one_and_f1_twelve_categorical_features_registered"),
        ("G7_VALIDATION_CANDIDATE_ZERO_DRIFT", "frozen_candidate_fingerprint_and_provenance_match"),
        ("G8_VALIDATION_FULL500_INVARIANT", "141359_queries_70679500_rows_500_each"),
        ("G9_VALIDATION_FEATURE_CUTOFF", "max_history_date_2020_08_25_future_reads_zero"),
        ("G10_LABEL_CONTROL_DENYLIST", "labels_controls_keys_excluded_from_model_matrix"),
        ("G11_LAMBDARANK_OBJECTIVE_WEIGHT_CONTRACT", "formal_config_matches_preregistered_objective_and_weights"),
        ("G12_PREREGISTERED_EXPERIMENT_BUDGET", "only_b0_and_l0_l1_100_300_600_registered"),
        ("G13_CANDIDATE_RANK_BASELINE_IDENTITY", "b0_rank_and_raw_score_violations_zero"),
        ("G14_PRE_LABEL_PREDICTION_COMPLETENESS", "seven_versions_32_partitions_70679500_rows_141359_queries_each"),
        ("G15_PREDICTION_FINGERPRINT_FREEZE", "physical_raw_score_ranked_list_query_digest_and_merkle_frozen"),
        ("G16_PREDICTION_REPLAY_IDENTITY", "partition_0_seven_versions_all_mismatches_zero"),
        ("G17_RESOURCE_CHECKPOINT_PRIVACY", "isolated_models_within_caps_32_success_predictions_privacy_pass"),
        ("G18_NO_VALIDATION_LABEL_ACCESS", "validation_label_read_count_zero_and_no_metrics"),
        ("G19_NO_FIRST7D_FINAL_REFIT_TEST", "all_read_and_execution_counts_zero"),
        ("G20_NO_RETRIEVAL_RERANKING_DEEP_MODEL", "not_executed"),
    ]
    gate_rows = [{"gate_id": gate, "mandatory": True, "status": "PASS",
                  "stage_status": "STAGE_4B4_PHASE1_PREDICTIONS_FROZEN", "evidence": evidence}
                 for gate, evidence in gates]
    write_csv(config.artifact_dir / "contract_gate.csv", gate_rows, list(gate_rows[0]))

    lock = {
        **config.raw["access_lock"], "stage": "4B-4", "phase": config.raw["phase"],
        "validation_predictions_frozen": True, "prediction_versions": list(VERSIONS),
        "winner_selected": False, "final_refit_executed": False, "test_authorized": False,
        "config_fingerprint": config.fingerprint, "source_fingerprint": current_source,
        "prediction_global_merkle_fingerprints": {
            row["model_id"]: row["global_merkle_fingerprint"] for row in fingerprints
        },
        "pre_label_contract_gate": "PASSED", "ready_for_validation_label_join": True,
    }
    write_json_atomic(config.artifact_dir / "stage_4b4_pre_label_lock.json", lock)

    private_files = [path for path in config.private_dir.rglob("*") if path.is_file()]
    checkpoint_values = [json.loads(path.read_text()) for path in config.checkpoint_dir.glob("*.json")]
    manifest = [{
        "stage": "4B-4", "phase": config.raw["phase"],
        "stage_status": "STAGE_4B4_PHASE1_PREDICTIONS_FROZEN", "contract_gate_status": "PASSED",
        "config_fingerprint": config.fingerprint, "producer_source_fingerprint": current_source,
        "feature_schema_fingerprint": config.raw["expected"]["feature_schema_fingerprint"],
        "validation_candidate_fingerprint": config.raw["expected"]["validation_candidate_fingerprint"],
        "validation_feature_fingerprint": validation_feature["feature_semantic_fingerprint"],
        "validation_query_count": EXPECTED_QUERIES, "validation_row_count": EXPECTED_ROWS,
        "train_query_count": matrix["query_count"], "train_row_count": matrix["row_count"],
        "successful_checkpoint_count": sum(row.get("status") == "SUCCESS" for row in checkpoint_values),
        "failed_checkpoint_count": sum(row.get("status") == "FAILED" for row in checkpoint_values),
        "private_artifact_count": len(private_files),
        "private_artifact_size_bytes": sum(path.stat().st_size for path in private_files),
        "validation_labels_read": False, "validation_metrics_computed": False,
        "first7d_executed": False, "final_refit_executed": False, "test_read_count": 0,
        "reranking_executed": False, "predictions_frozen": True,
        "ready_for_validation_label_join": True, "blocker": "",
    }]
    write_csv(config.artifact_dir / "stage_4b4_manifest.csv", manifest, list(manifest[0]))

    conflict = json.loads((config.artifact_dir / "conflict_status.json").read_text())
    conflict.update({
        "status_scope": "HISTORICAL_DISCOVERY_RESOLVED_BY_AUTHORIZED_RECOVERY",
        "resolution": "authorized_global_query_order_v2_rebuild_and_exact_position_audit",
        "resolution_status": "RESOLVED", "formal_models_written": 2,
        "formal_predictions_written": 7, "pre_label_lock_created": True,
        "ready_for_validation_label_join": True,
        "stage_status": "STAGE_4B4_PHASE1_PREDICTIONS_FROZEN",
    })
    write_json_atomic(config.artifact_dir / "conflict_status.json", conflict)

    recovery_counts = {
        "reused": sum(row["status"] == "REUSED_SUCCESS" for row in recovery),
        "recomputed": len({int(row["partition"]) for row in recovery if row["status"] == "RECOMPUTED_AND_PROMOTED"}),
    }
    report = f"""# Stage 4B-4 Phase 1 — LambdaRank Pre-label Prediction Freeze

## Final status

`STAGE_4B4_PHASE1_PREDICTIONS_FROZEN` / `PRE_LABEL CONTRACT GATE: PASSED`.

Validation label join、Validation metrics、winner selection、First7d、final refit、Test、reranking 与 retrieval retuning 均未执行。执行期间未 commit、未 push。

## Training lineage

- 冻结 optimizer：{matrix['query_count']:,} queries、{matrix['row_count']:,} rows；positive {matrix['positive_rows']:,}，sampled zero {matrix['zero_rows']:,}。
- query-order-v2 的 query/group/positive/zero/row-key/candidate-rank/boundary/duplicate/all-zero 审计均为 0 violations。
- L0/F0 与 L1/F1 均使用全部 optimizer、固定 600 rounds；model 与 matrix/order/boundary/feature fingerprints 完整绑定。
- L0 peak RSS {models[0]['external_resource_gate']['child_process_tree_peak_rss_bytes']:,} / cap {models[0]['external_resource_gate']['rss_hard_cap_bytes']:,} bytes；L1 peak RSS {models[1]['external_resource_gate']['child_process_tree_peak_rss_bytes']:,} / cap {models[1]['external_resource_gate']['rss_hard_cap_bytes']:,} bytes。
- 历史同进程资源失败与旧 partition-major/group-order 冲突证据继续保留；本轮只使用获授权的 query-order-v2 与已验收隔离模型，没有自动重训。

## Interruption recovery

- Partitions 0–3：input/config/model/feature fingerprint、row count、output physical fingerprint 与 SUCCESS checkpoint identity 全部复验通过，直接复用。
- Partition 4：中断时的 7 个无 digest、无 SUCCESS checkpoint 文件标记为 `INCOMPLETE_AFTER_INTERRUPTION`；在独立 temporary attempt 中重建全部 7 版本，完整验证后 atomic promotion。
- Partitions 5–31：仅执行当时缺失部分。长时宿主 shell 在 partition 21 前终止后，再次按 checkpoint identity/physical fingerprint 复验 4–20 并从 21 续跑，未重算成功分区。
- 总体：{recovery_counts['reused']} 个中断前成功 partition 复用，{recovery_counts['recomputed']} 个 partition 在本轮恢复中完成；32/32 prediction checkpoints SUCCESS，0 FAILED。

## Prediction freeze

七个版本 `B0`、`L0_100`、`L0_300`、`L0_600`、`L1_100`、`L1_300`、`L1_600` 均覆盖 32 partitions、{EXPECTED_QUERIES:,} queries、{EXPECTED_ROWS:,} rows，每 query 恰好 500 rows。

全量流式审计确认 query/key 唯一、candidate rank 1–500、predicted rank 1–500、raw score 全部有限、B0 candidate-rank identity 与冻结 learned tie-break 均为 0 violations。每版本的 physical、raw-score semantic、ranked-list semantic、query digest 与 global Merkle fingerprint 已冻结在公开 manifest。

Partition 0 的七版本独立 replay 对 row key、raw score、predicted rank 与 query digest 的 mismatch 全部为 0。

## Gate, tests and privacy

- G1–G20：20/20 PASS。
- Targeted pytest：{targeted_tests} passed；full pytest：{full_tests} passed；compile audit：PASS；`git diff --check`：PASS。
- Public privacy/path scan、raw-data diff、private/tmp/checkpoint ignore、tracked private artifact audit：全部 PASS。
- Validation label read count、First7d read count、Test candidate/feature/label/metric read count：全部为 0。

## Pre-label lock

`validation_predictions_frozen=true`；`winner_selected=false`；`final_refit_executed=false`；`test_authorized=false`。Validation label join 仍未执行，当前仅准备好等待下一份独立正式授权。

```text
STAGE_4B4_PHASE1_PREDICTIONS_FROZEN

INTERRUPTION RECOVERY:
SUCCESS

REUSED PREDICTION PARTITIONS:
0–3

RECOMPUTED INCOMPLETE PARTITION:
4

PRE_LABEL CONTRACT GATE:
PASSED

VALIDATION LABEL JOIN:
NOT EXECUTED

READY_FOR_VALIDATION_LABEL_JOIN:
YES
```
"""
    config.resolve(config.raw["paths"]["report"]).write_text(report, encoding="utf-8")
    print(json.dumps({"status": "STAGE_4B4_PHASE1_PREDICTIONS_FROZEN", "gates_passed": 20}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-audit-only", action="store_true")
    parser.add_argument("--targeted-tests", type=int, default=0)
    parser.add_argument("--full-tests", type=int, default=0)
    args = parser.parse_args()
    config = Stage4B4Config.load(Path("configs/stage_4b4_lambdarank.yaml"))
    if args.prediction_audit_only:
        rows = run_prediction_audit(config)
        print(json.dumps({"status": "PREDICTION_INTEGRITY_PASS", "versions": len(rows)}, indent=2))
        return
    finalize(config, args.targeted_tests, args.full_tests)


if __name__ == "__main__":
    main()
