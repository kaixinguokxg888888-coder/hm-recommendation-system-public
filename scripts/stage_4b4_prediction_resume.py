from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq

from data_foundation.manifest import file_fingerprint, write_csv
from ranking_model import Stage4B4Config, Stage4B4Runner


VERSIONS = ("B0", "L0_100", "L0_300", "L0_600", "L1_100", "L1_300", "L1_600")


class RecoveryStateError(RuntimeError):
    pass


def json_write_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def prediction_paths(private: Path, part: int, role: str = "formal") -> list[Path]:
    return [private / "predictions" / role / version / f"part_{part:03d}.parquet" for version in VERSIONS] + [
        private / "prediction_digests" / role / version / f"part_{part:03d}.parquet" for version in VERSIONS
    ]


def checkpoint_identity(runner: Stage4B4Runner, inputs: list[Path]) -> str:
    return hashlib.sha256(
        (runner.config.fingerprint + runner.source_fp + runner._set_fingerprint(inputs)).encode("ascii")
    ).hexdigest()


def verify_success_checkpoint(runner: Stage4B4Runner, part: int) -> dict[str, Any]:
    checkpoint = runner.checkpoint(f"prediction_{part:03d}")
    if not checkpoint.exists():
        raise RecoveryStateError(f"missing prediction checkpoint: {part}")
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    inputs = [runner.builder.feature_part(part), runner.model_path("L0"), runner.model_path("L1")]
    outputs = prediction_paths(runner.config.private_dir, part)
    if value.get("status") != "SUCCESS":
        raise RecoveryStateError(f"prediction checkpoint is not SUCCESS: {part}")
    if value.get("identity") != checkpoint_identity(runner, inputs):
        raise RecoveryStateError(f"prediction checkpoint identity mismatch: {part}")
    if not all(path.exists() for path in outputs):
        raise RecoveryStateError(f"prediction checkpoint output missing: {part}")
    if value.get("output_fingerprint") != runner._set_fingerprint(outputs):
        raise RecoveryStateError(f"prediction checkpoint physical mismatch: {part}")
    expected_rows = pq.ParquetFile(runner.builder.feature_part(part)).metadata.num_rows
    for path in outputs[:7]:
        if pq.ParquetFile(path).metadata.num_rows != expected_rows:
            raise RecoveryStateError(f"prediction row count mismatch: {part}")
    for path in outputs[7:]:
        if pq.ParquetFile(path).metadata.num_rows * 500 != expected_rows:
            raise RecoveryStateError(f"prediction digest query count mismatch: {part}")
    return {
        "partition": part, "status": "REUSED_SUCCESS", "checkpoint_identity": value["identity"],
        "output_fingerprint": value["output_fingerprint"], "prediction_rows_per_version": expected_rows,
    }


def mark_incomplete(runner: Stage4B4Runner, part: int) -> dict[str, Any]:
    marker_path = (
        runner.config.private_dir
        / "prediction_attempts"
        / f"part_{part:03d}"
        / "interruption_marker.json"
    )
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "INCOMPLETE_AFTER_INTERRUPTION" or marker.get("reuse_allowed") is not False:
            raise RecoveryStateError(f"invalid interruption marker: {part}")
        return {key: value for key, value in marker.items() if key != "partial_physical_fingerprints"}
    existing = prediction_paths(runner.config.private_dir, part)[:7]
    present = [path for path in existing if path.exists()]
    marker = {
        "partition": part, "status": "INCOMPLETE_AFTER_INTERRUPTION",
        "prediction_files_present": len(present), "digest_files_present": 0,
        "success_checkpoint_present": runner.checkpoint(f"prediction_{part:03d}").exists(),
        "partial_physical_fingerprints": [file_fingerprint(path) for path in present],
        "reuse_allowed": False,
    }
    json_write_atomic(marker_path, marker)
    return {key: value for key, value in marker.items() if key != "partial_physical_fingerprints"}


def atomic_prediction_attempt(
    runner: Stage4B4Runner, part: int, validation_feature_fingerprint: str,
) -> None:
    attempt_private = runner.config.private_dir / "prediction_attempts" / f"part_{part:03d}" / "attempt_002"
    temporary_raw = copy.deepcopy(runner.config.raw)
    temporary_raw["paths"]["private_directory"] = attempt_private.relative_to(runner.config.root).as_posix()
    temporary_config = Stage4B4Config(runner.config.root, runner.config.path, temporary_raw)
    temporary_runner = Stage4B4Runner(temporary_config)
    temporary_runner.builder = runner.builder
    temporary_runner.model_path = runner.model_path  # type: ignore[method-assign]
    temporary_models = temporary_config.private_dir / "models"
    temporary_models.mkdir(parents=True, exist_ok=True)
    for model_id in ("L0", "L1"):
        shutil.copy2(
            runner.config.private_dir / "models" / f"{model_id}_metadata.json",
            temporary_models / f"{model_id}_metadata.json",
        )
    temporary_outputs = temporary_runner.predict_partition(
        part, validation_feature_fingerprint=validation_feature_fingerprint
    )
    expected_rows = pq.ParquetFile(runner.builder.feature_part(part)).metadata.num_rows
    if len(temporary_outputs) != 14 or not all(path.exists() for path in temporary_outputs):
        raise RecoveryStateError(f"temporary prediction outputs incomplete: {part}")
    if any(pq.ParquetFile(path).metadata.num_rows != expected_rows for path in temporary_outputs[:7]):
        raise RecoveryStateError(f"temporary prediction row count mismatch: {part}")
    if any(pq.ParquetFile(path).metadata.num_rows * 500 != expected_rows for path in temporary_outputs[7:]):
        raise RecoveryStateError(f"temporary digest query count mismatch: {part}")
    formal_outputs = prediction_paths(runner.config.private_dir, part)
    for temporary, formal in zip(temporary_outputs, formal_outputs):
        formal.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, formal)


def run_missing_partition(
    runner: Stage4B4Runner, part: int, validation_feature_fingerprint: str,
) -> dict[str, Any]:
    inputs = [runner.builder.feature_part(part), runner.model_path("L0"), runner.model_path("L1")]
    outputs = prediction_paths(runner.config.private_dir, part)
    result = runner.run_task(
        f"prediction_{part:03d}",
        lambda: atomic_prediction_attempt(runner, part, validation_feature_fingerprint),
        outputs,
        inputs,
        partition=part,
    )
    checkpoint = json.loads(runner.checkpoint(f"prediction_{part:03d}").read_text(encoding="utf-8"))
    if result is None and not checkpoint.get("checkpoint_reused"):
        # operation has no return value; SUCCESS is established by the formal checkpoint below.
        pass
    if checkpoint.get("status") != "SUCCESS":
        raise RecoveryStateError(f"prediction partition did not reach SUCCESS: {part}")
    return {
        "partition": part, "status": "RECOMPUTED_AND_PROMOTED",
        "checkpoint_identity": checkpoint["identity"], "output_fingerprint": checkpoint["output_fingerprint"],
        "prediction_rows_per_version": pq.ParquetFile(runner.builder.feature_part(part)).metadata.num_rows,
    }


def resume_or_run_missing_partition(
    runner: Stage4B4Runner, part: int, validation_feature_fingerprint: str,
) -> dict[str, Any]:
    checkpoint = runner.checkpoint(f"prediction_{part:03d}")
    if checkpoint.exists():
        verified = verify_success_checkpoint(runner, part)
        verified["status"] = "RECOMPUTED_AND_PROMOTED"
        verified["resume_action"] = "REUSED_AFTER_RECOVERY_TIMEOUT"
        return verified
    result = run_missing_partition(runner, part, validation_feature_fingerprint)
    result["resume_action"] = "EXECUTED_MISSING_PARTITION"
    return result


def verify_model_and_matrix(runner: Stage4B4Runner) -> None:
    matrix = json.loads((runner.config.private_dir / "training_query_order_v2" / "matrix_metadata.json").read_text())
    if matrix["ordering_version"] != "stage_4b4_training_global_query_order_v2":
        raise RecoveryStateError("training matrix ordering version mismatch")
    if any(matrix["audit"][name] for name in (
        "query_position_mismatch", "group_size_position_mismatch", "positive_count_position_mismatch",
        "sampled_zero_count_position_mismatch", "row_key_mismatch", "candidate_rank_order_violations",
        "all_zero_groups", "cross_query_boundary_groups", "duplicate_row_keys",
    )):
        raise RecoveryStateError("training matrix audit no longer passes")
    for model_id in ("L0", "L1"):
        metadata = json.loads((runner.config.private_dir / "models" / f"{model_id}_metadata.json").read_text())
        if metadata["status"] != "ACCEPTED_AFTER_ISOLATED_RESOURCE_GATE" or metadata["rounds"] != 600:
            raise RecoveryStateError(f"model acceptance mismatch: {model_id}")
        if metadata["matrix_order_fingerprint"] != matrix["matrix_order_fingerprint"]:
            raise RecoveryStateError(f"model matrix lineage mismatch: {model_id}")
        if metadata["model_physical_fingerprint"] != file_fingerprint(runner.model_path(model_id)):
            raise RecoveryStateError(f"model physical fingerprint mismatch: {model_id}")


def main() -> None:
    config = Stage4B4Config.load(Path("configs/stage_4b4_lambdarank.yaml"))
    runner = Stage4B4Runner(config)
    runner.preflight()
    verify_model_and_matrix(runner)
    with (config.artifact_dir / "validation_feature_fingerprint_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        validation_feature_fingerprint = next(csv.DictReader(handle))["feature_semantic_fingerprint"]
    rows = [verify_success_checkpoint(runner, part) for part in range(4)]
    incomplete = mark_incomplete(runner, 4)
    rows.append({"partition": 4, **incomplete})
    for part in range(4, 32):
        rows.append(resume_or_run_missing_partition(runner, part, validation_feature_fingerprint))
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(config.artifact_dir / "prediction_interruption_recovery.csv", rows, fields)
    print(json.dumps({"status": "PARTITIONS_COMPLETE", "reused": 4, "recomputed": 28}, indent=2))


if __name__ == "__main__":
    main()
