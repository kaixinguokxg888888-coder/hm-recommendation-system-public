from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ranking_model.final_refit_post_training_recovery import (
    evaluate_raw_data_git_results,
    warning_categories,
)


class ReconciliationError(RuntimeError):
    pass


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def self_fingerprint(payload: dict[str, Any], field: str) -> str:
    return canonical_fingerprint({key: value for key, value in payload.items() if key != field})


def file_fingerprint(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def file_set_fingerprint(paths: Iterable[Path], root: Path) -> str:
    rows = [
        {
            "path": path.resolve().relative_to(root.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_fingerprint(path),
        }
        for path in sorted(paths, key=lambda value: value.as_posix())
    ]
    return canonical_fingerprint(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    os.replace(temporary, destination)


@dataclass(frozen=True)
class ReconciliationConfig:
    root: Path
    path: Path
    raw: dict[str, Any]
    fingerprint: str

    @classmethod
    def load(cls, path: Path) -> "ReconciliationConfig":
        resolved = path.resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        value = cls(
            root=resolved.parent.parent,
            path=resolved,
            raw=raw,
            fingerprint=canonical_fingerprint(raw),
        )
        value.validate()
        return value

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()

    def path_for(self, name: str) -> Path:
        return self.resolve(self.raw["paths"][name])

    def validate(self) -> None:
        if self.raw.get("phase") != "phase4_post_training_pytest_skip_audit_reconciliation_v1":
            raise ReconciliationError("wrong reconciliation phase")
        if self.raw.get("root_cause") != "AUDIT_PARSER_ERROR":
            raise ReconciliationError("root cause must be explicitly frozen")


def reconciliation_source_fingerprint(config: ReconciliationConfig) -> str:
    paths = [
        config.path,
        config.root / "src/ranking_model/pytest_skip_reconciliation.py",
        config.root / "scripts/run_stage_4b4_pytest_skip_reconciliation.py",
        config.root / "scripts/stage_4b4_pytest_structured_worker.py",
        config.root / "tests/test_stage_4b4_phase4_pytest_reconciliation.py",
    ]
    if any(not path.exists() for path in paths):
        raise ReconciliationError("reconciliation source inventory incomplete")
    return file_set_fingerprint(paths, config.root)


def _read_gate(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_original_state(config: ReconciliationConfig) -> dict[str, Any]:
    expected = config.raw["expected"]
    lock_path = config.path_for("lock")
    lock = read_json(lock_path)
    gate_path = config.path_for("gate")
    gate = _read_gate(gate_path)
    model_manifest = read_json(config.path_for("model_manifest"))
    binary_manifest = read_json(config.path_for("binary_manifest"))
    replay = read_json(config.path_for("replay_audit"))
    test_audit = read_json(config.path_for("test_audit"))
    is_original = file_fingerprint(lock_path) == expected["original_lock_physical_fingerprint"]
    state = "ORIGINAL_PHASE4_STATE" if is_original else "FAILED_ATTEMPT2_SUPERSEDED_STATE"
    state_hashes = {
        "test_audit": expected["original_test_audit_physical_fingerprint"] if is_original else None,
        "lock": expected[
            "original_lock_physical_fingerprint"
            if is_original else "failed_attempt2_lock_physical_fingerprint"
        ],
        "lock_semantic": expected[
            "original_lock_semantic_fingerprint"
            if is_original else "failed_attempt2_lock_semantic_fingerprint"
        ],
        "gate": expected[
            "original_gate_physical_fingerprint"
            if is_original else "failed_attempt2_gate_physical_fingerprint"
        ],
        "report": expected[
            "original_report_physical_fingerprint"
            if is_original else "failed_attempt2_report_physical_fingerprint"
        ],
        "manifest": expected[
            "original_manifest_physical_fingerprint"
            if is_original else "failed_attempt2_manifest_physical_fingerprint"
        ],
    }
    checks = {
        "original_test_audit": (
            file_fingerprint(config.path_for("test_audit"))
            in {
                expected["original_test_audit_physical_fingerprint"],
                *(() if is_original else (expected["failed_attempt2_test_audit_physical_fingerprint"],)),
            }
            and (
                file_fingerprint(config.path_for("test_audit"))
                != expected["original_test_audit_physical_fingerprint"]
                or test_audit.get("test_security_audit_fingerprint")
                == expected["original_test_audit_semantic_fingerprint"]
            )
        ),
        "original_lock": (
            file_fingerprint(lock_path) == state_hashes["lock"]
            and lock.get("lock_fingerprint") == state_hashes["lock_semantic"]
            and self_fingerprint(lock, "lock_fingerprint") == lock["lock_fingerprint"]
        ),
        "original_gate": (
            file_fingerprint(gate_path) == state_hashes["gate"]
            and len(gate) == 16
            and all(row["status"] == "PASS" for row in gate)
            and [row["evidence"] for row in gate[:14]] == expected["r1_to_r14_gate_evidence"]
        ),
        "original_report": file_fingerprint(config.path_for("report")) == state_hashes["report"],
        "original_manifest": file_fingerprint(config.path_for("manifest")) == state_hashes["manifest"],
        "model": (
            file_fingerprint(config.path_for("model_artifact")) == expected["model_physical_fingerprint"]
            and model_manifest["model_physical_fingerprint"] == expected["model_physical_fingerprint"]
            and model_manifest["model_semantic_fingerprint"] == expected["model_semantic_fingerprint"]
        ),
        "binary": (
            file_fingerprint(config.path_for("binary_artifact")) == expected["binary_physical_fingerprint"]
            and binary_manifest["binary_physical_fingerprint"] == expected["binary_physical_fingerprint"]
            and binary_manifest["binary_dataset_semantic_fingerprint"] == expected["binary_semantic_fingerprint"]
        ),
        "replay": (
            file_fingerprint(config.path_for("replay_audit")) == expected["replay_audit_physical_fingerprint"]
            and replay["first_reload"]["raw_score_fingerprint"] == expected["raw_score_fingerprint"]
            and replay["second_reload"]["raw_score_fingerprint"] == expected["raw_score_fingerprint"]
            and replay["first_reload"]["rank_tiebreak_fingerprint"] == expected["tie_break_fingerprint"]
            and replay["second_reload"]["rank_tiebreak_fingerprint"] == expected["tie_break_fingerprint"]
        ),
        "access": (
            lock["validation_labels_reopened"] is False
            and lock["first7d_read_count"] == 0
            and all(lock[name] == 0 for name in (
                "test_candidate_read_count",
                "test_feature_read_count",
                "test_label_read_count",
                "test_prediction_write_count",
                "test_metric_read_count",
            ))
        ),
    }
    if not all(checks.values()):
        raise ReconciliationError(f"REPORT_CONFLICT: {[name for name, value in checks.items() if not value]}")
    return {
        "status": f"{state}_VERIFIED",
        "checks": checks,
        "lock_fingerprint": lock["lock_fingerprint"],
        "test_audit_fingerprint": test_audit["test_security_audit_fingerprint"],
        "model_retrained": False,
        "binary_rebuilt": False,
        "prediction_recomputed": False,
        "replay_recomputed": False,
        "labels_opened": False,
        "test_access": False,
    }


def preregister_reconciliation(config: ReconciliationConfig) -> dict[str, Any]:
    identity = verify_original_state(config)
    source_fingerprint = reconciliation_source_fingerprint(config)
    payload: dict[str, Any] = {
        "status": "PYTEST_SKIP_RECONCILIATION_PREREGISTERED",
        "authorization": config.raw["authorization"],
        "config_canonical_fingerprint": config.fingerprint,
        "source_fingerprint": source_fingerprint,
        "root_cause_hypothesis": config.raw["root_cause"],
        "original_test_audit_physical_fingerprint": config.raw["expected"]["original_test_audit_physical_fingerprint"],
        "original_lock_physical_fingerprint": config.raw["expected"]["original_lock_physical_fingerprint"],
        "original_lock_semantic_fingerprint": config.raw["expected"]["original_lock_semantic_fingerprint"],
        "original_gate_physical_fingerprint": config.raw["expected"]["original_gate_physical_fingerprint"],
        "model_physical_fingerprint": config.raw["expected"]["model_physical_fingerprint"],
        "binary_physical_fingerprint": config.raw["expected"]["binary_physical_fingerprint"],
        "replay_audit_physical_fingerprint": config.raw["expected"]["replay_audit_physical_fingerprint"],
        "model_retraining_allowed": False,
        "binary_rebuild_allowed": False,
        "prediction_or_replay_allowed": False,
        "label_or_test_access_allowed": False,
    }
    payload["preregistration_fingerprint"] = self_fingerprint(payload, "preregistration_fingerprint")
    path = config.path_for("preregistration")
    if path.exists():
        existing = read_json(path)
        if existing.get("status") in {
            "FAILED_ATTEMPT1_PREREGISTRATION_SUPERSEDED",
            "FAILED_ATTEMPT2_PREREGISTRATION_SUPERSEDED",
            "ATTEMPT3_PREPARATION_PREREGISTRATION_SUPERSEDED",
        }:
            write_json_atomic(path, payload)
        elif existing != payload:
            raise ReconciliationError("reconciliation preregistration drift")
    else:
        write_json_atomic(path, payload)

    evidence_dir = config.path_for("private_evidence_directory")
    evidence = {
        "original_test_audit.json": (config.path_for("test_audit"), "original_test_audit_physical_fingerprint"),
        "original_final_refit_lock.json": (config.path_for("lock"), "original_lock_physical_fingerprint"),
        "original_resource_recovery_gate.csv": (config.path_for("gate"), "original_gate_physical_fingerprint"),
        "original_final_refit_report.md": (config.path_for("report"), "original_report_physical_fingerprint"),
        "original_resource_recovery_manifest.csv": (config.path_for("manifest"), "original_manifest_physical_fingerprint"),
    }
    for name, (source, expected_name) in evidence.items():
        destination = evidence_dir / name
        if destination.exists():
            if file_fingerprint(destination) != config.raw["expected"][expected_name]:
                raise ReconciliationError("private supersession evidence drift")
        else:
            if file_fingerprint(source) != config.raw["expected"][expected_name]:
                raise ReconciliationError("original supersession evidence unavailable")
            copy_atomic(source, destination)
    return payload


def supersede_failed_attempt1(config: ReconciliationConfig) -> dict[str, Any]:
    preregistration_path = config.path_for("preregistration")
    preregistration = read_json(preregistration_path)
    expected = config.raw["expected"]
    if (
        preregistration.get("source_fingerprint") != expected["failed_attempt1_source_fingerprint"]
        or preregistration.get("preregistration_fingerprint")
        != expected["failed_attempt1_preregistration_fingerprint"]
        or self_fingerprint(preregistration, "preregistration_fingerprint")
        != preregistration["preregistration_fingerprint"]
    ):
        raise ReconciliationError("failed attempt1 preregistration identity mismatch")
    full_result_path = config.path_for("private_test_results_directory") / "full.json"
    full_result = read_json(full_result_path)
    if full_result.get("return_code") != 2 or not full_result.get("collection_errors"):
        raise ReconciliationError("failed attempt1 evidence mismatch")
    archived_preregistration = config.path_for("private_evidence_directory") / "attempt1_preregistration.json"
    if archived_preregistration.exists():
        if file_fingerprint(archived_preregistration) != file_fingerprint(preregistration_path):
            raise ReconciliationError("failed attempt1 private evidence drift")
    else:
        copy_atomic(preregistration_path, archived_preregistration)
    failure: dict[str, Any] = {
        "phase": config.raw["phase"],
        "status": "FAILED_ATTEMPT1_SUPERSEDED_BEFORE_FORMAL_PROMOTION",
        "failure_classification": "RECONCILIATION_PARSER_LAUNCH_ENVIRONMENT_ERROR",
        "failed_source_fingerprint": preregistration["source_fingerprint"],
        "failed_preregistration_fingerprint": preregistration["preregistration_fingerprint"],
        "pytest_return_code": full_result["return_code"],
        "collection_errors": full_result["collection_errors"],
        "cause": (
            "The script-mode structured worker omitted the repository root from sys.path, so two tests that "
            "formally import scripts.* failed during collection. No corrected audit, lock, gate, report, "
            "model, binary, prediction, replay, label, or Test artifact was promoted or accessed."
        ),
        "formal_artifacts_promoted": False,
        "model_retrained": False,
        "binary_rebuilt": False,
        "prediction_recomputed": False,
        "replay_recomputed": False,
        "labels_opened": False,
        "test_access": False,
    }
    failure["failure_fingerprint"] = self_fingerprint(failure, "failure_fingerprint")
    write_json_atomic(config.path_for("failed_attempt1"), failure)
    tombstone = {
        "status": "FAILED_ATTEMPT2_PREREGISTRATION_SUPERSEDED",
        "failed_preregistration_fingerprint": preregistration["preregistration_fingerprint"],
        "failure_fingerprint": failure["failure_fingerprint"],
    }
    tombstone["tombstone_fingerprint"] = self_fingerprint(tombstone, "tombstone_fingerprint")
    write_json_atomic(preregistration_path, tombstone)
    return failure


def supersede_failed_attempt2(config: ReconciliationConfig) -> dict[str, Any]:
    expected = config.raw["expected"]
    preregistration_path = config.path_for("preregistration")
    preregistration = read_json(preregistration_path)
    if (
        preregistration.get("source_fingerprint") != expected["failed_attempt2_source_fingerprint"]
        or preregistration.get("preregistration_fingerprint")
        != expected["failed_attempt2_preregistration_fingerprint"]
    ):
        raise ReconciliationError("failed attempt2 preregistration identity mismatch")
    evidence = {
        "attempt2_test_audit.json": ("test_audit", "failed_attempt2_test_audit_physical_fingerprint"),
        "attempt2_final_refit_lock.json": ("lock", "failed_attempt2_lock_physical_fingerprint"),
        "attempt2_resource_recovery_gate.csv": ("gate", "failed_attempt2_gate_physical_fingerprint"),
        "attempt2_final_refit_report.md": ("report", "failed_attempt2_report_physical_fingerprint"),
        "attempt2_resource_recovery_manifest.csv": ("manifest", "failed_attempt2_manifest_physical_fingerprint"),
        "attempt2_reconciliation.json": ("reconciliation", "failed_attempt2_reconciliation_physical_fingerprint"),
        "attempt2_supersession.json": ("supersession", "failed_attempt2_supersession_physical_fingerprint"),
    }
    evidence_dir = config.path_for("private_evidence_directory")
    for name, (path_name, expected_name) in evidence.items():
        source = config.path_for(path_name)
        if file_fingerprint(source) != expected[expected_name]:
            raise ReconciliationError(f"failed attempt2 {path_name} identity mismatch")
        destination = evidence_dir / name
        if destination.exists():
            if file_fingerprint(destination) != expected[expected_name]:
                raise ReconciliationError("failed attempt2 private evidence drift")
        else:
            copy_atomic(source, destination)
    failure: dict[str, Any] = {
        "phase": config.raw["phase"],
        "status": "FAILED_ATTEMPT2_SUPERSEDED_BEFORE_FINAL_ACCEPTANCE",
        "failure_classification": "CORRECTED_AUDIT_SCHEMA_COMPATIBILITY_ERROR",
        "failed_source_fingerprint": preregistration["source_fingerprint"],
        "failed_preregistration_fingerprint": preregistration["preregistration_fingerprint"],
        "failed_test_nodeid": (
            "tests/test_stage_4b4_phase4_post_training_recovery.py::"
            "test_formal_test_audit_is_sanitized_when_passed"
        ),
        "cause": (
            "The promoted corrected audit omitted the legacy-compatible top-level raw_data_audit and "
            "privacy_audit fields. Post-promotion targeted and full pytest therefore failed the required "
            "formal-audit schema test. The entire attempt2 metadata chain is preserved as private evidence."
        ),
        "failed_test_audit_physical_fingerprint": expected["failed_attempt2_test_audit_physical_fingerprint"],
        "failed_lock_physical_fingerprint": expected["failed_attempt2_lock_physical_fingerprint"],
        "failed_lock_semantic_fingerprint": expected["failed_attempt2_lock_semantic_fingerprint"],
        "failed_gate_physical_fingerprint": expected["failed_attempt2_gate_physical_fingerprint"],
        "formal_metadata_requires_supersession": True,
        "model_retrained": False,
        "binary_rebuilt": False,
        "combined_matrix_rebuilt": False,
        "validation_optimizer_rebuilt": False,
        "prediction_recomputed": False,
        "replay_recomputed": False,
        "labels_opened": False,
        "test_access": False,
    }
    failure["failure_fingerprint"] = self_fingerprint(failure, "failure_fingerprint")
    write_json_atomic(config.path_for("failed_attempt2"), failure)
    tombstone = {
        "status": "FAILED_ATTEMPT1_PREREGISTRATION_SUPERSEDED",
        "failed_attempt2_preregistration_fingerprint": preregistration["preregistration_fingerprint"],
        "failure_fingerprint": failure["failure_fingerprint"],
    }
    tombstone["tombstone_fingerprint"] = self_fingerprint(tombstone, "tombstone_fingerprint")
    write_json_atomic(preregistration_path, tombstone)
    return failure


def prepare_attempt3(config: ReconciliationConfig) -> dict[str, Any]:
    expected = config.raw["expected"]
    preregistration_path = config.path_for("preregistration")
    preregistration = read_json(preregistration_path)
    if (
        preregistration.get("source_fingerprint") != expected["attempt3_preparation_source_fingerprint"]
        or preregistration.get("preregistration_fingerprint")
        != expected["attempt3_preparation_preregistration_fingerprint"]
    ):
        raise ReconciliationError("attempt3 preparation preregistration identity mismatch")
    failure = read_json(config.path_for("failed_attempt2"))
    if failure.get("status") != "FAILED_ATTEMPT2_SUPERSEDED_BEFORE_FINAL_ACCEPTANCE":
        raise ReconciliationError("attempt2 failure evidence missing")
    if file_fingerprint(config.path_for("test_audit")) != expected["failed_attempt2_test_audit_physical_fingerprint"]:
        raise ReconciliationError("attempt2 audit identity mismatch before preparation")
    evidence_dir = config.path_for("private_evidence_directory")
    original_audit = evidence_dir / "original_test_audit.json"
    if file_fingerprint(original_audit) != expected["original_test_audit_physical_fingerprint"]:
        raise ReconciliationError("original audit evidence unavailable")
    archived_preregistration = evidence_dir / "attempt3_preparation_preregistration.json"
    if archived_preregistration.exists():
        if file_fingerprint(archived_preregistration) != file_fingerprint(preregistration_path):
            raise ReconciliationError("attempt3 preparation preregistration drift")
    else:
        copy_atomic(preregistration_path, archived_preregistration)
    copy_atomic(original_audit, config.path_for("test_audit"))
    tombstone = {
        "status": "ATTEMPT3_PREPARATION_PREREGISTRATION_SUPERSEDED",
        "superseded_preregistration_fingerprint": preregistration["preregistration_fingerprint"],
        "restored_test_audit_physical_fingerprint": expected["original_test_audit_physical_fingerprint"],
        "reason": "Restore the preserved schema-compatible audit only for pre-promotion required tests.",
    }
    tombstone["tombstone_fingerprint"] = self_fingerprint(tombstone, "tombstone_fingerprint")
    write_json_atomic(preregistration_path, tombstone)
    return tombstone


EXPECTED_SKIP_NODEID = (
    "tests/test_stage_4b4_phase4.py::"
    "test_phase4_gate_has_exactly_thirteen_passes_when_present"
)


def classify_skip(nodeid: str, reason: str, suite: str, code_location: str) -> dict[str, Any]:
    if nodeid == EXPECTED_SKIP_NODEID and reason == "formal gate is created during finalization":
        return {
            "suite": suite,
            "nodeid": nodeid,
            "reason": reason,
            "classification": "EXPECTED_OBSOLETE_ARTIFACT_SKIP",
            "is_expected": True,
            "is_phase4_required_gate": False,
            "code_location": code_location,
            "evidence": (
                "The optional test targets legacy final_refit_contract_gate.csv (P4G1-P4G13); "
                "the formal recovered gate is final_refit_resource_recovery_gate.csv (R1-R16), "
                "whose dedicated tests execute and pass."
            ),
        }
    return {
        "suite": suite,
        "nodeid": nodeid,
        "reason": reason,
        "classification": "UNKNOWN_SKIP",
        "is_expected": False,
        "is_phase4_required_gate": True,
        "code_location": code_location,
        "evidence": "No frozen acceptance rule permits this skip.",
    }


def validate_structured_result(payload: dict[str, Any]) -> None:
    counts = payload["counts"]
    if counts["collected"] != counts["outcome_total"]:
        raise ReconciliationError("structured pytest outcome total mismatch")
    if payload["return_code"] != 0 or payload["status"] != "PASS":
        raise ReconciliationError("POST_TRAINING_ENGINEERING_GATE_FAILED")
    if counts["failed"] or counts["errors"] or counts["xfailed"] or counts["xpassed"]:
        raise ReconciliationError("POST_TRAINING_ENGINEERING_GATE_FAILED")
    if counts["skipped"] != len(payload["skipped"]):
        raise ReconciliationError("structured pytest skip count mismatch")
    if any(not row.get("reason") for row in payload["skipped"]):
        raise ReconciliationError("REPORT_CONFLICT: empty skip reason")


def canonical_pytest_command(args: list[str], output_name: str) -> list[str]:
    return [
        "python",
        "scripts/stage_4b4_pytest_structured_worker.py",
        "--output",
        f"<private-results>/{output_name}",
        "--",
        *args,
    ]


def _run_structured_suite(
    config: ReconciliationConfig,
    *,
    suite: str,
    args: list[str],
) -> dict[str, Any]:
    result_dir = config.path_for("private_test_results_directory")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"{suite}.json"
    stdout_path = result_dir / f"{suite}.stdout.log"
    stderr_path = result_dir / f"{suite}.stderr.log"
    worker = config.root / "scripts/stage_4b4_pytest_structured_worker.py"
    command = [sys.executable, str(worker), "--output", str(output_path), "--", *args]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=config.root, text=True, stdout=stdout, stderr=stderr)
    if not output_path.exists():
        raise ReconciliationError("structured pytest result missing")
    payload = read_json(output_path)
    if completed.returncode != payload["return_code"]:
        raise ReconciliationError("structured pytest return-code mismatch")
    validate_structured_result(payload)
    payload["command"] = canonical_pytest_command(args, output_path.name)
    payload["stdout_summary"] = "pytest output retained in Git-ignored private evidence"
    payload["stderr_summary"] = "empty" if stderr_path.stat().st_size == 0 else "sanitized private diagnostic retained"
    payload["public_paths_sanitized"] = True
    return payload


def _run(root: Path, command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=root, text=True, capture_output=True)


def _privacy_audit(config: ReconciliationConfig) -> dict[str, Any]:
    public_paths = [
        path
        for path in (config.root / "artifacts/stage_4b4").glob("final_refit*")
        if path.is_file()
    ] + [config.path_for("report")]
    drive = re.compile(r"(?i)\b[a-z]:[\\/](?![\\/])")
    root_text = str(config.root)
    user_name = Path.home().name
    violations = []
    for path in public_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if drive.search(text) or root_text in text or (user_name and user_name in text):
            violations.append(path.relative_to(config.root).as_posix())
    payload: dict[str, Any] = {
        "status": "PASS" if not violations else "FAIL",
        "scanned_file_count": len(public_paths),
        "violation_count": len(violations),
        "violation_files": violations,
    }
    payload["audit_fingerprint"] = canonical_fingerprint(payload)
    return payload


def _command_status(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "return_code": result.returncode,
        "warning_categories": warning_categories(result.stderr),
    }


def run_reconciliation_audit(config: ReconciliationConfig) -> dict[str, Any]:
    prereg = read_json(config.path_for("preregistration"))
    if reconciliation_source_fingerprint(config) != prereg["source_fingerprint"]:
        raise ReconciliationError("reconciliation source drift")
    commands = config.raw["commands"]
    targeted = _run_structured_suite(
        config,
        suite="targeted",
        args=list(commands["targeted_pytest_args"]),
    )
    reconciliation_targeted = _run_structured_suite(
        config,
        suite="reconciliation_targeted",
        args=list(commands["reconciliation_pytest_args"]),
    )
    full = _run_structured_suite(
        config,
        suite="full",
        args=list(commands["full_pytest_args"]),
    )

    skip_details = []
    for suite_name, suite in (("targeted", targeted), ("full", full)):
        skip_details.extend(
            classify_skip(row["nodeid"], row["reason"], suite_name, row["code_location"])
            for row in suite["skipped"]
        )
    if any(not row["is_expected"] or row["is_phase4_required_gate"] for row in skip_details):
        raise ReconciliationError("REPORT_CONFLICT: critical or unknown skip")
    if reconciliation_targeted["counts"]["skipped"] != 0:
        raise ReconciliationError("REPORT_CONFLICT: reconciliation test skipped")

    compile_result = _run(config.root, [sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"])
    diff_result = _run(config.root, ["git", "diff", "--check"])
    raw_working = _run(config.root, ["git", "diff", "--name-only", "--", "data/raw"])
    raw_staged = _run(config.root, ["git", "diff", "--cached", "--name-only", "--", "data/raw"])
    raw_audit = evaluate_raw_data_git_results(raw_working, raw_staged)
    raw_audit["audit_fingerprint"] = canonical_fingerprint(raw_audit)
    tracked = _run(config.root, [
        "git", "ls-files", "--",
        "artifacts/stage_4b4/private/**",
        "artifacts/stage_4b4/tmp/**",
        "artifacts/stage_4b4/checkpoints/**",
    ])
    tracked_count = len([line for line in tracked.stdout.splitlines() if line.strip()])
    probes = [
        "artifacts/stage_4b4/private/final_refit/pytest_skip_reconciliation/evidence/probe.json",
        "artifacts/stage_4b4/checkpoints/final_refit/pytest_skip_reconciliation/probe.json",
        "artifacts/stage_4b4/tmp/final_refit/pytest_skip_reconciliation/probe.json",
    ]
    ignore_pass = all(_run(config.root, ["git", "check-ignore", "-q", probe]).returncode == 0 for probe in probes)
    privacy = _privacy_audit(config)
    identity = verify_original_state(config)

    engineering = {
        "python_compile": _command_status(compile_result),
        "git_diff_check": _command_status(diff_result),
        "raw_data": raw_audit,
        "privacy": privacy,
        "ignore": {"status": "PASS" if ignore_pass else "FAIL", "probe_count": len(probes)},
        "tracked_private": {
            "status": "PASS" if tracked.returncode == 0 and tracked_count == 0 else "FAIL",
            "return_code": tracked.returncode,
            "tracked_file_count": tracked_count,
        },
    }
    if any(value["status"] != "PASS" for value in engineering.values()):
        raise ReconciliationError("POST_TRAINING_ENGINEERING_GATE_FAILED")

    corrected_audit: dict[str, Any] = {
        "status": "PASS",
        "phase": config.raw["phase"],
        "root_cause": "AUDIT_PARSER_ERROR",
        "parser_version": "pytest_hook_report_v1",
        "tests": {
            "targeted": targeted,
            "reconciliation_targeted": reconciliation_targeted,
            "full": full,
        },
        "skip_details": skip_details,
        "critical_phase4_test_skipped": False,
        "engineering": engineering,
        # 保留上一版正式审计的顶层兼容字段，避免下游 required audit test 失去合同字段。
        "raw_data_audit": raw_audit,
        "privacy_audit": privacy,
        "artifact_identity": {
            "model_physical_fingerprint": config.raw["expected"]["model_physical_fingerprint"],
            "model_semantic_fingerprint": config.raw["expected"]["model_semantic_fingerprint"],
            "binary_physical_fingerprint": config.raw["expected"]["binary_physical_fingerprint"],
            "binary_semantic_fingerprint": config.raw["expected"]["binary_semantic_fingerprint"],
            "replay_audit_physical_fingerprint": config.raw["expected"]["replay_audit_physical_fingerprint"],
            "model_retrained": False,
            "binary_rebuilt": False,
            "combined_matrix_rebuilt": False,
            "validation_optimizer_rebuilt": False,
            "prediction_recomputed": False,
            "replay_recomputed": False,
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
        "git": {"commit_created": False, "push_executed": False},
    }
    corrected_audit["test_security_audit_fingerprint"] = self_fingerprint(
        corrected_audit, "test_security_audit_fingerprint"
    )
    corrected_path = config.path_for("test_audit")
    write_json_atomic(corrected_path, corrected_audit)
    corrected_physical = file_fingerprint(corrected_path)

    reconciliation: dict[str, Any] = {
        "phase": config.raw["phase"],
        "status": "PYTEST_SKIP_AUDIT_RECONCILED",
        "historical_reported_counts": config.raw["historical_reported_counts"],
        "historical_formal_audit_counts": config.raw["historical_formal_audit_displayed_progress_markers"],
        "root_cause": "AUDIT_PARSER_ERROR",
        "root_cause_detail": (
            "The old auditor retained only the last non-empty pytest stdout line. The targeted suite fit on one "
            "progress line and displayed six pre-finalization skips; the full suite spanned multiple progress "
            "lines, so its persisted '..s...' fragment was not a total. The prior 1/2 narrative mixed later "
            "post-lock targeted/full output and also counted the targeted skip as if it belonged to full."
        ),
        "targeted_command": targeted["command"],
        "full_command": full["command"],
        "targeted_collection": targeted["counts"]["collected"],
        "full_collection": full["counts"]["collected"],
        "targeted_passed": targeted["counts"]["passed"],
        "targeted_failed": targeted["counts"]["failed"],
        "targeted_errors": targeted["counts"]["errors"],
        "targeted_skipped": targeted["counts"]["skipped"],
        "targeted_xfailed": targeted["counts"]["xfailed"],
        "targeted_xpassed": targeted["counts"]["xpassed"],
        "full_passed": full["counts"]["passed"],
        "full_failed": full["counts"]["failed"],
        "full_errors": full["counts"]["errors"],
        "full_skipped": full["counts"]["skipped"],
        "full_xfailed": full["counts"]["xfailed"],
        "full_xpassed": full["counts"]["xpassed"],
        "skip_details": skip_details,
        "skip_classification": {
            "EXPECTED_PROTOCOL_SKIP": 0,
            "EXPECTED_OBSOLETE_ARTIFACT_SKIP": sum(
                row["classification"] == "EXPECTED_OBSOLETE_ARTIFACT_SKIP" for row in skip_details
            ),
            "UNEXPECTED_REQUIRED_TEST_SKIP": 0,
            "ENVIRONMENT_OR_DEPENDENCY_SKIP": 0,
            "MISSING_FORMAL_ARTIFACT_SKIP": 0,
            "UNKNOWN_SKIP": 0,
        },
        "critical_gate_test_skipped": False,
        "model_retrained": False,
        "binary_rebuilt": False,
        "combined_matrix_rebuilt": False,
        "validation_optimizer_rebuilt": False,
        "prediction_recomputed": False,
        "replay_recomputed": False,
        "labels_opened": False,
        "test_access": False,
        "original_test_audit_physical_fingerprint": config.raw["expected"]["original_test_audit_physical_fingerprint"],
        "corrected_test_audit_physical_fingerprint": corrected_physical,
    }
    reconciliation["reconciliation_semantic_fingerprint"] = self_fingerprint(
        reconciliation, "reconciliation_semantic_fingerprint"
    )
    write_json_atomic(config.path_for("reconciliation"), reconciliation)
    return {
        "status": "PYTEST_SKIP_AUDIT_RECONCILED",
        "targeted": targeted["counts"],
        "full": full["counts"],
        "skip_details": skip_details,
        "corrected_test_audit_physical_fingerprint": corrected_physical,
        "reconciliation_semantic_fingerprint": reconciliation["reconciliation_semantic_fingerprint"],
        "identity": identity,
    }


def finalize_reconciliation(config: ReconciliationConfig) -> dict[str, Any]:
    expected = config.raw["expected"]
    evidence_dir = config.path_for("private_evidence_directory")
    original_audit = evidence_dir / "original_test_audit.json"
    original_lock_path = evidence_dir / "original_final_refit_lock.json"
    original_gate_path = evidence_dir / "original_resource_recovery_gate.csv"
    original_report_path = evidence_dir / "original_final_refit_report.md"
    original_manifest_path = evidence_dir / "original_resource_recovery_manifest.csv"
    if (
        file_fingerprint(original_audit) != expected["original_test_audit_physical_fingerprint"]
        or file_fingerprint(original_lock_path) != expected["original_lock_physical_fingerprint"]
        or file_fingerprint(original_gate_path) != expected["original_gate_physical_fingerprint"]
        or file_fingerprint(original_report_path) != expected["original_report_physical_fingerprint"]
        or file_fingerprint(original_manifest_path) != expected["original_manifest_physical_fingerprint"]
    ):
        raise ReconciliationError("REPORT_CONFLICT: original supersession evidence drift")
    corrected_audit = read_json(config.path_for("test_audit"))
    reconciliation = read_json(config.path_for("reconciliation"))
    if corrected_audit.get("status") != "PASS" or reconciliation.get("status") != "PYTEST_SKIP_AUDIT_RECONCILED":
        raise ReconciliationError("POST_TRAINING_ENGINEERING_GATE_FAILED")
    corrected_audit_physical = file_fingerprint(config.path_for("test_audit"))
    if corrected_audit_physical != reconciliation["corrected_test_audit_physical_fingerprint"]:
        raise ReconciliationError("corrected test audit binding mismatch")

    original_lock = read_json(original_lock_path)
    lock = dict(original_lock)
    lock.update({
        "status": "FINAL_REFIT_LOCKED_AFTER_PYTEST_AUDIT_RECONCILIATION",
        "pytest_skip_audit_reconciled": True,
        "pytest_skip_audit_root_cause": "AUDIT_PARSER_ERROR",
        "previous_lock_physical_fingerprint": expected["original_lock_physical_fingerprint"],
        "previous_lock_semantic_fingerprint": expected["original_lock_semantic_fingerprint"],
        "previous_test_audit_physical_fingerprint": expected["original_test_audit_physical_fingerprint"],
        "sanitized_test_security_audit_physical_fingerprint": corrected_audit_physical,
        "pytest_skip_reconciliation_physical_fingerprint": file_fingerprint(config.path_for("reconciliation")),
        "pytest_skip_reconciliation_semantic_fingerprint": reconciliation["reconciliation_semantic_fingerprint"],
        "model_retrained_during_reconciliation": False,
        "binary_rebuilt_during_reconciliation": False,
        "combined_matrix_rebuilt_during_reconciliation": False,
        "validation_optimizer_rebuilt_during_reconciliation": False,
        "prediction_recomputed_during_reconciliation": False,
        "replay_recomputed_during_reconciliation": False,
    })
    lock["lock_fingerprint"] = self_fingerprint(lock, "lock_fingerprint")
    write_json_atomic(config.path_for("lock"), lock)
    if self_fingerprint(read_json(config.path_for("lock")), "lock_fingerprint") != lock["lock_fingerprint"]:
        raise ReconciliationError("final-refit lock self-fingerprint failure")

    gate = _read_gate(original_gate_path)
    if [row["evidence"] for row in gate[:14]] != expected["r1_to_r14_gate_evidence"]:
        raise ReconciliationError("R1-R14 evidence drift")
    gate[14] = {
        "gate_id": "R15",
        "check": "structured targeted full pytest and engineering audits",
        "evidence": corrected_audit_physical,
        "status": "PASS",
    }
    gate[15] = {
        "gate_id": "R16",
        "check": "final-refit lock integrity after pytest audit reconciliation",
        "evidence": lock["lock_fingerprint"],
        "status": "PASS",
    }
    write_csv_atomic(config.path_for("gate"), gate, ["gate_id", "check", "evidence", "status"])

    with original_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != 1:
        raise ReconciliationError("resource recovery manifest row count drift")
    manifest_rows[0]["lock_fingerprint"] = lock["lock_fingerprint"]
    manifest_rows[0]["status"] = "STAGE_4B4_PHASE4_PYTEST_AUDIT_RECONCILED"
    write_csv_atomic(config.path_for("manifest"), manifest_rows, list(manifest_rows[0]))

    targeted = corrected_audit["tests"]["targeted"]["counts"]
    full = corrected_audit["tests"]["full"]["counts"]
    skip = reconciliation["skip_details"][0]
    targeted_command = " ".join(reconciliation["targeted_command"])
    full_command = " ".join(reconciliation["full_command"])
    original_report = original_report_path.read_text(encoding="utf-8")
    clarification = f"""

## 13. Pytest skip audit reconciliation

ChatGPT 独立验收发现上一轮自然语言汇报的 targeted/full skip 数与正式 audit 中保存的 pytest 进度片段不一致。本次核验确认根因为 `AUDIT_PARSER_ERROR`：旧 auditor 的 `safe_summary()` 只保留 stdout 最后一条非空行，不能作为结构化统计；上一轮 `1/2` 又混用了锁后另一批输出，并把 targeted 的 skip 误计入 full。

正式重跑命令通过 `scripts/stage_4b4_pytest_structured_worker.py` 调用 pytest，并固定使用 `-ra`。Targeted 收集 {targeted['collected']} 项，{targeted['passed']} passed，{targeted['skipped']} skipped，failed/errors/xfailed/xpassed 均为 0；Full 收集 {full['collected']} 项，{full['passed']} passed，{full['skipped']} skipped，failed/errors/xfailed/xpassed 均为 0。

Targeted command：`{targeted_command}`。

Full command：`{full_command}`。

两套 suite 唯一 skip 均为 `{skip['nodeid']}`，reason 为 `{skip['reason']}`，分类为 `EXPECTED_OBSOLETE_ARTIFACT_SKIP`。该测试只检查未生成的旧 `final_refit_contract_gate.csv`（P4G1–P4G13）；当前正式 gate 是 R1–R16，相关 required tests 已实际执行并通过，因此不会掩盖模型、binary、replay、隐私或 Test 防泄漏问题。

原 test/security audit physical fingerprint `{expected['original_test_audit_physical_fingerprint']}`、原 lock physical/semantic fingerprints `{expected['original_lock_physical_fingerprint']}` / `{expected['original_lock_semantic_fingerprint']}` 与原 gate physical fingerprint `{expected['original_gate_physical_fingerprint']}` 已作为 superseded evidence 保留。R1–R14 的语义与 evidence 字节保持不变；R15 现绑定 corrected structured test audit，R16 绑定新 lock `{lock['lock_fingerprint']}`。

本次未重训模型、未重建 binary/optimizer/combined matrix、未重新计算 prediction/replay、未打开 Validation/First7d/Test labels，也未读取或物化任何 Test artifact。Phase 4 已满足重新技术验收条件；pytest PASS 仅代表工程测试通过，不代表模型业务效果。
"""
    report_text = original_report.rstrip() + clarification
    report_path = config.path_for("report")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report_text, encoding="utf-8")
    os.replace(temporary_report, report_path)

    supersession: dict[str, Any] = {
        "phase": config.raw["phase"],
        "status": "PASS",
        "original_status": "PASS_WITH_NON_STRUCTURED_TRUNCATED_PYTEST_SUMMARY",
        "superseded_status": "SUPERSEDED_BY_STRUCTURED_PYTEST_AUDIT",
        "supersession_reason": "AUDIT_PARSER_ERROR",
        "original_physical_fingerprint": expected["original_test_audit_physical_fingerprint"],
        "corrected_physical_fingerprint": corrected_audit_physical,
        "original_lock_physical_fingerprint": expected["original_lock_physical_fingerprint"],
        "original_lock_semantic_fingerprint": expected["original_lock_semantic_fingerprint"],
        "current_lock_physical_fingerprint": file_fingerprint(config.path_for("lock")),
        "current_lock_semantic_fingerprint": lock["lock_fingerprint"],
        "original_gate_physical_fingerprint": expected["original_gate_physical_fingerprint"],
        "current_gate_physical_fingerprint": file_fingerprint(config.path_for("gate")),
        "original_report_physical_fingerprint": expected["original_report_physical_fingerprint"],
        "current_report_physical_fingerprint": file_fingerprint(config.path_for("report")),
        "r1_to_r14_evidence_unchanged": True,
        "r15_r16_rebound": True,
        "original_conflict_deleted": False,
    }
    supersession["supersession_fingerprint"] = self_fingerprint(
        supersession, "supersession_fingerprint"
    )
    write_json_atomic(config.path_for("supersession"), supersession)

    checkpoint = {
        "status": "SUCCESS",
        "artifact": "PYTEST_SKIP_AUDIT_RECONCILIATION",
        "reconciliation_semantic_fingerprint": reconciliation["reconciliation_semantic_fingerprint"],
        "corrected_test_audit_physical_fingerprint": corrected_audit_physical,
        "lock_fingerprint": lock["lock_fingerprint"],
        "gate_physical_fingerprint": file_fingerprint(config.path_for("gate")),
        "model_retrained": False,
        "binary_rebuilt": False,
        "combined_matrix_rebuilt": False,
        "validation_optimizer_rebuilt": False,
        "prediction_recomputed": False,
    }
    checkpoint["checkpoint_fingerprint"] = self_fingerprint(checkpoint, "checkpoint_fingerprint")
    checkpoint_path = config.path_for("checkpoint_directory") / "reconciliation_complete.json"
    write_json_atomic(checkpoint_path, checkpoint)
    return {
        "status": "STAGE_4B4_PHASE4_PYTEST_AUDIT_RECONCILED",
        "root_cause": "AUDIT_PARSER_ERROR",
        "targeted": targeted,
        "full": full,
        "skip_details": reconciliation["skip_details"],
        "original_test_audit": "SUPERSEDED",
        "original_final_refit_lock": "SUPERSEDED",
        "current_lock_fingerprint": lock["lock_fingerprint"],
        "gate_passed": 16,
        "gate_total": 16,
    }
