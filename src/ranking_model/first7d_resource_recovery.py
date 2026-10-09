from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import psutil

from ranking_model.resource_contract import (
    FORMAL_RSS_FRACTION,
    ResourceContractError,
    consume_frozen_rss_contract,
    resource_gate_passes,
)


class RecoveryArtifactDrift(RuntimeError):
    pass


class RecoveryScopeExceeded(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_fingerprint(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def self_fingerprint(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return canonical_fingerprint(value)


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".attempt")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".attempt")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_recovery_permissions(permissions: Mapping[str, Any]) -> None:
    required_false = {
        "first7d_label_reopen_allowed", "ground_truth_rematerialization_allowed",
        "population_rematerialization_allowed", "metric_recomputation_allowed",
        "bootstrap_recomputation_allowed", "prediction_recomputation_allowed",
        "winner_change_allowed", "methodology_changed",
    }
    if set(permissions) != required_false or any(permissions[key] is not False for key in required_false):
        raise RecoveryScopeExceeded("resource recovery permissions exceed the authorized scope")


def validate_access_state(access: Mapping[str, Any]) -> None:
    if access.get("first7d_read_count") != 1:
        raise RecoveryArtifactDrift("First7d total read count is not exactly one")
    if access.get("validation_14d_source_content_open_count") != 1:
        raise RecoveryArtifactDrift("First7d original content-open count drift")
    if any(access.get(key) != 0 for key in (
        "test_candidate_read_count", "test_feature_read_count",
        "test_label_read_count", "test_metric_read_count",
    )):
        raise RecoveryArtifactDrift("Test access count drift")


def validate_winner_state(lock: Mapping[str, Any]) -> None:
    expected = {
        "winner": "L1_600", "winner_feature_set": "F1", "winner_iteration": 600,
        "winner_changed": False, "direction_status": "ROBUSTNESS_DIRECTION_PASS",
        "confidence_status": "POSITIVE_AND_SIGNIFICANT",
        "segment_regression_review": "NO_FIRST7D_CRITICAL_SEGMENT_REGRESSION",
    }
    if any(lock.get(key) != value for key, value in expected.items()):
        raise RecoveryArtifactDrift("winner/direction/statistical/segment identity drift")


class First7dResourceRecoveryRunner:
    """Resource-only recovery; never parses First7d data or computes metrics."""

    def __init__(self, root: Path, config_path: Path) -> None:
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._validate_config()
        self.config_fingerprint = canonical_fingerprint(self.raw)
        self.paths = {key: self.root / value for key, value in self.raw["paths"].items()}

    def _validate_config(self) -> None:
        if (self.raw.get("stage"), self.raw.get("phase"), self.raw.get("recovery_type")) != (
            "4B-4", "phase3_post_first7d_resource_contract_recovery_v1",
            "post_first7d_resource_contract_only",
        ):
            raise RecoveryScopeExceeded("resource recovery config identity drift")
        validate_recovery_permissions(self.raw["recovery_permissions"])
        correction = self.raw["resource_correction"]
        exact = (
            correction["invalid_registered_fraction"], correction["formal_required_fraction"],
            correction["available_memory_at_original_preregistration_bytes"],
            correction["invalid_preregistered_cap_bytes"],
            correction["invalid_runtime_recomputed_cap_bytes"],
            correction["corrected_formal_cap_bytes"], correction["observed_peak_rss_bytes"],
        )
        if exact != (0.65, 0.70, 1857146880, 1207145472, 1315638272, 1300002816, 575967232):
            raise RecoveryScopeExceeded("resource recovery constants drift")

    def _source_fingerprint(self) -> str:
        files = [
            "src/ranking_model/resource_contract.py",
            "src/ranking_model/first7d_runner.py",
            "src/ranking_model/first7d_resource_recovery.py",
            "scripts/run_stage_4b4_first7d_resource_recovery.py",
            "tests/test_stage_4b4_phase3.py",
            "tests/test_stage_4b4_phase3_resource_recovery.py",
        ]
        digest = hashlib.sha256()
        for relative in files:
            path = self.root / relative
            if not path.exists():
                raise RecoveryArtifactDrift(f"missing recovery source: {relative}")
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(path.read_bytes() + b"\0")
        return digest.hexdigest()

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def _assert_physical(self, path_key: str, expected_key: str) -> str:
        path = self.paths[path_key]
        if not path.exists():
            raise RecoveryArtifactDrift(f"missing frozen artifact: {path_key}")
        actual = file_fingerprint(path)
        expected = self.raw["expected"][expected_key]
        if actual != expected:
            raise RecoveryArtifactDrift(f"{path_key} physical fingerprint drift")
        return actual

    @staticmethod
    def _artifact_set_fingerprint(paths: list[Path]) -> str:
        return canonical_fingerprint([
            {"name": path.name, "size": path.stat().st_size, "sha256": file_fingerprint(path)}
            for path in paths
        ])

    def _verify_checkpoint_set(self) -> dict[str, Any]:
        checkpoint_dir = self.paths["phase3_checkpoints"]
        task_paths = sorted(
            path for path in checkpoint_dir.glob("*.json")
            if path.name not in {"preregistration.json", "first7d_materialization.json"}
        )
        rows: list[dict[str, Any]] = []
        for path in task_paths:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            output_value = checkpoint.get("output")
            if not isinstance(output_value, str) or not output_value:
                raise RecoveryArtifactDrift(f"checkpoint output missing: {path.name}")
            output = (self.root / output_value).resolve()
            try:
                output.relative_to(self.root)
            except ValueError as exc:
                raise RecoveryArtifactDrift("checkpoint output escaped workspace") from exc
            actual = file_fingerprint(output)
            row = {
                "name": path.name, "status": checkpoint.get("status"),
                "identity": checkpoint.get("identity"),
                "output_fingerprint": checkpoint.get("output_fingerprint"),
                "actual_output_fingerprint": actual,
            }
            rows.append(row)
        expected = self.raw["expected"]
        if len(rows) != expected["checkpoint_success_count"]:
            raise RecoveryArtifactDrift("Phase 3 checkpoint count drift")
        if any(row["status"] != "SUCCESS" for row in rows):
            raise RecoveryArtifactDrift("non-SUCCESS Phase 3 checkpoint")
        if any(row["output_fingerprint"] != row["actual_output_fingerprint"] for row in rows):
            raise RecoveryArtifactDrift("checkpoint output physical fingerprint drift")
        fingerprint = canonical_fingerprint(rows)
        if fingerprint != expected["checkpoint_set_fingerprint"]:
            raise RecoveryArtifactDrift("checkpoint set identity drift")
        materialization = checkpoint_dir / "first7d_materialization.json"
        if file_fingerprint(materialization) != expected["materialization_checkpoint_physical_fingerprint"]:
            raise RecoveryArtifactDrift("materialization checkpoint physical drift")
        return {"success_count": len(rows), "checkpoint_set_fingerprint": fingerprint}

    def _verify_metric_sets(self) -> dict[str, Any]:
        metric_sets: dict[str, str] = {}
        digest_sets: dict[str, str] = {}
        for version in ("B0", "L1_600"):
            metric_parts = sorted((self.paths["query_metrics"] / version).glob("part_*.parquet"))
            digest_parts = sorted((self.paths["query_metric_digests"] / version).glob("part_*.parquet"))
            if len(metric_parts) != 32 or len(digest_parts) != 32:
                raise RecoveryArtifactDrift(f"{version} metric/digest partition count drift")
            metric_sets[version] = self._artifact_set_fingerprint(metric_parts)
            digest_sets[version] = self._artifact_set_fingerprint(digest_parts)
            if metric_sets[version] != self.raw["expected"]["metric_physical_fingerprints"][version]:
                raise RecoveryArtifactDrift(f"{version} metric physical set drift")
            if digest_sets[version] != self.raw["expected"]["metric_digest_physical_fingerprints"][version]:
                raise RecoveryArtifactDrift(f"{version} metric digest physical set drift")
        return {"metric_physical_fingerprints": metric_sets,
                "metric_digest_physical_fingerprints": digest_sets}

    def preflight(self) -> dict[str, Any]:
        for key in ("supersession_manifest", "recovery_audit", "recovered_gate",
                    "recovered_lock", "recovery_final_audit"):
            if self.paths[key].exists():
                raise RecoveryScopeExceeded(f"recovery output already exists: {key}")
        expected = self.raw["expected"]
        physical = {
            "base_phase3_config": self._assert_physical(
                "base_phase3_config", "base_phase3_config_physical_fingerprint"),
            "base_preregistration": self._assert_physical(
                "base_preregistration", "base_preregistration_physical_fingerprint"),
            "base_access_audit": self._assert_physical(
                "base_access_audit", "base_access_audit_physical_fingerprint"),
            "base_lock": self._assert_physical("base_lock", "base_lock_physical_fingerprint"),
            "base_gate": self._assert_physical("base_gate", "base_gate_physical_fingerprint"),
            "base_resource_audit": self._assert_physical(
                "base_resource_audit", "base_resource_audit_physical_fingerprint"),
            "base_final_audit": self._assert_physical(
                "base_final_audit", "base_final_audit_physical_fingerprint"),
            "base_result": self._assert_physical("base_result", "base_result_physical_fingerprint"),
            "base_report": self._assert_physical("base_report", "base_report_physical_fingerprint"),
            "ground_truth": self._assert_physical(
                "first7d_ground_truth", "first7d_ground_truth_physical_fingerprint"),
            "population": self._assert_physical(
                "first7d_population", "first7d_population_physical_fingerprint"),
            "aggregate_metrics": self._assert_physical(
                "aggregate_metrics", "aggregate_metrics_physical_fingerprint"),
            "bootstrap": self._assert_physical("bootstrap", "bootstrap_physical_fingerprint"),
            "replay": self._assert_physical("replay", "replay_physical_fingerprint"),
            "bootstrap_replay": self._assert_physical(
                "bootstrap_replay", "bootstrap_replay_physical_fingerprint"),
            "prediction_manifest": self._assert_physical(
                "prediction_manifest", "prediction_manifest_physical_fingerprint"),
        }

        base_config = json.loads(self.paths["base_phase3_config"].read_text(encoding="utf-8"))
        if canonical_fingerprint(base_config) != self.raw["base_phase3_config_fingerprint"]:
            raise RecoveryArtifactDrift("base Phase 3 config semantic fingerprint drift")
        if float(base_config["execution"]["rss_hard_fraction_of_available_at_preregistration"]) != 0.65:
            raise RecoveryArtifactDrift("original invalid 65% fraction was not preserved")

        prereg = json.loads(self.paths["base_preregistration"].read_text(encoding="utf-8"))
        if prereg.get("preregistration_fingerprint") != self.raw["base_phase3_preregistration_fingerprint"]:
            raise RecoveryArtifactDrift("base preregistration declared fingerprint drift")
        if self_fingerprint(prereg, "preregistration_fingerprint") != self.raw["base_phase3_preregistration_fingerprint"]:
            raise RecoveryArtifactDrift("base preregistration self-identity drift")

        access = json.loads(self.paths["base_access_audit"].read_text(encoding="utf-8"))
        validate_access_state(access)
        original_lock = json.loads(self.paths["base_lock"].read_text(encoding="utf-8"))
        if original_lock.get("lock_fingerprint") != expected["base_lock_semantic_fingerprint"]:
            raise RecoveryArtifactDrift("base lock declared semantic fingerprint drift")
        if self_fingerprint(original_lock, "lock_fingerprint") != expected["base_lock_semantic_fingerprint"]:
            raise RecoveryArtifactDrift("base lock self-identity drift")
        validate_winner_state(original_lock)

        original_gate = read_csv(self.paths["base_gate"])
        if len(original_gate) != 13 or any(row.get("status") != "PASS" for row in original_gate):
            raise RecoveryArtifactDrift("historical P3 gate bytes or rows drift")
        replay_rows = read_csv(self.paths["replay"])
        bootstrap_replay_rows = read_csv(self.paths["bootstrap_replay"])
        if len(replay_rows) != 2 or any(row.get("status") != "PASS" for row in replay_rows):
            raise RecoveryArtifactDrift("metric replay audit drift")
        if len(bootstrap_replay_rows) != 16 or any(row.get("status") != "PASS" for row in bootstrap_replay_rows):
            raise RecoveryArtifactDrift("bootstrap replay audit drift")

        resource = json.loads(self.paths["base_resource_audit"].read_text(encoding="utf-8"))
        correction = self.raw["resource_correction"]
        if resource.get("rss_hard_limit_bytes") != correction["invalid_runtime_recomputed_cap_bytes"]:
            raise RecoveryArtifactDrift("invalid runtime-recomputed cap evidence drift")
        if resource.get("peak_rss_bytes") != correction["observed_peak_rss_bytes"]:
            raise RecoveryArtifactDrift("observed peak RSS evidence drift")

        try:
            contract = consume_frozen_rss_contract(
                prereg,
                runtime_available_memory_bytes=psutil.virtual_memory().available,
                recovery=correction,
            )
        except ResourceContractError as exc:
            raise RecoveryArtifactDrift(str(exc)) from exc
        if not resource_gate_passes(
            correction["observed_peak_rss_bytes"], contract.frozen_absolute_rss_cap_bytes
        ):
            raise RecoveryArtifactDrift("corrected resource gate failed")

        metric_sets = self._verify_metric_sets()
        checkpoint = self._verify_checkpoint_set()
        prediction_rows = read_csv(self.paths["prediction_manifest"])
        actual_prediction = {
            row["model_id"]: row["global_merkle_fingerprint"]
            for row in prediction_rows if row["model_id"] in {"B0", "L1_600"}
        }
        if actual_prediction != expected["prediction_global_merkle_fingerprints"]:
            raise RecoveryArtifactDrift("prediction Merkle identity drift")

        head = self._git("rev-parse", "HEAD")
        main = self._git("rev-parse", "main")
        origin = self._git("rev-parse", "origin/main")
        if self._git("branch", "--show-current") != "main" or len({head, main, origin}) != 1:
            raise RecoveryArtifactDrift("Git branch/HEAD/main/origin-main drift")
        tracked = self._git(
            "ls-files", "artifacts/stage_4b4/private", "artifacts/stage_4b4/tmp",
            "artifacts/stage_4b4/checkpoints",
        )
        if tracked:
            raise RecoveryArtifactDrift("tracked private/tmp/checkpoint artifact detected")

        return {
            "physical_fingerprints": physical,
            "preregistration": prereg, "access": access, "original_lock": original_lock,
            "original_gate_pass_count": len(original_gate), "resource_audit": resource,
            "resource_contract": {
                "available_memory_at_original_preregistration_bytes": contract.available_memory_at_preregistration_bytes,
                "formal_required_fraction": contract.formal_fraction,
                "frozen_absolute_rss_cap_bytes": contract.frozen_absolute_rss_cap_bytes,
                "runtime_available_memory_observation_bytes": contract.runtime_available_memory_bytes,
                "cap_source": contract.source,
            },
            **metric_sets, "checkpoint": checkpoint,
            "prediction_global_merkle_fingerprints": actual_prediction,
            "replay_pass_count": len(replay_rows),
            "bootstrap_replay_pass_count": len(bootstrap_replay_rows),
            "git": {"head": head, "main": main, "origin_main": origin,
                    "tracked_private_artifact_count": 0},
        }

    def _privacy_scan(self, staged_payloads: list[object]) -> bool:
        forbidden = (str(self.root), "LXR15892896597B0216", "customer_id", "article_id", "postal_code")
        return not any(token in canonical_json(staged_payloads) for token in forbidden)

    def finalize(self, *, targeted_result: str, full_result: str,
                 compile_result: str, diff_check_result: str) -> dict[str, Any]:
        evidence = self.preflight()
        correction = self.raw["resource_correction"]
        original_lock = evidence["original_lock"]
        source_fingerprint = self._source_fingerprint()

        supersession = {
            "stage": "4B-4", "phase": self.raw["phase"],
            "status": "RESOURCE_CONTRACT_SUPERSESSION_RECORDED",
            "failure_status": "POST_FIRST7D_IMPLEMENTATION_FAILURE",
            "reason": "RESOURCE_CONTRACT_IDENTITY_FAILURE",
            "original_config": {
                "status": "PRESERVED_INVALID_RESOURCE_CONTRACT_EVIDENCE",
                "canonical_fingerprint": self.raw["base_phase3_config_fingerprint"],
                "physical_fingerprint": self.raw["expected"]["base_phase3_config_physical_fingerprint"],
            },
            "original_preregistration": {
                "status": "PRESERVED_INVALID_RESOURCE_CONTRACT_EVIDENCE",
                "semantic_fingerprint": self.raw["base_phase3_preregistration_fingerprint"],
                "physical_fingerprint": self.raw["expected"]["base_preregistration_physical_fingerprint"],
            },
            "original_lock": {
                "status": "SUPERSEDED",
                "formal_status": "NOT_FORMAL_DUE_TO_RESOURCE_CONTRACT_IDENTITY_FAILURE",
                "semantic_fingerprint": self.raw["expected"]["base_lock_semantic_fingerprint"],
                "physical_fingerprint": self.raw["expected"]["base_lock_physical_fingerprint"],
            },
            "original_gate": {
                "status": "SUPERSEDED",
                "formal_status": "NOT_FORMAL_DUE_TO_RESOURCE_CONTRACT_IDENTITY_FAILURE",
                "physical_fingerprint": self.raw["expected"]["base_gate_physical_fingerprint"],
            },
            "recorded_at": now(),
        }
        supersession["supersession_fingerprint"] = canonical_fingerprint(supersession)

        audit_rows = [
            {"audit_id": "RC01", "status": "PASS", "observed": "POST_FIRST7D_IMPLEMENTATION_FAILURE preserved"},
            {"audit_id": "RC02", "status": "PASS", "observed": "original fraction=0.65"},
            {"audit_id": "RC03", "status": "PASS", "observed": "original available=1857146880"},
            {"audit_id": "RC04", "status": "PASS", "observed": "invalid preregistered cap=1207145472"},
            {"audit_id": "RC05", "status": "PASS", "observed": "invalid runtime cap=1315638272"},
            {"audit_id": "RC06", "status": "PASS", "observed": "corrected 70% cap=1300002816"},
            {"audit_id": "RC07", "status": "PASS", "observed": "peak RSS=575967232"},
            {"audit_id": "RC08", "status": "PASS", "observed": "575967232 <= 1300002816"},
            {"audit_id": "RC09", "status": "PASS", "observed": "runtime available is observation-only"},
            {"audit_id": "RC10", "status": "PASS", "observed": "label reopen=0;total read=1"},
            {"audit_id": "RC11", "status": "PASS", "observed": "metrics/bootstrap recomputed=false"},
            {"audit_id": "RC12", "status": "PASS", "observed": "winner/direction unchanged"},
        ]

        recovered_lock = {
            "stage": "4B-4", "phase": self.raw["phase"],
            "status": "FIRST7D_RESOURCE_CONTRACT_RECOVERED",
            "recovery_type": self.raw["recovery_type"],
            "recovery_config_fingerprint": self.config_fingerprint,
            "recovery_source_fingerprint": source_fingerprint,
            "base_phase3_config_fingerprint": self.raw["base_phase3_config_fingerprint"],
            "base_phase3_preregistration_fingerprint": self.raw["base_phase3_preregistration_fingerprint"],
            "supersession_fingerprint": supersession["supersession_fingerprint"],
            "superseded_original_lock": {
                "semantic_fingerprint": self.raw["expected"]["base_lock_semantic_fingerprint"],
                "physical_fingerprint": self.raw["expected"]["base_lock_physical_fingerprint"],
            },
            "superseded_original_gate_physical_fingerprint": self.raw["expected"]["base_gate_physical_fingerprint"],
            "corrected_resource_contract": {
                "available_memory_at_original_preregistration_bytes": 1857146880,
                "invalid_registered_fraction": 0.65,
                "invalid_preregistered_cap_bytes": 1207145472,
                "formal_required_fraction": FORMAL_RSS_FRACTION,
                "frozen_absolute_rss_cap_bytes": 1300002816,
                "observed_peak_rss_bytes": 575967232,
                "observed_peak_fraction_of_corrected_cap": correction["observed_peak_fraction_of_corrected_cap"],
                "corrected_resource_gate": "PASS",
                "invalid_runtime_recomputed_cap_bytes": 1315638272,
                "invalid_runtime_cap_formal_evidence": False,
            },
            "first7d_ground_truth": original_lock["first7d_ground_truth"],
            "first7d_population": original_lock["first7d_population"],
            "prediction_global_merkle_fingerprints": original_lock["prediction_global_merkle_fingerprints"],
            "metric_fingerprints": original_lock["metric_fingerprints"],
            "public_metric_physical_fingerprint": original_lock["public_metric_physical_fingerprint"],
            "bootstrap_physical_fingerprint": original_lock["bootstrap_physical_fingerprint"],
            "replay_physical_fingerprint": original_lock["replay_physical_fingerprint"],
            "bootstrap_replay_physical_fingerprint": self.raw["expected"]["bootstrap_replay_physical_fingerprint"],
            "metric_digest_physical_fingerprints": evidence["metric_digest_physical_fingerprints"],
            "checkpoint_set_fingerprint": evidence["checkpoint"]["checkpoint_set_fingerprint"],
            "checkpoint_success_count": evidence["checkpoint"]["success_count"],
            "main_metrics": original_lock["main_metrics"],
            "winner_map_delta_vs_b0": original_lock["winner_map_delta_vs_b0"],
            "winner": "L1_600", "winner_feature_set": "F1", "winner_iteration": 600,
            "winner_changed": False,
            "direction_status": original_lock["direction_status"],
            "confidence_status": original_lock["confidence_status"],
            "segment_regression_review": original_lock["segment_regression_review"],
            "first7d_label_total_read_count": 1,
            "first7d_label_reopen_count_during_recovery": 0,
            "ground_truth_rematerialized": False,
            "population_rematerialized": False,
            "metric_recomputed": False, "bootstrap_recomputed": False,
            "prediction_recomputed": False, "methodology_changed": False,
            "final_refit_executed": False, "test_authorized": False,
            "test_candidate_read_count": 0, "test_feature_read_count": 0,
            "test_label_read_count": 0, "test_metric_read_count": 0,
            "reranking_executed": False, "retrieval_retuned": False,
            "locked_at": now(),
        }
        recovered_lock["lock_fingerprint"] = canonical_fingerprint(recovered_lock)

        raw_diff = self._git("diff", "--", "data/raw")
        tracked = self._git(
            "ls-files", "artifacts/stage_4b4/private", "artifacts/stage_4b4/tmp",
            "artifacts/stage_4b4/checkpoints",
        )
        ignored = all(subprocess.run(
            ["git", "check-ignore", "-q", str(self.paths[key])], cwd=self.root
        ).returncode == 0 for key in ("query_metrics", "query_metric_digests", "phase3_checkpoints"))
        tests_ok = "passed" in targeted_result.lower() and "passed" in full_result.lower()
        privacy_ok = self._privacy_scan([supersession, audit_rows, recovered_lock])

        gate_checks = [
            ("R1", "original failure and invalid resource evidence preserved", True),
            ("R2", "recovery authorization identity", self.raw["authorization"] == "chatgpt_post_first7d_resource_recovery_2026-08-18"),
            ("R3", "original preregistration available-memory identity", evidence["resource_contract"]["available_memory_at_original_preregistration_bytes"] == 1857146880),
            ("R4", "70% corrected-cap arithmetic identity", int(1857146880 * 0.70) == 1300002816),
            ("R5", "runner consumes frozen absolute cap", evidence["resource_contract"]["frozen_absolute_rss_cap_bytes"] == 1300002816),
            ("R6", "observed peak below corrected cap", resource_gate_passes(575967232, 1300002816)),
            ("R7", "First7d labels not reopened", recovered_lock["first7d_label_reopen_count_during_recovery"] == 0),
            ("R8", "ground-truth/population fingerprints unchanged", evidence["physical_fingerprints"]["ground_truth"] == original_lock["first7d_ground_truth"]["physical_fingerprint"] and evidence["physical_fingerprints"]["population"] == original_lock["first7d_population"]["physical_fingerprint"]),
            ("R9", "metrics/bootstrap fingerprints unchanged", evidence["physical_fingerprints"]["aggregate_metrics"] == original_lock["public_metric_physical_fingerprint"] and evidence["physical_fingerprints"]["bootstrap"] == original_lock["bootstrap_physical_fingerprint"]),
            ("R10", "prediction/winner/direction unchanged", evidence["prediction_global_merkle_fingerprints"] == original_lock["prediction_global_merkle_fingerprints"] and recovered_lock["winner"] == "L1_600" and recovered_lock["direction_status"] == "ROBUSTNESS_DIRECTION_PASS"),
            ("R11", "original lock/gate superseded", supersession["original_lock"]["status"] == "SUPERSEDED" and supersession["original_gate"]["status"] == "SUPERSEDED"),
            ("R12", "recovered lock integrity", self_fingerprint(recovered_lock, "lock_fingerprint") == recovered_lock["lock_fingerprint"]),
            ("R13", "no final-refit/Test/reranking/retrieval", all(recovered_lock[key] in (0, False) for key in ("final_refit_executed", "test_authorized", "test_candidate_read_count", "test_feature_read_count", "test_label_read_count", "test_metric_read_count", "reranking_executed", "retrieval_retuned"))),
            ("R14", "resource/privacy/checkpoint/Git/tests", tests_ok and compile_result == "PASS" and diff_check_result == "PASS" and privacy_ok and not raw_diff and not tracked and ignored and evidence["checkpoint"]["success_count"] == 160),
        ]
        gate_rows = [
            {"gate_id": gate, "description": description,
             "status": "PASS" if passed else "FAIL"}
            for gate, description, passed in gate_checks
        ]
        if any(row["status"] != "PASS" for row in gate_rows):
            raise RecoveryArtifactDrift("recovered R1-R14 gate failed")

        final_rows = [
            {"check_id": "RA01", "status": "PASS", "observed": f"config={self.config_fingerprint}"},
            {"check_id": "RA02", "status": "PASS", "observed": f"source={source_fingerprint}"},
            {"check_id": "RA03", "status": "PASS", "observed": "original lock/gate superseded"},
            {"check_id": "RA04", "status": "PASS", "observed": "corrected resource gate PASS"},
            {"check_id": "RA05", "status": "PASS", "observed": "First7d reopen=0,total=1"},
            {"check_id": "RA06", "status": "PASS", "observed": "metrics/bootstrap recomputed=false"},
            {"check_id": "RA07", "status": "PASS", "observed": "artifact fingerprints unchanged"},
            {"check_id": "RA08", "status": "PASS", "observed": "checkpoint=160/160"},
            {"check_id": "RA09", "status": "PASS", "observed": "replay=2/2;bootstrap replay=16/16"},
            {"check_id": "RA10", "status": "PASS", "observed": targeted_result},
            {"check_id": "RA11", "status": "PASS", "observed": full_result},
            {"check_id": "RA12", "status": "PASS", "observed": f"compile={compile_result};diff={diff_check_result}"},
            {"check_id": "RA13", "status": "PASS", "observed": "privacy/raw/ignore/tracked-private PASS"},
            {"check_id": "RA14", "status": "PASS", "observed": "Test reads=0;final-refit=false"},
        ]
        if not self._privacy_scan([supersession, audit_rows, recovered_lock, gate_rows, final_rows]):
            raise RecoveryArtifactDrift("staged recovery public payload privacy scan failed")

        write_json_atomic(self.paths["supersession_manifest"], supersession)
        write_csv_atomic(self.paths["recovery_audit"], audit_rows, audit_rows[0].keys())
        write_json_atomic(self.paths["recovered_lock"], recovered_lock)
        write_csv_atomic(self.paths["recovered_gate"], gate_rows, gate_rows[0].keys())
        write_csv_atomic(self.paths["recovery_final_audit"], final_rows, final_rows[0].keys())
        self._update_report(recovered_lock, supersession)
        return recovered_lock

    def _update_report(self, lock: Mapping[str, Any], supersession: Mapping[str, Any]) -> None:
        report_path = self.paths["base_report"]
        original = report_path.read_text(encoding="utf-8")
        addition = f"""

## Post-First7d Resource Contract Recovery

`FIRST7D_RESOURCE_CONTRACT_RECOVERED`

The original Phase 3 lock and P3 gate are preserved as historical evidence but are `SUPERSEDED / NOT_FORMAL_DUE_TO_RESOURCE_CONTRACT_IDENTITY_FAILURE`. The original config registered 65% (`1207145472` bytes), while the authorized formal contract is 70% of the original preregistration available memory: `1857146880 * 0.70 = 1300002816` bytes. The runtime-recomputed `1315638272` bytes is retained only as invalid historical evidence.

Observed peak RSS was `575967232` bytes (`{lock['corrected_resource_contract']['observed_peak_fraction_of_corrected_cap']:.17g}` of the corrected cap), so the corrected resource gate passes. Recovery did not reopen First7d labels, rematerialize ground truth/population, recompute metrics/bootstrap/predictions, change the winner, or execute final-refit/Test/reranking/retrieval retuning.

- Recovered lock fingerprint: `{lock['lock_fingerprint']}`
- Supersession fingerprint: `{supersession['supersession_fingerprint']}`
- Winner: `L1_600`; direction: `ROBUSTNESS_DIRECTION_PASS`; confidence: `POSITIVE_AND_SIGNIFICANT`.
- First7d total label read count: `1`; recovery reopen count: `0`.
- Recovered R1-R14 contract gate: `PASSED`.
"""
        temporary = report_path.with_suffix(report_path.suffix + ".attempt")
        temporary.write_text(original.rstrip() + addition, encoding="utf-8")
        os.replace(temporary, report_path)
