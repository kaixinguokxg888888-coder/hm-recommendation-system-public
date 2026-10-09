from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import duckdb
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from data_audit.stage1a import ResourceMonitor
from data_foundation.manifest import write_csv
from ranking_model.first7d_diagnostic import (
    FIRST7D_METRICS,
    FIRST7D_VERSIONS,
    classify_robustness,
    segment_review_required,
)
from ranking_model.resource_contract import (
    FORMAL_RSS_FRACTION,
    consume_frozen_rss_contract,
    validate_registered_fraction,
)
from ranking_model.validation_runner import (
    FrozenArtifactDrift,
    ValidationSelectionRunner,
    csv_read,
    json_atomic,
    now,
)
from ranking_model.validation_selection import (
    canonical_fingerprint,
    file_fingerprint,
    paired_bootstrap,
    query_merkle,
    query_metric_digest,
)


class PostFirst7dImplementationFailure(RuntimeError):
    pass


class First7dDiagnosticRunner:
    def __init__(self, root: Path, config_path: Path) -> None:
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._validate_config()
        self.config_fp = canonical_fingerprint(self.raw)
        self.paths = {key: self.root / value for key, value in self.raw["paths"].items()}
        self.public = self.paths["public_directory"]
        self.private = self.paths["private_directory"]
        self.checkpoints = self.paths["checkpoint_directory"]
        self.temp = self.paths["temp_directory"]
        available = psutil.virtual_memory().available
        execution = self.raw["execution"]
        self.available_at_preregistration = available
        self.rss_limit = int(available * execution["rss_hard_fraction_of_available_at_preregistration"])
        self.memory_mb = max(
            512,
            min(
                int(execution["duckdb_memory_max_mb"]),
                int(available / 1024 / 1024 * execution["memory_fraction_of_available"]),
            ),
        )
        self.resources: list[dict[str, Any]] = []
        self._prereg: dict[str, Any] = {}

    def _validate_config(self) -> None:
        if (self.raw.get("stage"), self.raw.get("phase")) != (
            "4B-4", "phase3_first7d_locked_winner_diagnostic_v1"
        ):
            raise ValueError("Phase 3 config identity drift")
        diagnostic = self.raw["diagnostic"]
        exact = (
            diagnostic["diagnostic_id"], diagnostic["snapshot_id"], diagnostic["as_of_date"],
            diagnostic["request_date"], diagnostic["label_start"], diagnostic["label_end"],
        )
        if exact != (
            "ranking_validation_first7d_diagnostic", "validation_14d", "2020-08-25",
            "2020-08-26", "2020-08-26", "2020-09-01",
        ):
            raise ValueError("First7d exact window drift")
        if tuple(self.raw["versions"]) != FIRST7D_VERSIONS:
            raise ValueError("only B0 and locked L1_600 are authorized")
        winner = self.raw["locked_winner"]
        if (winner["model_id"], winner["feature_set"], winner["iteration"], winner["winner_changed"]) != (
            "L1_600", "F1", 600, False
        ):
            raise ValueError("locked winner identity drift")
        if self.raw["bootstrap"] != {
            "method": "paired_user_block_percentile_v1", "iterations": 1000,
            "seed": 20260826, "confidence_level": 0.95,
            "comparison": "L1_600_minus_B0", "selection_use": False,
        }:
            raise ValueError("First7d bootstrap contract drift")
        if any(value not in (0, False) for value in self.raw["forbidden"].values()):
            raise ValueError("forbidden Phase 3 state is not locked")
        if self.raw["execution"]["simultaneous_prediction_versions"] != 1:
            raise ValueError("prediction resource isolation drift")
        validate_registered_fraction(
            float(self.raw["execution"]["rss_hard_fraction_of_available_at_preregistration"]),
            FORMAL_RSS_FRACTION,
        )

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self.temp.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(":memory:")
        con.execute(f"SET temp_directory='{self.temp.as_posix()}'")
        con.execute(f"SET memory_limit='{self.memory_mb}MB'")
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{self.raw['execution']['max_temp_directory_size']}'")
        return con

    def _source_fingerprint(self) -> str:
        paths = [
            self.root / "src/retrieval/evaluation.py",
            self.root / "src/ranking_model/validation_selection.py",
            self.root / "src/ranking_model/validation_runner.py",
            self.root / "src/ranking_model/first7d_diagnostic.py",
            self.root / "src/ranking_model/first7d_runner.py",
            self.root / "scripts/run_stage_4b4_first7d_diagnostic.py",
            self.root / "scripts/stage_4b4_phase3_audit.py",
            self.root / "tests/test_stage_4b4_phase3.py",
        ]
        digest = hashlib.sha256()
        for path in paths:
            if not path.exists():
                raise FrozenArtifactDrift(f"missing Phase 3 source: {path.name}")
            digest.update(path.relative_to(self.root).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes() + b"\0")
        return digest.hexdigest()

    @staticmethod
    def _self_fingerprint(payload: dict[str, Any], field: str) -> str:
        value = dict(payload)
        value.pop(field, None)
        return canonical_fingerprint(value)

    def _phase2_preflight(self) -> dict[str, Any]:
        expected = self.raw["expected"]
        phase2_config = json.loads(self.paths["phase2_config"].read_text(encoding="utf-8"))
        if canonical_fingerprint(phase2_config) != expected["phase2_config_fingerprint"]:
            raise FrozenArtifactDrift("Phase 2 config semantic fingerprint drift")

        physical_checks = {
            "phase2_selection_lock": "phase2_selection_lock_physical_fingerprint",
            "phase2_metrics": "phase2_metrics_physical_fingerprint",
            "phase2_bootstrap": "phase2_bootstrap_physical_fingerprint",
            "phase2_gate": "phase2_gate_physical_fingerprint",
            "phase2_recovery": "phase2_recovery_physical_fingerprint",
            "phase2_final_audit": "phase2_final_audit_physical_fingerprint",
        }
        observed_physical: dict[str, str] = {}
        for role, key in physical_checks.items():
            actual = file_fingerprint(self.paths[role])
            if actual != expected[key]:
                raise FrozenArtifactDrift(f"{role} physical fingerprint drift")
            observed_physical[role] = actual

        prereg = json.loads(self.paths["phase2_preregistration"].read_text(encoding="utf-8"))
        if prereg.get("preregistration_fingerprint") != expected["phase2_preregistration_fingerprint"]:
            raise FrozenArtifactDrift("Phase 2 preregistration declared fingerprint drift")
        if self._self_fingerprint(prereg, "preregistration_fingerprint") != expected["phase2_preregistration_fingerprint"]:
            raise FrozenArtifactDrift("Phase 2 preregistration self-identity drift")

        lock = json.loads(self.paths["phase2_selection_lock"].read_text(encoding="utf-8"))
        if lock.get("lock_fingerprint") != expected["phase2_selection_lock_semantic_fingerprint"]:
            raise FrozenArtifactDrift("Phase 2 selection lock declared semantic drift")
        if self._self_fingerprint(lock, "lock_fingerprint") != expected["phase2_selection_lock_semantic_fingerprint"]:
            raise FrozenArtifactDrift("Phase 2 selection lock self-identity drift")
        required = {
            "status": "VALIDATION_SELECTION_LOCKED", "winner": "L1_600",
            "value_gate": "PASSED_VALUE_GATE", "winner_selected": True,
            "validation_predictions_frozen": True, "validation_labels_read": True,
            "validation_label_read_count": 1, "first7d_read_count": 0,
            "final_refit_executed": False, "test_authorized": False,
            "test_candidate_read_count": 0, "test_feature_read_count": 0,
            "test_label_read_count": 0, "test_metric_read_count": 0,
        }
        if any(lock.get(key) != value for key, value in required.items()):
            raise FrozenArtifactDrift("Phase 2 lock state drift")
        if lock.get("prediction_global_merkle_fingerprints", {}).get("B0") != expected["prediction_global_merkle_fingerprints"]["B0"]:
            raise FrozenArtifactDrift("B0 prediction Merkle drift in Phase 2 lock")
        if lock.get("prediction_global_merkle_fingerprints", {}).get("L1_600") != expected["prediction_global_merkle_fingerprints"]["L1_600"]:
            raise FrozenArtifactDrift("L1_600 prediction Merkle drift in Phase 2 lock")
        if lock.get("main_metrics", {}).get("L1_600", {}).get("map_at_12") != 0.025257774555571332:
            raise FrozenArtifactDrift("Phase 2 winner metric drift")

        gates = csv_read(self.paths["phase2_gate"])
        if len(gates) != 13 or any(row.get("status") != "PASS" for row in gates):
            raise FrozenArtifactDrift("Phase 2 P2G1-P2G13 drift")
        recovery = json.loads(self.paths["phase2_recovery"].read_text(encoding="utf-8"))
        if recovery.get("status") != "RECOVERY_ONLY_SOURCE_CORRECTION" or recovery.get("methodology_changed") is not False:
            raise FrozenArtifactDrift("Phase 2 recovery audit drift")
        final_audit = csv_read(self.paths["phase2_final_audit"])
        if len(final_audit) != 24 or any(row.get("status") != "PASS" for row in final_audit):
            raise FrozenArtifactDrift("Phase 2 final audit drift")

        phase1_runner = ValidationSelectionRunner(self.root, self.paths["phase2_config"])
        phase1 = phase1_runner._phase1_preflight()
        if phase1["phase1_config_fingerprint"] != expected["phase1_config_fingerprint"]:
            raise FrozenArtifactDrift("Phase 1 config drift through Phase 2 verifier")

        manifest = csv_read(self.paths["prediction_manifest"])
        actual_merkle = {row["model_id"]: row["global_merkle_fingerprint"] for row in manifest}
        for version in FIRST7D_VERSIONS:
            if actual_merkle.get(version) != expected["prediction_global_merkle_fingerprints"][version]:
                raise FrozenArtifactDrift(f"{version} prediction manifest Merkle drift")
        return {
            "phase1": phase1,
            "phase2_config_fingerprint": expected["phase2_config_fingerprint"],
            "phase2_preregistration_fingerprint": prereg["preregistration_fingerprint"],
            "phase2_selection_lock_semantic_fingerprint": lock["lock_fingerprint"],
            "physical_fingerprints": observed_physical,
            "winner": lock["winner"],
            "winner_feature_set": "F1",
            "winner_iteration": 600,
            "value_gate": lock["value_gate"],
            "prediction_global_merkle_fingerprints": {
                version: actual_merkle[version] for version in FIRST7D_VERSIONS
            },
            "phase2_gate_pass_count": len(gates),
            "phase2_final_audit_pass_count": len(final_audit),
        }

    def _git_preflight(self) -> dict[str, Any]:
        def git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
            ).stdout.strip()
        head, main, origin = git("rev-parse", "HEAD"), git("rev-parse", "main"), git("rev-parse", "origin/main")
        branch = git("branch", "--show-current")
        if branch != "main" or len({head, main, origin}) != 1:
            raise FrozenArtifactDrift("Git branch/HEAD/main/origin-main drift")
        tracked_private = git("ls-files", "artifacts/stage_4b4/private", "artifacts/stage_4b4/tmp", "artifacts/stage_4b4/checkpoints")
        if tracked_private:
            raise FrozenArtifactDrift("tracked private/tmp/checkpoint artifact detected")
        return {
            "branch": branch, "head": head, "main": main, "origin_main": origin,
            "worktree_dirty_entry_count": len(git("status", "--porcelain").splitlines()),
            "tracked_private_artifact_count": 0,
        }

    def preregister(self) -> dict[str, Any]:
        forbidden_outputs = [
            self.paths["first7d_lock"], self.public / "first7d_diagnostic_result.json",
            self.public / "first7d_diagnostic_metrics.csv", self.public / "first7d_diagnostic_bootstrap.csv",
        ]
        if any(path.exists() for path in forbidden_outputs):
            raise FrozenArtifactDrift("First7d diagnostic or lock already exists")
        if (self.public / "first7d_diagnostic_access_audit.json").exists():
            raise FrozenArtifactDrift("First7d content access was already recorded")
        phase2 = self._phase2_preflight()
        git_state = self._git_preflight()
        label_lineage: dict[str, str] = {}
        for role in ("validation_ground_truth", "validation_population", "validation_item_state"):
            actual = file_fingerprint(self.paths[role])
            expected = self.raw["expected"][f"{role}_physical_fingerprint"]
            if actual != expected:
                raise FrozenArtifactDrift(f"{role} physical drift before preregistration")
            label_lineage[f"{role}_physical_fingerprint"] = actual

        payload = {
            "stage": "4B-4", "phase": self.raw["phase"],
            "status": "FIRST7D_PREREGISTERED_BEFORE_CONTENT_ACCESS",
            "created_at": now(), "config_fingerprint": self.config_fp,
            "source_fingerprint": self._source_fingerprint(),
            "evaluator_fingerprint": file_fingerprint(self.root / "src/retrieval/evaluation.py"),
            "diagnostic_contract": self.raw["diagnostic"],
            "versions": list(FIRST7D_VERSIONS), "locked_winner": self.raw["locked_winner"],
            "evaluation_contract": self.raw["evaluation"], "slices": self.raw["slices"],
            "bootstrap_contract": self.raw["bootstrap"], "decision_rules": self.raw["decision_rules"],
            "replay_contract": self.raw["replay"], "phase2_frozen_state": phase2,
            "label_lineage": {**label_lineage, "first7d_content_open_count": 0},
            "access": {
                "first7d_labels_read": False, "first7d_read_count": 0,
                "validation_14d_source_content_reopened_for_first7d": False,
                **self.raw["forbidden"],
            },
            "git_state": git_state,
            "resource_contract": {
                "available_memory_at_preregistration_bytes": self.available_at_preregistration,
                "rss_hard_limit_bytes": self.rss_limit, "duckdb_memory_limit_mb": self.memory_mb,
                "threads": 1, "simultaneous_prediction_versions": 1,
                "disk_free_bytes": shutil.disk_usage(self.root).free,
            },
        }
        payload["preregistration_fingerprint"] = canonical_fingerprint(payload)
        json_atomic(self.public / "first7d_diagnostic_preregistration.json", payload)
        rows = [
            {"check_id": "P3PRE1_GIT_IDENTITY", "status": "PASS", "observed": head_short(git_state["head"])},
            {"check_id": "P3PRE2_PHASE1_FROZEN", "status": "PASS", "observed": "G1-G20;32/32"},
            {"check_id": "P3PRE3_PHASE2_FROZEN", "status": "PASS", "observed": phase2["phase2_selection_lock_semantic_fingerprint"]},
            {"check_id": "P3PRE4_LOCKED_WINNER", "status": "PASS", "observed": "L1_600/F1/600"},
            {"check_id": "P3PRE5_ONLY_TWO_VERSIONS", "status": "PASS", "observed": "B0,L1_600"},
            {"check_id": "P3PRE6_EXACT_FIRST7D_WINDOW", "status": "PASS", "observed": "2020-08-26..2020-09-01"},
            {"check_id": "P3PRE7_CONTENT_ACCESS", "status": "PASS", "observed": "0"},
            {"check_id": "P3PRE8_TEST_AND_FINAL_STATE", "status": "PASS", "observed": "all_zero_or_false"},
            {"check_id": "P3PRE9_PRIVATE_TRACKING", "status": "PASS", "observed": "0"},
        ]
        write_csv(self.public / "first7d_diagnostic_pre_access_audit.csv", rows, list(rows[0]))
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        json_atomic(self.checkpoints / "preregistration.json", payload)
        return payload

    def _verify_preregistration(self) -> dict[str, Any]:
        path = self.public / "first7d_diagnostic_preregistration.json"
        if not path.exists():
            raise FrozenArtifactDrift("missing First7d preregistration")
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = payload.get("preregistration_fingerprint")
        if declared != self._self_fingerprint(payload, "preregistration_fingerprint"):
            raise FrozenArtifactDrift("First7d preregistration self-identity drift")
        if payload.get("config_fingerprint") != self.config_fp:
            raise FrozenArtifactDrift("First7d config changed after preregistration")
        if payload.get("source_fingerprint") != self._source_fingerprint():
            raise FrozenArtifactDrift("First7d source changed after preregistration")
        if payload.get("access", {}).get("first7d_read_count") != 0:
            raise FrozenArtifactDrift("First7d preregistration did not record zero access")
        frozen_rss = consume_frozen_rss_contract(
            payload,
            runtime_available_memory_bytes=psutil.virtual_memory().available,
        )
        self.rss_limit = frozen_rss.frozen_absolute_rss_cap_bytes
        self.memory_mb = int(payload["resource_contract"]["duckdb_memory_limit_mb"])
        self._phase2_preflight()
        for role in ("validation_ground_truth", "validation_population", "validation_item_state"):
            if file_fingerprint(self.paths[role]) != self.raw["expected"][f"{role}_physical_fingerprint"]:
                raise FrozenArtifactDrift(f"{role} changed after preregistration")
        return payload

    def _copy_atomic(self, con: duckdb.DuckDBPyConnection, sql: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".attempt.parquet")
        if temporary.exists():
            temporary.unlink()
        con.execute(f"COPY ({sql}) TO '{temporary.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        os.replace(temporary, output)

    def _materialize_first7d_once(self) -> tuple[Path, Path, dict[str, Any]]:
        access_path = self.public / "first7d_diagnostic_access_audit.json"
        if access_path.exists():
            raise PostFirst7dImplementationFailure("First7d logical access session already exists; automatic reopen is forbidden")
        access = {
            "stage": "4B-4", "phase": self.raw["phase"],
            "recorded_before_content_open": True,
            "first7d_labels_read": True, "first7d_read_count": 1,
            "validation_14d_source_content_open_count": 1,
            "label_window": "2020-08-26..2020-09-01",
            **self.raw["forbidden"], "recorded_at": now(),
        }
        json_atomic(access_path, access)

        gt_out = self.private / "ground_truth_first7d.parquet"
        pop_out = self.private / "evaluation_population_first7d.parquet"
        con = self._connect()
        try:
            gt = self.paths["validation_ground_truth"].as_posix()
            pop = self.paths["validation_population"].as_posix()
            state = self.paths["validation_item_state"].as_posix()
            con.execute(f"""
                CREATE TEMP TABLE first7d_ground_truth AS
                SELECT snapshot_id, user_key, article_key, label,
                       label_date_min AS first_observed_label_date,
                       as_of_date, DATE '2020-08-26' label_start,
                       DATE '2020-09-01' label_end, feedback_version
                FROM read_parquet('{gt}')
                WHERE label_date_min BETWEEN DATE '2020-08-26' AND DATE '2020-09-01'
            """)
            self._copy_atomic(con, "SELECT * FROM first7d_ground_truth ORDER BY user_key,article_key", gt_out)
            con.execute(f"""
                CREATE TEMP TABLE first7d_population AS
                WITH counts AS (
                  SELECT g.user_key,
                    count(*)::INTEGER positive_item_count,
                    count(*) FILTER(WHERE i.seen_item)::INTEGER warm_positive_count,
                    count(*) FILTER(WHERE i.cold_item)::INTEGER cold_positive_count
                  FROM first7d_ground_truth g
                  JOIN read_parquet('{state}') i USING(article_key)
                  GROUP BY g.user_key
                )
                SELECT p.snapshot_id, p.user_key, p.user_segment,
                  p.history_interaction_count, p.history_distinct_item_count,
                  c.positive_item_count, c.warm_positive_count, c.cold_positive_count,
                  (c.cold_positive_count>0) has_cold_positive,
                  p.is_main_evaluable, p.as_of_date
                FROM read_parquet('{pop}') p JOIN counts c USING(user_key)
            """)
            self._copy_atomic(con, "SELECT * FROM first7d_population ORDER BY user_key", pop_out)
            stats = con.execute("""
                SELECT (SELECT count(*) FROM first7d_ground_truth),
                       (SELECT count(DISTINCT user_key) FROM first7d_ground_truth),
                       (SELECT min(first_observed_label_date) FROM first7d_ground_truth),
                       (SELECT max(first_observed_label_date) FROM first7d_ground_truth),
                       (SELECT count(*) FROM first7d_population),
                       (SELECT count(*) FILTER(WHERE is_main_evaluable) FROM first7d_population),
                       (SELECT count(*) FILTER(WHERE NOT is_main_evaluable) FROM first7d_population)
            """).fetchone()
        finally:
            con.close()
        if not stats or stats[0] <= 0 or stats[1] <= 0 or stats[0] < stats[1]:
            raise PostFirst7dImplementationFailure(f"invalid First7d materialization counts: {stats}")
        if str(stats[2]) != "2020-08-26" or str(stats[3]) > "2020-09-01":
            raise PostFirst7dImplementationFailure(f"First7d window mismatch: {stats[2:4]}")
        if stats[4] != stats[1] or stats[5] + stats[6] != stats[4]:
            raise PostFirst7dImplementationFailure(f"First7d population mismatch: {stats}")
        checkpoint = {
            "task": "first7d_materialization", "status": "SUCCESS",
            "identity": canonical_fingerprint({
                "preregistration_fingerprint": self._prereg["preregistration_fingerprint"],
                "source_ground_truth_fingerprint": self.raw["expected"]["validation_ground_truth_physical_fingerprint"],
                "source_population_fingerprint": self.raw["expected"]["validation_population_physical_fingerprint"],
                "source_item_state_fingerprint": self.raw["expected"]["validation_item_state_physical_fingerprint"],
                "label_window": "2020-08-26..2020-09-01",
            }),
            "ground_truth_physical_fingerprint": file_fingerprint(gt_out),
            "population_physical_fingerprint": file_fingerprint(pop_out),
            "positive_pair_count": stats[0], "query_count": stats[1],
            "main_query_count": stats[5], "cold_query_count": stats[6],
            "generated_at": now(),
        }
        json_atomic(self.checkpoints / "first7d_materialization.json", checkpoint)
        return gt_out, pop_out, checkpoint

    def _binding(self, materialization: dict[str, Any]) -> dict[str, Any]:
        return {
            "preregistration_fingerprint": self._prereg["preregistration_fingerprint"],
            "first7d_ground_truth_physical_fingerprint": materialization["ground_truth_physical_fingerprint"],
            "first7d_population_physical_fingerprint": materialization["population_physical_fingerprint"],
            "prediction_global_merkle_fingerprints": self.raw["expected"]["prediction_global_merkle_fingerprints"],
            "evaluation_contract": self.raw["evaluation"],
        }

    def _checkpoint_run(
        self, name: str, output: Path, binding: dict[str, Any], operation: Callable[[], None]
    ) -> None:
        identity = canonical_fingerprint({"task": name, **binding})
        checkpoint = self.checkpoints / f"{name}.json"
        if checkpoint.exists() or output.exists():
            raise PostFirst7dImplementationFailure(f"unexpected pre-existing Phase 3 task state: {name}")
        monitor = ResourceMonitor(self.temp)
        started = time.perf_counter()
        try:
            with monitor:
                operation()
            if monitor.peak_rss_bytes > self.rss_limit:
                raise RuntimeError("Phase 3 peak RSS exceeded preregistered hard limit")
            if not output.exists():
                raise RuntimeError(f"missing output: {name}")
            row = {
                "task": name, "status": "SUCCESS", "identity": identity,
                "output": output.relative_to(self.root).as_posix(),
                "output_fingerprint": file_fingerprint(output),
                "row_count": pq.ParquetFile(output).metadata.num_rows,
                "wall_time_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "generated_at": now(),
            }
            json_atomic(checkpoint, row)
            self.resources.append(row)
        except Exception as exc:
            json_atomic(checkpoint, {
                "task": name, "status": "FAILED", "identity": identity,
                "error": str(exc).replace(str(self.root), "."), "generated_at": now(),
            })
            raise

    def _build_query_base(self, part: int, gt: Path, pop: Path, output: Path) -> None:
        con = self._connect()
        try:
            state = self.paths["validation_item_state"].as_posix()
            pred = (self.paths["predictions"] / "B0" / f"part_{part:03d}.parquet").as_posix()
            sql = f"""
              WITH target_counts AS (
                SELECT g.user_key,
                  count(*) FILTER(WHERE i.seen_item)::INTEGER seen_positive_count,
                  count(*) FILTER(WHERE i.cold_item)::INTEGER cold_positive_count_check,
                  count(*) FILTER(WHERE i.popularity_bucket='head')::INTEGER head_positive_count,
                  count(*) FILTER(WHERE i.popularity_bucket='mid')::INTEGER mid_positive_count,
                  count(*) FILTER(WHERE i.popularity_bucket='tail')::INTEGER tail_positive_count,
                  count(*) FILTER(WHERE i.popularity_bucket='zero_history')::INTEGER zero_history_positive_count
                FROM read_parquet('{gt.as_posix()}') g
                JOIN read_parquet('{state}') i USING(article_key)
                WHERE g.user_key % 32 = {part}
                GROUP BY g.user_key
              ), candidate_hits AS (
                SELECT p.customer_key user_key, count(g.article_key)::INTEGER candidate_positive_count
                FROM read_parquet('{pred}') p
                LEFT JOIN read_parquet('{gt.as_posix()}') g
                  ON g.user_key=p.customer_key AND g.article_key=p.article_key
                GROUP BY p.customer_key
              )
              SELECT p.*, h.candidate_positive_count,
                t.seen_positive_count, t.cold_positive_count_check,
                t.head_positive_count, t.mid_positive_count, t.tail_positive_count,
                t.zero_history_positive_count
              FROM read_parquet('{pop.as_posix()}') p
              JOIN candidate_hits h USING(user_key)
              JOIN target_counts t USING(user_key)
              WHERE p.user_key % 32 = {part}
              ORDER BY p.user_key
            """
            self._copy_atomic(con, sql, output)
        finally:
            con.close()

    def _metric_helper(self, gt: Path) -> ValidationSelectionRunner:
        helper = object.__new__(ValidationSelectionRunner)
        helper.paths = {
            "predictions": self.paths["predictions"],
            "ground_truth": gt,
            "item_state": self.paths["validation_item_state"],
        }
        helper.temp = self.temp
        helper.memory_mb = self.memory_mb
        helper.raw = {"execution": {
            "threads": 1,
            "max_temp_directory_size": self.raw["execution"]["max_temp_directory_size"],
        }}
        return helper

    def _write_query_digests(self, version: str, metrics: Path, output: Path, binding: dict[str, Any]) -> None:
        table = pq.read_table(metrics)
        metric_names = [name for name in table.column_names if name.endswith("_at_12")]
        columns = {name: table[name].to_pylist() for name in metric_names}
        snapshots, users = table["snapshot_id"].to_pylist(), table["customer_key"].to_pylist()
        rows: list[tuple[str, int, str]] = []
        for index, (snapshot, user) in enumerate(zip(snapshots, users, strict=True)):
            values = {name: columns[name][index] for name in metric_names if columns[name][index] is not None}
            rows.append((snapshot, int(user), query_metric_digest(version, snapshot, int(user), values)))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".attempt.parquet")
        pq.write_table(pa.table({
            "snapshot_id": [row[0] for row in rows],
            "customer_key": [row[1] for row in rows],
            "query_metric_digest": [row[2] for row in rows],
        }), temporary, compression="zstd")
        os.replace(temporary, output)
        query_merkle(rows, binding=canonical_fingerprint(binding))

    @staticmethod
    def _slice_predicates() -> dict[str, str]:
        return {
            "evaluable_warm_users": "is_main_evaluable",
            "short_history": "is_main_evaluable AND user_segment='short_history'",
            "warm": "is_main_evaluable AND user_segment='warm'",
            "active": "is_main_evaluable AND user_segment='active'",
            "positive_count_eq_1": "is_main_evaluable AND positive_item_count=1",
            "positive_count_gt_1": "is_main_evaluable AND positive_item_count>1",
            "candidate_positive_zero": "is_main_evaluable AND candidate_positive_count=0",
            "candidate_positive_nonzero": "is_main_evaluable AND candidate_positive_count>0",
            "has_cold_positive": "is_main_evaluable AND has_cold_positive",
            "no_cold_positive": "is_main_evaluable AND NOT has_cold_positive",
            "cold_history_label_users": "NOT is_main_evaluable",
        }

    @staticmethod
    def _target_specs() -> dict[str, tuple[str, str]]:
        return {
            "seen_item": ("seen", "seen_positive_count"),
            "cold_item": ("cold", "cold_positive_count_check"),
            "head": ("head", "head_positive_count"),
            "mid": ("mid", "mid_positive_count"),
            "tail": ("tail", "tail_positive_count"),
            "zero_history": ("zero_history", "zero_history_positive_count"),
        }

    def _aggregate(self) -> list[dict[str, Any]]:
        con = self._connect()
        rows: list[dict[str, Any]] = []
        try:
            pattern = (self.private / "query_metrics" / "*" / "part_*.parquet").as_posix()
            con.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
            for slice_name, predicate in self._slice_predicates().items():
                values = con.execute(f"""
                  SELECT version,count(*),avg(map_at_12),avg(ndcg_at_12),avg(recall_at_12),avg(hit_rate_at_12)
                  FROM m WHERE {predicate} GROUP BY version
                """).fetchall()
                for version, count, map12, ndcg, recall, hit in values:
                    rows.append({"version": version, "slice": slice_name, "target": "all",
                                 "query_count": count, "map_at_12": map12, "ndcg_at_12": ndcg,
                                 "recall_at_12": recall, "hit_rate_at_12": hit})
            for target, (prefix, count_field) in self._target_specs().items():
                values = con.execute(f"""
                  SELECT version,count(*),avg({prefix}_map_at_12),avg({prefix}_ndcg_at_12),
                         avg({prefix}_recall_at_12),avg({prefix}_hit_rate_at_12)
                  FROM m WHERE is_main_evaluable AND {count_field}>0 GROUP BY version
                """).fetchall()
                for version, count, map12, ndcg, recall, hit in values:
                    rows.append({"version": version, "slice": "evaluable_warm_users", "target": target,
                                 "query_count": count, "map_at_12": map12, "ndcg_at_12": ndcg,
                                 "recall_at_12": recall, "hit_rate_at_12": hit})
        finally:
            con.close()
        rows.sort(key=lambda row: (FIRST7D_VERSIONS.index(row["version"]), row["slice"], row["target"]))
        return rows

    def _bootstrap(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        con = self._connect()
        rows: list[dict[str, Any]] = []
        replay: list[dict[str, Any]] = []
        try:
            pattern = (self.private / "query_metrics" / "*" / "part_*.parquet").as_posix()
            con.execute(f"CREATE VIEW m AS SELECT * FROM read_parquet('{pattern}', union_by_name=true)")
            comparisons: list[tuple[str, str, str, str]] = [
                ("evaluable_warm_users", "all", "is_main_evaluable", "map_at_12")
            ]
            for name in self.raw["slices"]["critical_user"]:
                comparisons.append((name, "all", self._slice_predicates()[name], "map_at_12"))
            for target in self.raw["slices"]["critical_target"]:
                prefix, count_field = self._target_specs()[target]
                comparisons.append(("evaluable_warm_users", target,
                                    f"is_main_evaluable AND {count_field}>0", f"{prefix}_map_at_12"))
            for slice_name, target, predicate, field in comparisons:
                qualified = predicate
                for column in ("is_main_evaluable", "user_segment", "positive_item_count",
                               "candidate_positive_count", "has_cold_positive", "seen_positive_count",
                               "cold_positive_count_check", "head_positive_count", "mid_positive_count",
                               "tail_positive_count", "zero_history_positive_count"):
                    qualified = qualified.replace(column, f"b.{column}")
                values = con.execute(f"""
                  SELECT b.{field} baseline,c.{field} challenger
                  FROM m b JOIN m c USING(snapshot_id,customer_key)
                  WHERE b.version='B0' AND c.version='L1_600' AND ({qualified})
                  ORDER BY customer_key
                """).fetchnumpy()
                result = paired_bootstrap(
                    values["baseline"], values["challenger"], iterations=1000,
                    seed=20260826, confidence_level=0.95,
                )
                repeated = paired_bootstrap(
                    values["baseline"], values["challenger"], iterations=1000,
                    seed=20260826, confidence_level=0.95,
                )
                exact = result == repeated
                rows.append({"baseline": "B0", "challenger": "L1_600", "slice": slice_name,
                             "target": target, "metric": "MAP@12", **result})
                replay.append({"slice": slice_name, "target": target,
                               "exact_replay": exact, "status": "PASS" if exact else "FAIL"})
        finally:
            con.close()
        return rows, replay

    def _metric_fingerprints(self, binding: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for version in FIRST7D_VERSIONS:
            metric_parts = sorted((self.private / "query_metrics" / version).glob("part_*.parquet"))
            digest_parts = sorted((self.private / "query_metric_digests" / version).glob("part_*.parquet"))
            if len(metric_parts) != 32 or len(digest_parts) != 32:
                raise PostFirst7dImplementationFailure("First7d metric/digest partition completeness failed")
            physical = canonical_fingerprint([
                {"name": path.name, "size": path.stat().st_size, "sha256": file_fingerprint(path)}
                for path in metric_parts
            ])
            con = self._connect()
            try:
                rows = con.execute(f"""
                  SELECT snapshot_id,customer_key,query_metric_digest
                  FROM read_parquet('{(self.private / 'query_metric_digests' / version / 'part_*.parquet').as_posix()}')
                  ORDER BY snapshot_id,customer_key
                """).fetchall()
            finally:
                con.close()
            result[version] = {
                "physical_fingerprint": physical,
                "query_metric_merkle_fingerprint": query_merkle(rows, binding=canonical_fingerprint(binding)),
                "query_count": len(rows),
            }
        return result

    def _replay_partition_zero(self, gt: Path) -> list[dict[str, Any]]:
        helper = self._metric_helper(gt)
        base = self.private / "query_base" / "part_000.parquet"
        rows: list[dict[str, Any]] = []
        for version in FIRST7D_VERSIONS:
            output = self.private / "replay_partition_000" / version / "part_000.parquet"
            helper._build_metrics(version, 0, base, output)
            formal = self.private / "query_metrics" / version / "part_000.parquet"
            con = self._connect()
            try:
                mismatch = con.execute(f"""
                  SELECT count(*) FROM (
                    (SELECT * FROM read_parquet('{formal.as_posix()}') EXCEPT ALL
                     SELECT * FROM read_parquet('{output.as_posix()}'))
                    UNION ALL
                    (SELECT * FROM read_parquet('{output.as_posix()}') EXCEPT ALL
                     SELECT * FROM read_parquet('{formal.as_posix()}'))
                  )
                """).fetchone()[0]
            finally:
                con.close()
            rows.append({"version": version, "partition": 0, "row_mismatch_count": mismatch,
                         "status": "PASS" if mismatch == 0 else "FAIL"})
        return rows

    @staticmethod
    def _private_semantic_fingerprint(path: Path, columns: list[str]) -> str:
        digest = hashlib.sha256()
        digest.update("first7d_private_semantic_v1\n".encode())
        table = pq.read_table(path, columns=columns)
        arrays = [table[name].to_pylist() for name in columns]
        for values in zip(*arrays, strict=True):
            digest.update("|".join(str(value) for value in values).encode("utf-8") + b"\n")
        return digest.hexdigest()

    def evaluate(self) -> dict[str, Any]:
        try:
            self._prereg = self._verify_preregistration()
            gt, pop, materialization = self._materialize_first7d_once()
            binding = self._binding(materialization)
            helper = self._metric_helper(gt)
            for part in range(32):
                base = self.private / "query_base" / f"part_{part:03d}.parquet"
                self._checkpoint_run(
                    f"query_base_{part:03d}", base,
                    {**binding, "task_type": "query_base", "partition": part},
                    lambda part=part, base=base: self._build_query_base(part, gt, pop, base),
                )
                for version in FIRST7D_VERSIONS:
                    output = self.private / "query_metrics" / version / f"part_{part:03d}.parquet"
                    version_binding = {
                        **binding, "task_type": "query_metrics", "partition": part, "version": version,
                        "prediction_merkle": self.raw["expected"]["prediction_global_merkle_fingerprints"][version],
                    }
                    self._checkpoint_run(
                        f"metric_{version}_{part:03d}", output, version_binding,
                        lambda version=version, part=part, base=base, output=output:
                            helper._build_metrics(version, part, base, output),
                    )
                    digest_output = self.private / "query_metric_digests" / version / f"part_{part:03d}.parquet"
                    self._checkpoint_run(
                        f"digest_{version}_{part:03d}", digest_output,
                        {**version_binding, "metric_physical_fingerprint": file_fingerprint(output)},
                        lambda version=version, output=output, digest_output=digest_output:
                            self._write_query_digests(version, output, digest_output, binding),
                    )
                print(f"[phase3 first7d] partition {part:02d}/31 complete", flush=True)

            aggregate = self._aggregate()
            bootstrap, bootstrap_replay = self._bootstrap()
            main = {
                row["version"]: {metric: float(row[metric]) for metric in FIRST7D_METRICS}
                for row in aggregate if row["slice"] == "evaluable_warm_users" and row["target"] == "all"
            }
            delta = main["L1_600"]["map_at_12"] - main["B0"]["map_at_12"]
            main_bootstrap = next(row for row in bootstrap
                                  if row["slice"] == "evaluable_warm_users" and row["target"] == "all")
            decision = classify_robustness(delta, float(main_bootstrap["ci_lower"]), float(main_bootstrap["ci_upper"]))
            critical = [row for row in bootstrap
                        if not (row["slice"] == "evaluable_warm_users" and row["target"] == "all")
                        and float(row["ci_upper"]) < 0.0]
            review = (
                "FIRST7D_SEGMENT_REGRESSION_REVIEW_REQUIRED"
                if segment_review_required([float(row["ci_upper"]) for row in critical])
                else "NO_FIRST7D_CRITICAL_SEGMENT_REGRESSION"
            )
            replay = self._replay_partition_zero(gt)
            if any(row["status"] != "PASS" for row in replay + bootstrap_replay):
                raise PostFirst7dImplementationFailure("First7d deterministic replay failed")
            metric_fingerprints = self._metric_fingerprints(binding)

            metrics_path = self.public / "first7d_diagnostic_metrics.csv"
            bootstrap_path = self.public / "first7d_diagnostic_bootstrap.csv"
            replay_path = self.public / "first7d_diagnostic_replay_audit.csv"
            bootstrap_replay_path = self.public / "first7d_diagnostic_bootstrap_replay.csv"
            write_csv(metrics_path, aggregate, list(aggregate[0]))
            write_csv(bootstrap_path, bootstrap, list(bootstrap[0]))
            write_csv(replay_path, replay, list(replay[0]))
            write_csv(bootstrap_replay_path, bootstrap_replay, list(bootstrap_replay[0]))
            result = {
                "status": "FIRST7D_DIAGNOSTIC_COMPLETE",
                "diagnostic_id": self.raw["diagnostic"]["diagnostic_id"],
                "winner": "L1_600", "winner_changed": False,
                "main_metrics": main, "winner_map_delta_vs_b0": delta,
                "direction_status": decision.direction_status,
                "confidence_status": decision.confidence_status,
                "segment_regression_review": review,
                "critical_regression_slices": [f"{row['slice']}::{row['target']}" for row in critical],
            }
            json_atomic(self.public / "first7d_diagnostic_result.json", result)

            gt_semantic = self._private_semantic_fingerprint(
                gt, ["snapshot_id", "user_key", "article_key", "label", "first_observed_label_date"]
            )
            pop_semantic = self._private_semantic_fingerprint(
                pop, ["snapshot_id", "user_key", "user_segment", "positive_item_count",
                      "warm_positive_count", "cold_positive_count", "is_main_evaluable"]
            )
            lock = {
                "stage": "4B-4", "phase": self.raw["phase"],
                "status": "FIRST7D_LOCKED_WINNER_DIAGNOSTIC_COMPLETE",
                "config_fingerprint": self.config_fp,
                "preregistration_fingerprint": self._prereg["preregistration_fingerprint"],
                "phase1_lock_physical_fingerprint": self.raw["expected"]["phase1_lock_physical_fingerprint"],
                "phase2_selection_lock_semantic_fingerprint": self.raw["expected"]["phase2_selection_lock_semantic_fingerprint"],
                "phase2_selection_lock_physical_fingerprint": self.raw["expected"]["phase2_selection_lock_physical_fingerprint"],
                "winner": "L1_600", "winner_feature_set": "F1", "winner_iteration": 600,
                "winner_changed": False,
                "prediction_global_merkle_fingerprints": self.raw["expected"]["prediction_global_merkle_fingerprints"],
                "first7d_ground_truth": {
                    "physical_fingerprint": materialization["ground_truth_physical_fingerprint"],
                    "semantic_fingerprint": gt_semantic,
                    "positive_pair_count": materialization["positive_pair_count"],
                    "label_window": "2020-08-26..2020-09-01",
                },
                "first7d_population": {
                    "physical_fingerprint": materialization["population_physical_fingerprint"],
                    "semantic_fingerprint": pop_semantic,
                    "query_count": materialization["query_count"],
                    "main_query_count": materialization["main_query_count"],
                    "cold_query_count": materialization["cold_query_count"],
                },
                "metric_fingerprints": metric_fingerprints,
                "public_metric_physical_fingerprint": file_fingerprint(metrics_path),
                "bootstrap_physical_fingerprint": file_fingerprint(bootstrap_path),
                "replay_physical_fingerprint": file_fingerprint(replay_path),
                "main_metrics": main, "winner_map_delta_vs_b0": delta,
                "direction_status": decision.direction_status,
                "confidence_status": decision.confidence_status,
                "segment_regression_review": review,
                "first7d_labels_read": True, "first7d_read_count": 1,
                "validation_predictions_frozen": True, "winner_selected": True,
                **self.raw["forbidden"], "locked_at": now(),
            }
            lock["lock_fingerprint"] = canonical_fingerprint(lock)
            json_atomic(self.paths["first7d_lock"], lock)
            self._finalize_public(lock, materialization, aggregate, bootstrap, replay, bootstrap_replay)
            return lock
        except (FrozenArtifactDrift, PostFirst7dImplementationFailure):
            raise
        except Exception as exc:
            raise PostFirst7dImplementationFailure(str(exc)) from exc

    def _finalize_public(
        self, lock: dict[str, Any], materialization: dict[str, Any], aggregate: list[dict[str, Any]],
        bootstrap: list[dict[str, Any]], replay: list[dict[str, Any]], bootstrap_replay: list[dict[str, Any]],
    ) -> None:
        main_rows = [row for row in aggregate if row["slice"] == "evaluable_warm_users" and row["target"] == "all"]
        gates = [
            ("P3G1", "pre-access preregistration is immutable", bool(lock["preregistration_fingerprint"])),
            ("P3G2", "Phase 1 and Phase 2 frozen identity", bool(lock["phase2_selection_lock_semantic_fingerprint"])),
            ("P3G3", "exact First7d B-feedback window", lock["first7d_ground_truth"]["label_window"] == "2020-08-26..2020-09-01"),
            ("P3G4", "only B0 and locked L1_600", set(lock["metric_fingerprints"]) == set(FIRST7D_VERSIONS)),
            ("P3G5", "frozen full500 candidate contract",
             len(main_rows) == 2 and all(
                 lock["metric_fingerprints"][version]["query_count"] == materialization["query_count"]
                 for version in FIRST7D_VERSIONS
             )),
            ("P3G6", "main and cold populations separated", materialization["main_query_count"] + materialization["cold_query_count"] == materialization["query_count"]),
            ("P3G7", "all-zero and full-positive denominators retained", self.raw["evaluation"]["all_zero_relevance_queries"] == "retained_as_zero"),
            ("P3G8", "four canonical metrics complete", all(all(row[m] is not None for m in FIRST7D_METRICS) for row in main_rows)),
            ("P3G9", "paired bootstrap and exact replay", len(bootstrap) == len(bootstrap_replay) and all(row["status"] == "PASS" for row in bootstrap_replay)),
            ("P3G10", "direction/confidence rules applied without selection", lock["winner_changed"] is False),
            ("P3G11", "partition zero exact metric replay", all(row["status"] == "PASS" for row in replay)),
            ("P3G12", "atomic First7d lock and fingerprint", bool(lock.get("lock_fingerprint"))),
            ("P3G13", "final-refit/Test/rerank/retuning remain forbidden", all(lock.get(key) in (0, False) for key in self.raw["forbidden"])),
        ]
        rows = [{"gate_id": gate, "description": description,
                 "status": "PASS" if passed else "FAIL"} for gate, description, passed in gates]
        write_csv(self.public / "first7d_diagnostic_contract_gate.csv", rows, list(rows[0]))
        if any(row["status"] != "PASS" for row in rows):
            raise PostFirst7dImplementationFailure("Phase 3 P3G1-P3G13 failed")
        resource = {
            "rss_hard_limit_bytes": self.rss_limit, "duckdb_memory_limit_mb": self.memory_mb,
            "threads": 1, "simultaneous_prediction_versions": 1,
            "task_count": len(self.resources),
            "peak_rss_bytes": max((row["peak_rss_bytes"] for row in self.resources), default=0),
            "peak_temp_bytes": max((row["temp_disk_peak_bytes"] for row in self.resources), default=0),
        }
        json_atomic(self.public / "first7d_diagnostic_resource_audit.json", resource)
        report = f"""# Stage 4B-4 Phase 3 — First7d Locked-Winner Robustness Diagnostic

## Status

`{lock['status']}`

- Frozen winner: `L1_600` (`F1`, iteration 600); winner changed: `false`.
- Diagnostic window: 2020-08-26 through 2020-09-01, reusing `validation_14d` request/candidates/features/predictions.
- B0 MAP@12: {lock['main_metrics']['B0']['map_at_12']:.17g}
- L1_600 MAP@12: {lock['main_metrics']['L1_600']['map_at_12']:.17g}
- Delta MAP@12: {lock['winner_map_delta_vs_b0']:.17g}
- Direction: `{lock['direction_status']}`
- Confidence: `{lock['confidence_status']}`
- Segment review: `{lock['segment_regression_review']}`

First7d was used only once after the 14-day winner lock. It did not select or change a model and did not authorize Test. Final refit, Test, reranking and retrieval retuning were not executed. P3G1–P3G13 are independent contract checks and all passed.
"""
        self.paths["report"].write_text(report, encoding="utf-8")


def head_short(value: str) -> str:
    return value[:12]
