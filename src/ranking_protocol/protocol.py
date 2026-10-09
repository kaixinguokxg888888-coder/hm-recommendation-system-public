from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data_audit.config import find_project_root
from data_foundation.manifest import file_fingerprint


EXPECTED_CALENDAR = {
    "ranking_train_14d_v1": ("2020-08-11", "2020-08-12", "2020-08-11", "2020-08-12", "2020-08-25", 14),
    "validation_14d": ("2020-08-25", "2020-08-26", "2020-08-25", "2020-08-26", "2020-09-08", 14),
    "test_14d": ("2020-09-08", "2020-09-09", "2020-09-08", "2020-09-09", "2020-09-22", 14),
    "ranking_validation_first7d_diagnostic": ("2020-08-25", "2020-08-26", "2020-08-25", "2020-08-26", "2020-09-01", 7),
}
EXPECTED_CAPS = {"repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100}
EXPECTED_ORDER = ["repurchase", "itemcf", "two_tower", "trend", "popularity_fallback"]
EXPECTED_COMPONENTS = ["repurchase", "itemcf", "two_tower", "ann", "trend", "popularity", "fusion"]
EXPECTED_QUERY_KEY = ["snapshot_id", "customer_key"]
CONTRACT_VERSION = "stage_3e_candidate_contract_v1"
PROTOCOL_VERSION = "source_rank_cap_pre_dedup_v1"


class Stage4AProtocolError(ValueError):
    """Stage 4A formal fact, calendar, or protocol validation failed."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_date(value: str) -> date:
    return date.fromisoformat(value)


def _inclusive_days(start: str, end: str) -> int:
    return (_as_date(end) - _as_date(start)).days + 1


def _is_subset(inner_start: str, inner_end: str, outer_start: str, outer_end: str) -> bool:
    return _as_date(outer_start) <= _as_date(inner_start) <= _as_date(inner_end) <= _as_date(outer_end)


@dataclass(frozen=True)
class Stage4AConfig:
    root: Path
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Stage4AConfig":
        resolved = path.resolve()
        value = cls(find_project_root(resolved), resolved, json.loads(resolved.read_text(encoding="utf-8")))
        value.validate()
        return value

    def resolve(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise Stage4AProtocolError("Stage 4A 只允许项目相对路径") from exc
        return path

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    @property
    def artifact_dir(self) -> Path:
        return self.resolve(self.raw["execution"]["paths"]["artifact_directory"])

    @property
    def decision_path(self) -> Path:
        return self.resolve(self.raw["execution"]["paths"]["decision_doc"])

    @property
    def report_path(self) -> Path:
        return self.resolve(self.raw["execution"]["paths"]["stage_report"])

    @property
    def fingerprint(self) -> str:
        return file_fingerprint(self.path)

    def validate(self) -> None:
        if self.raw.get("stage") != "4A" or self.raw.get("stage_version") != "1.0.0":
            raise Stage4AProtocolError("Stage/version 必须冻结为 4A/1.0.0")
        if self.raw.get("protocol_status") != "PASSED_PROTOCOL_GATE":
            raise Stage4AProtocolError("Stage 4A protocol gate status 漂移")
        if self.raw.get("final_acceptance_status") != "ACCEPTED" or self.raw.get("stage_4b_authorized") is not True:
            raise Stage4AProtocolError("Final acceptance 必须 accepted 并允许后续独立 Prompt 消费 Stage 4B 协议")
        if self.raw.get("stage_4b_protocol_consumption_status") != "ALLOWED_AFTER_FINAL_ACCEPTED_WITH_SEPARATE_PROMPT":
            raise Stage4AProtocolError("Stage 4B protocol consumption status 漂移")
        if self.raw.get("stage_4b_entered_during_stage_4a") is not False:
            raise Stage4AProtocolError("Stage 4A closeout 不得执行 Stage 4B")
        if self.raw.get("ranking_training_protocol") != "single_historical_14d_snapshot_v1":
            raise Stage4AProtocolError("V1 Ranking 主训练协议漂移")
        rolling = self.raw["rolling_historical_snapshots"]
        if rolling != {"priority": "P1", "enabled_in_stage_4a": False, "implemented": False}:
            raise Stage4AProtocolError("Rolling snapshot 必须保持 P1 且 Stage 4A 未实现")
        self._validate_calendar()
        self._validate_population_contract()
        self._validate_retrieval_contract()
        self._validate_label_and_sampling_contract()
        self._validate_feature_and_execution_contract()

    def _validate_calendar(self) -> None:
        calendar = self.raw.get("calendar", {})
        if set(calendar) != set(EXPECTED_CALENDAR):
            raise Stage4AProtocolError("Ranking calendar snapshot 集合漂移")
        fields = ("as_of_date", "request_date", "history_cutoff", "label_start", "label_end", "label_window_days")
        for snapshot, expected in EXPECTED_CALENDAR.items():
            row = calendar[snapshot]
            actual = tuple(row.get(field) for field in fields)
            if actual != expected:
                raise Stage4AProtocolError(f"{snapshot} calendar 漂移")
            if _inclusive_days(row["label_start"], row["label_end"]) != row["label_window_days"]:
                raise Stage4AProtocolError(f"{snapshot} label window 非闭区间声明天数")
            if row["history_cutoff"] != row["as_of_date"]:
                raise Stage4AProtocolError(f"{snapshot} history cutoff/as-of 不一致")
            if _as_date(row["request_date"]) != _as_date(row["as_of_date"]) + timedelta(days=1):
                raise Stage4AProtocolError(f"{snapshot} request date 必须紧随 as-of")
            if _as_date(row["history_cutoff"]) >= _as_date(row["label_start"]):
                raise Stage4AProtocolError(f"{snapshot} history 未严格早于 label")
        diagnostic = calendar["ranking_validation_first7d_diagnostic"]
        if diagnostic.get("candidate_snapshot_id") != "validation_14d" or diagnostic.get("is_independent_request_snapshot") is not False:
            raise Stage4AProtocolError("first7d 必须复用 Validation request/candidate snapshot")

    def _validate_population_contract(self) -> None:
        population = self.raw["population_contract"]
        if population.get("version") != "historical_evaluable_warm_users_v1":
            raise Stage4AProtocolError("Ranking training population version 漂移")
        if population.get("construction_stage") != "4B" or population.get("full_historical_population_materialized_in_stage_4a") is not False:
            raise Stage4AProtocolError("Stage 4A 不得物化 full historical population")
        if population.get("main_optimization_population") != "historical_evaluable_warm_users_v1":
            raise Stage4AProtocolError("Main Ranking optimization population 漂移")
        warm = population["historical_evaluable_warm_users_v1"]
        if not all(warm.get(field) is True for field in (
            "requires_future_B_feedback_positive", "requires_pre_cutoff_B_feedback_history_interaction",
            "requires_valid_internal_customer_article_keys",
        )) or warm.get("history_cutoff") != "2020-08-11":
            raise Stage4AProtocolError("Historical evaluable warm user definition 不完整")
        cold = population["cold_history_label_users"]
        if cold.get("included_in_main_optimization_population") or cold.get("included_in_main_evaluation_average"):
            raise Stage4AProtocolError("Cold-history 不得混入 main population")
        zero = population["all_zero_relevance_group"]
        if zero.get("classification") != "retrieval_miss" or zero.get("future_lambdarank_optimization") != "exclude" or zero.get("end_to_end_evaluation") != "retain":
            raise Stage4AProtocolError("All-zero relevance group semantics 漂移")
        if population.get("candidate_membership_label_aware") is not False:
            raise Stage4AProtocolError("Population semantics 不得赋予 label-aware candidate membership")

    def _validate_retrieval_contract(self) -> None:
        replay = self.raw["candidate_replay_contract"]
        checks = {
            "candidate_contract_version": CONTRACT_VERSION,
            "candidate_protocol_version": PROTOCOL_VERSION,
            "candidate_budget": 500,
            "historical_replay_stage": "4B",
            "full_replay_in_stage_4a": False,
            "membership_label_access_allowed": False,
            "label_aware_candidate_injection_allowed": False,
            "semantic_drift_allowed": False,
        }
        for field, expected in checks.items():
            if replay.get(field) != expected:
                raise Stage4AProtocolError(f"Candidate replay contract 漂移: {field}")
        frozen = replay["frozen_retrieval"]
        expected_frozen = {
            "model": "TT1",
            "quota": "T100",
            "ann_ef_search": 200,
            "source_caps": EXPECTED_CAPS,
            "quota_semantics": "source_rank_cap_before_cross_source_dedup",
            "insertion_order": EXPECTED_ORDER,
            "duplicate_semantics": "duplicates_consume_pre_dedup_source_rank_cap_and_merge_provenance",
            "score_semantics": "source_local_raw_scores_never_summed_or_cross_source_calibrated",
            "fallback": "popularity_unique_fill_to_500",
        }
        if frozen != expected_frozen:
            raise Stage4AProtocolError("Stage 3E frozen retrieval identity 漂移")
        if list(replay["components"]) != EXPECTED_COMPONENTS:
            raise Stage4AProtocolError("Historical replay component 集合或顺序漂移")
        fingerprint = replay["historical_fingerprint_contract"]
        if fingerprint.get("protocol_identity_name") != "historical_retrieval_protocol_identity":
            raise Stage4AProtocolError("Historical protocol identity 命名漂移")
        if fingerprint.get("historical_formal_snapshot_id") != "ranking_train_14d_v1":
            raise Stage4AProtocolError("Historical fingerprint snapshot 漂移")
        if fingerprint.get("membership_or_rank_may_differ_from_validation_and_test") is not True:
            raise Stage4AProtocolError("跨 snapshot membership/rank 差异必须允许")
        if fingerprint.get("equality_with_validation_fingerprint_required") or fingerprint.get("equality_with_test_fingerprint_required"):
            raise Stage4AProtocolError("Historical fingerprint 不得要求等于 Validation/Test")
        if fingerprint.get("same_snapshot_replay_requirement") != "historical_replay_fingerprint_equals_historical_formal_fingerprint":
            raise Stage4AProtocolError("Same-snapshot replay fingerprint contract 漂移")

    def _validate_label_and_sampling_contract(self) -> None:
        labels = self.raw["label_semantics"]
        if labels.get("feedback") != "B_user_item_day_binary":
            raise Stage4AProtocolError("Ranking label 必须沿用 B-feedback")
        if labels.get("row_key") != ["snapshot_id", "customer_key", "article_key"]:
            raise Stage4AProtocolError("Ranking row key 漂移")
        if labels.get("zero") != "candidate_item_not_observed_as_purchased_in_future_label_window":
            raise Stage4AProtocolError("label=0 的未观察购买语义漂移")
        if "not_injected" not in labels.get("candidate_miss", ""):
            raise Stage4AProtocolError("Retrieval miss 不得注入 candidate")
        sampling = self.raw["negative_sampling"]
        if not sampling.get("keep_all_positives") or sampling.get("max_unobserved_candidates_per_query") != 100:
            raise Stage4AProtocolError("V1 unobserved candidate sampling cap 漂移")
        if sampling.get("key") != ["snapshot_id", "customer_key", "article_key", "fixed_seed"]:
            raise Stage4AProtocolError("Deterministic sampling key 漂移")
        exact_sampling = {
            "hash_algorithm": "SHA-256",
            "encoding": "UTF-8",
            "canonical_payload": "{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}",
            "digest_expression": "sha256(canonical_payload.encode(\"utf-8\")).hexdigest()",
            "sort_order": ["digest_ASC", "article_key_ASC"],
            "forbidden_inputs": ["filesystem_path", "partition_id", "row_order", "process_id", "python_runtime_hash_seed"],
            "python_builtin_hash_allowed": False,
        }
        for field, expected in exact_sampling.items():
            if sampling.get(field) != expected:
                raise Stage4AProtocolError(f"Deterministic negative hash contract 漂移: {field}")
        if sampling.get("validation_candidate_downsampling") or sampling.get("test_candidate_downsampling"):
            raise Stage4AProtocolError("Validation/Test 必须保留完整 500 candidates")
        if self.raw["query_group"].get("key") != EXPECTED_QUERY_KEY:
            raise Stage4AProtocolError("Ranking query key 必须包含 snapshot_id")

    def _validate_feature_and_execution_contract(self) -> None:
        feature = self.raw["feature_time_contract"]
        if feature.get("dynamic_feature_rule") != "feature_timestamp_le_snapshot_as_of_date":
            raise Stage4AProtocolError("Feature cutoff contract 漂移")
        if feature.get("historical_rows_recomputed_after_final_refit") is not False:
            raise Stage4AProtocolError("Final refit 不得用晚期历史重算旧 snapshot feature")
        final_refit = self.raw["final_refit_discipline"]
        if final_refit.get("supervision_snapshots") != ["ranking_train_14d_v1", "validation_14d"]:
            raise Stage4AProtocolError("Final-refit supervision snapshots 漂移")
        if final_refit.get("test_supervision_allowed") or final_refit.get("internal_validation_selection"):
            raise Stage4AProtocolError("Final refit 不得使用 Test 或内部选择")
        if self.raw["test_discipline"].get("selection_allowed") is not False:
            raise Stage4AProtocolError("Test selection 必须禁止")
        first7d = self.raw["first7d_diagnostic_discipline"]
        if first7d.get("execution_order") != [
            "complete_full_validation_14d_selection", "freeze_single_validation_14d_winner",
            "run_one_first7d_locked_winner_vs_stage3e_baseline",
        ]:
            raise Stage4AProtocolError("First7d execution order 漂移")
        if first7d.get("selection_allowed") is not False or first7d.get("run_count") != 1:
            raise Stage4AProtocolError("First7d 只能执行一次锁后 diagnostic")
        if first7d.get("direction_failure_status") != "VALIDATION_ROBUSTNESS_NOT_MET":
            raise Stage4AProtocolError("First7d robustness failure status 漂移")
        if first7d.get("automatic_retuning_allowed") or first7d.get("automatic_test_authorization_allowed"):
            raise Stage4AProtocolError("First7d 失败不得自动调参或授权 Test")
        execution = self.raw["execution"]
        forbidden_true = (
            "historical_full_candidate_replay", "ranking_dataset_materialization",
            "ranking_feature_materialization", "ranking_training", "ranking_prediction",
            "test_ranking_evaluation", "reranking",
        )
        if any(execution.get(field) is not False for field in forbidden_true):
            raise Stage4AProtocolError("Stage 4A 提前进入 Ranking implementation")


def validate_upstream(config: Stage4AConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in config.raw["upstream_artifacts"]:
        path = config.resolve(record["path"])
        if not path.is_file():
            raise Stage4AProtocolError(f"Upstream artifact 缺失: {record['role']}")
        actual = file_fingerprint(path)
        status = "PASS" if actual == record["fingerprint"] else "FAIL"
        rows.append({
            "lineage_role": record["role"], "artifact_path": record["path"],
            "expected_fingerprint": record["fingerprint"], "actual_fingerprint": actual,
            "status": status,
        })
    failed = [row for row in rows if row["status"] != "PASS"]
    if failed:
        raise Stage4AProtocolError(f"BLOCKED_BY_CONFLICT: upstream fingerprint mismatch: {failed}")
    _validate_upstream_identity(config)
    return rows


def _validate_upstream_identity(config: Stage4AConfig) -> None:
    stage3c = json.loads(config.resolve("artifacts/stage_3c/stage_3c_lock.json").read_text(encoding="utf-8"))
    stage3d = json.loads(config.resolve("artifacts/stage_3d/stage_3d_lock.json").read_text(encoding="utf-8"))
    stage3e = json.loads(config.resolve("configs/stage_3e_candidate_contract.yaml").read_text(encoding="utf-8"))
    if stage3c.get("recommended_stage_status") != "FAILED_VALUE_GATE":
        raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3C status")
    checks = {
        "stage_status": "PASSED_VALUE_GATE", "recommended_stage_status": "PASSED_VALUE_GATE",
        "test_status": "SUCCESS", "locked_model": "TT1", "locked_quota": "T100",
        "candidate_protocol_version": PROTOCOL_VERSION, "locked_quota_caps": EXPECTED_CAPS,
        "final_refit_epochs": 1, "final_refit_internal_dev_examples": 0,
    }
    for field, expected in checks.items():
        if stage3d.get(field) != expected:
            raise Stage4AProtocolError(f"BLOCKED_BY_CONFLICT: Stage 3D {field}")
    if stage3d.get("ann_parameters", {}).get("ef_search") != 200:
        raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3D ANN efSearch")
    if stage3e.get("candidate_contract_version") != CONTRACT_VERSION:
        raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3E candidate contract")
    if stage3e.get("candidate_protocol_version") != PROTOCOL_VERSION:
        raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3E candidate protocol")
    if stage3e.get("frozen_retrieval") != config.raw["candidate_replay_contract"]["frozen_retrieval"] | {
        "candidate_budget": 500
    }:
        # Stage 3E config does not have the explanatory score_semantics field.
        comparable = dict(config.raw["candidate_replay_contract"]["frozen_retrieval"])
        comparable.pop("score_semantics")
        comparable["candidate_budget"] = 500
        if stage3e.get("frozen_retrieval") != comparable:
            raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3E frozen retrieval")
    manifest = _read_csv(config.resolve("artifacts/stage_3e/stage_3e_manifest.csv"))
    if not manifest or {row["stage_status"] for row in manifest} != {"PASSED_CONTRACT_GATE"}:
        raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3E status")
    contract = _read_csv(config.resolve("artifacts/stage_3e/candidate_contract_manifest.csv"))
    if {row["snapshot_id"] for row in contract} != {"validation_14d", "test_14d"}:
        raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3E snapshots")
    if {row["candidate_contract_version"] for row in contract} != {CONTRACT_VERSION}:
        raise Stage4AProtocolError("BLOCKED_BY_CONFLICT: Stage 3E contract manifest version")


def calendar_rows(config: Stage4AConfig) -> list[dict[str, Any]]:
    replay = config.raw["candidate_replay_contract"]
    rows: list[dict[str, Any]] = []
    for snapshot_id, values in config.raw["calendar"].items():
        rows.append({
            "snapshot_id": snapshot_id, **values,
            "candidate_contract_version": replay["candidate_contract_version"],
            "candidate_protocol_version": replay["candidate_protocol_version"],
        })
    return rows


def replay_contract_rows(config: Stage4AConfig) -> list[dict[str, Any]]:
    rows = []
    for component, values in config.raw["candidate_replay_contract"]["components"].items():
        rows.append({
            "component": component, **values,
            "label_access_allowed": False,
            "parameter_selection_allowed": False,
            "semantic_drift_allowed": False,
            "replay_execution_stage": "4B",
            "stage_4a_full_replay_performed": False,
        })
    return rows


def replay_capability_rows(config: Stage4AConfig) -> list[dict[str, str]]:
    checks = [
        ("repurchase", "READY", "src/retrieval/repurchase.py", "def pure_repurchase", "Core function accepts as_of_date and filters history."),
        ("itemcf", "NEEDS_STAGE4B_ADAPTATION", "src/retrieval/stage3b/runner.py", "SnapshotSpec", "Core cutoff logic exists; formal runner snapshot registry and lineage need ranking_train registration."),
        ("tt1_training", "NEEDS_STAGE4B_ADAPTATION", "src/retrieval/stage3d/runner.py", "def build_training_targets", "Snapshot-scoped causal training/full-refit exists; runner paths and formal lock flow cover only Validation/Test."),
        ("tt1_full_refit", "NEEDS_STAGE4B_ADAPTATION", "src/retrieval/stage3d/runner.py", "def train_final_model", "Locked epoch full-refit exists; historical checkpoint role and lineage must be added in Stage 4B."),
        ("ann", "NEEDS_STAGE4B_ADAPTATION", "src/retrieval/stage3d/ann.py", "def build_hnsw_index", "ANN is embedding/index specific; historical snapshot wiring and audit lineage must be added."),
        ("trend", "READY", "src/retrieval/trend.py", "def trend_ranking", "Core function accepts as_of_date and computes the trailing window only."),
        ("popularity", "READY", "src/retrieval/popularity.py", "def popularity_ranking", "Core function accepts as_of_date and filters future interactions."),
        ("fusion", "NEEDS_STAGE4B_ADAPTATION", "src/retrieval/stage3d/union.py", "def source_rank_cap_union", "Frozen union primitive is parameterized; Stage 3E formalizer currently admits only Validation/Test snapshots."),
    ]
    rows: list[dict[str, str]] = []
    for component, status, relative, marker, reason in checks:
        text = config.resolve(relative).read_text(encoding="utf-8")
        if marker not in text:
            status = "BLOCKED"
            reason = f"Expected capability marker missing: {marker}"
        rows.append({"component": component, "status": status, "evidence": relative, "reason": reason})
    return rows


def leakage_audit_rows(config: Stage4AConfig) -> list[dict[str, str]]:
    calendar = config.raw["calendar"]
    train = calendar["ranking_train_14d_v1"]
    validation = calendar["validation_14d"]
    test = calendar["test_14d"]
    diagnostic = calendar["ranking_validation_first7d_diagnostic"]
    split = {row["split_id"]: row for row in _read_csv(config.resolve("artifacts/stage_2a/split_manifest.csv"))}
    checks = [
        ("L1_OUTER_TRAIN_CONTAINMENT", _as_date(train["label_end"]) <= _as_date(split["validation_14d"]["history_end"]), f"{train['label_end']} <= outer Train end {split['validation_14d']['history_end']}"),
        ("L2_FEATURE_BEFORE_LABEL", all(_as_date(row["history_cutoff"]) < _as_date(row["label_start"]) for row in calendar.values()), "Every history_cutoff is strictly earlier than label_start."),
        ("L3_TRAIN_VALIDATION_BOUNDARY", _as_date(train["label_end"]) < _as_date(validation["label_start"]), f"{train['label_end']} < {validation['label_start']}"),
        ("L4_VALIDATION_TEST_BOUNDARY", _as_date(validation["label_end"]) < _as_date(test["label_start"]), f"{validation['label_end']} < {test['label_start']}"),
        ("L5_TEST_LABEL_SELECTION_ISOLATION", config.raw["test_discipline"]["selection_allowed"] is False and config.raw["validation_discipline"]["test_access_allowed"] is False, "Test is excluded from selection inputs."),
        ("L6_FIRST7D_TEST_ISOLATION", _is_subset(diagnostic["label_start"], diagnostic["label_end"], validation["label_start"], validation["label_end"]) and _as_date(diagnostic["label_end"]) < _as_date(test["label_start"]), "Diagnostic is a Validation subset and ends before Test."),
        ("L7_CANDIDATE_LABEL_SEPARATION", config.raw["candidate_replay_contract"]["membership_label_access_allowed"] is False and config.raw["candidate_replay_contract"]["label_aware_candidate_injection_allowed"] is False, "Future labels may annotate but never alter candidate membership."),
        ("L8_HISTORICAL_FEATURE_CUTOFF", config.raw["feature_time_contract"]["dynamic_feature_rule"] == "feature_timestamp_le_snapshot_as_of_date", "feature_timestamp <= corresponding snapshot as_of_date."),
        ("L9_HISTORICAL_REPLAY", config.raw["candidate_replay_contract"]["full_replay_in_stage_4a"] is False and all(row["historical_rebuild_required"] for row in config.raw["candidate_replay_contract"]["components"].values()), "Historical sources must rebuild from the historical cutoff; no later snapshot backfill."),
        ("L10_RETRIEVAL_SEMANTIC_FREEZE", config.raw["candidate_replay_contract"]["semantic_drift_allowed"] is False, "Stage 3E retrieval semantics are immutable during historical replay."),
        ("L11_RANKING_POPULATION_CONTRACT", config.raw["population_contract"]["main_optimization_population"] == "historical_evaluable_warm_users_v1" and config.raw["population_contract"]["candidate_membership_label_aware"] is False, "Historical warm optimization population is frozen; query population semantics cannot alter candidate membership."),
        ("L12_DETERMINISTIC_NEGATIVE_HASH", config.raw["negative_sampling"]["hash_algorithm"] == "SHA-256" and config.raw["negative_sampling"]["canonical_payload"] == "{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}" and config.raw["negative_sampling"]["python_builtin_hash_allowed"] is False, "UTF-8 SHA-256 payload and digest/article tie-break are fully specified; built-in hash is forbidden."),
        ("L13_FIRST7D_POST_SELECTION_ONLY", config.raw["first7d_diagnostic_discipline"]["selection_allowed"] is False and config.raw["first7d_diagnostic_discipline"]["challenger_scope"] == "single_locked_14d_winner" and config.raw["first7d_diagnostic_discipline"]["automatic_retuning_allowed"] is False, "First7d runs once after locking the 14d winner and cannot select or retune."),
        ("L14_HISTORICAL_FINGERPRINT_SEMANTICS", config.raw["candidate_replay_contract"]["historical_fingerprint_contract"]["membership_or_rank_may_differ_from_validation_and_test"] is True and config.raw["candidate_replay_contract"]["historical_fingerprint_contract"]["equality_with_validation_fingerprint_required"] is False and config.raw["candidate_replay_contract"]["historical_fingerprint_contract"]["equality_with_test_fingerprint_required"] is False, "Protocol identity is cross-snapshot; semantic fingerprint equality applies only to formal/replay of the same historical snapshot."),
        ("L15_STAGE4B_NOT_ENTERED_DURING_STAGE4A", config.raw["final_acceptance_status"] == "ACCEPTED" and config.raw["stage_4b_authorized"] is True and config.raw["stage_4b_entered_during_stage_4a"] is False and all(config.raw["execution"][field] is False for field in ("historical_full_candidate_replay", "ranking_dataset_materialization", "ranking_feature_materialization", "ranking_training", "ranking_prediction", "test_ranking_evaluation", "reranking")), "FINAL ACCEPTED permits future Stage 4B protocol consumption under a separate Prompt; Stage 4B was not entered during Stage 4A."),
    ]
    rows = [{"leakage_gate_id": gate, "status": "PASS" if passed else "FAIL", "evidence": evidence} for gate, passed, evidence in checks]
    if any(row["status"] != "PASS" for row in rows):
        raise Stage4AProtocolError("FAILED_PROTOCOL_GATE: leakage audit")
    return rows


def negative_sampling_digest(row: Mapping[str, Any], *, fixed_seed: int) -> str:
    required = {"snapshot_id", "customer_key", "article_key"}
    if not required.issubset(row):
        raise Stage4AProtocolError("Deterministic sampling row 缺少 key")
    payload = f"{row['snapshot_id']}|{row['customer_key']}|{row['article_key']}|{fixed_seed}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_unobserved_sample(
    rows: Iterable[Mapping[str, Any]], *, fixed_seed: int, max_rows: int = 100,
) -> list[Mapping[str, Any]]:
    if max_rows != 100:
        raise Stage4AProtocolError("V1 max_unobserved_candidates_per_query 已冻结为 100")
    values = list(rows)
    return sorted(
        values,
        key=lambda row: (negative_sampling_digest(row, fixed_seed=fixed_seed), int(row["article_key"])),
    )[:max_rows]


def historical_replay_fingerprint_matches(
    *, formal_snapshot_id: str, replay_snapshot_id: str,
    formal_fingerprint: str, replay_fingerprint: str,
) -> bool:
    expected = "ranking_train_14d_v1"
    if formal_snapshot_id != expected or replay_snapshot_id != expected:
        raise Stage4AProtocolError(
            "Historical replay identity 只能比较 ranking_train_14d_v1 formal 与同 snapshot replay；"
            "不得比较 Validation/Test fingerprint"
        )
    return formal_fingerprint == replay_fingerprint


def no_ranking_implementation_audit(config: Stage4AConfig) -> tuple[bool, str]:
    python_paths = list((config.root / "src" / "ranking_protocol").glob("*.py"))
    python_paths.append(config.root / "scripts" / "finalize_stage_4a_protocol.py")
    forbidden_code = (
        r"^\s*import\s+lightgbm", r"^\s*from\s+lightgbm", r"LGBMRanker\s*\(",
        r"objective\s*=\s*['\"]lambdarank", r"def\s+(train_ranker|score_candidates|rerank)\s*\(",
    )
    findings: list[str] = []
    for path in python_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if any(re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE) for pattern in forbidden_code):
            findings.append(config.relative(path))
    forbidden_paths = [
        "src/ranking/lightgbm_ranker.py", "artifacts/stage_4a/private/ranking_dataset",
        "artifacts/stage_4a/private/ranking_features", "artifacts/stage_4a/test_ranking_scores.csv",
    ]
    findings.extend(relative for relative in forbidden_paths if config.resolve(relative).exists())
    return not findings, "No Stage 4B model/dataset/feature/prediction/rerank implementation found." if not findings else ";".join(findings)


def privacy_audit(config: Stage4AConfig, paths: Sequence[Path]) -> tuple[bool, str]:
    findings: list[str] = []
    username = os.environ.get("USERNAME", "").lower()
    for path in paths:
        if not path.exists() or path.suffix.lower() not in {".csv", ".md", ".yaml", ".py"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        scrubbed = re.sub(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", "<sha256>", text, flags=re.IGNORECASE)
        try:
            display = config.relative(path)
        except ValueError:
            # Synthetic tests may live outside the project; never echo their machine path.
            display = path.name
        if re.search(r"[A-Za-z]:[\\/]", scrubbed):
            findings.append(f"{display}:absolute_path")
        if username and username in scrubbed.lower():
            findings.append(f"{display}:local_username")
        if re.search(r"(?<![0-9])0[0-9]{9}(?![0-9])", scrubbed):
            findings.append(f"{display}:raw_article_value")
    return not findings, "No entity values, local absolute paths, or local username found." if not findings else ";".join(findings)


def protocol_gate_rows(
    config: Stage4AConfig, upstream: Sequence[Mapping[str, Any]], leakage: Sequence[Mapping[str, Any]],
    capabilities: Sequence[Mapping[str, str]], *, no_ranking: tuple[bool, str], privacy: tuple[bool, str],
) -> list[dict[str, Any]]:
    leakage_map = {row["leakage_gate_id"]: row for row in leakage}
    frozen = config.raw["candidate_replay_contract"]["frozen_retrieval"]
    checks = [
        ("G1_UPSTREAM_LINEAGE", all(row["status"] == "PASS" for row in upstream), "All declared upstream fingerprints and formal statuses match."),
        ("G2_STAGE3E_CONTRACT_IDENTITY", config.raw["candidate_replay_contract"]["candidate_contract_version"] == CONTRACT_VERSION and frozen["quota"] == "T100", "Stage 3E contract v1 and locked TT1-T100 identity match."),
        ("G3_RANKING_CALENDAR_VALIDITY", all(_inclusive_days(row["label_start"], row["label_end"]) == row["label_window_days"] for row in config.raw["calendar"].values()), "All calendar windows and request cutoffs validate."),
        ("G4_OUTER_TRAIN_CONTAINMENT", leakage_map["L1_OUTER_TRAIN_CONTAINMENT"]["status"] == "PASS", leakage_map["L1_OUTER_TRAIN_CONTAINMENT"]["evidence"]),
        ("G5_VALIDATION_TEST_ISOLATION", all(leakage_map[name]["status"] == "PASS" for name in ("L3_TRAIN_VALIDATION_BOUNDARY", "L4_VALIDATION_TEST_BOUNDARY", "L5_TEST_LABEL_SELECTION_ISOLATION")), "Train/Validation/Test are disjoint and Test is not selectable."),
        ("G6_FIRST7D_DIAGNOSTIC_ISOLATION", leakage_map["L6_FIRST7D_TEST_ISOLATION"]["status"] == "PASS", leakage_map["L6_FIRST7D_TEST_ISOLATION"]["evidence"]),
        ("G7_HISTORICAL_REPLAY_CONTRACT", leakage_map["L9_HISTORICAL_REPLAY"]["status"] == "PASS" and not any(row["status"] == "BLOCKED" for row in capabilities), "Replay semantics are frozen; Stage 4B adaptations are explicitly recorded."),
        ("G8_FEATURE_TIME_CONTRACT", leakage_map["L8_HISTORICAL_FEATURE_CUTOFF"]["status"] == "PASS", leakage_map["L8_HISTORICAL_FEATURE_CUTOFF"]["evidence"]),
        ("G9_LABEL_NEGATIVE_SEMANTICS", config.raw["negative_sampling"]["max_unobserved_candidates_per_query"] == 100 and config.raw["label_semantics"]["feedback"] == "B_user_item_day_binary", "B-feedback labels and deterministic cap=100 are frozen; label=0 is unobserved purchase."),
        ("G10_CANDIDATE_LABEL_SEPARATION", leakage_map["L7_CANDIDATE_LABEL_SEPARATION"]["status"] == "PASS", leakage_map["L7_CANDIDATE_LABEL_SEPARATION"]["evidence"]),
        ("G11_FINAL_REFIT_DISCIPLINE", config.raw["final_refit_discipline"]["test_supervision_allowed"] is False and config.raw["final_refit_discipline"]["historical_feature_snapshots_retained"] is True, "Final refit uses locked settings and retains snapshot-time features."),
        ("G12_TEST_LOCK_DISCIPLINE", config.raw["test_discipline"]["sequence"][2] == "ranking_lock_written" and config.raw["test_discipline"]["selection_allowed"] is False, "Ranking lock precedes final refit/Test; Test selection is forbidden."),
        ("G13_NO_RANKING_IMPLEMENTATION", no_ranking[0], no_ranking[1]),
        ("G14_PRIVACY", privacy[0], privacy[1]),
        ("G15_RANKING_POPULATION_CONTRACT", leakage_map["L11_RANKING_POPULATION_CONTRACT"]["status"] == "PASS", leakage_map["L11_RANKING_POPULATION_CONTRACT"]["evidence"]),
        ("G16_DETERMINISTIC_NEGATIVE_HASH_CONTRACT", leakage_map["L12_DETERMINISTIC_NEGATIVE_HASH"]["status"] == "PASS", leakage_map["L12_DETERMINISTIC_NEGATIVE_HASH"]["evidence"]),
        ("G17_FIRST7D_POST_SELECTION_ONLY", leakage_map["L13_FIRST7D_POST_SELECTION_ONLY"]["status"] == "PASS" and config.raw["first7d_diagnostic_discipline"]["automatic_test_authorization_allowed"] is False, leakage_map["L13_FIRST7D_POST_SELECTION_ONLY"]["evidence"]),
        ("G18_HISTORICAL_FINGERPRINT_SEMANTICS", leakage_map["L14_HISTORICAL_FINGERPRINT_SEMANTICS"]["status"] == "PASS", leakage_map["L14_HISTORICAL_FINGERPRINT_SEMANTICS"]["evidence"]),
        ("G19_STAGE4B_NOT_ENTERED_DURING_STAGE4A", leakage_map["L15_STAGE4B_NOT_ENTERED_DURING_STAGE4A"]["status"] == "PASS", leakage_map["L15_STAGE4B_NOT_ENTERED_DURING_STAGE4A"]["evidence"]),
    ]
    rows = [{"gate_id": gate, "mandatory": True, "status": "PASS" if passed else "FAIL", "reason": reason, "stage_status": "PASSED_PROTOCOL_GATE" if passed else "FAILED_PROTOCOL_GATE"} for gate, passed, reason in checks]
    if any(row["status"] != "PASS" for row in rows):
        raise Stage4AProtocolError("FAILED_PROTOCOL_GATE")
    return rows


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "ranking_protocol").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("*stage_4a_protocol*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
