from __future__ import annotations

import copy
import csv
import hashlib
from pathlib import Path

import pytest

from ranking_protocol.protocol import (
    CONTRACT_VERSION,
    EXPECTED_CAPS,
    EXPECTED_ORDER,
    PROTOCOL_VERSION,
    Stage4AConfig,
    Stage4AProtocolError,
    calendar_rows,
    deterministic_unobserved_sample,
    historical_replay_fingerprint_matches,
    leakage_audit_rows,
    negative_sampling_digest,
    no_ranking_implementation_audit,
    privacy_audit,
    replay_capability_rows,
    replay_contract_rows,
    source_fingerprint,
    validate_upstream,
)


CONFIG_PATH = Path("configs/stage_4a_ranking_time_protocol.yaml")


def _config() -> Stage4AConfig:
    return Stage4AConfig.load(CONFIG_PATH)


def _mutated(config: Stage4AConfig) -> tuple[Stage4AConfig, dict[str, object]]:
    raw = copy.deepcopy(config.raw)
    return Stage4AConfig(config.root, config.path, raw), raw


def _audit_map() -> dict[str, dict[str, str]]:
    return {row["leakage_gate_id"]: row for row in leakage_audit_rows(_config())}


def test_ranking_train_calendar_is_exact() -> None:
    row = _config().raw["calendar"]["ranking_train_14d_v1"]
    assert (row["as_of_date"], row["request_date"], row["history_cutoff"]) == (
        "2020-08-11", "2020-08-12", "2020-08-11",
    )
    assert (row["label_start"], row["label_end"]) == ("2020-08-12", "2020-08-25")


@pytest.mark.parametrize("snapshot", ["ranking_train_14d_v1", "validation_14d", "test_14d"])
def test_main_label_windows_are_exactly_14_inclusive_days(snapshot: str) -> None:
    assert _config().raw["calendar"][snapshot]["label_window_days"] == 14


def test_ranking_train_labels_are_contained_in_outer_train() -> None:
    assert _audit_map()["L1_OUTER_TRAIN_CONTAINMENT"]["status"] == "PASS"


def test_ranking_train_and_validation_do_not_overlap() -> None:
    assert _audit_map()["L3_TRAIN_VALIDATION_BOUNDARY"]["status"] == "PASS"


def test_validation_and_test_do_not_overlap() -> None:
    assert _audit_map()["L4_VALIDATION_TEST_BOUNDARY"]["status"] == "PASS"


def test_first7d_is_validation_subset_and_not_an_independent_request() -> None:
    config = _config()
    row = config.raw["calendar"]["ranking_validation_first7d_diagnostic"]
    assert row["candidate_snapshot_id"] == "validation_14d"
    assert row["is_independent_request_snapshot"] is False
    assert _audit_map()["L6_FIRST7D_TEST_ISOLATION"]["status"] == "PASS"


def test_feature_cutoff_is_strictly_before_future_labels() -> None:
    assert _audit_map()["L2_FEATURE_BEFORE_LABEL"]["status"] == "PASS"
    assert _config().raw["feature_time_contract"]["dynamic_feature_rule"] == "feature_timestamp_le_snapshot_as_of_date"


def test_candidate_membership_forbids_label_aware_injection() -> None:
    replay = _config().raw["candidate_replay_contract"]
    assert replay["membership_label_access_allowed"] is False
    assert replay["label_aware_candidate_injection_allowed"] is False
    assert "not_injected" in _config().raw["label_semantics"]["candidate_miss"]


def test_ranking_training_population_contract_is_historical_warm_equivalent() -> None:
    population = _config().raw["population_contract"]
    assert population["main_optimization_population"] == "historical_evaluable_warm_users_v1"
    assert population["ranking_train_all_label_users"]["definition"] == "at_least_one_B_feedback_positive_in_2020_08_12_through_2020_08_25"
    warm = population["historical_evaluable_warm_users_v1"]
    assert warm["requires_future_B_feedback_positive"] is True
    assert warm["requires_pre_cutoff_B_feedback_history_interaction"] is True
    assert warm["history_cutoff"] == "2020-08-11"
    assert warm["requires_valid_internal_customer_article_keys"] is True
    assert population["full_historical_population_materialized_in_stage_4a"] is False


def test_cold_history_population_is_separate_from_main_optimization_and_average() -> None:
    cold = _config().raw["population_contract"]["cold_history_label_users"]
    assert cold["included_in_main_optimization_population"] is False
    assert cold["included_in_main_evaluation_average"] is False
    assert cold["audit_and_evaluation_semantics"] == "separate"


def test_stage3e_candidate_and_protocol_versions_match_exactly() -> None:
    replay = _config().raw["candidate_replay_contract"]
    assert replay["candidate_contract_version"] == CONTRACT_VERSION
    assert replay["candidate_protocol_version"] == PROTOCOL_VERSION


def test_source_caps_match_locked_t100_exactly() -> None:
    assert _config().raw["candidate_replay_contract"]["frozen_retrieval"]["source_caps"] == EXPECTED_CAPS


def test_insertion_order_and_duplicate_semantics_match_stage3e() -> None:
    frozen = _config().raw["candidate_replay_contract"]["frozen_retrieval"]
    assert frozen["insertion_order"] == EXPECTED_ORDER
    assert frozen["quota_semantics"] == "source_rank_cap_before_cross_source_dedup"
    assert frozen["duplicate_semantics"] == "duplicates_consume_pre_dedup_source_rank_cap_and_merge_provenance"


def test_test_is_forbidden_for_selection() -> None:
    config = _config()
    assert config.raw["test_discipline"]["selection_allowed"] is False
    assert config.raw["validation_discipline"]["test_access_allowed"] is False
    assert _audit_map()["L5_TEST_LABEL_SELECTION_ISOLATION"]["status"] == "PASS"


def test_final_refit_forbids_test_supervision_and_retains_historical_features() -> None:
    final_refit = _config().raw["final_refit_discipline"]
    assert final_refit["supervision_snapshots"] == ["ranking_train_14d_v1", "validation_14d"]
    assert final_refit["test_supervision_allowed"] is False
    assert final_refit["historical_feature_snapshots_retained"] is True
    assert final_refit["internal_validation_selection"] is False


def test_negative_cap_is_frozen_at_100_and_not_searchable() -> None:
    sampling = _config().raw["negative_sampling"]
    assert sampling["max_unobserved_candidates_per_query"] == 100
    assert sampling["searchable_in_stage_4b"] is False
    config, raw = _mutated(_config())
    raw["negative_sampling"]["max_unobserved_candidates_per_query"] = 200  # type: ignore[index]
    with pytest.raises(Stage4AProtocolError, match="sampling cap"):
        config.validate()


def test_deterministic_unobserved_sampling_is_order_invariant_and_capped() -> None:
    rows = [{"snapshot_id": "ranking_train_14d_v1", "customer_key": 7, "article_key": value} for value in range(130)]
    first = deterministic_unobserved_sample(rows, fixed_seed=20260811)
    second = deterministic_unobserved_sample(reversed(rows), fixed_seed=20260811)
    assert len(first) == 100
    assert [row["article_key"] for row in first] == [row["article_key"] for row in second]


def test_negative_sampling_digest_uses_exact_utf8_sha256_payload() -> None:
    row = {"snapshot_id": "ranking_train_14d_v1", "customer_key": 7, "article_key": 11}
    payload = "ranking_train_14d_v1|7|11|20260811"
    assert negative_sampling_digest(row, fixed_seed=20260811) == hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_deterministic_sampling_is_partition_order_invariant() -> None:
    rows = [{"snapshot_id": "ranking_train_14d_v1", "customer_key": 9, "article_key": value} for value in range(160)]
    partitions = [rows[:40], rows[40:80], rows[80:120], rows[120:]]
    forward = deterministic_unobserved_sample((row for part in partitions for row in part), fixed_seed=20260811)
    reverse = deterministic_unobserved_sample((row for part in reversed(partitions) for row in part), fixed_seed=20260811)
    assert [row["article_key"] for row in forward] == [row["article_key"] for row in reverse]


def test_deterministic_sampling_repeated_run_and_seed_binding() -> None:
    rows = [{"snapshot_id": "ranking_train_14d_v1", "customer_key": 5, "article_key": value} for value in range(140)]
    first = deterministic_unobserved_sample(rows, fixed_seed=20260811)
    repeat = deterministic_unobserved_sample(rows, fixed_seed=20260811)
    other_seed = deterministic_unobserved_sample(rows, fixed_seed=20260812)
    assert [row["article_key"] for row in first] == [row["article_key"] for row in repeat]
    assert [row["article_key"] for row in first] != [row["article_key"] for row in other_seed]


def test_deterministic_sampling_keeps_every_row_when_n_le_100() -> None:
    rows = [{"snapshot_id": "ranking_train_14d_v1", "customer_key": 3, "article_key": value} for value in range(100)]
    sampled = deterministic_unobserved_sample(reversed(rows), fixed_seed=20260811)
    assert len(sampled) == 100
    assert {row["article_key"] for row in sampled} == set(range(100))


def test_negative_hash_contract_forbids_runtime_and_partition_inputs() -> None:
    sampling = _config().raw["negative_sampling"]
    assert sampling["hash_algorithm"] == "SHA-256"
    assert sampling["encoding"] == "UTF-8"
    assert sampling["canonical_payload"] == "{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}"
    assert sampling["sort_order"] == ["digest_ASC", "article_key_ASC"]
    assert sampling["python_builtin_hash_allowed"] is False
    assert set(sampling["forbidden_inputs"]) == {
        "filesystem_path", "partition_id", "row_order", "process_id", "python_runtime_hash_seed",
    }


def test_deterministic_sampling_rejects_cap_search() -> None:
    with pytest.raises(Stage4AProtocolError, match="冻结为 100"):
        deterministic_unobserved_sample([], fixed_seed=1, max_rows=50)


def test_query_group_key_includes_snapshot() -> None:
    assert _config().raw["query_group"]["key"] == ["snapshot_id", "customer_key"]


def test_all_zero_group_excluded_from_optimizer_but_retained_in_evaluation() -> None:
    group = _config().raw["query_group"]
    assert group["all_zero_relevance_training"] == "exclude_from_lambdarank_optimization"
    assert group["all_zero_relevance_evaluation"] == "retain_in_end_to_end_population"
    population_zero = _config().raw["population_contract"]["all_zero_relevance_group"]
    assert population_zero["classification"] == "retrieval_miss"
    assert population_zero["future_lambdarank_optimization"] == "exclude"
    assert population_zero["end_to_end_evaluation"] == "retain"


def test_first7d_is_single_post_selection_diagnostic_not_secondary_selection() -> None:
    discipline = _config().raw["first7d_diagnostic_discipline"]
    assert discipline["execution_order"] == [
        "complete_full_validation_14d_selection", "freeze_single_validation_14d_winner",
        "run_one_first7d_locked_winner_vs_stage3e_baseline",
    ]
    assert discipline["challenger_scope"] == "single_locked_14d_winner"
    assert discipline["run_count"] == 1
    assert discipline["selection_allowed"] is False
    assert discipline["automatic_retuning_allowed"] is False


def test_first7d_direction_failure_requires_review_and_does_not_authorize_test() -> None:
    discipline = _config().raw["first7d_diagnostic_discipline"]
    assert discipline["direction_failure_status"] == "VALIDATION_ROBUSTNESS_NOT_MET"
    assert discipline["automatic_test_authorization_allowed"] is False
    assert discipline["failure_action"] == "return_for_technical_review"


def test_historical_fingerprint_contract_separates_protocol_and_semantic_identity() -> None:
    contract = _config().raw["candidate_replay_contract"]["historical_fingerprint_contract"]
    assert contract["protocol_identity_name"] == "historical_retrieval_protocol_identity"
    assert contract["first_formal_materialization"] == "create_new_snapshot_specific_historical_candidate_semantic_fingerprint"
    assert contract["membership_or_rank_may_differ_from_validation_and_test"] is True
    assert contract["equality_with_validation_fingerprint_required"] is False
    assert contract["equality_with_test_fingerprint_required"] is False


def test_historical_replay_fingerprint_only_compares_same_historical_snapshot() -> None:
    assert historical_replay_fingerprint_matches(
        formal_snapshot_id="ranking_train_14d_v1", replay_snapshot_id="ranking_train_14d_v1",
        formal_fingerprint="a" * 64, replay_fingerprint="a" * 64,
    )
    assert not historical_replay_fingerprint_matches(
        formal_snapshot_id="ranking_train_14d_v1", replay_snapshot_id="ranking_train_14d_v1",
        formal_fingerprint="a" * 64, replay_fingerprint="b" * 64,
    )
    with pytest.raises(Stage4AProtocolError, match="不得比较 Validation/Test"):
        historical_replay_fingerprint_matches(
            formal_snapshot_id="ranking_train_14d_v1", replay_snapshot_id="validation_14d",
            formal_fingerprint="a" * 64, replay_fingerprint="a" * 64,
        )


def test_final_acceptance_allows_future_stage4b_consumption_but_did_not_enter_it() -> None:
    config = _config()
    assert config.raw["protocol_status"] == "PASSED_PROTOCOL_GATE"
    assert config.raw["final_acceptance_status"] == "ACCEPTED"
    assert config.raw["stage_4b_authorized"] is True
    assert config.raw["stage_4b_protocol_consumption_status"] == "ALLOWED_AFTER_FINAL_ACCEPTED_WITH_SEPARATE_PROMPT"
    assert config.raw["stage_4b_entered_during_stage_4a"] is False


def test_no_ranking_implementation_gate_passes() -> None:
    passed, evidence = no_ranking_implementation_audit(_config())
    assert passed, evidence


def test_privacy_scan_rejects_machine_path_and_entity_shaped_value(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.csv"
    unsafe.write_text("value\n" + "C" + ":/temp\n" + "0" + "123456789\n", encoding="utf-8")
    passed, evidence = privacy_audit(_config(), [unsafe])
    assert not passed
    assert "absolute_path" in evidence and "raw_article_value" in evidence


def test_replay_contract_covers_all_components_and_forbids_labels_and_drift() -> None:
    rows = replay_contract_rows(_config())
    assert [row["component"] for row in rows] == [
        "repurchase", "itemcf", "two_tower", "ann", "trend", "popularity", "fusion",
    ]
    assert all(row["label_access_allowed"] is False for row in rows)
    assert all(row["parameter_selection_allowed"] is False for row in rows)
    assert all(row["semantic_drift_allowed"] is False for row in rows)


def test_replay_capability_has_no_blocked_component() -> None:
    rows = replay_capability_rows(_config())
    assert len(rows) == 8
    assert {row["status"] for row in rows} <= {"READY", "NEEDS_STAGE4B_ADAPTATION"}


def test_upstream_fingerprints_and_formal_statuses_match() -> None:
    rows = validate_upstream(_config())
    assert len(rows) >= 20
    assert {row["status"] for row in rows} == {"PASS"}


def test_calendar_rows_carry_candidate_contract_identity() -> None:
    rows = calendar_rows(_config())
    assert len(rows) == 4
    assert {row["candidate_contract_version"] for row in rows} == {CONTRACT_VERSION}
    assert {row["candidate_protocol_version"] for row in rows} == {PROTOCOL_VERSION}


def test_validation_and_test_evaluate_all_500_candidates() -> None:
    config = _config()
    assert config.raw["validation_discipline"]["evaluation_candidate_count"] == 500
    assert config.raw["test_discipline"]["evaluation_candidate_count"] == 500
    assert config.raw["negative_sampling"]["validation_candidate_downsampling"] is False
    assert config.raw["negative_sampling"]["test_candidate_downsampling"] is False


def test_primary_ranking_metric_is_map12_not_recall500() -> None:
    evaluation = _config().raw["ranking_evaluation"]
    assert evaluation["primary"] == "MAP@12"
    assert evaluation["auxiliary"] == ["NDCG@12", "Recall@12", "HitRate@12"]
    assert evaluation["recall_at_500_is_primary_ranking_metric"] is False


def test_stage4a_execution_flags_prohibit_stage4b_work() -> None:
    execution = _config().raw["execution"]
    for field in (
        "historical_full_candidate_replay", "ranking_dataset_materialization",
        "ranking_feature_materialization", "ranking_training", "ranking_prediction",
        "test_ranking_evaluation", "reranking",
    ):
        assert execution[field] is False


def test_static_article_attribute_limitation_is_explicit() -> None:
    limitation = _config().raw["feature_time_contract"]["static_article_attribute_limitation"]
    assert "historical_effective_time_not_verifiable" in limitation


def test_source_fingerprint_is_deterministic() -> None:
    root = _config().root
    assert source_fingerprint(root) == source_fingerprint(root)


def test_generated_public_audits_have_expected_gate_counts_when_present() -> None:
    artifact_dir = Path("artifacts/stage_4a")
    leakage_path = artifact_dir / "leakage_audit.csv"
    gate_path = artifact_dir / "protocol_gate.csv"
    if not leakage_path.exists() or not gate_path.exists():
        pytest.skip("Formal Stage 4A finalizer has not run yet")
    with leakage_path.open(encoding="utf-8-sig", newline="") as handle:
        leakage = list(csv.DictReader(handle))
    with gate_path.open(encoding="utf-8-sig", newline="") as handle:
        gates = list(csv.DictReader(handle))
    assert len(leakage) == 15 and {row["status"] for row in leakage} == {"PASS"}
    assert len(gates) == 19 and {row["status"] for row in gates} == {"PASS"}
    assert gates[-1]["gate_id"] == "G19_STAGE4B_NOT_ENTERED_DURING_STAGE4A"
