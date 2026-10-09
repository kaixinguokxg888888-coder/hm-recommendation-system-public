from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import psutil


class Stage4B1ConfigError(ValueError):
    """Raised when the historical replay configuration changes frozen semantics."""


@dataclass(frozen=True)
class HistoricalSnapshot:
    snapshot_id: str
    history_start: str
    history_end: str
    request_date: str
    label_start: str
    label_end: str

    @property
    def as_of_date(self) -> str:
        return self.history_end


@dataclass(frozen=True)
class Stage4B1Config:
    root: Path
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Stage4B1Config":
        resolved = path.resolve()
        config = cls(resolved.parents[1], resolved, json.loads(resolved.read_text(encoding="utf-8")))
        config.validate()
        return config

    def resolve(self, value: str) -> Path:
        return self.root / value

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    @property
    def artifact_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["artifact_directory"])

    @property
    def private_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["private_directory"])

    @property
    def checkpoint_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["checkpoint_directory"])

    @property
    def temp_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["temp_directory"])

    @property
    def report_path(self) -> Path:
        return self.resolve(self.raw["paths"]["report"])

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def memory_limit_mb(self) -> int:
        values = self.raw["duckdb"]
        available_mb = int(psutil.virtual_memory().available / (1024 * 1024))
        requested = int(available_mb * float(values["memory_fraction_of_available"]))
        return max(int(values["memory_min_mb"]), min(int(values["memory_max_mb"]), requested))

    @property
    def snapshot(self) -> HistoricalSnapshot:
        value = self.raw["snapshot"]
        return HistoricalSnapshot(
            snapshot_id=value["snapshot_id"],
            history_start=value["history_start"],
            history_end=value["history_cutoff"],
            request_date=value["request_date"],
            label_start=value["query_user_label_start"],
            label_end=value["query_user_label_end"],
        )

    def validate(self) -> None:
        if self.raw.get("stage") != "4B-1":
            raise Stage4B1ConfigError("Stage 必须为 4B-1")
        snapshot = self.raw["snapshot"]
        expected_snapshot = {
            "snapshot_id": "ranking_train_14d_v1",
            "as_of_date": "2020-08-11",
            "request_date": "2020-08-12",
            "history_cutoff": "2020-08-11",
            "query_user_label_start": "2020-08-12",
            "query_user_label_end": "2020-08-25",
        }
        for key, expected in expected_snapshot.items():
            if snapshot.get(key) != expected:
                raise Stage4B1ConfigError(f"historical snapshot 冻结字段变化: {key}")
        if date.fromisoformat(snapshot["history_cutoff"]) >= date.fromisoformat(snapshot["request_date"]):
            raise Stage4B1ConfigError("history cutoff 必须早于 request date")

        frozen = self.raw["frozen_retrieval"]
        expected = {
            "candidate_contract_version": "stage_3e_candidate_contract_v1",
            "candidate_protocol_version": "source_rank_cap_pre_dedup_v1",
            "semantic_fingerprint_version": "candidate_semantic_sha256_user_merkle_v1",
            "model": "TT1",
            "quota": "T100",
            "locked_epoch": 1,
            "candidate_budget": 500,
            "source_caps": {"repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100},
            "insertion_order": ["repurchase", "itemcf", "two_tower", "trend", "popularity_fallback"],
            "quota_semantics": "source_rank_cap_before_cross_source_dedup",
            "duplicate_semantics": "duplicates_consume_pre_dedup_source_rank_cap_and_merge_provenance",
            "fallback_semantics": "popularity_unique_fill_to_500",
            "source_score_semantics": "source_local_raw_scores_never_summed_or_cross_source_calibrated",
        }
        for key, value in expected.items():
            if frozen.get(key) != value:
                raise Stage4B1ConfigError(f"frozen retrieval 字段变化: {key}")
        ann_expected = {
            "algorithm": "hnsw", "metric": "cosine", "m": 32,
            "ef_construction": 200, "ef_search": 200,
            "initial_query_k": 1000, "retry_query_k": 2000,
            "exact_rerank": True, "mean_overlap_min": 0.98, "p5_overlap_min": 0.95,
        }
        if frozen.get("ann") != ann_expected:
            raise Stage4B1ConfigError("ANN 冻结参数变化")
        execution = self.raw["execution"]
        if int(execution["partitions"]) != 32 or execution["replay_partition_order"] != "reverse":
            raise Stage4B1ConfigError("partition/replay scheduling 协议变化")
