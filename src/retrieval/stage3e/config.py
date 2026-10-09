from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from data_audit.config import find_project_root


EXPECTED_SNAPSHOTS = ["validation_14d", "test_14d"]
EXPECTED_CAPS = {"repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100}
EXPECTED_ORDER = ["repurchase", "itemcf", "two_tower", "trend", "popularity_fallback"]


@dataclass(frozen=True)
class Stage3EConfig:
    root: Path
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> "Stage3EConfig":
        resolved = path.resolve()
        value = cls(find_project_root(resolved), json.loads(resolved.read_text(encoding="utf-8")), resolved)
        value.validate()
        return value

    def validate(self) -> None:
        if self.raw.get("stage") != "3E":
            raise ValueError("配置 stage 必须为 3E")
        if self.raw.get("candidate_contract_version") != "stage_3e_candidate_contract_v1":
            raise ValueError("Stage 3E candidate contract version 漂移")
        if self.raw.get("candidate_protocol_version") != "source_rank_cap_pre_dedup_v1":
            raise ValueError("Stage 3E candidate protocol version 漂移")
        if self.raw.get("migration_semantics") != "stage3d_locked_t100_rank_preserving_v1":
            raise ValueError("Stage 3E 必须执行 rank-preserving contract migration")
        if self.raw.get("snapshots") != EXPECTED_SNAPSHOTS:
            raise ValueError("Stage 3E 只能物化 Validation/Test 正式 snapshot")
        frozen = self.raw["frozen_retrieval"]
        checks = {
            "model": "TT1",
            "quota": "T100",
            "ann_ef_search": 200,
            "candidate_budget": 500,
            "source_caps": EXPECTED_CAPS,
            "quota_semantics": "source_rank_cap_before_cross_source_dedup",
            "duplicate_semantics": "duplicates_consume_pre_dedup_source_rank_cap_and_merge_provenance",
            "insertion_order": EXPECTED_ORDER,
            "fallback": "popularity_unique_fill_to_500",
        }
        for field, expected in checks.items():
            if frozen.get(field) != expected:
                raise ValueError(f"Stage 3E frozen retrieval definition 漂移: {field}")
        semantic = self.raw["semantic_fingerprint"]
        if semantic.get("version") != "candidate_semantic_sha256_user_merkle_v1":
            raise ValueError("Stage 3E semantic fingerprint version 漂移")
        expected_fields = ["snapshot_id", "customer_key", "candidate_rank", "article_key"]
        if semantic.get("membership_fields") != expected_fields or semantic.get("canonical_order") != expected_fields:
            raise ValueError("Stage 3E semantic fingerprint canonical fields 漂移")
        execution = self.raw["execution"]
        if execution.get("partitions") != 32 or execution.get("replay_partition_order") != "reverse":
            raise ValueError("Stage 3E partition/replay protocol 漂移")
        duckdb = self.raw["duckdb"]
        if duckdb.get("threads") != 1 or duckdb.get("preserve_insertion_order") is not False:
            raise ValueError("Stage 3E DuckDB determinism/resource protocol 漂移")

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

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
    def memory_limit_mb(self) -> int:
        values = self.raw["duckdb"]
        available = psutil.virtual_memory().available // (1024 * 1024)
        selected = int(available * float(values["memory_fraction_of_available"]))
        return max(int(values["memory_min_mb"]), min(int(values["memory_max_mb"]), selected))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.path.read_bytes())
        for name in (
            "split_manifest", "population_manifest", "stage_3a_manifest", "stage_3b_manifest",
            "stage_3c_manifest", "stage_3d_manifest", "stage_3a_lock", "stage_3b_lock",
            "stage_3c_lock", "stage_3d_lock", "stage_3d_config",
        ):
            digest.update(b"\0")
            digest.update(self.resolve(self.raw["upstream"][name]).read_bytes())
        return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "retrieval" / "stage3e").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("run_stage_3e*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
