from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import psutil


class Stage4B2ConfigError(ValueError):
    """Raised when Stage 4B-2 configuration drifts from the frozen protocol."""


@dataclass(frozen=True)
class Stage4B2Config:
    root: Path
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Stage4B2Config":
        resolved = path.resolve()
        value = cls(resolved.parents[1], resolved, json.loads(resolved.read_text(encoding="utf-8")))
        value.validate()
        return value

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
        available = int(psutil.virtual_memory().available / 1024 / 1024)
        requested = int(available * float(values["memory_fraction_of_available"]))
        return max(int(values["memory_min_mb"]), min(int(values["memory_max_mb"]), requested))

    def validate(self) -> None:
        if self.raw.get("stage") != "4B-2":
            raise Stage4B2ConfigError("stage 必须为 4B-2")
        snapshot = self.raw["snapshot"]
        expected = {
            "snapshot_id": "ranking_train_14d_v1", "as_of_date": "2020-08-11",
            "request_date": "2020-08-12", "label_start": "2020-08-12",
            "label_end": "2020-08-25", "label_window_days": 14,
        }
        for key, value in expected.items():
            if snapshot.get(key) != value:
                raise Stage4B2ConfigError(f"frozen calendar 漂移: {key}")
        start, end = date.fromisoformat(snapshot["label_start"]), date.fromisoformat(snapshot["label_end"])
        if end - start != timedelta(days=13) or date.fromisoformat(snapshot["as_of_date"]) >= start:
            raise Stage4B2ConfigError("label window 必须是 cutoff 后连续 14 天")
        versions = self.raw["versions"]
        required_versions = {
            "feedback": "B_user_item_day_binary",
            "main_population": "historical_evaluable_warm_users_v1",
            "candidate_contract": "stage_3e_candidate_contract_v1",
            "candidate_protocol": "source_rank_cap_pre_dedup_v1",
            "negative_sampling_policy": "sha256_zero_cap100_v1",
        }
        for key, value in required_versions.items():
            if versions.get(key) != value:
                raise Stage4B2ConfigError(f"frozen version 漂移: {key}")
        sampling = self.raw["negative_sampling"]
        if sampling != {
            "fixed_seed": 20260811, "algorithm": "sha256", "encoding": "utf-8",
            "max_zero_per_query": 100,
            "canonical_payload": "{snapshot_id}|{customer_key}|{article_key}|{fixed_seed}",
            "sort": ["digest_ASC", "article_key_ASC"], "keep_all_positives": True,
            "python_builtin_hash_allowed": False, "partition_id_in_payload": False,
        }:
            raise Stage4B2ConfigError("deterministic sampling contract 漂移")
        if self.raw["candidate"]["budget"] != 500 or self.raw["execution"]["partitions"] != 32:
            raise Stage4B2ConfigError("candidate budget/partition count 漂移")
        forbidden = self.raw["strictly_forbidden_downstream_execution"]
        if any(value not in {"NOT ENTERED", "NOT EXECUTED"} for value in forbidden.values()):
            raise Stage4B2ConfigError("下游禁止状态必须保持未进入/未执行")
