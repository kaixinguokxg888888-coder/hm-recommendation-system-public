from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import psutil


class Stage4B3ConfigError(ValueError):
    """Raised when Stage 4B-3 configuration drifts from its formal contract."""


@dataclass(frozen=True)
class Stage4B3Config:
    root: Path
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Stage4B3Config":
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
        if self.raw.get("stage") != "4B-3":
            raise Stage4B3ConfigError("stage 必须为 4B-3")
        snapshot = self.raw["snapshot"]
        expected = {
            "snapshot_id": "ranking_train_14d_v1",
            "as_of_date": "2020-08-11",
            "request_date": "2020-08-12",
        }
        if snapshot != expected:
            raise Stage4B3ConfigError("formal snapshot/calendar 漂移")
        if date.fromisoformat(snapshot["request_date"]) != date.fromisoformat(snapshot["as_of_date"]) + timedelta(days=1):
            raise Stage4B3ConfigError("request_date 必须是 as-of 后一日")
        versions = self.raw["versions"]
        required = {
            "feedback": "B_user_item_day_binary",
            "f0": "core_temporal_retrieval_v1",
            "f1": "core_plus_affinity_v1",
            "candidate_contract": "stage_3e_candidate_contract_v1",
            "candidate_protocol": "source_rank_cap_pre_dedup_v1",
            "optimizer_view": "ranking_optimizer_sha256_100_v1",
        }
        if any(versions.get(key) != value for key, value in required.items()):
            raise Stage4B3ConfigError("frozen feature/upstream version 漂移")
        if self.raw["execution"]["partitions"] != 32:
            raise Stage4B3ConfigError("formal partition count 必须保持 32")
        if self.raw["execution"]["query_partition_expression"] != "customer_key % 32":
            raise Stage4B3ConfigError("query partition function 漂移")
        fields = self.raw["canonical_fields"]
        if len(fields) != 11 or len({row["semantic_field"] for row in fields}) != len(fields):
            raise Stage4B3ConfigError("canonical field contract 必须是 Stage 3C 唯一 11 字段集合")
        denylist = set(self.raw["feature_denylist"])
        if not {"label", "future_positive_count", "candidate_positive_count", "is_retrieval_miss", "optimizer_eligible"} <= denylist:
            raise Stage4B3ConfigError("feature leakage denylist 不完整")
        if self.raw["price_features"]["status"] != "OMITTED_BY_LINEAGE_UNCERTAINTY":
            raise Stage4B3ConfigError("price lineage 未正式解决，不得启用 price feature")
        forbidden = self.raw["strictly_forbidden_downstream_execution"]
        if any(value not in {"NOT ENTERED", "NOT EXECUTED"} for value in forbidden.values()):
            raise Stage4B3ConfigError("下游禁止状态必须保持未进入/未执行")
