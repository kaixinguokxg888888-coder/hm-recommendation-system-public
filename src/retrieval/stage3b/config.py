from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from data_audit.config import find_project_root


@dataclass(frozen=True)
class Stage3BConfig:
    root: Path
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> "Stage3BConfig":
        resolved = path.resolve()
        value = cls(
            root=find_project_root(resolved),
            raw=json.loads(resolved.read_text(encoding="utf-8")),
            path=resolved,
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.raw["stage"] != "3B":
            raise ValueError("配置 stage 必须为 3B")
        itemcf = self.raw["itemcf"]
        frozen = {
            "variants": ["cosine", "iuf"],
            "max_items_per_user_for_cooccurrence": 50,
            "top_neighbors_per_item": 200,
            "max_seed_items_per_user": 20,
            "seen_item_exclusion": True,
            "candidate_max_k": 500,
        }
        for field, expected in frozen.items():
            if itemcf[field] != expected:
                raise ValueError(f"Stage 3B 冻结参数漂移: {field}")
        if self.raw["evaluation_k"] != [12, 50, 100, 500]:
            raise ValueError("Stage 3B evaluation_k 必须为 [12,50,100,500]")
        union = self.raw["union"]
        if union["candidate_budget"] != 500:
            raise ValueError("Stage 3B union 总预算必须为 500")
        if union["base_quotas"] != {"repurchase": 50, "trend": 100, "itemcf": 0}:
            raise ValueError("Base Union 固定配额漂移")
        if union["itemcf_quotas"] != {"repurchase": 50, "itemcf": 300, "trend": 100}:
            raise ValueError("ItemCF Union 固定配额漂移")
        if self.raw["bootstrap"]["iterations"] < 1000:
            raise ValueError("paired bootstrap 至少 1000 次")
        if self.raw["duckdb"]["threads"] != 1:
            raise ValueError("Stage 3B 正式运行固定 DuckDB 单线程")

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
    def lock_path(self) -> Path:
        return self.resolve(self.raw["test"]["lock_path"])

    @property
    def memory_limit_mb(self) -> int:
        values = self.raw["duckdb"]
        available = psutil.virtual_memory().available // (1024 * 1024)
        selected = int(available * float(values["memory_fraction_of_available"]))
        return max(int(values["memory_min_mb"]), min(int(values["memory_max_mb"]), selected))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.path.read_bytes())
        for name in ("split", "population", "stage_3a"):
            digest.update(b"\0")
            digest.update(self.resolve(self.raw["input_manifests"][name]).read_bytes())
        digest.update(b"\0")
        digest.update(self.resolve(self.raw["upstream"]["stage_3a_lock"]).read_bytes())
        return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "retrieval" / "stage3b").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("run_stage_3b*.py")))
    paths.extend(sorted((root / "scripts").glob("finalize_stage_3b*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
