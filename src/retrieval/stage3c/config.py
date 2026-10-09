from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from data_audit.config import find_project_root


@dataclass(frozen=True)
class Stage3CConfig:
    root: Path
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> "Stage3CConfig":
        resolved = path.resolve()
        value = cls(find_project_root(resolved), json.loads(resolved.read_text(encoding="utf-8")), resolved)
        value.validate()
        return value

    def validate(self) -> None:
        if self.raw.get("stage") != "3C":
            raise ValueError("配置 stage 必须为 3C")
        content = self.raw["content"]
        frozen = {
            "max_seed_items_per_user": 20,
            "top_neighbors_per_seed": 200,
            "candidate_max_k": 500,
            "seen_item_exclusion": True,
            "candidate_shortfall_fallback": False,
        }
        for field, expected in frozen.items():
            if content.get(field) != expected:
                raise ValueError(f"Stage 3C 冻结 Content 参数漂移: {field}")
        if list(self.raw["content_variants"]) != ["C0", "C1", "C2", "C3"]:
            raise ValueError("Content 版本只能按 C0-C3 预注册")
        union = self.raw["union"]
        if union["candidate_budget"] != 500:
            raise ValueError("Stage 3C union 总预算必须为 500")
        expected = {
            "Q0": {"repurchase": 50, "itemcf": 300, "content": 0, "trend": 100},
            "Q1": {"repurchase": 50, "itemcf": 250, "content": 50, "trend": 100},
            "Q2": {"repurchase": 50, "itemcf": 200, "content": 100, "trend": 100},
        }
        if union["variants"] != expected:
            raise ValueError("Q0/Q1/Q2 source_rank_cap 漂移")
        if union["quota_semantics"] != "source_rank_cap_before_cross_source_dedup":
            raise ValueError("Stage 3C 必须沿用 Stage 3B 预去重 source_rank 截断")
        if union["near_tie_tolerance"] != 0.0005 or union["strong_pass_delta"] != 0.002:
            raise ValueError("Stage 3C 预注册阈值漂移")
        if self.raw["bootstrap"]["iterations"] < 1000:
            raise ValueError("paired bootstrap 至少 1000 次")
        sources = [row["source_column"] for row in self.raw["canonical_fields"]]
        semantics = [row["semantic_field"] for row in self.raw["canonical_fields"]]
        if len(sources) != len(set(sources)) or len(semantics) != len(set(semantics)):
            raise ValueError("canonical source/semantic field 不得重复")

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
        for name in ("split", "population", "stage_3a", "stage_3b"):
            digest.update(b"\0")
            digest.update(self.resolve(self.raw["input_manifests"][name]).read_bytes())
        for name in ("stage_3a_lock", "stage_3b_lock"):
            digest.update(b"\0")
            digest.update(self.resolve(self.raw["upstream"][name]).read_bytes())
        return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "retrieval" / "stage3c").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("run_stage_3c*.py")))
    paths.extend(sorted((root / "scripts").glob("finalize_stage_3c*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
