from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from data_audit.config import find_project_root


@dataclass(frozen=True)
class RetrievalConfig:
    root: Path
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> "RetrievalConfig":
        resolved = path.resolve()
        raw = json.loads(resolved.read_text(encoding="utf-8"))
        config = cls(find_project_root(resolved), raw, resolved)
        config.validate()
        return config

    def validate(self) -> None:
        if self.raw["stage"] != "3A":
            raise ValueError("配置 stage 必须为 3A")
        if self.raw["candidate_max_k"] != 500 or self.raw["final_k"] != 12:
            raise ValueError("Stage 3A 冻结 candidate_max_k=500、final_k=12")
        if self.raw["evaluation_k"] != [12, 50, 100, 500]:
            raise ValueError("Stage 3A evaluation_k 必须为 [12,50,100,500]")
        if self.raw["trend"]["windows_days"] != [7, 14, 28]:
            raise ValueError("Trend 只允许预注册窗口 7/14/28")
        if self.raw["duckdb"]["threads"] != 1:
            raise ValueError("Stage 3A 正式运行固定 DuckDB 单线程")

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
        return max(
            int(values["memory_min_mb"]),
            min(int(values["memory_max_mb"]), selected),
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.path.read_bytes())
        for key in ("split", "population"):
            digest.update(b"\0")
            digest.update(
                self.resolve(self.raw["input_manifests"][key]).read_bytes()
            )
        return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "retrieval").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("run_stage_3a*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
