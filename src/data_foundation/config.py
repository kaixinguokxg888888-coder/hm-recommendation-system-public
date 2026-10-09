from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from data_audit.config import find_project_root


@dataclass(frozen=True)
class FoundationConfig:
    root: Path
    raw: dict[str, Any]
    config_path: Path

    @classmethod
    def load(cls, path: Path) -> "FoundationConfig":
        absolute = path.resolve()
        raw = json.loads(absolute.read_text(encoding="utf-8"))
        return cls(find_project_root(absolute), raw, absolute)

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    @property
    def data_paths(self) -> dict[str, Path]:
        return {
            name: self.resolve(self.raw["paths"][name])
            for name in ("articles", "customers", "transactions")
        }

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
        cfg = self.raw["duckdb"]
        available_mb = psutil.virtual_memory().available // (1024 * 1024)
        chosen = int(available_mb * float(cfg["memory_fraction_of_available"]))
        return max(int(cfg["memory_min_mb"]), min(int(cfg["memory_max_mb"]), chosen))

    @property
    def fingerprint(self) -> str:
        contracts = self.resolve(self.raw["contracts_path"]).read_bytes()
        return hashlib.sha256(self.config_path.read_bytes() + b"\0" + contracts).hexdigest()
