from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import psutil


class Stage4B4ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Stage4B4Config:
    root: Path
    path: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "Stage4B4Config":
        resolved = path.resolve()
        value = cls(resolved.parents[1], resolved, json.loads(resolved.read_text(encoding="utf-8")))
        value.validate()
        return value

    def resolve(self, value: str) -> Path:
        return self.root / value

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        available_mb = int(psutil.virtual_memory().available / 1024 / 1024)
        return max(512, min(int(self.raw["execution"]["duckdb_memory_max_mb"]), int(available_mb * 0.35)))

    def validate(self) -> None:
        if (self.raw.get("stage"), self.raw.get("phase")) != (
            "4B-4", "phase1_pre_label_prediction_freeze_v1"
        ):
            raise Stage4B4ConfigError("Stage/phase contract 漂移")
        expected = {
            "train": ("ranking_train_14d_v1", "2020-08-11", "2020-08-12"),
            "validation": ("validation_14d", "2020-08-25", "2020-08-26"),
        }
        for role, values in expected.items():
            row = self.raw["snapshots"][role]
            if tuple(row[key] for key in ("snapshot_id", "as_of_date", "request_date")) != values:
                raise Stage4B4ConfigError(f"{role} calendar 漂移")
            if date.fromisoformat(row["request_date"]) != date.fromisoformat(row["as_of_date"]) + timedelta(days=1):
                raise Stage4B4ConfigError(f"{role} request_date 非 as-of+1")
        params = self.raw["lightgbm"]["params"]
        locked = {
            "objective": "lambdarank", "metric": "None", "label_gain": [0, 1],
            "lambdarank_truncation_level": 15, "lambdarank_norm": True,
            "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 2000,
            "deterministic": True, "force_col_wise": True, "num_threads": 4,
        }
        if any(params.get(key) != value for key, value in locked.items()):
            raise Stage4B4ConfigError("LambdaRank formal parameters 漂移")
        if self.raw["lightgbm"]["num_boost_round"] != 600:
            raise Stage4B4ConfigError("只允许 600-round full booster")
        if self.raw["lightgbm"]["checkpoints"] != [100, 300, 600]:
            raise Stage4B4ConfigError("prediction checkpoint budget 漂移")
        expected_predictions = [
            "B0", "L0_100", "L0_300", "L0_600", "L1_100", "L1_300", "L1_600"
        ]
        if self.raw["predictions"]["versions"] != expected_predictions:
            raise Stage4B4ConfigError("pre-registered prediction budget 漂移")
        if any(self.raw["access_lock"].values()):
            raise Stage4B4ConfigError("pre-label access lock 初始值必须全部 false/0")
