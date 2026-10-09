from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from data_audit.config import find_project_root


@dataclass(frozen=True)
class Stage3DConfig:
    root: Path
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: Path) -> "Stage3DConfig":
        resolved = path.resolve()
        value = cls(find_project_root(resolved), json.loads(resolved.read_text(encoding="utf-8")), resolved)
        value.validate()
        return value

    def validate(self) -> None:
        if self.raw.get("stage") != "3D":
            raise ValueError("配置 stage 必须为 3D")
        model = self.raw["model"]
        expected_model = {
            "variants": ["TT0", "TT1"],
            "output_dimension": 64,
            "article_id_embedding_dimension": 64,
            "user_id_embedding_dimension": 32,
            "attribute_embedding_dimension": 8,
            "hidden_dimension": 128,
            "max_history_items": 20,
            "max_training_targets_per_user": 3,
            "temperature": 0.07,
            "l2_normalize_output": True,
        }
        for field, expected in expected_model.items():
            if model.get(field) != expected:
                raise ValueError(f"Stage 3D 冻结模型参数漂移: {field}")
        training = self.raw["training"]
        expected_training = {
            "framework": "pytorch_cpu", "optimizer": "adamw", "learning_rate": 0.001,
            "weight_decay": 0.00001, "epochs": 3, "gradient_clip_norm": 1.0,
            "seed": 20260805, "dataloader_num_workers": 0,
        }
        for field, expected in expected_training.items():
            if training.get(field) != expected:
                raise ValueError(f"Stage 3D 冻结训练参数漂移: {field}")
        if training.get("batch_size") not in {256, 512}:
            raise ValueError("batch_size 只允许预注册的 512 或 OOM 前置降级 256")
        expected_refit = {
            "required_for_formal_candidates": True,
            "reinitialize_after_epoch_selection": True,
            "sample_scope": "all_eligible_samples_without_internal_dev_holdout",
            "epoch_source": "locked_validation_selected_epoch",
            "validation_models": ["locked_model_only"],
            "test_models": ["locked_model_only"],
        }
        if self.raw.get("final_refit") != expected_refit:
            raise ValueError("Stage 3D 正式候选必须来自锁定 epoch 后的全量重新初始化 refit")
        samples = self.raw["training_samples"]
        if samples["max_training_targets_per_user"] != 3 or samples["max_history_items"] != 20:
            raise ValueError("训练 target/history 上限漂移")
        ann = self.raw["ann"]
        frozen_ann = {
            "algorithm": "hnsw", "metric": "cosine", "m": 32,
            "ef_construction": 200, "ef_search_candidates": [200, 400],
            "initial_query_k": 1000, "retry_query_k": 2000,
        }
        for field, expected in frozen_ann.items():
            if ann.get(field) != expected:
                raise ValueError(f"Stage 3D ANN 参数漂移: {field}")
        union = self.raw["union"]
        expected_union = {
            "Q0": {"repurchase": 50, "itemcf": 300, "two_tower": 0, "trend": 100},
            "T50": {"repurchase": 50, "itemcf": 250, "two_tower": 50, "trend": 100},
            "T100": {"repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100},
        }
        if union["candidate_budget"] != 500 or union["variants"] != expected_union:
            raise ValueError("Q0/T50/T100 固定预算漂移")
        if union["quota_semantics"] != "source_rank_cap_before_cross_source_dedup":
            raise ValueError("Stage 3D 必须沿用预去重 source-rank cap")
        if (union["near_tie_tolerance"] != 0.0005 or union["strong_pass_delta"] != 0.002
                or union["severe_segment_regression_delta"] != -0.002):
            raise ValueError("value gate 阈值漂移")
        fields = self.raw["canonical_fields"]
        sources = [row["source_column"] for row in fields]
        semantics = [row["semantic_field"] for row in fields]
        if len(sources) != len(set(sources)) or len(semantics) != len(set(semantics)):
            raise ValueError("canonical code/name 不得重复建模")
        if {row["semantic_group"] for row in fields} != {
            "taxonomy", "appearance", "merchandising_index"
        }:
            raise ValueError("TT1 只能使用 Stage 3C 三个 semantic group")
        if self.raw["bootstrap"]["iterations"] != 1000:
            raise ValueError("paired bootstrap 固定 1000 次")

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
        for name in ("split", "population", "stage_3a", "stage_3b", "stage_3c"):
            digest.update(b"\0")
            digest.update(self.resolve(self.raw["input_manifests"][name]).read_bytes())
        for name in ("stage_3a_lock", "stage_3b_lock", "stage_3c_lock"):
            digest.update(b"\0")
            digest.update(self.resolve(self.raw["upstream"][name]).read_bytes())
        return digest.hexdigest()


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src" / "retrieval" / "stage3d").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("run_stage_3d*.py")))
    paths.extend(sorted((root / "scripts").glob("finalize_stage_3d*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
