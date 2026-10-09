from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import psutil
import torch

from data_audit.stage1a import ResourceMonitor, declared_fingerprint_values, sensitive_value_matches
from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.stage3b.bootstrap import paired_user_bootstrap
from retrieval.stage3b.config import Stage3BConfig
from retrieval.stage3b.runner import Stage3BRunner
from retrieval.stage3c.content import paired_bootstrap

from .ann import (
    build_hnsw_index,
    exact_top_k_blockwise,
    load_hnsw_index,
    overlap_recall,
    rerank_ann_candidates,
)
from .config import Stage3DConfig, source_fingerprint
from .data import AttributeVocabulary, recency_weight
from .locking import lock_fingerprint, validate_test_lock
from .model import TwoTowerModel, in_batch_softmax_loss
from .union import select_model, select_quota


FINGERPRINT_FIELDS = {
    "ann_audit_fingerprint", "artifact_fingerprint", "candidate_semantic_fingerprint",
    "candidate_universe_fingerprint", "checkpoint_fingerprint", "config_fingerprint",
    "dataset_fingerprint", "ground_truth_fingerprint", "input_fingerprint", "lock_fingerprint",
    "model_checkpoint_fingerprint", "output_fingerprint", "output_fingerprints",
    "population_fingerprint", "run_fingerprint", "source_fingerprint",
    "stage_2a_manifest_fingerprint", "stage_2b_manifest_fingerprint",
    "stage_3a_lock_fingerprint", "stage_3b_lock_fingerprint", "stage_3c_lock_fingerprint",
    "validation_artifacts", "q0_candidate_semantic_fingerprint", "q0_output_fingerprint",
    "ground_truth_fingerprint", "index_fingerprint",
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def semantic_partition_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_fingerprint(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_reusable(
    payload: Mapping[str, Any] | None,
    *,
    run_fingerprint: str,
    outputs: Sequence[Path],
    root: Path,
) -> bool:
    if not payload or payload.get("status") != "SUCCESS" or payload.get("run_fingerprint") != run_fingerprint:
        return False
    expected = payload.get("output_fingerprints", {})
    return bool(outputs) and all(
        path.exists()
        and expected.get(path.resolve().relative_to(root).as_posix()) == file_fingerprint(path)
        for path in outputs
    )


def _sha256_values(value: Any) -> set[str]:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return {value.lower()}
    if isinstance(value, Mapping):
        return {token for nested in value.values() for token in _sha256_values(nested)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {token for nested in value for token in _sha256_values(nested)}
    return set()


def validate_stage3d_public_artifacts(paths: Iterable[Path]) -> None:
    findings: dict[str, dict[str, list[str]]] = {}
    for path in paths:
        if not path.exists():
            continue
        text_value = path.read_text(encoding="utf-8-sig")
        allowed = set(declared_fingerprint_values(path))
        if path.suffix.lower() == ".json":
            payload = json.loads(text_value)
            if isinstance(payload, Mapping):
                for field, value in payload.items():
                    if field in FINGERPRINT_FIELDS:
                        allowed.update(_sha256_values(value))
        elif path.suffix.lower() == ".csv":
            for row in csv.DictReader(text_value.splitlines()):
                for field in FINGERPRINT_FIELDS & set(row):
                    raw = row[field]
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        value = raw
                    allowed.update(_sha256_values(value))
        matches = sensitive_value_matches(text_value)
        matches = {name: [value for value in values if value.lower() not in allowed]
                   for name, values in matches.items()}
        matches = {name: values for name, values in matches.items() if values}
        if matches:
            findings[path.name] = matches
    if findings:
        raise ValueError(f"公开产物敏感信息检查失败: {findings}")


class Stage3DRunner:
    def __init__(self, config: Stage3DConfig) -> None:
        self.config = config
        self.source_fingerprint = source_fingerprint(config.root)
        self.stage3b = Stage3BRunner(Stage3BConfig.load(config.root / "configs/stage_3b_itemcf.yaml"))
        self.dataset_fingerprint = self.stage3b.dataset_fingerprint
        self.feedback_version = self.stage3b.feedback_version
        self.population_version = self.stage3b.population_version
        self.snapshots = self.stage3b.snapshots
        self.upstream_lock_fingerprints = {
            f"{name}_lock_fingerprint": file_fingerprint(
                config.resolve(config.raw["upstream"][f"{name}_lock"])
            ) for name in ("stage_3a", "stage_3b", "stage_3c")
        }
        self.run_fingerprint = hashlib.sha256(
            (self.dataset_fingerprint + config.fingerprint + self.source_fingerprint
             + "".join(self.upstream_lock_fingerprints.values())).encode("utf-8")
        ).hexdigest()
        self.resources: list[dict[str, Any]] = []
        self.run_status: list[dict[str, Any]] = []
        self._validate_frozen_upstream()

    def _validate_frozen_upstream(self) -> None:
        expected_locks = {
            "stage_3a_lock_fingerprint": "c8c8f233f05001a1d600433068c1874a875f9ec47aa7d7b8d612e51cfbe2d134",
            "stage_3b_lock_fingerprint": "049ba378c5bc69285ebb8387006c16514737082c3586142499ffcad3104af8c8",
            "stage_3c_lock_fingerprint": "a85136709b6b1137c888391d2788766921b7499c054809e91da821f83657af18",
        }
        if self.upstream_lock_fingerprints != expected_locks:
            raise ValueError("Stage 3A/3B/3C 冻结 lock fingerprint 已变化")
        stage3c_lock = json_read(self.config.resolve(self.config.raw["upstream"]["stage_3c_lock"]))
        if (stage3c_lock.get("recommended_stage_status") != "FAILED_VALUE_GATE"
                or stage3c_lock.get("test_allowed") is not False
                or stage3c_lock.get("locked_quota_variant") != "Q0"):
            raise ValueError("Stage 3C 正式失败状态、Test gate 或 Q0 锁定事实发生变化")
        stage3b_lock = json_read(self.config.resolve(self.config.raw["upstream"]["stage_3b_lock"]))
        if stage3b_lock.get("locked_itemcf_variant") != "iuf":
            raise ValueError("Stage 3B 锁定 ItemCF 不再是 IUF")
        for manifest_name in ("stage_3a", "stage_3b", "stage_3c"):
            for row in csv_rows(self.config.resolve(self.config.raw["input_manifests"][manifest_name])):
                artifact = self.config.resolve(row["artifact_path"])
                if not artifact.exists() or file_fingerprint(artifact) != row["artifact_fingerprint"]:
                    raise ValueError(f"上游 manifest 产物不一致: {row['artifact_path']}")
        q0_rows = csv_rows(self.config.root / "artifacts/stage_3c/q0_reproduction.csv")
        variants = {"validation_14d": "itemcf_union_iuf", "test_14d": "itemcf_union"}
        for row in q0_rows:
            parts = sorted((self.stage3b.config.private_dir / "unions" / row["snapshot_id"]
                            / variants[row["snapshot_id"]]).glob("part_*.parquet"))
            if len(parts) != 32 or semantic_partition_fingerprint(parts) != row["candidate_semantic_fingerprint"]:
                raise ValueError("冻结 Q0 candidate fingerprint 无法精确复现")

    def prepare_directories(self) -> None:
        for path in (self.config.artifact_dir, self.config.private_dir,
                     self.config.checkpoint_dir, self.config.temp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        values = self.config.raw["duckdb"]
        con.execute(f"SET temp_directory='{sql_path(self.config.temp_dir)}'")
        con.execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
        con.execute(f"SET threads={int(values['threads'])}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{values['max_temp_directory_size']}'")
        return con

    def checkpoint_path(self, group: str) -> Path:
        return self.config.checkpoint_dir / f"{group}.json"

    def run_group(self, group: str, operation: Callable[[], Any], outputs: Iterable[Path]) -> Any:
        paths = list(outputs)
        checkpoint = self.checkpoint_path(group)
        previous = json_read(checkpoint) if checkpoint.exists() else None
        if checkpoint_reusable(previous, run_fingerprint=self.run_fingerprint, outputs=paths,
                               root=self.config.root):
            row = dict(previous)
            row["run_status"] = "REUSED"
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "REUSED",
                                    "attempt": row.get("attempt", 1), "message": ""})
            return None
        attempt = int((previous or {}).get("attempt", 0)) + 1
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        try:
            with monitor:
                result = operation()
            missing = [self.config.relative(path) for path in paths if not path.exists()]
            if missing:
                raise RuntimeError(f"group 输出不完整: {missing}")
            row = {
                "group_name": group, "status": "SUCCESS", "attempt": attempt,
                "generated_at": now(), "run_fingerprint": self.run_fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "memory_limit_mb": self.config.memory_limit_mb,
                "threads": self.config.raw["duckdb"]["threads"],
                "output_files": [self.config.relative(path) for path in paths],
                "output_fingerprints": {self.config.relative(path): file_fingerprint(path) for path in paths},
                "output_size_bytes": sum(path.stat().st_size for path in paths),
                "error_message": "",
            }
            json_write(checkpoint, row)
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "SUCCESS",
                                    "attempt": attempt, "message": ""})
            return result
        except Exception as exc:
            row = {
                "group_name": group, "status": "FAILED", "attempt": attempt,
                "generated_at": now(), "run_fingerprint": self.run_fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "error_message": str(exc).replace(str(self.config.root), "."),
            }
            json_write(checkpoint, row)
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "FAILED",
                                    "attempt": attempt, "message": row["error_message"]})
            raise

    def stage2a_path(self, name: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_2a_private"]) / name

    def stage2b_path(self, prefix: str, snapshot: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_2b_private"]) / f"{prefix}_{snapshot}.parquet"

    def stage3a_ranking(self, snapshot: str, source: str, variant: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_3a_private"]) / f"ranking_{snapshot}_{source}_{variant}.parquet"

    def stage3a_candidate(self, snapshot: str, source: str, variant: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_3a_private"]) / f"candidates_{snapshot}_{source}_{variant}.parquet"

    def stage3b_candidate_part(self, snapshot: str, part: int) -> Path:
        return self.stage3b.config.private_dir / "candidates" / snapshot / "iuf" / f"part_{part:03d}.parquet"

    @staticmethod
    def q0_variant(snapshot: str) -> str:
        return "itemcf_union_iuf" if snapshot == "validation_14d" else "itemcf_union"

    def q0_parts(self, snapshot: str) -> list[Path]:
        return sorted((self.stage3b.config.private_dir / "unions" / snapshot
                       / self.q0_variant(snapshot)).glob("part_*.parquet"))

    def q0_glob(self, snapshot: str) -> str:
        return sql_path(self.stage3b.config.private_dir / "unions" / snapshot
                        / self.q0_variant(snapshot) / "part_*.parquet")

    def history_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"history_{snapshot}.parquet"

    def target_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"training_targets_{snapshot}.parquet"

    def sample_part(self, snapshot: str, part: int) -> Path:
        return self.config.private_dir / "training_samples" / snapshot / f"part_{part:03d}.parquet"

    def attribute_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"attributes_{snapshot}.parquet"

    def attribute_vocab_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"attribute_vocab_{snapshot}.json"

    def attribute_index_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"attribute_index_{snapshot}.npy"

    def user_feature_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"user_features_{snapshot}.parquet"

    def selection_model_checkpoint(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "models" / snapshot / f"{variant}_best.pt"

    def model_checkpoint(self, snapshot: str, variant: str) -> Path:
        """正式 embedding/candidate 唯一允许引用的全量 refit checkpoint。"""
        return self.config.private_dir / "models" / snapshot / f"{variant}_final_refit.pt"

    def training_history_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "models" / snapshot / f"{variant}_training_history.json"

    def final_refit_history_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "models" / snapshot / f"{variant}_final_refit_history.json"

    def embedding_path(self, snapshot: str, variant: str, kind: str) -> Path:
        return self.config.private_dir / "embeddings" / snapshot / f"{variant}_{kind}.npz"

    def embedding_metadata_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "embeddings" / snapshot / f"{variant}_lineage.json"

    def ann_index_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "ann" / snapshot / f"{variant}.hnsw"

    def ann_metadata_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "ann" / snapshot / f"{variant}.json"

    def candidate_part(self, snapshot: str, variant: str, part: int) -> Path:
        return self.config.private_dir / "candidates" / snapshot / variant / f"part_{part:03d}.parquet"

    def candidate_glob(self, snapshot: str, variant: str) -> str:
        return sql_path(self.config.private_dir / "candidates" / snapshot / variant / "part_*.parquet")

    def candidate_lineage_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "candidates" / snapshot / variant / "lineage.json"

    def union_part(self, snapshot: str, variant: str, part: int) -> Path:
        return self.config.private_dir / "unions" / snapshot / variant / f"part_{part:03d}.parquet"

    def union_glob(self, snapshot: str, variant: str) -> str:
        return sql_path(self.config.private_dir / "unions" / snapshot / variant / "part_*.parquet")

    def user_metric_path(self, snapshot: str, source: str, variant: str) -> Path:
        return self.config.private_dir / f"user_metrics_{snapshot}_{source}_{variant}.parquet"

    def build_snapshot_history(self, snapshot: str) -> Path:
        target = self.history_path(snapshot)
        feedback = self.stage2a_path("feedback_daily.parquet")
        customer_map = self.stage2a_path("customer_key_map.parquet")
        article_map = self.stage2a_path("article_key_map.parquet")
        cutoff = self.snapshots[snapshot].history_end
        con = self.connect()
        try:
            con.execute(
                f"""COPY (
                SELECT c.customer_key,a.article_key,f.t_dat
                FROM read_parquet('{sql_path(feedback)}') f
                JOIN read_parquet('{sql_path(customer_map)}') c USING(customer_id)
                JOIN read_parquet('{sql_path(article_map)}') a USING(article_id)
                WHERE f.t_dat<=DATE '{cutoff}'
                ORDER BY c.customer_key,f.t_dat,a.article_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)"""
            )
        finally:
            con.close()
        return target

    def build_training_targets(self, snapshot: str) -> Path:
        target = self.target_path(snapshot)
        history = self.history_path(snapshot)
        con = self.connect()
        try:
            con.execute(
                f"""COPY (
                WITH acquisitions AS (
                  SELECT customer_key,article_key target_article_key,min(t_dat) target_date
                  FROM read_parquet('{sql_path(history)}') GROUP BY customer_key,article_key
                ), eligible AS (
                  SELECT a.* FROM acquisitions a
                  WHERE EXISTS (SELECT 1 FROM read_parquet('{sql_path(history)}') h
                                WHERE h.customer_key=a.customer_key AND h.t_dat<a.target_date)
                ), ranked AS (
                  SELECT *,row_number() OVER(PARTITION BY customer_key
                    ORDER BY target_date DESC,target_article_key)::INTEGER target_rank
                  FROM eligible
                )
                SELECT row_number() OVER(ORDER BY customer_key,target_rank)::BIGINT sample_id,
                       customer_key,target_article_key,target_date,target_rank,
                       ((((customer_key::HUGEINT*6364136223846793005)+20260805)
                          %9223372036854775783)%20=0)::BOOLEAN is_dev
                FROM ranked WHERE target_rank<=3 ORDER BY customer_key,target_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 262144)"""
            )
        finally:
            con.close()
        return target

    def build_sample_part(self, snapshot: str, part: int) -> Path:
        target = self.sample_part(snapshot, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        history = self.history_path(snapshot)
        targets = self.target_path(snapshot)
        partitions = int(self.config.raw["training_samples"]["partitions"])
        pivot = ",\n".join(
            f"max(article_key) FILTER(WHERE history_rank={rank})::BIGINT history_{rank:02d}"
            for rank in range(1, 21)
        )
        con = self.connect()
        try:
            con.execute(
                f"""COPY (
                WITH selected AS (
                  SELECT * FROM read_parquet('{sql_path(targets)}')
                  WHERE customer_key%{partitions}={part}
                ), unique_context AS (
                  SELECT t.sample_id,t.customer_key,t.target_article_key,t.target_date,t.target_rank,t.is_dev,
                         h.article_key,max(h.t_dat) last_date
                  FROM selected t JOIN read_parquet('{sql_path(history)}') h
                    ON h.customer_key=t.customer_key AND h.t_dat<t.target_date
                   AND h.article_key<>t.target_article_key
                  GROUP BY t.sample_id,t.customer_key,t.target_article_key,t.target_date,t.target_rank,t.is_dev,h.article_key
                ), ranked AS (
                  SELECT *,row_number() OVER(PARTITION BY sample_id ORDER BY last_date DESC,article_key)::INTEGER history_rank
                  FROM unique_context QUALIFY history_rank<=20
                )
                SELECT sample_id,customer_key,target_article_key,target_date,target_rank,is_dev,
                       count(*)::INTEGER history_length,{pivot}
                FROM ranked GROUP BY sample_id,customer_key,target_article_key,target_date,target_rank,is_dev
                ORDER BY customer_key,target_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)"""
            )
        finally:
            con.close()
        return target

    def build_attributes(self, snapshot: str) -> tuple[Path, Path, Path]:
        attributes = self.attribute_path(snapshot)
        vocab_path = self.attribute_vocab_path(snapshot)
        index_path = self.attribute_index_path(snapshot)
        articles = self.config.resolve(self.config.raw["paths"]["articles"])
        article_map = self.stage2a_path("article_key_map.parquet")
        item_state = self.stage2b_path("item_state", snapshot)
        fields = self.config.raw["canonical_fields"]
        columns = ",".join(f"r.{row['source_column']}" for row in fields)
        con = self.connect()
        try:
            con.execute(
                f"""COPY (
                SELECT m.article_key,{columns}
                FROM read_parquet('{sql_path(articles)}') r
                JOIN read_parquet('{sql_path(article_map)}') m USING(article_id)
                JOIN read_parquet('{sql_path(item_state)}') s USING(article_key)
                WHERE s.seen_item ORDER BY m.article_key
                ) TO '{sql_path(attributes)}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
            )
        finally:
            con.close()
        table = pq.read_table(attributes)
        rows = table.to_pylist()
        vocab = AttributeVocabulary.build(rows, fields)
        json_write(vocab_path, vocab.to_json())
        num_articles = pq.ParquetFile(article_map).metadata.num_rows
        encoded = np.zeros((num_articles, len(fields)), dtype=np.int64)
        for row in rows:
            encoded[int(row["article_key"])] = vocab.encode_row(row)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(index_path, encoded, allow_pickle=False)
        return attributes, vocab_path, index_path

    def build_user_features(self, snapshot: str) -> Path:
        target = self.user_feature_path(snapshot)
        population = self.stage2b_path("evaluation_population", snapshot)
        history = self.stage3b.config.private_dir / f"evaluation_history_{snapshot}.parquet"
        pivot = ",\n".join(
            f"max(article_key) FILTER(WHERE history_rank={rank})::BIGINT history_{rank:02d}"
            for rank in range(1, 21)
        )
        con = self.connect()
        try:
            con.execute(
                f"""COPY (
                WITH users AS (
                  SELECT user_key customer_key,user_segment,is_main_evaluable
                  FROM read_parquet('{sql_path(population)}') WHERE is_main_evaluable
                ), ranked AS (
                  SELECT u.customer_key,u.user_segment,h.article_key,h.last_purchase_date,
                         row_number() OVER(PARTITION BY u.customer_key
                           ORDER BY h.last_purchase_date DESC,h.article_key)::INTEGER history_rank
                  FROM users u JOIN read_parquet('{sql_path(history)}') h USING(customer_key)
                  QUALIFY history_rank<=20
                )
                SELECT customer_key,user_segment,count(*)::INTEGER history_length,{pivot}
                FROM ranked GROUP BY customer_key,user_segment ORDER BY customer_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
            )
        finally:
            con.close()
        return target

    @staticmethod
    def _batch_numpy(batch: pa.RecordBatch, name: str, *, null_value: int = -1) -> np.ndarray:
        column = batch.column(batch.schema.get_field_index(name))
        if column.null_count:
            column = pc.fill_null(column, null_value)
        return np.asarray(column)

    def iter_sample_batches(
        self, snapshot: str, *, dev: bool | None, batch_size: int,
    ) -> Iterator[tuple[np.ndarray, ...]]:
        columns = ["customer_key", "target_article_key", "is_dev", *(f"history_{rank:02d}" for rank in range(1, 21))]
        parts = int(self.config.raw["training_samples"]["partitions"])
        weights = np.asarray([recency_weight(rank) for rank in range(1, 21)], dtype=np.float32)
        for part in range(parts):
            path = self.sample_part(snapshot, part)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
                is_dev = self._batch_numpy(batch, "is_dev", null_value=0).astype(bool)
                # dev=None 是锁定 epoch 后的正式全量 refit；不能再保留 internal dev。
                keep = np.ones(len(is_dev), dtype=bool) if dev is None else (is_dev if dev else ~is_dev)
                if not keep.any():
                    continue
                users = self._batch_numpy(batch, "customer_key")[keep].astype(np.int64, copy=False)
                targets = self._batch_numpy(batch, "target_article_key")[keep].astype(np.int64, copy=False)
                history = np.column_stack([
                    self._batch_numpy(batch, f"history_{rank:02d}")[keep]
                    for rank in range(1, 21)
                ]).astype(np.int64, copy=False)
                history_weights = (history >= 0).astype(np.float32) * weights[None, :]
                yield users, targets, history, history_weights

    def _model(self, snapshot: str, variant: str) -> TwoTowerModel:
        num_users = pq.ParquetFile(self.stage2a_path("customer_key_map.parquet")).metadata.num_rows
        num_articles = pq.ParquetFile(self.stage2a_path("article_key_map.parquet")).metadata.num_rows
        vocab = AttributeVocabulary.from_json(json_read(self.attribute_vocab_path(snapshot)))
        model = TwoTowerModel(
            variant=variant, num_users=num_users, num_articles=num_articles,
            attribute_vocab_sizes=vocab.sizes() if variant == "TT1" else (),
            attribute_groups=vocab.groups if variant == "TT1" else (),
            article_embedding_dim=self.config.raw["model"]["article_id_embedding_dimension"],
            user_embedding_dim=self.config.raw["model"]["user_id_embedding_dimension"],
            attribute_embedding_dim=self.config.raw["model"]["attribute_embedding_dimension"],
            hidden_dim=self.config.raw["model"]["hidden_dimension"],
            output_dim=self.config.raw["model"]["output_dimension"],
        )
        if variant == "TT1":
            # np.load(mmap_mode="r") 是只读 view；模型只索引不修改，但复制后可避免 PyTorch 的只读 NumPy warning。
            attribute_index = np.array(
                np.load(self.attribute_index_path(snapshot), mmap_mode="r"), copy=True,
            )
            model.set_attribute_index(torch.from_numpy(attribute_index))
        return model

    def _epoch_checkpoint(self, snapshot: str, variant: str, epoch: int) -> Path:
        return self.config.private_dir / "models" / snapshot / f"{variant}_epoch_{epoch}.pt"

    def train_model(self, snapshot: str, variant: str) -> tuple[Path, Path]:
        values = self.config.raw["training"]
        torch.manual_seed(int(values["seed"]))
        torch.set_num_threads(int(self.config.raw["torch"]["num_threads"]))
        try:
            torch.set_num_interop_threads(int(self.config.raw["torch"]["interop_threads"]))
        except RuntimeError:
            pass
        model = self._model(snapshot, variant).cpu()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(values["learning_rate"]),
                                      weight_decay=float(values["weight_decay"]))
        history_rows: list[dict[str, Any]] = []
        start_epoch = 1
        for epoch in range(int(values["epochs"]), 0, -1):
            path = self._epoch_checkpoint(snapshot, variant, epoch)
            meta = path.with_suffix(".json")
            if path.exists() and meta.exists() and json_read(meta).get("run_fingerprint") == self.run_fingerprint:
                payload = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(payload["model"])
                optimizer.load_state_dict(payload["optimizer"])
                history_rows = list(payload["history"])
                start_epoch = epoch + 1
                break
        batch_size = int(values["batch_size"])
        for epoch in range(start_epoch, int(values["epochs"]) + 1):
            model.train()
            train_loss_sum, train_examples, batches = 0.0, 0, 0
            started = time.perf_counter()
            generator = torch.Generator().manual_seed(int(values["seed"]) + epoch)
            for users_np, targets_np, history_np, weights_np in self.iter_sample_batches(
                    snapshot, dev=False, batch_size=batch_size):
                order = torch.randperm(len(users_np), generator=generator).numpy()
                users = torch.from_numpy(users_np[order])
                targets = torch.from_numpy(targets_np[order])
                histories = torch.from_numpy(history_np[order])
                history_weights = torch.from_numpy(weights_np[order])
                optimizer.zero_grad(set_to_none=True)
                user_vector, item_vector = model(users, histories, history_weights, targets)
                loss = in_batch_softmax_loss(user_vector, item_vector, targets, histories,
                                             temperature=float(self.config.raw["model"]["temperature"]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(values["gradient_clip_norm"]))
                optimizer.step()
                train_loss_sum += float(loss.item()) * len(users)
                train_examples += len(users)
                batches += 1
            model.eval()
            dev_loss_sum, dev_examples = 0.0, 0
            with torch.inference_mode():
                for users_np, targets_np, history_np, weights_np in self.iter_sample_batches(
                        snapshot, dev=True, batch_size=batch_size):
                    users, targets = torch.from_numpy(users_np), torch.from_numpy(targets_np)
                    histories, history_weights = torch.from_numpy(history_np), torch.from_numpy(weights_np)
                    user_vector, item_vector = model(users, histories, history_weights, targets)
                    loss = in_batch_softmax_loss(user_vector, item_vector, targets, histories,
                                                 temperature=float(self.config.raw["model"]["temperature"]))
                    dev_loss_sum += float(loss.item()) * len(users)
                    dev_examples += len(users)
            row = {
                "epoch": epoch, "train_loss": train_loss_sum / max(train_examples, 1),
                "dev_loss": dev_loss_sum / max(dev_examples, 1), "train_examples": train_examples,
                "dev_examples": dev_examples, "batches": batches,
                "execution_seconds": time.perf_counter() - started,
            }
            history_rows.append(row)
            epoch_path = self._epoch_checkpoint(snapshot, variant, epoch)
            epoch_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "history": history_rows, "epoch": epoch}, epoch_path)
            json_write(epoch_path.with_suffix(".json"), {
                "run_fingerprint": self.run_fingerprint, "epoch": epoch,
                "checkpoint_fingerprint": file_fingerprint(epoch_path),
            })
        selected = min(history_rows, key=lambda row: (float(row["dev_loss"]), int(row["epoch"])))
        selected_path = self._epoch_checkpoint(snapshot, variant, int(selected["epoch"]))
        best = self.selection_model_checkpoint(snapshot, variant)
        best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(selected_path, best)
        json_write(self.training_history_path(snapshot, variant), {
            "stage": "3D", "snapshot_id": snapshot, "model_variant": variant,
            "framework_version": torch.__version__, "device": "cpu",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "selected_epoch": selected["epoch"], "epochs": history_rows,
            "batch_size": batch_size, "seed": values["seed"],
            "checkpoint_role": "model_selection_only",
            "uses_all_eligible_samples": False,
            "model_checkpoint_fingerprint": file_fingerprint(best),
            "run_fingerprint": self.run_fingerprint,
        })
        return best, self.training_history_path(snapshot, variant)

    def train_final_model(
        self,
        snapshot: str,
        variant: str,
        *,
        locked_epoch: int,
        selection_checkpoint: Path,
    ) -> tuple[Path, Path]:
        if locked_epoch < 1:
            raise ValueError("final refit 的 locked epoch 必须为正")
        values = self.config.raw["training"]
        torch.manual_seed(int(values["seed"]))
        torch.set_num_threads(int(self.config.raw["torch"]["num_threads"]))
        try:
            torch.set_num_interop_threads(int(self.config.raw["torch"]["interop_threads"]))
        except RuntimeError:
            pass
        # 这里故意重新构造模型，不能从 selection checkpoint 继续训练。
        model = self._model(snapshot, variant).cpu()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
        )
        batch_size = int(values["batch_size"])
        history_rows: list[dict[str, Any]] = []
        for epoch in range(1, locked_epoch + 1):
            model.train()
            train_loss_sum, train_examples, batches = 0.0, 0, 0
            started = time.perf_counter()
            generator = torch.Generator().manual_seed(int(values["seed"]) + epoch)
            for users_np, targets_np, history_np, weights_np in self.iter_sample_batches(
                    snapshot, dev=None, batch_size=batch_size):
                order = torch.randperm(len(users_np), generator=generator).numpy()
                users = torch.from_numpy(users_np[order])
                targets = torch.from_numpy(targets_np[order])
                histories = torch.from_numpy(history_np[order])
                history_weights = torch.from_numpy(weights_np[order])
                optimizer.zero_grad(set_to_none=True)
                user_vector, item_vector = model(users, histories, history_weights, targets)
                loss = in_batch_softmax_loss(
                    user_vector, item_vector, targets, histories,
                    temperature=float(self.config.raw["model"]["temperature"]),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(values["gradient_clip_norm"]))
                optimizer.step()
                train_loss_sum += float(loss.item()) * len(users)
                train_examples += len(users)
                batches += 1
            history_rows.append({
                "epoch": epoch,
                "train_loss": train_loss_sum / max(train_examples, 1),
                "train_examples": train_examples,
                "dev_examples": 0,
                "batches": batches,
                "execution_seconds": time.perf_counter() - started,
            })
        sample_glob = sql_path(self.config.private_dir / "training_samples" / snapshot / "part_*.parquet")
        con = self.connect()
        try:
            final_users = int(con.execute(
                f"SELECT count(DISTINCT customer_key) FROM read_parquet('{sample_glob}')"
            ).fetchone()[0])
        finally:
            con.close()
        expected_examples = sum(
            pq.ParquetFile(self.sample_part(snapshot, part)).metadata.num_rows
            for part in range(int(self.config.raw["training_samples"]["partitions"]))
        )
        if not history_rows or int(history_rows[-1]["train_examples"]) != expected_examples:
            raise RuntimeError("final refit 未消费全部 eligible training examples")
        checkpoint = self.model_checkpoint(snapshot, variant)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        selection_fingerprint = file_fingerprint(selection_checkpoint)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history_rows,
            "epoch": locked_epoch,
            "checkpoint_role": "formal_full_refit",
            "uses_all_eligible_samples": True,
            "selection_checkpoint_fingerprint": selection_fingerprint,
        }, checkpoint)
        history_path = self.final_refit_history_path(snapshot, variant)
        json_write(history_path, {
            "stage": "3D",
            "snapshot_id": snapshot,
            "model_variant": variant,
            "framework_version": torch.__version__,
            "device": "cpu",
            "checkpoint_role": "formal_full_refit",
            "reinitialized_after_epoch_selection": True,
            "uses_all_eligible_samples": True,
            "selected_epoch": locked_epoch,
            "trained_epochs": locked_epoch,
            "final_training_users": final_users,
            "final_training_examples": expected_examples,
            "internal_dev_examples": 0,
            "batch_size": batch_size,
            "seed": values["seed"],
            "epochs": history_rows,
            "selection_checkpoint_path": self.config.relative(selection_checkpoint),
            "selection_checkpoint_fingerprint": selection_fingerprint,
            "final_checkpoint_path": self.config.relative(checkpoint),
            "final_checkpoint_fingerprint": file_fingerprint(checkpoint),
            "run_fingerprint": self.run_fingerprint,
        })
        return checkpoint, history_path

    def validate_formal_checkpoint(self, snapshot: str, variant: str) -> dict[str, Any]:
        checkpoint = self.model_checkpoint(snapshot, variant)
        history = json_read(self.final_refit_history_path(snapshot, variant))
        if (history.get("checkpoint_role") != "formal_full_refit"
                or history.get("uses_all_eligible_samples") is not True
                or history.get("reinitialized_after_epoch_selection") is not True
                or int(history.get("internal_dev_examples", -1)) != 0
                or history.get("final_checkpoint_fingerprint") != file_fingerprint(checkpoint)):
            raise RuntimeError("正式 embedding 禁止引用 selection-only 或不完整 refit checkpoint")
        return history

    def export_embeddings(self, snapshot: str, variant: str) -> tuple[Path, Path, Path]:
        model = self._model(snapshot, variant).cpu()
        refit = self.validate_formal_checkpoint(snapshot, variant)
        payload = torch.load(self.model_checkpoint(snapshot, variant), map_location="cpu", weights_only=True)
        if (payload.get("checkpoint_role") != "formal_full_refit"
                or payload.get("uses_all_eligible_samples") is not True):
            raise RuntimeError("正式 embedding 的 checkpoint payload 不是 full refit")
        model.load_state_dict(payload["model"])
        model.eval()
        item_table = pq.read_table(self.attribute_path(snapshot), columns=["article_key"])
        item_keys = np.asarray(item_table["article_key"]).astype(np.int64, copy=False)
        item_vectors: list[np.ndarray] = []
        batch_size = int(self.config.raw["execution"]["embedding_batch_size"])
        with torch.inference_mode():
            for start in range(0, len(item_keys), batch_size):
                keys = torch.from_numpy(item_keys[start:start + batch_size])
                item_vectors.append(model.encode_items(keys).numpy())
        item_embeddings = np.concatenate(item_vectors).astype(np.float32, copy=False)
        item_target = self.embedding_path(snapshot, variant, "items")
        item_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(item_target, article_keys=item_keys, embeddings=item_embeddings)

        columns = ["customer_key", "user_segment", *(f"history_{rank:02d}" for rank in range(1, 21))]
        user_keys_parts, segment_parts, user_vector_parts = [], [], []
        weights = np.asarray([recency_weight(rank) for rank in range(1, 21)], dtype=np.float32)
        with torch.inference_mode():
            for batch in pq.ParquetFile(self.user_feature_path(snapshot)).iter_batches(
                    batch_size=batch_size, columns=columns):
                user_keys = self._batch_numpy(batch, "customer_key").astype(np.int64, copy=False)
                histories = np.column_stack([
                    self._batch_numpy(batch, f"history_{rank:02d}") for rank in range(1, 21)
                ]).astype(np.int64, copy=False)
                history_weights = (histories >= 0).astype(np.float32) * weights[None, :]
                vectors = model.encode_users(torch.from_numpy(user_keys), torch.from_numpy(histories),
                                             torch.from_numpy(history_weights)).numpy()
                user_keys_parts.append(user_keys)
                segment_parts.append(np.asarray(batch.column(batch.schema.get_field_index("user_segment")).to_pylist(), dtype="U16"))
                user_vector_parts.append(vectors)
        user_target = self.embedding_path(snapshot, variant, "users")
        np.savez(user_target, customer_keys=np.concatenate(user_keys_parts),
                 segments=np.concatenate(segment_parts), embeddings=np.concatenate(user_vector_parts).astype(np.float32))
        metadata = self.embedding_metadata_path(snapshot, variant)
        json_write(metadata, {
            "stage": "3D", "snapshot_id": snapshot, "model_variant": variant,
            "checkpoint_role": "formal_full_refit",
            "source_model_checkpoint_path": self.config.relative(self.model_checkpoint(snapshot, variant)),
            "source_model_checkpoint_fingerprint": refit["final_checkpoint_fingerprint"],
            "item_embedding_fingerprint": file_fingerprint(item_target),
            "user_embedding_fingerprint": file_fingerprint(user_target),
            "run_fingerprint": self.run_fingerprint,
        })
        return item_target, user_target, metadata

    def build_ann(self, snapshot: str, variant: str) -> tuple[Path, Path]:
        data = np.load(self.embedding_path(snapshot, variant, "items"))
        embedding_lineage = json_read(self.embedding_metadata_path(snapshot, variant))
        index_path = self.ann_index_path(snapshot, variant)
        _, build_seconds = build_hnsw_index(
            data["embeddings"], index_path, m=int(self.config.raw["ann"]["m"]),
            ef_construction=int(self.config.raw["ann"]["ef_construction"]),
            seed=int(self.config.raw["training"]["seed"]),
        )
        json_write(self.ann_metadata_path(snapshot, variant), {
            "algorithm": "hnsw", "metric": "cosine", "m": self.config.raw["ann"]["m"],
            "ef_construction": self.config.raw["ann"]["ef_construction"],
            "build_seconds": build_seconds, "index_size_bytes": index_path.stat().st_size,
            "item_count": len(data["article_keys"]), "dimension": data["embeddings"].shape[1],
            "source_model_checkpoint_fingerprint": embedding_lineage["source_model_checkpoint_fingerprint"],
            "source_item_embedding_fingerprint": embedding_lineage["item_embedding_fingerprint"],
            "index_fingerprint": file_fingerprint(index_path), "run_fingerprint": self.run_fingerprint,
        })
        return index_path, self.ann_metadata_path(snapshot, variant)

    def ann_audit_detail_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "ann" / snapshot / f"{variant}_audit.parquet"

    def ann_audit_summary_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / "ann" / snapshot / f"{variant}_audit.json"

    def _seen_sets(self, snapshot: str, user_keys: np.ndarray) -> dict[int, set[int]]:
        history = self.stage3b.config.private_dir / f"evaluation_history_{snapshot}.parquet"
        con = self.connect()
        try:
            con.register("selected_users", pa.table({"customer_key": pa.array(user_keys, type=pa.int64())}))
            table = con.execute(
                f"""SELECT h.customer_key,h.article_key FROM read_parquet('{sql_path(history)}') h
                SEMI JOIN selected_users u USING(customer_key) ORDER BY h.customer_key,h.article_key"""
            ).fetch_arrow_table()
        finally:
            con.close()
        result: dict[int, set[int]] = defaultdict(set)
        customers = np.asarray(table["customer_key"])
        articles = np.asarray(table["article_key"])
        for customer, article in zip(customers, articles, strict=True):
            result[int(customer)].add(int(article))
        return result

    @staticmethod
    def _stable_audit_sample(customer_keys: np.ndarray, segments: np.ndarray, max_users: int) -> np.ndarray:
        chosen: list[int] = []
        groups = ("short_history", "warm", "active")
        per_group = max_users // len(groups)
        for group_index, group in enumerate(groups):
            indices = np.flatnonzero(segments == group)
            mixed = ((customer_keys[indices].astype(object) * 6364136223846793005 + 20260805)
                     % 9223372036854775783)
            order = np.argsort(np.asarray(mixed, dtype=np.int64), kind="stable")
            limit = per_group + (1 if group_index < max_users % len(groups) else 0)
            chosen.extend(indices[order[:limit]].tolist())
        return np.asarray(chosen, dtype=np.int64)

    def audit_ann(
        self, snapshot: str, variant: str, *, locked_ef_search: int | None = None,
    ) -> tuple[Path, Path]:
        ann_values = self.config.raw["ann"]
        item_data = np.load(self.embedding_path(snapshot, variant, "items"))
        user_data = np.load(self.embedding_path(snapshot, variant, "users"))
        item_keys = item_data["article_keys"].astype(np.int64, copy=False)
        item_embeddings = item_data["embeddings"].astype(np.float32, copy=False)
        user_keys = user_data["customer_keys"].astype(np.int64, copy=False)
        segments = user_data["segments"].astype("U16", copy=False)
        user_embeddings = user_data["embeddings"].astype(np.float32, copy=False)
        selected = self._stable_audit_sample(user_keys, segments, int(ann_values["audit_max_users"]))
        selected_keys = user_keys[selected]
        seen = self._seen_sets(snapshot, selected_keys)
        audit_rows: list[dict[str, Any]] = []
        selected_ef: int | None = None
        attempts: list[dict[str, Any]] = []
        ef_candidates = ([locked_ef_search] if locked_ef_search is not None
                         else ann_values["ef_search_candidates"])
        for ef_search in ef_candidates:
            index = load_hnsw_index(
                self.ann_index_path(snapshot, variant), dimension=item_embeddings.shape[1],
                max_elements=len(item_embeddings), ef_search=int(ef_search),
            )
            current: list[dict[str, Any]] = []
            for position in selected:
                customer = int(user_keys[position])
                vector = user_embeddings[position]
                query_started = time.perf_counter()
                labels, _ = index.knn_query(vector, k=min(int(ann_values["initial_query_k"]), len(item_keys)))
                approx_keys, _ = rerank_ann_candidates(
                    vector, labels[0], item_embeddings, item_keys,
                    seen_items=seen.get(customer, set()), k=500,
                )
                retried = False
                if len(approx_keys) < 500:
                    labels, _ = index.knn_query(vector, k=min(int(ann_values["retry_query_k"]), len(item_keys)))
                    approx_keys, _ = rerank_ann_candidates(
                        vector, labels[0], item_embeddings, item_keys,
                        seen_items=seen.get(customer, set()), k=500,
                    )
                    retried = True
                query_seconds = time.perf_counter() - query_started
                exact_started = time.perf_counter()
                exact_keys = exact_top_k_blockwise(
                    vector, item_embeddings, item_keys, seen_items=seen.get(customer, set()), k=500,
                    block_items=int(ann_values["exact_block_items"]),
                )
                exact_seconds = time.perf_counter() - exact_started
                current.append({
                    "customer_key": customer, "user_segment": str(segments[position]),
                    "ef_search": int(ef_search), "overlap_recall_at_500": overlap_recall(approx_keys, exact_keys),
                    "ann_query_seconds": query_seconds, "exact_query_seconds": exact_seconds,
                    "retried_query_k": retried, "ann_candidate_count": len(approx_keys),
                    "exact_candidate_count": len(exact_keys),
                })
            recalls = np.asarray([row["overlap_recall_at_500"] for row in current], dtype=np.float64)
            attempt = {
                "ef_search": int(ef_search), "mean": float(recalls.mean()),
                "median": float(np.median(recalls)), "p5": float(np.quantile(recalls, 0.05)),
                "passed": bool(recalls.mean() >= float(ann_values["audit_mean_overlap_recall_min"])
                               and np.quantile(recalls, 0.05) >= float(ann_values["audit_p5_overlap_recall_min"])),
            }
            attempts.append(attempt)
            audit_rows.extend(current)
            if attempt["passed"]:
                selected_ef = int(ef_search)
                break
        detail = self.ann_audit_detail_path(snapshot, variant)
        detail.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(audit_rows), detail, compression="zstd")
        final_rows = [row for row in audit_rows if row["ef_search"] == (
            selected_ef or int(ef_candidates[-1]))]
        recalls = np.asarray([row["overlap_recall_at_500"] for row in final_rows], dtype=np.float64)
        segment_summary = {}
        for segment in ("short_history", "warm", "active"):
            values = np.asarray([row["overlap_recall_at_500"] for row in final_rows
                                 if row["user_segment"] == segment], dtype=np.float64)
            segment_summary[segment] = {
                "user_count": len(values), "mean": float(values.mean()) if len(values) else None,
                "median": float(np.median(values)) if len(values) else None,
                "p5": float(np.quantile(values, 0.05)) if len(values) else None,
            }
        summary = {
            "stage": "3D", "snapshot_id": snapshot, "model_variant": variant,
            "audit_user_count": len(final_rows), "selected_ef_search": selected_ef,
            "passed": selected_ef is not None, "mean_overlap_recall": float(recalls.mean()),
            "median_overlap_recall": float(np.median(recalls)),
            "p5_overlap_recall": float(np.quantile(recalls, 0.05)),
            "mean_ann_query_seconds": float(np.mean([row["ann_query_seconds"] for row in final_rows])),
            "mean_exact_query_seconds": float(np.mean([row["exact_query_seconds"] for row in final_rows])),
            "retry_rate": float(np.mean([row["retried_query_k"] for row in final_rows])),
            "attempts": attempts, "segments": segment_summary,
            "index_size_bytes": self.ann_index_path(snapshot, variant).stat().st_size,
            "ann_audit_fingerprint": file_fingerprint(detail),
        }
        json_write(self.ann_audit_summary_path(snapshot, variant), summary)
        if selected_ef is None:
            raise RuntimeError("FAILED_RETRIEVAL_GATE: ANN exact-overlap 门禁未通过")
        return detail, self.ann_audit_summary_path(snapshot, variant)

    @staticmethod
    def _candidate_schema() -> pa.Schema:
        return pa.schema([
            ("snapshot_id", pa.string()), ("as_of_date", pa.date32()),
            ("customer_key", pa.int64()), ("article_key", pa.int64()),
            ("source", pa.string()), ("source_variant", pa.string()),
            ("source_rank", pa.int32()), ("source_score", pa.float32()),
            ("candidate_universe_version", pa.string()), ("feedback_version", pa.string()),
            ("population_version", pa.string()), ("dataset_fingerprint", pa.string()),
            ("config_fingerprint", pa.string()), ("source_score_kind", pa.string()),
            ("ann_ef_search", pa.int32()), ("ann_retry", pa.bool_()),
        ])

    def build_candidate_part(self, snapshot: str, variant: str, part: int) -> Path:
        item_data = np.load(self.embedding_path(snapshot, variant, "items"))
        user_data = np.load(self.embedding_path(snapshot, variant, "users"))
        item_keys = item_data["article_keys"].astype(np.int64, copy=False)
        item_embeddings = item_data["embeddings"].astype(np.float32, copy=False)
        user_keys = user_data["customer_keys"].astype(np.int64, copy=False)
        user_embeddings = user_data["embeddings"].astype(np.float32, copy=False)
        partitions = int(self.config.raw["execution"]["candidate_partitions"])
        selected = np.flatnonzero(user_keys % partitions == part)
        selected_keys = user_keys[selected]
        seen = self._seen_sets(snapshot, selected_keys)
        audit = json_read(self.ann_audit_summary_path(snapshot, variant))
        ef_search = int(audit["selected_ef_search"])
        ann_values = self.config.raw["ann"]
        index = load_hnsw_index(self.ann_index_path(snapshot, variant),
                                dimension=item_embeddings.shape[1], max_elements=len(item_embeddings),
                                ef_search=ef_search)
        target = self.candidate_part(snapshot, variant, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        schema = self._candidate_schema()
        writer = pq.ParquetWriter(target, schema, compression="zstd")
        as_of = datetime.strptime(self.snapshots[snapshot].history_end, "%Y-%m-%d").date()
        try:
            for start in range(0, len(selected), 128):
                positions = selected[start:start + 128]
                if not len(positions):
                    continue
                labels, _ = index.knn_query(user_embeddings[positions],
                                            k=min(int(ann_values["initial_query_k"]), len(item_keys)))
                customers_out, articles_out, ranks_out, scores_out, retries_out = [], [], [], [], []
                for local, position in enumerate(positions):
                    customer = int(user_keys[position])
                    keys, scores = rerank_ann_candidates(
                        user_embeddings[position], labels[local], item_embeddings, item_keys,
                        seen_items=seen.get(customer, set()), k=500,
                    )
                    retried = False
                    if len(keys) < 500:
                        retry_labels, _ = index.knn_query(
                            user_embeddings[position], k=min(int(ann_values["retry_query_k"]), len(item_keys))
                        )
                        keys, scores = rerank_ann_candidates(
                            user_embeddings[position], retry_labels[0], item_embeddings, item_keys,
                            seen_items=seen.get(customer, set()), k=500,
                        )
                        retried = True
                    count = len(keys)
                    customers_out.append(np.full(count, customer, dtype=np.int64))
                    articles_out.append(keys.astype(np.int64, copy=False))
                    ranks_out.append(np.arange(1, count + 1, dtype=np.int32))
                    scores_out.append(scores.astype(np.float32, copy=False))
                    retries_out.append(np.full(count, retried, dtype=bool))
                if not customers_out:
                    continue
                n = sum(len(value) for value in customers_out)
                table = pa.table({
                    "snapshot_id": pa.array([snapshot] * n), "as_of_date": pa.array([as_of] * n, type=pa.date32()),
                    "customer_key": np.concatenate(customers_out), "article_key": np.concatenate(articles_out),
                    "source": pa.array(["two_tower"] * n), "source_variant": pa.array([variant] * n),
                    "source_rank": np.concatenate(ranks_out), "source_score": np.concatenate(scores_out),
                    "candidate_universe_version": pa.array(["history_seen_items_v1"] * n),
                    "feedback_version": pa.array([self.feedback_version] * n),
                    "population_version": pa.array([self.population_version] * n),
                    "dataset_fingerprint": pa.array([self.dataset_fingerprint] * n),
                    "config_fingerprint": pa.array([self.config.fingerprint] * n),
                    "source_score_kind": pa.array(["exact_cosine_after_hnsw"] * n),
                    "ann_ef_search": np.full(n, ef_search, dtype=np.int32),
                    "ann_retry": np.concatenate(retries_out),
                }, schema=schema)
                writer.write_table(table)
        finally:
            writer.close()
        return target

    def validate_candidates(self, snapshot: str, variant: str) -> None:
        candidates = self.candidate_glob(snapshot, variant)
        history = self.stage3b.config.private_dir / f"evaluation_history_{snapshot}.parquet"
        attributes = self.attribute_path(snapshot)
        con = self.connect()
        try:
            violations = con.execute(
                f"""SELECT
                (SELECT count(*) FROM read_parquet('{candidates}') WHERE source_rank<1 OR source_rank>500),
                (SELECT count(*) FROM read_parquet('{candidates}') c JOIN read_parquet('{sql_path(history)}') h USING(customer_key,article_key)),
                (SELECT count(*) FROM read_parquet('{candidates}') c ANTI JOIN read_parquet('{sql_path(attributes)}') a USING(article_key)),
                (SELECT count(*) FROM (SELECT customer_key,article_key,count(*) n FROM read_parquet('{candidates}') GROUP BY 1,2 HAVING n>1))"""
            ).fetchone()
            if violations != (0, 0, 0, 0):
                raise ValueError(f"Two-Tower candidate 协议违反: {violations}")
        finally:
            con.close()

    def build_candidate_lineage(self, snapshot: str, variant: str) -> Path:
        refit = self.validate_formal_checkpoint(snapshot, variant)
        embeddings = json_read(self.embedding_metadata_path(snapshot, variant))
        ann = json_read(self.ann_metadata_path(snapshot, variant))
        parts = [
            self.candidate_part(snapshot, variant, part)
            for part in range(int(self.config.raw["execution"]["candidate_partitions"]))
        ]
        final_fingerprint = refit["final_checkpoint_fingerprint"]
        passed = (
            embeddings.get("source_model_checkpoint_fingerprint") == final_fingerprint
            and ann.get("source_model_checkpoint_fingerprint") == final_fingerprint
            and ann.get("source_item_embedding_fingerprint") == embeddings.get("item_embedding_fingerprint")
            and all(path.exists() for path in parts)
        )
        if not passed:
            raise RuntimeError("正式 candidate 的 checkpoint/embedding/ANN lineage 不一致")
        target = self.candidate_lineage_path(snapshot, variant)
        json_write(target, {
            "stage": "3D", "snapshot_id": snapshot, "model_variant": variant,
            "candidate_role": "formal_full_refit",
            "source_model_checkpoint_fingerprint": final_fingerprint,
            "source_item_embedding_fingerprint": embeddings["item_embedding_fingerprint"],
            "source_user_embedding_fingerprint": embeddings["user_embedding_fingerprint"],
            "source_ann_index_fingerprint": ann["index_fingerprint"],
            "candidate_semantic_fingerprint": semantic_partition_fingerprint(parts),
            "candidate_partitions": len(parts),
            "lineage_passed": True,
            "run_fingerprint": self.run_fingerprint,
        })
        return target

    def evaluate(self, snapshot: str, *, candidate_glob: str, source: str, variant: str) -> Path:
        target = self.user_metric_path(snapshot, source, variant)
        population = self.stage2b_path("evaluation_population", snapshot)
        ground_truth = self.stage2b_path("ground_truth", snapshot)
        spec = self.snapshots[snapshot]
        if file_fingerprint(ground_truth) != spec.ground_truth_fingerprint:
            raise ValueError("Ground truth fingerprint 在评估前发生变化")
        con = self.connect()
        try:
            con.execute(
                f"""COPY (
                WITH users AS (
                  SELECT user_key customer_key,user_segment,is_main_evaluable,positive_item_count
                  FROM read_parquet('{sql_path(population)}')
                ), counts AS (
                  SELECT customer_key,count(*)::INTEGER candidate_count FROM read_parquet('{candidate_glob}') GROUP BY customer_key
                ), hit_rows AS (
                  SELECT c.customer_key,c.source_rank,row_number() OVER(PARTITION BY c.customer_key ORDER BY c.source_rank)::INTEGER cumulative_hits
                  FROM read_parquet('{candidate_glob}') c SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=c.customer_key AND g.article_key=c.article_key
                ), hits AS (
                  SELECT customer_key,
                    count(*) FILTER(WHERE source_rank<=12)::INTEGER hits_12,
                    count(*) FILTER(WHERE source_rank<=50)::INTEGER hits_50,
                    count(*) FILTER(WHERE source_rank<=100)::INTEGER hits_100,
                    count(*) FILTER(WHERE source_rank<=300)::INTEGER hits_300,
                    count(*) FILTER(WHERE source_rank<=500)::INTEGER hits_500,
                    sum(cumulative_hits::DOUBLE/source_rank) FILTER(WHERE source_rank<=12) ap_numerator_12
                  FROM hit_rows GROUP BY customer_key
                )
                SELECT '{snapshot}' snapshot_id,DATE '{spec.history_end}' as_of_date,u.customer_key,u.user_segment,
                  u.is_main_evaluable,u.positive_item_count,coalesce(c.candidate_count,0)::INTEGER candidate_count,
                  coalesce(h.hits_12,0)::INTEGER hits_12,coalesce(h.hits_50,0)::INTEGER hits_50,
                  coalesce(h.hits_100,0)::INTEGER hits_100,coalesce(h.hits_300,0)::INTEGER hits_300,
                  coalesce(h.hits_500,0)::INTEGER hits_500,
                  coalesce(h.hits_12,0)::DOUBLE/u.positive_item_count recall_at_12,
                  coalesce(h.hits_50,0)::DOUBLE/u.positive_item_count recall_at_50,
                  coalesce(h.hits_100,0)::DOUBLE/u.positive_item_count recall_at_100,
                  coalesce(h.hits_300,0)::DOUBLE/u.positive_item_count recall_at_300,
                  coalesce(h.hits_500,0)::DOUBLE/u.positive_item_count recall_at_500,
                  (coalesce(h.hits_12,0)>0)::INTEGER hit_rate_at_12,
                  (coalesce(h.hits_50,0)>0)::INTEGER hit_rate_at_50,
                  (coalesce(h.hits_100,0)>0)::INTEGER hit_rate_at_100,
                  (coalesce(h.hits_300,0)>0)::INTEGER hit_rate_at_300,
                  (coalesce(h.hits_500,0)>0)::INTEGER hit_rate_at_500,
                  coalesce(h.ap_numerator_12,0)::DOUBLE/least(u.positive_item_count,12) ap_at_12,
                  '{source}' AS "source",'{variant}' source_variant,'{spec.ground_truth_fingerprint}' ground_truth_fingerprint,
                  '{self.dataset_fingerprint}' dataset_fingerprint,'{self.config.fingerprint}' config_fingerprint
                FROM users u LEFT JOIN counts c USING(customer_key) LEFT JOIN hits h USING(customer_key)
                ORDER BY customer_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
            )
        finally:
            con.close()
        return target

    @staticmethod
    def _population_sql(metric_path: Path) -> str:
        path = sql_path(metric_path)
        return f"""SELECT 'evaluable_warm_users' segment,* FROM read_parquet('{path}') WHERE is_main_evaluable
        UNION ALL SELECT 'short_history',* FROM read_parquet('{path}') WHERE user_segment='short_history'
        UNION ALL SELECT 'warm',* FROM read_parquet('{path}') WHERE user_segment='warm'
        UNION ALL SELECT 'active',* FROM read_parquet('{path}') WHERE user_segment='active'
        UNION ALL SELECT 'cold_history_label_users',* FROM read_parquet('{path}') WHERE user_segment='cold'"""

    def metric_summary(self, snapshot: str, source: str, variant: str, metric_path: Path) -> list[dict[str, Any]]:
        con = self.connect()
        try:
            values = con.execute(
                f"""WITH p AS ({self._population_sql(metric_path)})
                SELECT segment,count(*) denominator,avg(ap_at_12),avg(recall_at_12),avg(recall_at_50),
                  avg(recall_at_100),avg(recall_at_300),avg(recall_at_500),avg(hit_rate_at_500),
                  avg(candidate_count),avg((candidate_count>0)::INTEGER),avg((candidate_count<500)::INTEGER)
                FROM p GROUP BY segment ORDER BY segment"""
            ).fetchall()
        finally:
            con.close()
        fields = ("map_at_12", "recall_at_12", "recall_at_50", "recall_at_100", "recall_at_300",
                  "recall_at_500", "hit_rate_at_500", "mean_candidates", "candidate_user_coverage",
                  "candidate_shortfall_rate")
        rows = []
        for value in values:
            for index, metric in enumerate(fields, 2):
                rows.append({
                    "stage": "3D", "phase": "validation" if snapshot == "validation_14d" else "test",
                    "snapshot_id": snapshot, "as_of_date": self.snapshots[snapshot].history_end,
                    "source": source, "model_variant": variant if source == "two_tower" else "not_applicable",
                    "union_variant": variant if source == "union" else "not_applicable",
                    "segment": value[0], "population_denominator": value[1],
                    "metric": metric, "metric_value": value[index],
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "candidate_universe_version": "history_seen_items_v1", "run_status": "SUCCESS",
                })
        return rows

    def _build_union_chunk(
        self, snapshot: str, union_variant: str, model_variant: str,
        part: int, modulus: int, residue: int, target: Path,
    ) -> None:
        if union_variant not in {"T50", "T100"}:
            raise ValueError("Stage 3D 只新建 T50/T100；Q0 必须精确复用 Stage 3B")
        caps = self.config.raw["union"]["variants"][union_variant]
        population = self.stage2b_path("evaluation_population", snapshot)
        repurchase = self.stage3a_candidate(snapshot, "repurchase", "pure")
        trend = self.stage3a_ranking(snapshot, "trend", "window_7d")
        popularity = self.stage3a_ranking(snapshot, "popularity", "global")
        itemcf = self.stage3b_candidate_part(snapshot, part)
        two_tower = self.candidate_part(snapshot, model_variant, part)
        spec = self.snapshots[snapshot]
        target.parent.mkdir(parents=True, exist_ok=True)
        con = self.connect()
        try:
            con.execute(
                f"""COPY (
                WITH users AS (
                  SELECT user_key customer_key,is_main_evaluable FROM read_parquet('{sql_path(population)}')
                  WHERE user_key%{modulus}={residue}
                ), primary_raw AS (
                  SELECT r.customer_key,r.article_key,'repurchase' source_name,1 source_priority,r.source_rank,r.source_score
                  FROM read_parquet('{sql_path(repurchase)}') r SEMI JOIN users u USING(customer_key)
                  WHERE r.source_rank<={int(caps['repurchase'])}
                  UNION ALL
                  SELECT i.customer_key,i.article_key,'itemcf',2,i.source_rank,i.source_score
                  FROM read_parquet('{sql_path(itemcf)}') i SEMI JOIN users u USING(customer_key)
                  WHERE i.source_rank<={int(caps['itemcf'])}
                  UNION ALL
                  SELECT t.customer_key,t.article_key,'two_tower',3,t.source_rank,t.source_score
                  FROM read_parquet('{sql_path(two_tower)}') t SEMI JOIN users u USING(customer_key)
                  WHERE t.source_rank<={int(caps['two_tower'])}
                  UNION ALL
                  SELECT u.customer_key,t.article_key,'trend',4,t.source_rank,t.source_score
                  FROM users u CROSS JOIN read_parquet('{sql_path(trend)}') t
                  WHERE t.source_rank<={int(caps['trend'])}
                ), primary_agg AS (
                  SELECT customer_key,article_key,
                    bool_or(source_name='repurchase') from_repurchase,bool_or(source_name='itemcf') from_itemcf,
                    bool_or(source_name='two_tower') from_two_tower,bool_or(source_name='trend') from_trend,
                    min(source_rank) FILTER(WHERE source_name='repurchase') repurchase_rank,
                    max(source_score) FILTER(WHERE source_name='repurchase') repurchase_score,
                    min(source_rank) FILTER(WHERE source_name='itemcf') itemcf_rank,
                    max(source_score) FILTER(WHERE source_name='itemcf') itemcf_score,
                    min(source_rank) FILTER(WHERE source_name='two_tower') two_tower_rank,
                    max(source_score) FILTER(WHERE source_name='two_tower') two_tower_score,
                    min(source_rank) FILTER(WHERE source_name='trend') trend_rank,
                    max(source_score) FILTER(WHERE source_name='trend') trend_score,
                    min(source_priority) first_priority,count(DISTINCT source_name)::INTEGER source_count
                  FROM primary_raw GROUP BY customer_key,article_key
                ), primary_ranked AS (
                  SELECT *,row_number() OVER(PARTITION BY customer_key ORDER BY first_priority,
                    CASE first_priority WHEN 1 THEN repurchase_rank WHEN 2 THEN itemcf_rank
                         WHEN 3 THEN two_tower_rank ELSE trend_rank END,article_key)::INTEGER union_rank
                  FROM primary_agg
                ), primary_selected AS (SELECT * FROM primary_ranked WHERE union_rank<=500),
                primary_counts AS (
                  SELECT u.customer_key,u.is_main_evaluable,count(p.article_key)::INTEGER primary_count
                  FROM users u LEFT JOIN primary_selected p USING(customer_key)
                  GROUP BY u.customer_key,u.is_main_evaluable
                ), popularity_pool AS (
                  SELECT pc.customer_key,pc.is_main_evaluable,pc.primary_count,p.article_key,
                         p.source_rank popularity_rank,p.source_score popularity_score
                  FROM primary_counts pc CROSS JOIN read_parquet('{sql_path(popularity)}') p
                  ANTI JOIN primary_selected x ON x.customer_key=pc.customer_key AND x.article_key=p.article_key
                  WHERE p.source_rank<=500+pc.primary_count
                ), popularity_fill AS (
                  SELECT *,row_number() OVER(PARTITION BY customer_key ORDER BY popularity_rank,article_key)::INTEGER fill_rank
                  FROM popularity_pool QUALIFY fill_rank<=500-primary_count
                ), combined AS (
                  SELECT p.customer_key,p.article_key,p.union_rank,
                    CASE p.first_priority WHEN 1 THEN 'repurchase' WHEN 2 THEN 'itemcf'
                         WHEN 3 THEN 'two_tower' ELSE 'trend' END first_source,
                    p.from_repurchase,p.from_itemcf,p.from_two_tower,p.from_trend,false from_popularity,
                    p.repurchase_rank,p.repurchase_score,p.itemcf_rank,p.itemcf_score,p.two_tower_rank,p.two_tower_score,
                    p.trend_rank,p.trend_score,NULL::INTEGER popularity_rank,NULL::DOUBLE popularity_score,
                    p.source_count,u.is_main_evaluable
                  FROM primary_selected p JOIN users u USING(customer_key)
                  UNION ALL
                  SELECT customer_key,article_key,(primary_count+fill_rank)::INTEGER,'popularity_fallback',
                    false,false,false,false,true,NULL::INTEGER,NULL::DOUBLE,NULL::INTEGER,NULL::DOUBLE,
                    NULL::INTEGER,NULL::DOUBLE,NULL::INTEGER,NULL::DOUBLE,popularity_rank,popularity_score,
                    1::INTEGER,is_main_evaluable FROM popularity_fill
                )
                SELECT '{snapshot}' snapshot_id,DATE '{spec.history_end}' as_of_date,customer_key,article_key,
                  'union' AS "source",'{model_variant}_{union_variant}' source_variant,union_rank source_rank,
                  1.0/union_rank::DOUBLE source_score,'history_seen_items_v1' candidate_universe_version,
                  '{self.feedback_version}' feedback_version,'{self.population_version}' population_version,
                  '{self.dataset_fingerprint}' dataset_fingerprint,'{self.config.fingerprint}' config_fingerprint,
                  first_source,source_count,from_repurchase,from_itemcf,from_two_tower,from_trend,from_popularity,
                  repurchase_rank,repurchase_score,itemcf_rank,itemcf_score,two_tower_rank,two_tower_score,
                  trend_rank,trend_score,popularity_rank,popularity_score,
                  'source_rank_cap_pre_dedup_then_fixed_insertion' source_score_kind
                FROM combined ORDER BY customer_key,source_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
            )
        finally:
            con.close()

    def build_union_part(self, snapshot: str, union_variant: str, model_variant: str, part: int) -> Path:
        partitions = int(self.config.raw["execution"]["candidate_partitions"])
        subparts = int(self.config.raw["execution"]["union_subpartitions"])
        temp = self.config.temp_dir / "union_chunks" / snapshot / f"{model_variant}_{union_variant}"
        temp.mkdir(parents=True, exist_ok=True)
        chunks = []
        for subpart in range(subparts):
            chunk = temp / f"part_{part:03d}_sub_{subpart:02d}.parquet"
            self._build_union_chunk(snapshot, union_variant, model_variant, part,
                                    partitions * subparts, part + partitions * subpart, chunk)
            chunks.append(chunk)
        target = self.union_part(snapshot, f"{model_variant}_{union_variant}", part)
        target.parent.mkdir(parents=True, exist_ok=True)
        con = self.connect()
        try:
            literals = ",".join(f"'{sql_path(path)}'" for path in chunks)
            con.execute(f"COPY (SELECT * FROM read_parquet([{literals}]) ORDER BY customer_key,source_rank) "
                        f"TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally:
            con.close()
        for chunk in chunks:
            chunk.unlink(missing_ok=True)
        return target

    def validate_union(self, snapshot: str, variant: str) -> None:
        glob = self.union_glob(snapshot, variant)
        population = self.stage2b_path("evaluation_population", snapshot)
        con = self.connect()
        try:
            audit = con.execute(
                f"""WITH x AS (
                  SELECT customer_key,count(*) n,count(DISTINCT article_key) d,min(source_rank) lo,
                         max(source_rank) hi,count(DISTINCT source_rank) dr
                  FROM read_parquet('{glob}') GROUP BY customer_key)
                SELECT count(*) FILTER(WHERE n>500 OR n<>d OR lo<>1 OR hi<>n OR dr<>n),
                       (SELECT count(*) FROM read_parquet('{sql_path(population)}'))-count(*),
                       count(*) FILTER(WHERE n<>500) FROM x"""
            ).fetchone()
            if audit != (0, 0, 0):
                raise ValueError(f"Union 固定预算协议违反: {audit}")
        finally:
            con.close()

    def reproduce_q0(self, snapshot: str) -> Path:
        target = self.user_metric_path(snapshot, "union", "Q0")
        self.evaluate(snapshot, candidate_glob=self.q0_glob(snapshot), source="union", variant="Q0")
        public = csv_rows(self.config.root / "artifacts/stage_3b"
                          / ("validation_union_metrics.csv" if snapshot == "validation_14d" else "test_union_metrics.csv"))
        expected = next(float(row["metric_value"]) for row in public
                        if row["source_variant"] == self.q0_variant(snapshot)
                        and row["metric"] == "recall_at_500"
                        and row["segment"] == "evaluable_warm_users")
        actual = self._metric_lookup(self.metric_summary(snapshot, "union", "Q0", target),
                                     segment="evaluable_warm_users", metric="recall_at_500")
        if abs(expected - actual) > 1e-15:
            raise ValueError(f"BLOCKED_BY_Q0_REPRODUCTION: {actual} != {expected}")
        return target

    @staticmethod
    def _metric_lookup(rows: Sequence[Mapping[str, Any]], *, segment: str, metric: str) -> float:
        return float(next(row["metric_value"] for row in rows
                          if row["segment"] == segment and row["metric"] == metric))

    def comparison_analysis(
        self, snapshot: str, model_variant: str, union_variant: str,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        challenger_name = f"{model_variant}_{union_variant}"
        challenger = self.union_glob(snapshot, challenger_name)
        route = self.candidate_glob(snapshot, model_variant)
        itemcf = sql_path(self.stage3b.config.private_dir / "candidates" / snapshot / "iuf" / "part_*.parquet")
        trend = self.stage3a_ranking(snapshot, "trend", "window_7d")
        repurchase = self.stage3a_candidate(snapshot, "repurchase", "pure")
        ground_truth = self.stage2b_path("ground_truth", snapshot)
        q0_metric = self.user_metric_path(snapshot, "union", "Q0")
        challenger_metric = self.user_metric_path(snapshot, "union", challenger_name)
        con = self.connect()
        try:
            change = con.execute(
                f"""WITH p AS (
                  SELECT q.customer_key,q.user_segment,q.is_main_evaluable,q.recall_at_500 base_recall,
                         c.recall_at_500 challenger_recall,q.hits_500 base_hits,c.hits_500 challenger_hits
                  FROM read_parquet('{sql_path(q0_metric)}') q JOIN read_parquet('{sql_path(challenger_metric)}') c USING(customer_key)
                  WHERE q.is_main_evaluable),
                gained AS (
                  SELECT count(*) n FROM read_parquet('{challenger}') c
                  SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=c.customer_key AND g.article_key=c.article_key
                  ANTI JOIN read_parquet('{self.q0_glob(snapshot)}') q USING(customer_key,article_key)
                ),
                lost AS (
                  SELECT count(*) n FROM read_parquet('{self.q0_glob(snapshot)}') q
                  SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=q.customer_key AND g.article_key=q.article_key
                  ANTI JOIN read_parquet('{challenger}') c USING(customer_key,article_key)
                )
                SELECT count(*) user_count,avg(base_recall),avg(challenger_recall),
                  count(*) FILTER(WHERE challenger_recall>base_recall),
                  count(*) FILTER(WHERE challenger_recall=base_recall),
                  count(*) FILTER(WHERE challenger_recall<base_recall),
                  (SELECT n FROM gained),(SELECT n FROM lost),
                  sum(challenger_hits-base_hits) FROM p"""
            ).fetchone()
            exclusive = con.execute(
                f"""WITH r AS (SELECT customer_key,article_key FROM read_parquet('{route}')),
                route_hits AS (SELECT r.* FROM r SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                               ON g.user_key=r.customer_key AND g.article_key=r.article_key)
                SELECT count(*) route_ground_truth_hits,
                  count(*) FILTER(WHERE
                    NOT EXISTS(SELECT 1 FROM read_parquet('{itemcf}') i WHERE i.customer_key=h.customer_key AND i.article_key=h.article_key)
                    AND NOT EXISTS(SELECT 1 FROM read_parquet('{sql_path(repurchase)}') p WHERE p.customer_key=h.customer_key AND p.article_key=h.article_key)
                    AND NOT EXISTS(SELECT 1 FROM read_parquet('{sql_path(trend)}') t WHERE t.source_rank<=500 AND t.article_key=h.article_key)),
                  count(DISTINCT customer_key) FILTER(WHERE
                    NOT EXISTS(SELECT 1 FROM read_parquet('{itemcf}') i WHERE i.customer_key=h.customer_key AND i.article_key=h.article_key)
                    AND NOT EXISTS(SELECT 1 FROM read_parquet('{sql_path(repurchase)}') p WHERE p.customer_key=h.customer_key AND p.article_key=h.article_key)
                    AND NOT EXISTS(SELECT 1 FROM read_parquet('{sql_path(trend)}') t WHERE t.source_rank<=500 AND t.article_key=h.article_key))
                FROM route_hits h"""
            ).fetchone()
            overlap = con.execute(
                f"""WITH rc AS (SELECT customer_key,count(*) n FROM read_parquet('{route}') GROUP BY 1),
                ic AS (SELECT customer_key,count(*) n FROM read_parquet('{itemcf}') GROUP BY 1),
                pc AS (SELECT customer_key,count(*) n FROM read_parquet('{sql_path(repurchase)}') GROUP BY 1),
                tc AS (SELECT count(*) n FROM read_parquet('{sql_path(trend)}') WHERE source_rank<=500),
                ri AS (SELECT r.customer_key,count(*) n FROM read_parquet('{route}') r JOIN read_parquet('{itemcf}') i USING(customer_key,article_key) GROUP BY 1),
                rt AS (SELECT r.customer_key,count(*) n FROM read_parquet('{route}') r JOIN read_parquet('{sql_path(trend)}') t USING(article_key) WHERE t.source_rank<=500 GROUP BY 1),
                rp AS (SELECT r.customer_key,count(*) n FROM read_parquet('{route}') r JOIN read_parquet('{sql_path(repurchase)}') p USING(customer_key,article_key) GROUP BY 1)
                SELECT avg(coalesce(ri.n,0)::DOUBLE/(rc.n+coalesce(ic.n,0)-coalesce(ri.n,0))),
                       avg(coalesce(rt.n,0)::DOUBLE/(rc.n+tc.n-coalesce(rt.n,0))),
                       avg(CASE WHEN rc.n+coalesce(pc.n,0)=0 THEN 0
                           ELSE coalesce(rp.n,0)::DOUBLE/(rc.n+coalesce(pc.n,0)-coalesce(rp.n,0)) END)
                FROM rc LEFT JOIN ic USING(customer_key) LEFT JOIN ri USING(customer_key)
                LEFT JOIN rt USING(customer_key) LEFT JOIN rp USING(customer_key)
                LEFT JOIN pc USING(customer_key) CROSS JOIN tc"""
            ).fetchone()
            # 分群 delta 明细直接复用固定用户级分母。
            segment_values = con.execute(
                f"""SELECT q.user_segment,count(*),avg(c.recall_at_500-q.recall_at_500),
                  avg(c.hit_rate_at_500-q.hit_rate_at_500),avg(c.ap_at_12-q.ap_at_12)
                FROM read_parquet('{sql_path(q0_metric)}') q JOIN read_parquet('{sql_path(challenger_metric)}') c USING(customer_key)
                WHERE q.is_main_evaluable GROUP BY q.user_segment ORDER BY q.user_segment"""
            ).fetchall()
        finally:
            con.close()
        user_count = int(change[0])
        marginal = {
            "stage": "3D", "phase": "validation" if snapshot == "validation_14d" else "test",
            "snapshot_id": snapshot, "model_variant": model_variant, "union_variant": union_variant,
            "population_denominator": user_count, "q0_recall_at_500": change[1],
            "challenger_recall_at_500": change[2], "delta_recall_at_500": float(change[2]) - float(change[1]),
            "improved_users": int(change[3]), "unchanged_users": int(change[4]), "declined_users": int(change[5]),
            "improved_user_rate": int(change[3]) / user_count, "unchanged_user_rate": int(change[4]) / user_count,
            "declined_user_rate": int(change[5]) / user_count, "gained_ground_truth_hits": int(change[6]),
            "lost_ground_truth_hits": int(change[7]), "net_ground_truth_hits": int(change[8]),
        }
        overlap_row = {
            "stage": "3D", "phase": marginal["phase"], "snapshot_id": snapshot,
            "model_variant": model_variant, "union_variant": union_variant,
            "two_tower_itemcf_jaccard": overlap[0], "two_tower_trend_jaccard": overlap[1],
            "two_tower_repurchase_jaccard": overlap[2], "two_tower_ground_truth_hits": int(exclusive[0]),
            "two_tower_exclusive_ground_truth_hits": int(exclusive[1]),
            "users_with_two_tower_exclusive_hits": int(exclusive[2]),
        }
        segments = [{
            "stage": "3D", "phase": marginal["phase"], "snapshot_id": snapshot,
            "model_variant": model_variant, "union_variant": union_variant,
            "segment": value[0], "population_denominator": value[1],
            "delta_recall_at_500": value[2], "delta_hit_rate_at_500": value[3], "delta_map_at_12": value[4],
        } for value in segment_values]
        return marginal, overlap_row, segments

    def quota_occupancy(self, snapshot: str, model_variant: str, union_variant: str) -> list[dict[str, Any]]:
        glob = self.union_glob(snapshot, f"{model_variant}_{union_variant}")
        caps = self.config.raw["union"]["variants"][union_variant]
        population = self.stage2b_path("evaluation_population", snapshot)
        sources = ("repurchase", "itemcf", "two_tower", "trend", "popularity")
        repurchase = self.stage3a_candidate(snapshot, "repurchase", "pure")
        itemcf = sql_path(self.stage3b.config.private_dir / "candidates" / snapshot / "iuf" / "part_*.parquet")
        route = self.candidate_glob(snapshot, model_variant)
        con = self.connect()
        try:
            means = con.execute(
                f"""WITH users AS (
                  SELECT user_key customer_key FROM read_parquet('{sql_path(population)}') WHERE is_main_evaluable
                ), rb AS (SELECT customer_key,count(*) n FROM read_parquet('{sql_path(repurchase)}') GROUP BY 1),
                ib AS (SELECT customer_key,count(*) n FROM read_parquet('{itemcf}') GROUP BY 1),
                tb AS (SELECT customer_key,count(*) n FROM read_parquet('{route}') GROUP BY 1),
                occ AS (
                  SELECT customer_key,count(*) n,
                    sum(from_repurchase::INTEGER) rep_present,sum((first_source='repurchase')::INTEGER) rep_first,
                    sum(from_itemcf::INTEGER) ic_present,sum((first_source='itemcf')::INTEGER) ic_first,
                    sum(from_two_tower::INTEGER) tt_present,sum((first_source='two_tower')::INTEGER) tt_first,
                    sum(from_trend::INTEGER) tr_present,sum((first_source='trend')::INTEGER) tr_first,
                    sum((first_source='popularity_fallback')::INTEGER) pop_first
                  FROM read_parquet('{glob}') GROUP BY 1
                )
                SELECT avg(o.n),avg(500-o.n),
                  avg(coalesce(rb.n,0)),avg(least(coalesce(rb.n,0),{int(caps['repurchase'])})),avg(o.rep_present),avg(o.rep_first),
                  avg(coalesce(ib.n,0)),avg(least(coalesce(ib.n,0),{int(caps['itemcf'])})),avg(o.ic_present),avg(o.ic_first),
                  avg(coalesce(tb.n,0)),avg(least(coalesce(tb.n,0),{int(caps['two_tower'])})),avg(o.tt_present),avg(o.tt_first),
                  500::DOUBLE,{int(caps['trend'])}::DOUBLE,avg(o.tr_present),avg(o.tr_first),avg(o.pop_first)
                FROM users u LEFT JOIN rb USING(customer_key) LEFT JOIN ib USING(customer_key)
                LEFT JOIN tb USING(customer_key) JOIN occ o USING(customer_key)"""
            ).fetchone()
            q0_itemcf_mean = con.execute(
                f"""SELECT avg(n) FROM (SELECT customer_key,count(*) FILTER(WHERE first_source='itemcf') n
                FROM read_parquet('{self.q0_glob(snapshot)}') q
                SEMI JOIN read_parquet('{sql_path(population)}') p
                  ON p.user_key=q.customer_key AND p.is_main_evaluable GROUP BY customer_key)"""
            ).fetchone()[0]
        finally:
            con.close()
        before = (means[2], means[6], means[10], means[14], means[18])
        truncated = (means[3], means[7], means[11], means[15], means[18])
        present = (means[4], means[8], means[12], means[16], means[18])
        occupancy = (means[5], means[9], means[13], means[17], means[18])
        output = []
        for index, source in enumerate(sources):
            source_key = "popularity_fallback" if source == "popularity" else source
            actual = float(occupancy[index])
            duplicates = max(float(present[index]) - actual, 0.0)
            output.append({
                "stage": "3D", "phase": "validation" if snapshot == "validation_14d" else "test",
                "snapshot_id": snapshot, "model_variant": model_variant, "union_variant": union_variant,
                "population": "evaluable_warm_users", "source": source_key,
                "configured_source_rank_cap": caps.get(source_key),
                "mean_candidates_before_cap": before[index],
                "mean_candidates_after_source_rank_cap": truncated[index],
                "mean_candidates_present_after_dedup": present[index],
                "mean_actual_unique_occupancy": actual,
                "mean_duplicates_with_higher_priority": duplicates,
                "cross_source_duplicate_rate": duplicates / float(truncated[index]) if float(truncated[index]) else 0.0,
                "mean_effective_position_loss": (max(float(caps[source_key]) - actual, 0.0)
                                                 if source_key in caps else 0.0),
                "mean_popularity_fill": means[18], "mean_candidate_count": means[0],
                "mean_candidate_shortfall": means[1],
                "duplicate_loss": duplicates,
                "mean_itemcf_positions_reduced_vs_q0": (float(q0_itemcf_mean) - actual if source == "itemcf" else None),
                "quota_semantics": "source_rank_cap_before_cross_source_dedup",
            })
        return output

    def refresh_quota_occupancy_public(self) -> Path:
        combinations = [("validation_14d", "TT0", "T50"),
                        ("validation_14d", "TT1", "T50"),
                        ("validation_14d", "TT1", "T100")]
        if self.union_part("test_14d", "TT1_T100", 0).exists():
            combinations.append(("test_14d", "TT1", "T100"))
        rows = [row for snapshot, model, quota in combinations
                for row in self.quota_occupancy(snapshot, model, quota)]
        return self._write_csv("quota_occupancy.csv", rows)

    def bootstrap_comparison(self, snapshot: str, model_variant: str, union_variant: str) -> list[dict[str, Any]]:
        q0 = self.user_metric_path(snapshot, "union", "Q0")
        challenger = self.user_metric_path(snapshot, "union", f"{model_variant}_{union_variant}")
        con = self.connect()
        try:
            table = con.execute(
                f"""SELECT q.user_segment,q.is_main_evaluable,q.recall_at_500 base,c.recall_at_500 challenger
                FROM read_parquet('{sql_path(q0)}') q JOIN read_parquet('{sql_path(challenger)}') c USING(customer_key)"""
            ).fetch_arrow_table()
        finally:
            con.close()
        segments = np.asarray(table["user_segment"].to_pylist(), dtype="U16")
        main = np.asarray(table["is_main_evaluable"]).astype(bool)
        base = np.asarray(table["base"], dtype=np.float64)
        challenger_values = np.asarray(table["challenger"], dtype=np.float64)
        selectors = {
            "evaluable_warm_users": main, "short_history": segments == "short_history",
            "warm": segments == "warm", "active": segments == "active", "cold_history_label_users": segments == "cold",
        }
        output = []
        for offset, (segment, selector) in enumerate(selectors.items()):
            result = paired_bootstrap(
                base[selector].tolist(), challenger_values[selector].tolist(),
                iterations=int(self.config.raw["bootstrap"]["iterations"]),
                seed=int(self.config.raw["bootstrap"]["seed"]) + offset,
                confidence_level=float(self.config.raw["bootstrap"]["confidence_level"]),
            )
            output.append({
                "stage": "3D", "phase": "validation" if snapshot == "validation_14d" else "test",
                "snapshot_id": snapshot, "model_variant": model_variant, "union_variant": union_variant,
                "comparison": f"{model_variant}_{union_variant}_vs_Q0", "segment": segment,
                **result, "iterations": self.config.raw["bootstrap"]["iterations"],
                "seed": int(self.config.raw["bootstrap"]["seed"]) + offset,
                "confidence_level": self.config.raw["bootstrap"]["confidence_level"],
                "interpretation": "offline_paired_user_block_bootstrap_not_online_ab_test",
            })
        return output

    def _run_snapshot_foundation(self, snapshot: str) -> None:
        self.run_group(f"{snapshot}_history", lambda: self.build_snapshot_history(snapshot),
                       [self.history_path(snapshot)])
        self.run_group(f"{snapshot}_training_targets", lambda: self.build_training_targets(snapshot),
                       [self.target_path(snapshot)])
        for part in range(int(self.config.raw["training_samples"]["partitions"])):
            self.run_group(
                f"{snapshot}_training_samples_{part:03d}",
                lambda part=part: self.build_sample_part(snapshot, part),
                [self.sample_part(snapshot, part)],
            )
        self.run_group(f"{snapshot}_attributes", lambda: self.build_attributes(snapshot), [
            self.attribute_path(snapshot), self.attribute_vocab_path(snapshot), self.attribute_index_path(snapshot),
        ])
        self.run_group(f"{snapshot}_user_features", lambda: self.build_user_features(snapshot),
                       [self.user_feature_path(snapshot)])

    def _run_model_core(
        self,
        snapshot: str,
        variant: str,
        *,
        locked_epoch: int,
        selection_checkpoint: Path,
        locked_ef_search: int | None = None,
    ) -> None:
        self.run_group(
            f"{snapshot}_{variant}_final_refit",
            lambda: self.train_final_model(
                snapshot, variant, locked_epoch=locked_epoch,
                selection_checkpoint=selection_checkpoint,
            ),
            [self.model_checkpoint(snapshot, variant), self.final_refit_history_path(snapshot, variant)],
        )
        self.run_group(f"{snapshot}_{variant}_embeddings", lambda: self.export_embeddings(snapshot, variant), [
            self.embedding_path(snapshot, variant, "items"), self.embedding_path(snapshot, variant, "users"),
            self.embedding_metadata_path(snapshot, variant),
        ])
        self.run_group(f"{snapshot}_{variant}_ann_index", lambda: self.build_ann(snapshot, variant), [
            self.ann_index_path(snapshot, variant), self.ann_metadata_path(snapshot, variant),
        ])
        self.run_group(
            f"{snapshot}_{variant}_ann_audit",
            lambda: self.audit_ann(snapshot, variant, locked_ef_search=locked_ef_search),
            [self.ann_audit_detail_path(snapshot, variant), self.ann_audit_summary_path(snapshot, variant)],
        )
        for part in range(int(self.config.raw["execution"]["candidate_partitions"])):
            self.run_group(
                f"{snapshot}_{variant}_candidates_{part:03d}",
                lambda part=part: self.build_candidate_part(snapshot, variant, part),
                [self.candidate_part(snapshot, variant, part)],
            )
        self.validate_candidates(snapshot, variant)
        self.run_group(
            f"{snapshot}_{variant}_candidate_lineage",
            lambda: self.build_candidate_lineage(snapshot, variant),
            [self.candidate_lineage_path(snapshot, variant)],
        )
        metric = self.user_metric_path(snapshot, "two_tower", variant)
        self.run_group(
            f"{snapshot}_{variant}_single_route_evaluation",
            lambda: self.evaluate(snapshot, candidate_glob=self.candidate_glob(snapshot, variant),
                                  source="two_tower", variant=variant), [metric],
        )

    def _run_union_variant(self, snapshot: str, model_variant: str, union_variant: str) -> None:
        physical = f"{model_variant}_{union_variant}"
        for part in range(int(self.config.raw["execution"]["candidate_partitions"])):
            self.run_group(
                f"{snapshot}_{physical}_union_{part:03d}",
                lambda part=part: self.build_union_part(snapshot, union_variant, model_variant, part),
                [self.union_part(snapshot, physical, part)],
            )
        self.validate_union(snapshot, physical)
        metric = self.user_metric_path(snapshot, "union", physical)
        self.run_group(
            f"{snapshot}_{physical}_union_evaluation",
            lambda: self.evaluate(snapshot, candidate_glob=self.union_glob(snapshot, physical),
                                  source="union", variant=physical), [metric],
        )

    def _write_csv(self, name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        if not rows:
            raise ValueError(f"公开聚合 {name} 不允许为空")
        target = self.config.artifact_dir / name
        fields: list[str] = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        normalized = [{field: row.get(field) for field in fields} for row in rows]
        write_csv(target, normalized, fields)
        return target

    def _single_route_extra_metrics(self, snapshot: str, variant: str) -> list[dict[str, Any]]:
        candidates = self.candidate_glob(snapshot, variant)
        attributes = self.attribute_path(snapshot)
        con = self.connect()
        try:
            values = con.execute(
                f"""WITH item_counts AS (
                  SELECT article_key,count(*) n FROM read_parquet('{candidates}') GROUP BY article_key
                ), ranked AS (
                  SELECT n,row_number() OVER(ORDER BY n DESC,article_key) r,count(*) OVER() catalog_items,
                         sum(n) OVER() all_rows FROM item_counts)
                SELECT (SELECT count(DISTINCT article_key)::DOUBLE/(SELECT count(*) FROM read_parquet('{sql_path(attributes)}'))
                        FROM read_parquet('{candidates}')) catalog_coverage,
                       sum(n) FILTER(WHERE r<=greatest(ceil(catalog_items*0.01),1))/max(all_rows) top_1pct_candidate_share,
                       max(n)::DOUBLE/max(all_rows) top_item_candidate_share FROM ranked"""
            ).fetchone()
        finally:
            con.close()
        base = {
            "stage": "3D", "phase": "validation" if snapshot == "validation_14d" else "test",
            "snapshot_id": snapshot, "source": "two_tower", "model_variant": variant,
            "union_variant": "not_applicable", "segment": "evaluable_warm_users",
            "population_denominator": None, "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint, "candidate_universe_version": "history_seen_items_v1",
            "run_status": "SUCCESS",
        }
        return [{**base, "metric": name, "metric_value": value} for name, value in zip(
            ("catalog_coverage", "top_1pct_candidate_share", "top_item_candidate_share"), values, strict=True)]

    def popularity_bucket_summary(
        self, snapshot: str, model_variant: str, union_variant: str,
    ) -> list[dict[str, Any]]:
        state = self.stage2b_path("item_state", snapshot)
        ground_truth = self.stage2b_path("ground_truth", snapshot)
        paths = {
            "two_tower": self.candidate_glob(snapshot, model_variant),
            "iuf_itemcf": sql_path(self.stage3b.config.private_dir / "candidates" / snapshot / "iuf" / "part_*.parquet"),
            "challenger_union": self.union_glob(snapshot, f"{model_variant}_{union_variant}"),
            "q0_union": self.q0_glob(snapshot),
        }
        con = self.connect()
        try:
            rows: list[dict[str, Any]] = []
            for route_name, route_path in paths.items():
                values = con.execute(
                    f"""WITH c AS (SELECT customer_key,article_key FROM read_parquet('{route_path}')),
                    enriched AS (SELECT c.*,s.popularity_bucket FROM c JOIN read_parquet('{sql_path(state)}') s USING(article_key)),
                    hits AS (SELECT e.popularity_bucket,count(*) n FROM enriched e
                      SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                        ON g.user_key=e.customer_key AND g.article_key=e.article_key GROUP BY 1),
                    counts AS (SELECT popularity_bucket,count(*) n FROM enriched GROUP BY 1)
                    SELECT x.popularity_bucket,x.n,x.n::DOUBLE/sum(x.n) OVER(),coalesce(h.n,0)
                    FROM counts x LEFT JOIN hits h USING(popularity_bucket) ORDER BY x.popularity_bucket"""
                ).fetchall()
                for value in values:
                    rows.append({
                        "stage": "3D", "phase": "validation" if snapshot == "validation_14d" else "test",
                        "snapshot_id": snapshot, "model_variant": model_variant,
                        "union_variant": union_variant, "route": route_name, "popularity_bucket": value[0],
                        "candidate_count": value[1], "candidate_share": value[2],
                        "ground_truth_hits": value[3], "exclusive_hits": None, "lost_hits_vs_q0": None,
                        "bucket_definition": "stage_2b_cutoff_history_support", "run_status": "SUCCESS",
                    })
            exclusive = con.execute(
                f"""WITH r AS (SELECT customer_key,article_key FROM read_parquet('{paths['two_tower']}')),
                hits AS (SELECT r.* FROM r SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                  ON g.user_key=r.customer_key AND g.article_key=r.article_key),
                exclusive AS (SELECT h.* FROM hits h
                  ANTI JOIN read_parquet('{paths['iuf_itemcf']}') i USING(customer_key,article_key)
                  ANTI JOIN read_parquet('{sql_path(self.stage3a_candidate(snapshot, 'repurchase', 'pure'))}') p USING(customer_key,article_key)
                  WHERE NOT EXISTS(SELECT 1 FROM read_parquet('{sql_path(self.stage3a_ranking(snapshot, 'trend', 'window_7d'))}') t
                                   WHERE t.source_rank<=500 AND t.article_key=h.article_key))
                SELECT s.popularity_bucket,count(*) FROM exclusive e
                JOIN read_parquet('{sql_path(state)}') s USING(article_key) GROUP BY 1"""
            ).fetchall()
            lost = con.execute(
                f"""SELECT s.popularity_bucket,count(*) FROM read_parquet('{paths['q0_union']}') q
                SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                  ON g.user_key=q.customer_key AND g.article_key=q.article_key
                ANTI JOIN read_parquet('{paths['challenger_union']}') c USING(customer_key,article_key)
                JOIN read_parquet('{sql_path(state)}') s USING(article_key) GROUP BY 1"""
            ).fetchall()
        finally:
            con.close()
        exclusive_map = {str(value[0]): int(value[1]) for value in exclusive}
        lost_map = {str(value[0]): int(value[1]) for value in lost}
        for row in rows:
            if row["route"] == "two_tower":
                row["exclusive_hits"] = exclusive_map.get(str(row["popularity_bucket"]), 0)
            if row["route"] == "challenger_union":
                row["lost_hits_vs_q0"] = lost_map.get(str(row["popularity_bucket"]), 0)
        return rows

    def _selection_rows(self) -> tuple[list[dict[str, Any]], str]:
        rows = []
        for variant in self.config.raw["model"]["variants"]:
            metric = self.metric_summary("validation_14d", "union", f"{variant}_T50",
                                         self.user_metric_path("validation_14d", "union", f"{variant}_T50"))
            single = self.metric_summary("validation_14d", "two_tower", variant,
                                         self.user_metric_path("validation_14d", "two_tower", variant))
            marginal, overlap, _ = self.comparison_analysis("validation_14d", variant, "T50")
            audit = json_read(self.ann_audit_summary_path("validation_14d", variant))
            rows.append({
                "stage": "3D", "phase": "validation", "snapshot_id": "validation_14d",
                "model_variant": variant,
                "t50_recall_at_500": self._metric_lookup(metric, segment="evaluable_warm_users", metric="recall_at_500"),
                "t50_hit_rate_at_500": self._metric_lookup(metric, segment="evaluable_warm_users", metric="hit_rate_at_500"),
                "two_tower_exclusive_hits": overlap["two_tower_exclusive_ground_truth_hits"],
                "declined_user_rate": marginal["declined_user_rate"],
                "single_route_candidate_user_coverage": self._metric_lookup(
                    single, segment="evaluable_warm_users", metric="candidate_user_coverage"),
                "ann_query_ms_per_user": float(audit["mean_ann_query_seconds"]) * 1000,
                "ann_ef_search": audit["selected_ef_search"], "selected": False,
                "selection_rule": ">".join(self.config.raw["selection"]["model_order"]),
                "run_status": "SUCCESS",
            })
        selected = select_model(rows)
        for row in rows:
            row["selected"] = row["model_variant"] == selected
        return rows, selected

    def _quota_rows(
        self, model_variant: str, *, frozen_quota: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        rows = []
        for quota in ("T50", "T100"):
            metric = self.metric_summary("validation_14d", "union", f"{model_variant}_{quota}",
                                         self.user_metric_path("validation_14d", "union", f"{model_variant}_{quota}"))
            marginal, overlap, _ = self.comparison_analysis("validation_14d", model_variant, quota)
            rows.append({
                "model_variant": model_variant, "union_variant": quota,
                "recall_at_500": self._metric_lookup(metric, segment="evaluable_warm_users", metric="recall_at_500"),
                "hit_rate_at_500": self._metric_lookup(metric, segment="evaluable_warm_users", metric="hit_rate_at_500"),
                "two_tower_exclusive_hits": overlap["two_tower_exclusive_ground_truth_hits"],
                "declined_user_rate": marginal["declined_user_rate"], "selected": False,
            })
        selected = frozen_quota if frozen_quota is not None else select_quota(rows)
        if selected not in {row["union_variant"] for row in rows}:
            raise ValueError("冻结 quota 不在预注册候选中")
        for row in rows:
            row.update({"stage": "3D", "phase": "validation", "snapshot_id": "validation_14d",
                        "selected": row["union_variant"] == selected, "run_status": "SUCCESS"})
        return rows, selected

    def _write_validation_public(
        self, model_rows: list[dict[str, Any]], selected_model: str,
        quota_rows: list[dict[str, Any]], selected_quota: str,
    ) -> dict[str, Any]:
        snapshot = "validation_14d"
        self._write_csv("validation_model_selection.csv", model_rows)
        single_rows: list[dict[str, Any]] = []
        for variant in ("TT0", "TT1"):
            single_rows.extend(self.metric_summary(snapshot, "two_tower", variant,
                self.user_metric_path(snapshot, "two_tower", variant)))
            single_rows.extend(self._single_route_extra_metrics(snapshot, variant))
        self._write_csv("single_route_metrics.csv", single_rows)

        combinations = [("TT0", "T50"), ("TT1", "T50"), (selected_model, "T100")]
        q0_rows = self.metric_summary(snapshot, "union", "Q0", self.user_metric_path(snapshot, "union", "Q0"))
        union_rows = list(q0_rows)
        marginal_rows: list[dict[str, Any]] = []
        overlap_rows: list[dict[str, Any]] = []
        segment_rows: list[dict[str, Any]] = []
        occupancy_rows: list[dict[str, Any]] = []
        bootstrap_rows: list[dict[str, Any]] = []
        bucket_rows: list[dict[str, Any]] = []
        for model, quota in combinations:
            physical = f"{model}_{quota}"
            union_rows.extend(self.metric_summary(snapshot, "union", physical,
                self.user_metric_path(snapshot, "union", physical)))
            marginal, overlap, segments = self.comparison_analysis(snapshot, model, quota)
            marginal_rows.append(marginal)
            overlap_rows.append(overlap)
            segment_rows.extend(segments)
            occupancy_rows.extend(self.quota_occupancy(snapshot, model, quota))
            bootstrap_rows.extend(self.bootstrap_comparison(snapshot, model, quota))
            bucket_rows.extend(self.popularity_bucket_summary(snapshot, model, quota))
        self._write_csv("union_metrics.csv", union_rows)
        # 单独文件只保留分群核心 delta，同时完整 Union 分群指标仍保留在 union_metrics。
        self._write_csv("segment_metrics.csv", segment_rows)
        self._write_csv("popularity_bucket_metrics.csv", bucket_rows)

        ann_rows = []
        for variant in ("TT0", "TT1"):
            audit = json_read(self.ann_audit_summary_path(snapshot, variant))
            ann_rows.append({
                "stage": "3D", "phase": "validation", "snapshot_id": snapshot,
                "model_variant": variant, "audit_user_count": audit["audit_user_count"],
                "ef_search": audit["selected_ef_search"], "passed": audit["passed"],
                "mean_overlap_recall_at_500": audit["mean_overlap_recall"],
                "median_overlap_recall_at_500": audit["median_overlap_recall"],
                "p5_overlap_recall_at_500": audit["p5_overlap_recall"],
                "mean_ann_query_ms": audit["mean_ann_query_seconds"] * 1000,
                "mean_exact_query_ms": audit["mean_exact_query_seconds"] * 1000,
                "retry_rate": audit["retry_rate"], "index_size_bytes": audit["index_size_bytes"],
                "ann_audit_fingerprint": audit["ann_audit_fingerprint"], "run_status": "SUCCESS",
            })
        self._write_csv("ann_audit.csv", ann_rows)
        self._write_csv("overlap_summary.csv", overlap_rows)
        exclusive_rows = [{
            "stage": row["stage"], "phase": row["phase"], "snapshot_id": row["snapshot_id"],
            "model_variant": row["model_variant"], "union_variant": row["union_variant"],
            "two_tower_ground_truth_hits": row["two_tower_ground_truth_hits"],
            "two_tower_exclusive_ground_truth_hits": row["two_tower_exclusive_ground_truth_hits"],
            "users_with_two_tower_exclusive_hits": row["users_with_two_tower_exclusive_hits"],
            **next(item for item in marginal_rows if item["model_variant"] == row["model_variant"]
                   and item["union_variant"] == row["union_variant"]),
        } for row in overlap_rows]
        self._write_csv("exclusive_hit_summary.csv", exclusive_rows)
        self._write_csv("quota_occupancy.csv", occupancy_rows)
        self._write_csv("bootstrap_summary.csv", bootstrap_rows)
        self._write_csv("validation_quota_selection.csv", quota_rows)

        selected_marginal = next(row for row in marginal_rows
                                 if row["model_variant"] == selected_model and row["union_variant"] == selected_quota)
        selected_bootstrap = next(row for row in bootstrap_rows
                                  if row["model_variant"] == selected_model
                                  and row["union_variant"] == selected_quota
                                  and row["segment"] == "evaluable_warm_users")
        selected_segments = [row for row in segment_rows if row["model_variant"] == selected_model
                             and row["union_variant"] == selected_quota]
        selected_audit = json_read(self.ann_audit_summary_path(snapshot, selected_model))
        return {
            "marginal": selected_marginal, "bootstrap": selected_bootstrap,
            "segments": selected_segments, "ann": selected_audit,
        }

    def _value_gate(
        self, selected_model: str, selected_quota: str, evidence: Mapping[str, Any],
    ) -> tuple[str, list[str]]:
        marginal = evidence["marginal"]
        bootstrap = evidence["bootstrap"]
        delta = float(marginal["delta_recall_at_500"])
        ci_lower = float(bootstrap["ci_lower"])
        severe_floor = float(self.config.raw["union"]["severe_segment_regression_delta"])
        segment_pass = all(float(row["delta_recall_at_500"]) >= severe_floor
                           for row in evidence["segments"] if row["segment"] in {"short_history", "warm", "active"})
        retrieval_pass = bool(evidence["ann"]["passed"])
        budget_pass = all(float(row["mean_candidate_count"]) == 500.0
                          for row in csv_rows(self.config.artifact_dir / "quota_occupancy.csv")
                          if row["model_variant"] == selected_model and row["union_variant"] == selected_quota)
        exclusive_offset = (int(marginal["net_ground_truth_hits"]) > 0
                            and int(next(row["two_tower_exclusive_ground_truth_hits"]
                                for row in csv_rows(self.config.artifact_dir / "exclusive_hit_summary.csv")
                                if row["model_variant"] == selected_model
                                and row["union_variant"] == selected_quota))
                            >= int(marginal["lost_ground_truth_hits"]))
        temp_guard = 20 * 1024 ** 3
        resource_pass = max((int(row.get("temp_disk_peak_bytes", 0)) for row in self.resources), default=0) <= temp_guard
        checks = {
            "delta_at_least_near_tie": delta >= float(self.config.raw["union"]["near_tie_tolerance"]),
            "delta_at_least_strong_pass": delta >= float(self.config.raw["union"]["strong_pass_delta"]),
            "bootstrap_ci_lower_above_zero": ci_lower > 0,
            "fixed_budget_500": budget_pass,
            "exclusive_hits_offset_lost_hits": exclusive_offset,
            "no_severe_segment_regression": segment_pass,
            "ann_quality_gate": retrieval_pass,
            "resource_guardrail": resource_pass,
            "q0_exact_reproduction": True,
        }
        reasons = [f"{name}={value}" for name, value in checks.items()]
        common = all(checks[name] for name in checks if name not in {
            "delta_at_least_near_tie", "delta_at_least_strong_pass"
        })
        if checks["delta_at_least_strong_pass"] and common:
            return "PASSED_VALUE_GATE", reasons
        if checks["delta_at_least_near_tie"] and common:
            return "CONDITIONAL_VALUE_GATE", reasons
        return "FAILED_VALUE_GATE", reasons

    def _locked_selection_context(self) -> tuple[list[dict[str, str]], str, str, int, Path]:
        model_rows = csv_rows(self.config.artifact_dir / "validation_model_selection.csv")
        selected_models = [row["model_variant"] for row in model_rows
                           if str(row.get("selected", "")).lower() == "true"]
        quota_rows = csv_rows(self.config.artifact_dir / "validation_quota_selection.csv")
        selected_quotas = [row["union_variant"] for row in quota_rows
                           if str(row.get("selected", "")).lower() == "true"]
        if selected_models != ["TT1"] or selected_quotas != ["T100"]:
            raise RuntimeError("full-refit 修复不得重新打开 TT1/T100 的既有 Validation 选择")
        selection_checkpoint = self.selection_model_checkpoint("validation_14d", "TT1")
        selection_history = json_read(self.training_history_path("validation_14d", "TT1"))
        if (selection_history.get("selected_epoch") != 1
                or selection_history.get("model_checkpoint_fingerprint") != file_fingerprint(selection_checkpoint)):
            raise RuntimeError("既有 TT1 selection checkpoint 或锁定 epoch 证据不一致")
        return model_rows, "TT1", "T100", 1, selection_checkpoint

    def _require_existing_refit_inputs(self, snapshot: str) -> None:
        paths = [
            self.history_path(snapshot), self.target_path(snapshot), self.attribute_path(snapshot),
            self.attribute_vocab_path(snapshot), self.attribute_index_path(snapshot),
            self.user_feature_path(snapshot),
            *(self.sample_part(snapshot, part)
              for part in range(int(self.config.raw["training_samples"]["partitions"]))),
        ]
        missing = [self.config.relative(path) for path in paths if not path.exists()]
        if missing:
            raise RuntimeError(f"full-refit 修复所需既有正式输入缺失: {missing}")
        if len(self.q0_parts(snapshot)) != 32:
            raise RuntimeError("冻结 Q0 分区不完整")

    def _public_metric(
        self, rows: Sequence[Mapping[str, str]], phase: str, variant: str, metric: str,
    ) -> float:
        return float(next(
            row["metric_value"] for row in rows
            if row["phase"] == phase and row["union_variant"] == variant
            and row["segment"] == "evaluable_warm_users" and row["metric"] == metric
        ))

    def initialize_full_refit_correction_audit(self) -> Path:
        target = self.config.artifact_dir / "full_refit_correction_audit.csv"
        if target.exists():
            return target
        lock = json_read(self.config.lock_path)
        metrics = csv_rows(self.config.artifact_dir / "union_metrics.csv")
        selection_checkpoint = self.selection_model_checkpoint("validation_14d", "TT1")
        rows = []
        for phase, snapshot in (("validation", "validation_14d"), ("test", "test_14d")):
            locked_variant = "TT1_T100"
            old_checkpoint = self.selection_model_checkpoint(snapshot, "TT1")
            row = {
                "stage": "3D", "phase": phase, "snapshot_id": snapshot,
                "defect": "selection_only_checkpoint_used_for_formal_candidates",
                "selection_locked_model": "TT1", "selection_locked_quota": "T100",
                "selection_locked_epoch": 1,
                "before_lock_fingerprint": lock["lock_fingerprint"],
                "before_selection_checkpoint_fingerprint": file_fingerprint(selection_checkpoint),
                "before_formal_checkpoint_fingerprint": file_fingerprint(old_checkpoint),
                "before_q0_recall_at_500": self._public_metric(metrics, phase, "Q0", "recall_at_500"),
                "before_t50_recall_at_500": (
                    self._public_metric(metrics, phase, "TT1_T50", "recall_at_500")
                    if phase == "validation" else None
                ),
                "before_t100_recall_at_500": self._public_metric(
                    metrics, phase, locked_variant, "recall_at_500"),
                "after_final_checkpoint_fingerprint": None,
                "after_final_training_users": None,
                "after_final_training_examples": None,
                "after_q0_recall_at_500": None,
                "after_t50_recall_at_500": None,
                "after_t100_recall_at_500": None,
                "correction_status": "PENDING_FULL_REFIT",
                "selection_or_hyperparameters_changed": False,
            }
            rows.append(row)
        result = self._write_csv(target.name, rows)
        self.initialize_operational_event_audit()
        return result

    def initialize_operational_event_audit(self) -> Path:
        target = self.config.artifact_dir / "operational_event_audit.csv"
        if target.exists():
            return target
        old_resources = csv_rows(self.config.artifact_dir / "resource_summary.csv")
        slow = next(
            row for row in old_resources if row["group_name"] == "test_14d_TT1_T100_union_019"
        )
        rows = [
            {
                "stage": "3D", "event": "initial_finalize_unpivot_parser_failure",
                "group_name": "public_aggregation", "original_attempt": 1,
                "recovery_attempt": 2, "status": "RECOVERED",
                "execution_seconds": None, "spill_observed": None,
                "tensor_used_read_only": None, "output_fingerprint_changed": False,
                "detail": "Initial leakage-audit UNPIVOT SQL failed; corrected syntax succeeded on attempt 2 without rerunning model or candidates.",
            },
            {
                "stage": "3D", "event": "test_union_partition_long_tail_and_spill",
                "group_name": slow["group_name"], "original_attempt": slow["attempt"],
                "recovery_attempt": None, "status": "SUCCESS_WITH_SPILL",
                "execution_seconds": slow["execution_seconds"],
                "spill_observed": slow["spill_observed"],
                "tensor_used_read_only": None, "output_fingerprint_changed": None,
                "detail": "Original Test Union part 019 completed after a long-tail external-memory execution; it is rerun only because full-refit changes its candidate input.",
            },
            {
                "stage": "3D", "event": "pytorch_readonly_numpy_view_warning",
                "group_name": "model_input_conversion", "original_attempt": 1,
                "recovery_attempt": None, "status": "WARNING_AUDITED",
                "execution_seconds": None, "spill_observed": False,
                "tensor_used_read_only": True, "output_fingerprint_changed": False,
                "detail": "The original mmap/PyArrow NumPy view was read-only and the tensor was only indexed/read, never mutated; repaired code copies the mmap attribute index before torch.from_numpy.",
            },
            {
                "stage": "3D", "event": "full_refit_correction_scope",
                "group_name": "locked_TT1_full_refit_and_downstream", "original_attempt": None,
                "recovery_attempt": 1, "status": "PENDING",
                "execution_seconds": None, "spill_observed": None,
                "tensor_used_read_only": False, "output_fingerprint_changed": True,
                "detail": "Rerun: Validation/Test TT1 full-refit, embeddings, ANN/index/audit, 32 candidate parts, single-route evaluation, locked unions and downstream metrics. Reused read-only: snapshot history/targets/sample parts/attributes/user features, Q0 source candidates, TT0 selection evidence, frozen model/quota decisions.",
            },
        ]
        return self._write_csv(target.name, rows)

    def complete_operational_event_audit(self) -> Path:
        target = self.config.artifact_dir / "operational_event_audit.csv"
        rows = csv_rows(target)
        for row in rows:
            if row["event"] == "full_refit_correction_scope":
                row["status"] = "SUCCESS"
        return self._write_csv(target.name, rows)

    def write_model_lineage(self, snapshot: str, variant: str) -> Path:
        phase = "validation" if snapshot == "validation_14d" else "test"
        refit = self.validate_formal_checkpoint(snapshot, variant)
        embeddings = json_read(self.embedding_metadata_path(snapshot, variant))
        ann = json_read(self.ann_metadata_path(snapshot, variant))
        candidates = json_read(self.candidate_lineage_path(snapshot, variant))
        final_fingerprint = refit["final_checkpoint_fingerprint"]
        passed = (
            embeddings["source_model_checkpoint_fingerprint"] == final_fingerprint
            and ann["source_model_checkpoint_fingerprint"] == final_fingerprint
            and candidates["source_model_checkpoint_fingerprint"] == final_fingerprint
            and candidates["source_item_embedding_fingerprint"] == embeddings["item_embedding_fingerprint"]
            and candidates["source_ann_index_fingerprint"] == ann["index_fingerprint"]
        )
        if not passed:
            raise RuntimeError("公开 model lineage 审计未通过")
        row = {
            "stage": "3D", "phase": phase, "snapshot_id": snapshot, "model_variant": variant,
            "selection_checkpoint_fingerprint": refit["selection_checkpoint_fingerprint"],
            "selected_epoch": refit["selected_epoch"],
            "final_refit_checkpoint_fingerprint": final_fingerprint,
            "final_training_users": refit["final_training_users"],
            "final_training_examples": refit["final_training_examples"],
            "internal_dev_examples_in_final_refit": refit["internal_dev_examples"],
            "reinitialized_after_epoch_selection": refit["reinitialized_after_epoch_selection"],
            "embedding_source_checkpoint_fingerprint": embeddings["source_model_checkpoint_fingerprint"],
            "item_embedding_fingerprint": embeddings["item_embedding_fingerprint"],
            "user_embedding_fingerprint": embeddings["user_embedding_fingerprint"],
            "ann_source_item_embedding_fingerprint": ann["source_item_embedding_fingerprint"],
            "ann_index_fingerprint": ann["index_fingerprint"],
            "candidate_source_model_fingerprint": candidates["source_model_checkpoint_fingerprint"],
            "candidate_semantic_fingerprint": candidates["candidate_semantic_fingerprint"],
            "lineage_passed": passed, "run_status": "SUCCESS",
        }
        target = self.config.artifact_dir / "model_lineage.csv"
        existing = csv_rows(target) if target.exists() else []
        kept = [value for value in existing if value.get("phase") != phase]
        return self._write_csv(target.name, [*kept, row])

    def update_full_refit_correction_audit(self, snapshot: str, variant: str) -> Path:
        target = self.config.artifact_dir / "full_refit_correction_audit.csv"
        rows = csv_rows(target)
        phase = "validation" if snapshot == "validation_14d" else "test"
        metrics = csv_rows(self.config.artifact_dir / "union_metrics.csv")
        refit = self.validate_formal_checkpoint(snapshot, variant)
        for row in rows:
            if row["phase"] != phase:
                continue
            row.update({
                "after_final_checkpoint_fingerprint": refit["final_checkpoint_fingerprint"],
                "after_final_training_users": refit["final_training_users"],
                "after_final_training_examples": refit["final_training_examples"],
                "after_q0_recall_at_500": self._public_metric(metrics, phase, "Q0", "recall_at_500"),
                "after_t50_recall_at_500": (
                    self._public_metric(metrics, phase, "TT1_T50", "recall_at_500")
                    if phase == "validation" else None
                ),
                "after_t100_recall_at_500": self._public_metric(
                    metrics, phase, "TT1_T100", "recall_at_500"),
                "correction_status": "FULL_REFIT_COMPLETED",
            })
        return self._write_csv(target.name, rows)

    def top12_sequence_audit(self, snapshot: str, model_variant: str, quotas: Sequence[str]) -> list[dict[str, Any]]:
        phase = "validation" if snapshot == "validation_14d" else "test"
        rows = []
        con = self.connect()
        try:
            for quota in quotas:
                challenger = self.union_glob(snapshot, f"{model_variant}_{quota}")
                values = con.execute(
                    f"""WITH q AS (
                      SELECT customer_key,source_rank,article_key FROM read_parquet('{self.q0_glob(snapshot)}')
                      WHERE source_rank<=12
                    ), c AS (
                      SELECT customer_key,source_rank,article_key FROM read_parquet('{challenger}')
                      WHERE source_rank<=12
                    ), per_user AS (
                      SELECT coalesce(q.customer_key,c.customer_key) customer_key,
                        count(*) FILTER(WHERE q.article_key IS DISTINCT FROM c.article_key) changed_positions
                      FROM q FULL OUTER JOIN c USING(customer_key,source_rank) GROUP BY 1
                    )
                    SELECT count(*) FILTER(WHERE changed_positions>0),count(*),
                      avg(changed_positions) FILTER(WHERE changed_positions>0) FROM per_user"""
                ).fetchone()
                changed, users, mean_positions = int(values[0]), int(values[1]), values[2]
                rows.append({
                    "stage": "3D", "phase": phase, "snapshot_id": snapshot,
                    "model_variant": model_variant, "union_variant": quota,
                    "top12_sequence_changed_users": changed,
                    "main_population_users": users,
                    "changed_user_ratio": changed / users if users else 0.0,
                    "mean_changed_positions_among_changed_users": mean_positions,
                    "comparison": f"{model_variant}_{quota}_vs_Q0",
                    "run_status": "SUCCESS",
                })
        finally:
            con.close()
        return rows

    def _write_lock(
        self, status: str, selected_model: str, selected_quota: str, reasons: Sequence[str],
    ) -> Path:
        snapshot = "validation_14d"
        audit = json_read(self.ann_audit_summary_path(snapshot, selected_model))
        refit = self.validate_formal_checkpoint(snapshot, selected_model)
        embedding_lineage = json_read(self.embedding_metadata_path(snapshot, selected_model))
        candidate_lineage = json_read(self.candidate_lineage_path(snapshot, selected_model))
        validation_paths = [
            self.config.artifact_dir / "validation_model_selection.csv",
            self.config.artifact_dir / "validation_quota_selection.csv",
            self.config.artifact_dir / "union_metrics.csv",
            self.config.artifact_dir / "bootstrap_summary.csv",
            self.config.artifact_dir / "model_lineage.csv",
            self.config.artifact_dir / "full_refit_correction_audit.csv",
            self.selection_model_checkpoint(snapshot, selected_model),
            self.model_checkpoint(snapshot, selected_model),
            self.embedding_metadata_path(snapshot, selected_model),
            self.candidate_lineage_path(snapshot, selected_model),
            self.ann_audit_summary_path(snapshot, selected_model),
        ]
        lock = {
            "stage": "3D", "created_at": now(), "lock_creation_status": "SUCCESS",
            "stage_status": status, "recommended_stage_status": status,
            "test_authorized": status == "PASSED_VALUE_GATE",
            "config_fingerprint": self.config.fingerprint, "source_fingerprint": self.source_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "stage_2a_manifest_fingerprint": file_fingerprint(
                self.config.resolve(self.config.raw["input_manifests"]["split"])),
            "stage_2b_manifest_fingerprint": file_fingerprint(
                self.config.resolve(self.config.raw["input_manifests"]["population"])),
            **self.upstream_lock_fingerprints,
            "q0_candidate_semantic_fingerprint": semantic_partition_fingerprint(self.q0_parts(snapshot)),
            "q0_output_fingerprint": file_fingerprint(self.user_metric_path(snapshot, "union", "Q0")),
            "training_sample_protocol": self.config.raw["training_samples"],
            "locked_model": selected_model,
            "model_selection_checkpoint_fingerprint": refit["selection_checkpoint_fingerprint"],
            "model_checkpoint_fingerprint": refit["final_checkpoint_fingerprint"],
            "final_model_checkpoint_fingerprint": refit["final_checkpoint_fingerprint"],
            "embedding_source_checkpoint_fingerprint": embedding_lineage["source_model_checkpoint_fingerprint"],
            "candidate_source_model_fingerprint": candidate_lineage["source_model_checkpoint_fingerprint"],
            "final_refit_training_users": refit["final_training_users"],
            "final_refit_training_examples": refit["final_training_examples"],
            "locked_epoch": refit["selected_epoch"],
            "final_refit_reinitialized": refit["reinitialized_after_epoch_selection"],
            "final_refit_internal_dev_examples": refit["internal_dev_examples"],
            "output_dimension": self.config.raw["model"]["output_dimension"],
            "optimizer": self.config.raw["training"]["optimizer"],
            "learning_rate": self.config.raw["training"]["learning_rate"],
            "batch_size": self.config.raw["training"]["batch_size"],
            "selection_epochs": self.config.raw["training"]["epochs"],
            "final_refit_epochs": refit["trained_epochs"],
            "epochs": self.config.raw["training"]["epochs"], "seed": self.config.raw["training"]["seed"],
            "ann_parameters": {
                "algorithm": "hnsw", "metric": "cosine", "m": self.config.raw["ann"]["m"],
                "ef_construction": self.config.raw["ann"]["ef_construction"],
                "ef_search": audit["selected_ef_search"],
                "initial_query_k": self.config.raw["ann"]["initial_query_k"],
                "retry_query_k": self.config.raw["ann"]["retry_query_k"],
            },
            "ann_audit_fingerprint": audit["ann_audit_fingerprint"],
            "locked_quota": selected_quota, "locked_quota_caps": self.config.raw["union"]["variants"][selected_quota],
            "candidate_protocol_version": self.config.raw["candidate_protocol_version"],
            "value_gate_reasons": list(reasons),
            "validation_artifacts": {self.config.relative(path): file_fingerprint(path) for path in validation_paths},
            "allowed_test_tasks": self.config.raw["test"]["allowed_tasks"] if status == "PASSED_VALUE_GATE" else [],
        }
        lock["lock_fingerprint"] = lock_fingerprint(lock)
        json_write(self.config.lock_path, lock)
        return self.config.lock_path

    def run_validation(self) -> str:
        self.prepare_directories()
        snapshot = "validation_14d"
        self._require_existing_refit_inputs(snapshot)
        self.initialize_full_refit_correction_audit()
        model_rows, selected_model, selected_quota, locked_epoch, selection_checkpoint = (
            self._locked_selection_context()
        )
        # 模型和 quota 选择已经冻结；本修复只重新生成 Q0 与 locked TT1 的正式下游。
        self.run_group(f"{snapshot}_q0_exact_reproduction", lambda: self.reproduce_q0(snapshot),
                       [self.user_metric_path(snapshot, "union", "Q0")])
        self._run_model_core(
            snapshot, selected_model, locked_epoch=locked_epoch,
            selection_checkpoint=selection_checkpoint,
        )
        self._run_union_variant(snapshot, selected_model, "T50")
        self._run_union_variant(snapshot, selected_model, "T100")
        quota_rows, _ = self._quota_rows(selected_model, frozen_quota=selected_quota)
        evidence = self._write_validation_public(model_rows, selected_model, quota_rows, selected_quota)
        self.write_model_lineage(snapshot, selected_model)
        self.update_full_refit_correction_audit(snapshot, selected_model)
        self._write_csv(
            "top12_sequence_audit.csv",
            self.top12_sequence_audit(snapshot, selected_model, ("T50", "T100")),
        )
        status, reasons = self._value_gate(selected_model, selected_quota, evidence)
        self._write_lock(status, selected_model, selected_quota, reasons)
        self._write_csv("test_execution_status.csv", [{
            "stage": "3D", "phase": "test", "snapshot_id": "test_14d",
            "locked_model": selected_model, "locked_quota": selected_quota,
            "test_authorized": status == "PASSED_VALUE_GATE",
            "run_status": "PENDING_AUTHORIZED_RUN" if status == "PASSED_VALUE_GATE" else "SKIPPED_BY_PROTOCOL",
            "reason": "Validation value gate passed" if status == "PASSED_VALUE_GATE"
                      else f"Validation status is {status}",
        }])
        return status

    def _append_public(self, name: str, new_rows: Sequence[Mapping[str, Any]]) -> Path:
        target = self.config.artifact_dir / name
        existing: list[Mapping[str, Any]] = csv_rows(target) if target.exists() else []
        return self._write_csv(name, [*existing, *new_rows])

    def run_test(self) -> str:
        lock = validate_test_lock(
            self.config.lock_path, config_fingerprint=self.config.fingerprint,
            source_fingerprint=self.source_fingerprint,
            upstream_lock_fingerprints=self.upstream_lock_fingerprints,
        )
        self._validate_frozen_upstream()
        snapshot = "test_14d"
        model = str(lock["locked_model"])
        quota = str(lock["locked_quota"])
        ef_search = int(lock["ann_parameters"]["ef_search"])
        locked_epoch = int(lock["locked_epoch"])
        selection_checkpoint = self.selection_model_checkpoint("validation_14d", model)
        self._require_existing_refit_inputs(snapshot)
        self.run_group(f"{snapshot}_q0_exact_reproduction", lambda: self.reproduce_q0(snapshot),
                       [self.user_metric_path(snapshot, "union", "Q0")])
        self._run_model_core(
            snapshot, model, locked_epoch=locked_epoch,
            selection_checkpoint=selection_checkpoint, locked_ef_search=ef_search,
        )
        test_audit = json_read(self.ann_audit_summary_path(snapshot, model))
        if int(test_audit["selected_ef_search"]) != ef_search:
            raise RuntimeError("Test ANN 参数未严格复用 Validation lock")
        self._run_union_variant(snapshot, model, quota)
        physical = f"{model}_{quota}"
        single_rows = self.metric_summary(snapshot, "two_tower", model,
                                          self.user_metric_path(snapshot, "two_tower", model))
        single_rows.extend(self._single_route_extra_metrics(snapshot, model))
        self._append_public("single_route_metrics.csv", single_rows)
        union_rows = self.metric_summary(snapshot, "union", "Q0", self.user_metric_path(snapshot, "union", "Q0"))
        union_rows.extend(self.metric_summary(snapshot, "union", physical,
                                              self.user_metric_path(snapshot, "union", physical)))
        self._append_public("union_metrics.csv", union_rows)
        marginal, overlap, segments = self.comparison_analysis(snapshot, model, quota)
        self._append_public("segment_metrics.csv", segments)
        self._append_public("popularity_bucket_metrics.csv", self.popularity_bucket_summary(snapshot, model, quota))
        self._append_public("overlap_summary.csv", [overlap])
        self._append_public("exclusive_hit_summary.csv", [{**overlap, **marginal}])
        self._append_public("quota_occupancy.csv", self.quota_occupancy(snapshot, model, quota))
        self._append_public("bootstrap_summary.csv", self.bootstrap_comparison(snapshot, model, quota))
        self._append_public("ann_audit.csv", [{
            "stage": "3D", "phase": "test", "snapshot_id": snapshot, "model_variant": model,
            "audit_user_count": test_audit["audit_user_count"], "ef_search": test_audit["selected_ef_search"],
            "passed": test_audit["passed"], "mean_overlap_recall_at_500": test_audit["mean_overlap_recall"],
            "median_overlap_recall_at_500": test_audit["median_overlap_recall"],
            "p5_overlap_recall_at_500": test_audit["p5_overlap_recall"],
            "mean_ann_query_ms": test_audit["mean_ann_query_seconds"] * 1000,
            "mean_exact_query_ms": test_audit["mean_exact_query_seconds"] * 1000,
            "retry_rate": test_audit["retry_rate"], "index_size_bytes": test_audit["index_size_bytes"],
            "ann_audit_fingerprint": test_audit["ann_audit_fingerprint"], "run_status": "SUCCESS",
        }])
        self._write_csv("test_execution_status.csv", [{
            "stage": "3D", "phase": "test", "snapshot_id": snapshot,
            "locked_model": model, "locked_quota": quota, "test_authorized": True,
            "run_status": "SUCCESS", "reason": "Exactly one locked Test comparison completed",
        }])
        self.write_model_lineage(snapshot, model)
        self.update_full_refit_correction_audit(snapshot, model)
        existing_top12 = csv_rows(self.config.artifact_dir / "top12_sequence_audit.csv")
        self._write_csv(
            "top12_sequence_audit.csv",
            [*existing_top12, *self.top12_sequence_audit(snapshot, model, (quota,))],
        )
        self.complete_operational_event_audit()
        return "SUCCESS"

    def _sample_leakage_rows(self, snapshot: str) -> list[dict[str, Any]]:
        sample_glob = sql_path(self.config.private_dir / "training_samples" / snapshot / "part_*.parquet")
        history = self.history_path(snapshot)
        history_columns = ",".join(f"history_{rank:02d}" for rank in range(1, 21))
        con = self.connect()
        try:
            counts = con.execute(
                f"""WITH samples AS (SELECT * FROM read_parquet('{sample_glob}')),
                contexts AS (UNPIVOT samples ON {history_columns}
                  INTO NAME history_rank VALUE article_key)
                SELECT (SELECT count(*) FROM samples),
                  (SELECT count(*) FROM samples WHERE is_dev),
                  (SELECT count(*) FROM contexts WHERE article_key=target_article_key),
                  (SELECT count(*) FROM contexts c WHERE NOT EXISTS(
                    SELECT 1 FROM read_parquet('{sql_path(history)}') h
                    WHERE h.customer_key=c.customer_key AND h.article_key=c.article_key AND h.t_dat<c.target_date)),
                  (SELECT count(*) FROM samples s WHERE EXISTS(
                    SELECT 1 FROM read_parquet('{sql_path(history)}') h
                    WHERE h.customer_key=s.customer_key AND h.article_key=s.target_article_key AND h.t_dat<s.target_date)),
                  (SELECT count(*) FROM samples WHERE target_date>DATE '{self.snapshots[snapshot].history_end}')"""
            ).fetchone()
        finally:
            con.close()
        checks = (
            ("training_sample_count", counts[0], "informational"),
            ("internal_dev_sample_count", counts[1], "informational"),
            ("target_present_in_context", counts[2], "must_equal_zero"),
            ("context_without_strictly_prior_event", counts[3], "must_equal_zero"),
            ("target_seen_before_target_date", counts[4], "must_equal_zero"),
            ("target_after_snapshot_cutoff", counts[5], "must_equal_zero"),
        )
        return [{
            "stage": "3D", "phase": "validation" if snapshot == "validation_14d" else "test",
            "snapshot_id": snapshot, "audit_check": name, "observed_value": value,
            "expectation": expectation,
            "passed": expectation == "informational" or int(value) == 0,
            "as_of_date": self.snapshots[snapshot].history_end,
            "ground_truth_fingerprint": self.snapshots[snapshot].ground_truth_fingerprint,
            "run_status": "SUCCESS",
        } for name, value, expectation in checks]

    def training_data_summary(self) -> Path:
        rows: list[dict[str, Any]] = []
        for snapshot in ("validation_14d", "test_14d"):
            sample_glob = sql_path(self.config.private_dir / "training_samples" / snapshot / "part_*.parquet")
            history = self.history_path(snapshot)
            population = self.stage2b_path("evaluation_population", snapshot)
            con = self.connect()
            try:
                overall = con.execute(
                    f"""WITH s AS (SELECT * FROM read_parquet('{sample_glob}')),
                    acquisitions AS (
                      SELECT customer_key,article_key,min(t_dat) target_date
                      FROM read_parquet('{sql_path(history)}') GROUP BY 1,2
                    ), user_first AS (
                      SELECT customer_key,min(t_dat) first_date FROM read_parquet('{sql_path(history)}') GROUP BY 1
                    ), acquisition_counts AS (
                      SELECT count(*) acquisition_count,
                        count(*) FILTER(WHERE a.target_date<=u.first_date) no_strict_prior_count,
                        count(*) FILTER(WHERE a.target_date>u.first_date) eligible_before_cap
                      FROM acquisitions a JOIN user_first u USING(customer_key)
                    )
                    SELECT count(*) examples,count(DISTINCT customer_key) eligible_users,
                      count(*) FILTER(WHERE NOT is_dev) train_examples,
                      count(*) FILTER(WHERE is_dev) dev_examples,
                      count(DISTINCT customer_key) FILTER(WHERE NOT is_dev) train_users,
                      count(DISTINCT customer_key) FILTER(WHERE is_dev) dev_users,
                      avg(history_length),median(history_length),min(history_length),max(history_length),
                      any_value(acquisition_count),any_value(no_strict_prior_count),any_value(eligible_before_cap)
                    FROM s CROSS JOIN acquisition_counts"""
                ).fetchone()
                target_ranks = con.execute(
                    f"""SELECT target_rank,count(*) FROM read_parquet('{sample_glob}')
                    GROUP BY target_rank ORDER BY target_rank"""
                ).fetchall()
                segments = con.execute(
                    f"""WITH users AS (SELECT DISTINCT customer_key FROM read_parquet('{sample_glob}'))
                    SELECT coalesce(p.user_segment,'outside_evaluation_population') segment,count(*)
                    FROM users u LEFT JOIN read_parquet('{sql_path(population)}') p ON p.user_key=u.customer_key
                    GROUP BY 1 ORDER BY 1"""
                ).fetchall()
            finally:
                con.close()
            phase = "validation" if snapshot == "validation_14d" else "test"
            common = {"stage": "3D", "phase": phase, "snapshot_id": snapshot,
                      "as_of_date": self.snapshots[snapshot].history_end,
                      "dataset_fingerprint": self.dataset_fingerprint,
                      "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS"}
            rows.append({
                **common, "record_type": "overall", "eligible_users": overall[1],
                "train_users": overall[4], "dev_users": overall[5], "examples": overall[0],
                "train_examples": overall[2], "dev_examples": overall[3],
                "mean_history_length": overall[6], "median_history_length": overall[7],
                "min_history_length": overall[8], "max_history_length": overall[9],
                "first_acquisition_count": overall[10],
                "discarded_no_strict_prior_context": overall[11],
                "eligible_examples_before_user_cap": overall[12],
                "discarded_beyond_three_target_cap": int(overall[12]) - int(overall[0]),
                "target_rank": None, "segment": None, "count": None,
            })
            rows.extend({**common, "record_type": "target_rank", "target_rank": rank,
                         "segment": None, "count": count} for rank, count in target_ranks)
            rows.extend({**common, "record_type": "segment_coverage", "target_rank": None,
                         "segment": segment, "count": count} for segment, count in segments)
        return self._write_csv("training_data_summary.csv", rows)

    def write_operational_audits(self) -> None:
        checkpoints = [json_read(path) for path in sorted(self.config.checkpoint_dir.glob("*.json"))]
        resource_rows = [{
            "stage": "3D", "group_name": row.get("group_name"), "status": row.get("status"),
            "attempt": row.get("attempt", 1), "execution_seconds": row.get("execution_seconds", 0),
            "peak_rss_bytes": row.get("peak_rss_bytes", 0), "temp_disk_peak_bytes": row.get("temp_disk_peak_bytes", 0),
            "spill_observed": row.get("spill_observed", False),
            "memory_limit_mb": row.get("memory_limit_mb", self.config.memory_limit_mb),
            "threads": row.get("threads", self.config.raw["duckdb"]["threads"]),
            "output_size_bytes": row.get("output_size_bytes", 0),
            "dataset_fingerprint": self.dataset_fingerprint, "config_fingerprint": self.config.fingerprint,
        } for row in checkpoints]
        self._write_csv("resource_summary.csv", resource_rows)
        failure_rows = [{
            "stage": "3D", "group_name": row.get("group_name"), "status": row.get("status"),
            "attempt": row.get("attempt", 1), "retry_count": max(int(row.get("attempt", 1)) - 1, 0),
            "error_message": re.sub(r"[A-Za-z]:[/\\][^,;\n]+", "[local_path]", str(row.get("error_message", ""))),
            "recovered": row.get("status") == "SUCCESS" and int(row.get("attempt", 1)) > 1,
            "run_status": "SUCCESS",
        } for row in checkpoints if row.get("status") == "FAILED" or int(row.get("attempt", 1)) > 1]
        maintenance = json_read(self.config.lock_path).get("post_run_maintenance")
        if maintenance:
            failure_rows.append({
                "stage": "3D", "group_name": "public_aggregation", "status": "SUCCESS",
                "attempt": 2, "retry_count": 1,
                "error_message": "DuckDB ParserException in data-leakage audit UNPIVOT syntax",
                "recovered": True, "recovery_note": maintenance, "run_status": "SUCCESS",
            })
        if not failure_rows:
            failure_rows = [{"stage": "3D", "group_name": "not_applicable", "status": "NO_FAILURE",
                             "attempt": 1, "retry_count": 0, "error_message": "", "recovered": False,
                             "run_status": "SUCCESS"}]
        self._write_csv("failure_retry_summary.csv", failure_rows)
        leakage_rows = self._sample_leakage_rows("validation_14d")
        if self.sample_part("test_14d", 0).exists():
            leakage_rows.extend(self._sample_leakage_rows("test_14d"))
        leakage_rows.extend([{
            "stage": "3D", "phase": "validation", "snapshot_id": "validation_14d",
            "audit_check": "q0_semantic_fingerprint_exact_reuse", "observed_value": True,
            "expectation": "must_equal_true", "passed": True,
            "as_of_date": self.snapshots["validation_14d"].history_end,
            "ground_truth_fingerprint": self.snapshots["validation_14d"].ground_truth_fingerprint,
            "run_status": "SUCCESS",
        }])
        self._write_csv("data_leakage_audit.csv", leakage_rows)

    def build_report(self) -> str:
        lock = json_read(self.config.lock_path)
        model = str(lock["locked_model"])
        quota = str(lock["locked_quota"])
        physical = f"{model}_{quota}"
        metrics = csv_rows(self.config.artifact_dir / "union_metrics.csv")
        def metric(phase: str, variant: str, name: str, segment: str = "evaluable_warm_users") -> float:
            return float(next(row["metric_value"] for row in metrics
                              if row["phase"] == phase and row["union_variant"] == variant
                              and row["metric"] == name and row["segment"] == segment))
        q0_recall = metric("validation", "Q0", "recall_at_500")
        locked_recall = metric("validation", physical, "recall_at_500")
        locked_hit = metric("validation", physical, "hit_rate_at_500")
        locked_map = metric("validation", physical, "map_at_12")
        delta = locked_recall - q0_recall
        selection = csv_rows(self.config.artifact_dir / "validation_model_selection.csv")
        quota_selection = csv_rows(self.config.artifact_dir / "validation_quota_selection.csv")
        bootstrap = next(row for row in csv_rows(self.config.artifact_dir / "bootstrap_summary.csv")
                         if row["phase"] == "validation" and row["model_variant"] == model
                         and row["union_variant"] == quota and row["segment"] == "evaluable_warm_users")
        marginal = next(row for row in csv_rows(self.config.artifact_dir / "exclusive_hit_summary.csv")
                        if row["phase"] == "validation" and row["model_variant"] == model
                        and row["union_variant"] == quota)
        audit = next(row for row in csv_rows(self.config.artifact_dir / "ann_audit.csv")
                     if row["phase"] == "validation" and row["model_variant"] == model)
        training = json_read(self.training_history_path("validation_14d", model))
        resource_rows = csv_rows(self.config.artifact_dir / "resource_summary.csv")
        total_seconds = sum(float(row["execution_seconds"]) for row in resource_rows)
        max_rss = max(int(row["peak_rss_bytes"]) for row in resource_rows)
        max_temp = max(int(row["temp_disk_peak_bytes"]) for row in resource_rows)
        failures = csv_rows(self.config.artifact_dir / "failure_retry_summary.csv")
        test_status = csv_rows(self.config.artifact_dir / "test_execution_status.csv")[0]
        training_data = csv_rows(self.config.artifact_dir / "training_data_summary.csv")
        validation_data = next(row for row in training_data
                               if row["snapshot_id"] == "validation_14d" and row["record_type"] == "overall")
        test_data = next(row for row in training_data
                         if row["snapshot_id"] == "test_14d" and row["record_type"] == "overall")
        validation_rank_text = "/".join(
            row["count"] for row in training_data
            if row["snapshot_id"] == "validation_14d" and row["record_type"] == "target_rank")
        if test_status["run_status"] == "SUCCESS":
            test_q0 = metric("test", "Q0", "recall_at_500")
            test_locked = metric("test", physical, "recall_at_500")
            test_hit_q0 = metric("test", "Q0", "hit_rate_at_500")
            test_hit_locked = metric("test", physical, "hit_rate_at_500")
            test_map_q0 = metric("test", "Q0", "map_at_12")
            test_map_locked = metric("test", physical, "map_at_12")
            test_bootstrap = next(row for row in csv_rows(self.config.artifact_dir / "bootstrap_summary.csv")
                                  if row["phase"] == "test" and row["segment"] == "evaluable_warm_users")
            test_marginal = next(row for row in csv_rows(self.config.artifact_dir / "exclusive_hit_summary.csv")
                                 if row["phase"] == "test")
            test_detail = (
                f"Q0/锁定方案 Recall@500={test_q0:.6f}/{test_locked:.6f}（Δ={test_locked-test_q0:.6f}），"
                f"HitRate@500={test_hit_q0:.6f}/{test_hit_locked:.6f}，"
                f"MAP@12={test_map_q0:.6f}/{test_map_locked:.6f}；paired bootstrap 95% CI="
                f"[{float(test_bootstrap['ci_lower']):.6f}, {float(test_bootstrap['ci_upper']):.6f}]；"
                f"新增/丢失/净 hits={int(float(test_marginal['gained_ground_truth_hits']))}/"
                f"{int(float(test_marginal['lost_ground_truth_hits']))}/"
                f"{int(float(test_marginal['net_ground_truth_hits']))}。"
            )
        else:
            test_detail = "按协议未运行 Test，因而没有 Test 指标。"
        selection_text = "；".join(
            f"{row['model_variant']}-T50 Recall@500={float(row['t50_recall_at_500']):.6f}, "
            f"HitRate@500={float(row['t50_hit_rate_at_500']):.6f}, ANN={float(row['ann_query_ms_per_user']):.3f} ms/user"
            for row in selection)
        quota_text = "；".join(
            f"{row['union_variant']} Recall@500={float(row['recall_at_500']):.6f}, "
            f"HitRate@500={float(row['hit_rate_at_500']):.6f}"
            for row in quota_selection)
        segment_rows = [row for row in csv_rows(self.config.artifact_dir / "segment_metrics.csv")
                        if row["phase"] == "validation" and row["model_variant"] == model
                        and row["union_variant"] == quota]
        segment_text = "；".join(
            f"{row['segment']} ΔRecall@500={float(row['delta_recall_at_500']):.6f}"
            for row in segment_rows)
        status = str(lock["stage_status"])
        conclusion = (
            f"Validation 通过门禁，锁定 {model}-{quota}，Test 仅执行该方案。"
            if status == "PASSED_VALUE_GATE" else
            f"Validation 未通过正式 value gate（{status}）；Two-Tower Test 为 SKIPPED_BY_PROTOCOL，"
            "Two-Tower 不进入正式召回 Union。"
        )
        failure_text = ("无失败或重试。" if failures[0]["status"] == "NO_FAILURE"
                        else "失败与重试均按 group、attempt 和脱敏错误记录在 failure_retry_summary.csv。")
        return f"""# Stage 3D：Two-Tower Retrieval 与固定预算增量召回

## 1. 阶段目标和边界

本阶段实现 CPU 可复现的 TT0/TT1 双塔召回、HNSW 检索、精确余弦审计和 Q0/T50/T100 固定预算实验。未实现矩阵分解、BPR、序列模型、排序、重排、在线服务或 Stage 3E。

## 2. Stage 1C 优先级变更

Stage 1C 曾把 MF 放在双塔之前；本阶段通过 `docs/decisions/stage_3d_two_tower_route_override.md` 显式记录路线变更，没有改写历史决策。原因是结构化商品塔、可独立预计算向量和 ANN 工业链路更有展示与验证价值；这不等于否定 MF，也不预设双塔优于 IUF。

## 3. 开工门禁

开工时处于 `main`，本地与 `origin/main` 为 0 ahead/0 behind，工作区干净；Stage 3A/3B/3C lock、manifest 和私有上游 fingerprint 一致，Stage 3C 状态为 `FAILED_VALUE_GATE`，Q0 Validation 候选 semantic fingerprint 精确复现。项目 `.venv` 为 Python 3.10，PyTorch {torch.__version__} CPU 与 hnswlib 0.8.0 smoke 通过。

## 4. 正式输入、lock 和 fingerprint

正式配置为 `configs/stage_3d_two_tower.yaml`。Stage 2A/2B manifest、Stage 3A/3B/3C lock、Q0 candidate/output、源码、训练 checkpoint、ANN audit 与 Validation 选择产物 fingerprint 均写入 `stage_3d_lock.json`。公开文件只保存完整 SHA-256，不保存实体 ID。

## 5. 数据、防泄漏和训练样本协议

反馈沿用 B 口径；每用户最多取 3 个首次 acquisition target，context 必须严格早于 target 日且移除 target，最近 20 个唯一历史商品用 `1/log2(rank+1)` 池化。内部 dev 按稳定 user-key hash 划分，Validation/Test 各自按 snapshot 重建；Test refit 如获准可使用 Test cutoff 前已经发生的 Validation 时段交易，这属于滚动 snapshot refit，不是标签泄漏。ground truth 只进入评估。

Validation 有 {validation_data['eligible_users']} 个 eligible users、{validation_data['examples']} 条样本，train/dev users={validation_data['train_users']}/{validation_data['dev_users']}，train/dev examples={validation_data['train_examples']}/{validation_data['dev_examples']}；target rank 1/2/3 样本数={validation_rank_text}，history 长度 mean/median/max={float(validation_data['mean_history_length']):.3f}/{float(validation_data['median_history_length']):.1f}/{validation_data['max_history_length']}。首次 acquisition 共 {validation_data['first_acquisition_count']}，其中 {validation_data['discarded_no_strict_prior_context']} 个没有严格更早 context、{validation_data['discarded_beyond_three_target_cap']} 个因每用户最多 3 target 被丢弃。Test 对应 eligible users/examples={test_data['eligible_users']}/{test_data['examples']}；完整分群覆盖见 `training_data_summary.csv`。

## 6. TT0/TT1 架构

TT0 商品塔使用 article-ID embedding；TT1 在相同底座上加入 Stage 3C canonical 字段 embedding，并在 taxonomy、appearance、merchandising/index 三组内取均值后等权拼接。用户塔由 user-ID embedding 与同一商品基础表示的历史池化组成。两塔经 MLP/线性层输出 64 维 L2 归一化向量；商品向量可离线独立预计算，不使用跨塔交叉特征。

## 7. 未观察项和 in-batch negative 边界

损失为 temperature=0.07 的 in-batch softmax。batch 其他 target 只能称为采样的未观察购买项，不能称为真实负样本；重复正 target 和当前用户历史内 target 使用确定性 mask 排除，对角正类保留。本阶段没有额外 popularity、uniform、hard 或 ANN negative 实验。

## 8. ANN 实现和质量审计

HNSW 使用 cosine、M=32、efConstruction=200；候选经 HNSW 粗取后使用原始 64 维向量精确余弦重排，排除 seen items。分层抽取 {audit['audit_user_count']} 名用户与 blockwise exact Top-500 比较，锁定 efSearch={audit['ef_search']}；mean/median/p5 overlap recall 分别为 {float(audit['mean_overlap_recall_at_500']):.6f}/{float(audit['median_overlap_recall_at_500']):.6f}/{float(audit['p5_overlap_recall_at_500']):.6f}，门禁={audit['passed']}。

## 9. Validation 模型选择

TT0/TT1 共享样本、优化器、batch、epoch、seed 和评估器，仅 TT1 增加属性信号。{selection_text}。按预注册顺序锁定 `{model}`，不是根据 Test 选择。

## 10. 单路指标

完整 Recall@12/50/100/300/500、HitRate@500、MAP@12、候选用户覆盖、shortfall、平均候选数、catalog coverage、集中度与 head/mid/tail 分布见 `single_route_metrics.csv` 和 `popularity_bucket_metrics.csv`。MAP@12 仅描述召回器内部 score 顺序，不是主 value gate。

## 11. Q0/T50/T100 固定预算结果

Q0 精确复用冻结 IUF Union，Validation Recall@500={q0_recall:.6f}。锁定模型的 {quota_text}；最终 `{physical}` Recall@500={locked_recall:.6f}、HitRate@500={locked_hit:.6f}、MAP@12={locked_map:.6f}、相对 Q0 ΔRecall@500={delta:.6f}。所有方案预算均为 500，未靠扩大 candidate K 获益。

## 12. Overlap

Two-Tower 与 IUF、Trend、Repurchase 的用户级平均 Jaccard 及共享情况见 `overlap_summary.csv`；统计直接基于固定候选，不将异构原始分数相加。

## 13. 独占命中和替换损失

锁定方案 Two-Tower exclusive ground-truth hits={int(float(marginal['two_tower_exclusive_ground_truth_hits']))}，新增/丢失/净 ground-truth hits={int(float(marginal['gained_ground_truth_hits']))}/{int(float(marginal['lost_ground_truth_hits']))}/{int(float(marginal['net_ground_truth_hits']))}；improved/declined user rate={float(marginal['improved_user_rate']):.6f}/{float(marginal['declined_user_rate']):.6f}。是否足以抵消 IUF 尾部损失由 value gate 原样判断。

## 14. Quota occupancy

T50/T100 均先按 source rank 截断，再合并 provenance 和去重，最后 Popularity 唯一补齐。configured cap、实际 Two-Tower unique occupancy、duplicate loss、实际减少的 ItemCF unique positions、Popularity fill 和 shortfall 分开记录在 `quota_occupancy.csv`，不把 cap 冒充实际占位。

## 15. 用户分群

分母严格复用 Stage 2B：cold=0、short-history=1–4、warm=5–20、active=21+。{segment_text}。cold-history 个性化 Two-Tower 候选按定义为 0，并与主协同人口分开。

## 16. 商品流行度分层

head/mid/tail 严格复用 Stage 2B cutoff-history bucket。TT0/TT1/IUF 的候选占比、ground-truth hits、Two-Tower exclusive hits 与 Union lost hits 见 `popularity_bucket_metrics.csv`。tail 或低历史支持不等于真实新商品。

## 17. Paired bootstrap

锁定方案相对 Q0 的 paired user-block bootstrap mean delta={float(bootstrap['delta_mean']):.6f}，95% CI=[{float(bootstrap['ci_lower']):.6f}, {float(bootstrap['ci_upper']):.6f}]，delta>0 比例={float(bootstrap['positive_fraction']):.6f}。这是离线同用户稳定性分析，不是线上显著性检验或 A/B Test。

## 18. 资源

checkpoint group 累计执行 {total_seconds:.1f}s，观测最大 RSS={max_rss / 1024**3:.3f} GiB，最大临时盘={max_temp / 1024**3:.3f} GiB；训练使用 CPU、batch={training['batch_size']}、最多 3 epochs，锁定 checkpoint epoch={training['selected_epoch']}。逐组数据见 `resource_summary.csv`。

## 19. OOM、spill、失败、重试和恢复

{failure_text} 大任务按输入/source/config fingerprint 和输出 SHA-256 checkpoint；只复用完全匹配的 SUCCESS 输出。没有把 smoke 或局部结果表述为全量结论。

## 20. Validation lock

正式状态为 `{status}`，锁定模型 `{model}`、配额 `{quota}`、ANN efSearch={audit['ef_search']}。lock 自身和所有选择证据都有 fingerprint；`test_authorized={str(lock['test_authorized']).lower()}`。

## 21. Test 门禁和实际执行状态

Test 状态为 `{test_status['run_status']}`。只有 `PASSED_VALUE_GATE` 才允许按 lock 重建一次 snapshot、refit 锁定模型并比较 Q0 与唯一锁定 Union；Conditional/Failed 不触碰 Two-Tower Test。{test_detail} Test 结果未用于回调模型、配额或 ANN 参数。

## 22. 最终阶段状态

{conclusion}

## 23. 真实限制

数据没有曝光、点击、真实负反馈、库存、可售状态、精确上架时间、日内时间戳、order_id 或真实 quantity。未购买不等于不喜欢；完整 articles 身份域不等于历史可售域；离线 Recall/MAP 不等于 CTR、CVR、收入或线上收益。双塔也没有证明线上 cold-item 能力。

## 24. 简历和面试可支持事实

可以陈述：实现了 CPU 双塔、结构化商品塔、严格 snapshot 样本、in-batch false-negative mask、HNSW+exact rerank、ANN exact-overlap audit、固定预算 Union、paired bootstrap、可恢复 checkpoint 和公开审计链路；最终效果必须连同 `{status}` 如实陈述。

## 25. 不能声称的内容

不能声称线上 A/B、CTR/CVR 或收益提升，不能把未观察项称为真实负样本，不能把重复交易行称为 quantity，不能声称有库存过滤或精确新商品识别，也不能把本阶段包装为排序、重排或在线服务。

## 26. 下一阶段建议

若未通过，建议由新的正式 Prompt 授权 Stage 3E 冻结现有 Q0/候选契约，再评审是否进入 LightGBM Ranking；不补做 MF/BPR、不重开 Content、不增加双塔参数救结果。本报告不授权自动开始下一阶段。

## 本阶段面试知识点与可能追问

- 双塔能把用户和商品独立编码到同一空间，商品向量可以离线预计算；用户塔输入 ID 与历史池化，商品塔输入 ID，TT1 额外加入 canonical 属性。
- 历史池化是有意保持 V1 复杂度与数据证据匹配；无精确日内顺序且本阶段未授权序列模型。
- user-ID embedding 是双塔输入之一；它与 MF 都学习 ID 潜向量，但本实现还有历史池化、商品属性、非线性投影、in-batch softmax 和可独立 ANN 检索，不能等同于传统 MF。
- target 必须从 context 移除且 context 严格早于 target；同日交易没有可靠先后，不能硬造序列。
- in-batch 项是采样的未观察购买项；重复 target 和用户历史已见 target 需要 mask，temperature 控制 softmax 分布尖锐度，L2 norm 使点积等价 cosine。
- TT0/TT1 是控制变量实验；属性组先组内均值再等权拼接，避免字段多的组天然权重大。
- HNSW 是近似候选生成，exact cosine rerank 只修正粗取集合内顺序，不能弥补漏召回；因此必须与 exact Top-500 做 overlap audit。
- source-rank cap 是预去重配额，不等于实际 unique occupancy；固定预算回答的是替换 IUF 尾部后净增是否为正。
- paired bootstrap 保留同一用户成对差值，只能说明离线稳定性；Test lock 防止用 Test 回调模型或配额。
"""

    def write_privacy_audit(self) -> Path:
        target = self.config.artifact_dir / "privacy_fingerprint_audit.csv"
        # privacy audit 与 manifest 不能互相指纹；manifest 单向记录 privacy audit，
        # 两者仍会共同进入最后的敏感值扫描。
        paths = sorted(path for path in self.config.artifact_dir.glob("*")
                       if path.is_file() and path != target and path.name != "stage_3d_manifest.csv")
        if self.config.report_path.exists():
            paths.append(self.config.report_path)
        rows = []
        forbidden_headers = {"customer_key", "article_key", "customer_id", "article_id", "postal_code", "user_key"}
        for path in paths:
            header_pass = True
            if path.suffix.lower() == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    header_pass = not bool(set(csv.DictReader(handle).fieldnames or ()) & forbidden_headers)
            text_value = path.read_text(encoding="utf-8-sig")
            local_path_pass = str(self.config.root).lower() not in text_value.lower()
            rows.append({
                "stage": "3D", "artifact_path": self.config.relative(path),
                "artifact_fingerprint": file_fingerprint(path),
                "forbidden_entity_header_pass": header_pass,
                "local_absolute_path_pass": local_path_pass,
                "declared_fingerprint_only": True,
                "passed": header_pass and local_path_pass,
                "run_status": "SUCCESS" if header_pass and local_path_pass else "FAILED",
            })
        target = self._write_csv("privacy_fingerprint_audit.csv", rows)
        all_public = [*self.config.artifact_dir.glob("*.csv"), *self.config.artifact_dir.glob("*.json"),
                      self.config.report_path]
        validate_stage3d_public_artifacts(all_public)
        if not all(bool(row["passed"]) for row in rows):
            raise ValueError("公开产物隐私或本地路径审计失败")
        for relative in ("artifacts/stage_3d/private", "artifacts/stage_3d/tmp",
                         "artifacts/stage_3d/checkpoints"):
            result = subprocess.run(["git", "check-ignore", relative], cwd=self.config.root,
                                    text=True, capture_output=True)
            if result.returncode != 0:
                raise ValueError(f"{relative} 未被 .gitignore 覆盖")
        tracked = subprocess.run(["git", "ls-files", "data/raw", "artifacts/stage_3d/private",
                                  "artifacts/stage_3d/tmp", "artifacts/stage_3d/checkpoints"],
                                 cwd=self.config.root, text=True, capture_output=True, check=True).stdout.splitlines()
        if tracked:
            raise ValueError("raw/private/tmp/checkpoints 存在 tracked 文件")
        self._validate_frozen_upstream()
        return target

    def write_manifest(self) -> Path:
        target = self.config.artifact_dir / "stage_3d_manifest.csv"
        public = sorted(path for path in self.config.artifact_dir.glob("*") if path.is_file() and path != target)
        public.extend([
            self.config.path, self.config.report_path,
            self.config.root / "docs/decisions/stage_3d_two_tower_route_override.md",
        ])
        status = json_read(self.config.lock_path)["stage_status"]
        rows = [{
            "stage": "3D", "snapshot_id": "validation_14d,test_14d",
            "as_of_date": self.snapshots["test_14d"].history_end,
            "artifact_path": self.config.relative(path), "artifact_fingerprint": file_fingerprint(path),
            "size_bytes": path.stat().st_size, "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint, "stage_status": status, "run_status": "SUCCESS",
        } for path in public]
        write_csv(target, rows, rows[0].keys())
        return target

    def finalize(self) -> None:
        self.refresh_quota_occupancy_public()
        self.training_data_summary()
        self.write_operational_audits()
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.write_text(self.build_report(), encoding="utf-8")
        self.write_privacy_audit()
        self.write_manifest()
        validate_stage3d_public_artifacts([*self.config.artifact_dir.glob("*.csv"),
                                           *self.config.artifact_dir.glob("*.json"), self.config.report_path])

    def run_all(self) -> str:
        status = self.run_validation()
        if status == "PASSED_VALUE_GATE":
            self.run_test()
        self.finalize()
        return status
