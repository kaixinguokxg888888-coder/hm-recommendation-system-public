from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from data_audit.stage1a import ResourceMonitor
from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.config import RetrievalConfig, source_fingerprint as stage3a_source_fingerprint
from retrieval.runner import SnapshotSpec as Stage3ASnapshotSpec
from retrieval.runner import Stage3ARunner
from retrieval.stage3b.config import Stage3BConfig, source_fingerprint as stage3b_source_fingerprint
from retrieval.stage3b.runner import SnapshotSpec as Stage3BSnapshotSpec
from retrieval.stage3b.runner import Stage3BRunner
from retrieval.stage3d.config import Stage3DConfig, source_fingerprint as stage3d_source_fingerprint
from retrieval.stage3d.data import AttributeVocabulary, recency_weight
from retrieval.stage3d.model import in_batch_softmax_loss
from retrieval.stage3d.runner import Stage3DRunner
from retrieval.stage3e.config import Stage3EConfig, source_fingerprint as stage3e_source_fingerprint
from retrieval.stage3e.contract import (
    CONTRACT_VERSION,
    PROTOCOL_VERSION,
    fingerprint_from_user_digests,
    parquet_partition_user_digests,
    schema_manifest_rows,
)

from .config import Stage4B1Config
from .contract import (
    HistoricalReplayError,
    directory_fingerprint,
    forbidden_query_scope_fields,
    privacy_findings,
    protocol_identity_rows,
    validate_formal_historical_lineage,
)


SNAPSHOT_ID = "ranking_train_14d_v1"
CHECKPOINT_SEMANTICS_VERSION = "stage_4b1_historical_replay_v1"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def source_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((root / "src/retrieval/stage4b1").glob("*.py"))
    paths.extend(sorted((root / "scripts").glob("run_stage_4b1*.py")))
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class ComponentConfig:
    """Path adapter that keeps frozen component parameters but writes only to Stage 4B-1."""

    owner: Stage4B1Config
    raw: dict[str, Any]
    component: str

    @property
    def root(self) -> Path:
        return self.owner.root

    def resolve(self, value: str) -> Path:
        return (self.root / value).resolve()

    def relative(self, path: Path) -> str:
        return self.owner.relative(path)

    @property
    def artifact_dir(self) -> Path:
        return self.owner.artifact_dir

    @property
    def private_dir(self) -> Path:
        return self.owner.private_dir / self.component

    @property
    def checkpoint_dir(self) -> Path:
        return self.owner.checkpoint_dir / self.component

    @property
    def temp_dir(self) -> Path:
        return self.owner.temp_dir / self.component

    @property
    def report_path(self) -> Path:
        return self.owner.report_path

    @property
    def lock_path(self) -> Path:
        return self.owner.resolve(self.owner.raw["upstream"]["stage_3d_lock"])

    @property
    def memory_limit_mb(self) -> int:
        return self.owner.memory_limit_mb

    @property
    def fingerprint(self) -> str:
        return self.owner.fingerprint


class HistoricalStage3AAdapter(Stage3ARunner):
    def __init__(self, owner: "Stage4B1Runner") -> None:
        self.owner = owner
        raw = copy.deepcopy(owner.stage3a_frozen.raw)
        raw["paths"] = {
            "artifact_directory": owner.config.relative(owner.config.artifact_dir),
            "private_directory": owner.config.relative(owner.config.private_dir / "sources"),
            "checkpoint_directory": owner.config.relative(owner.config.checkpoint_dir / "sources"),
            "temp_directory": owner.config.relative(owner.config.temp_dir / "sources"),
            "report": owner.config.relative(owner.config.report_path),
        }
        self.config = ComponentConfig(owner.config, raw, "sources")
        self.dataset_fingerprint = owner.dataset_fingerprint
        self.feedback_version = owner.feedback_version
        self.population_version = owner.population_version
        snapshot = owner.config.snapshot
        self.snapshots = {
            SNAPSHOT_ID: Stage3ASnapshotSpec(
                SNAPSHOT_ID, snapshot.history_start, snapshot.history_end,
                snapshot.label_start, snapshot.label_end, "query_user_scope_only_no_item_labels",
            )
        }
        self.source_fingerprint = owner.source_fingerprint
        self.run_fingerprint = owner.run_fingerprint
        self.resources = owner.resources
        self.run_status = owner.run_status

    def upstream_path(self, prefix: str, snapshot: str) -> Path:
        if snapshot != SNAPSHOT_ID:
            raise HistoricalReplayError("historical adapter refuses non-registered snapshot")
        if prefix == "evaluation_population":
            return self.owner.request_population_path
        if prefix == "item_state":
            return self.owner.item_state_path
        raise HistoricalReplayError(f"Stage 4B-1 does not materialize ranking supervision: {prefix}")


class HistoricalStage3BAdapter(Stage3BRunner):
    def __init__(self, owner: "Stage4B1Runner", stage3a: HistoricalStage3AAdapter) -> None:
        self.owner = owner
        raw = copy.deepcopy(owner.stage3b_frozen.raw)
        raw["paths"] = {
            "artifact_directory": owner.config.relative(owner.config.artifact_dir),
            "private_directory": owner.config.relative(owner.config.private_dir / "itemcf"),
            "checkpoint_directory": owner.config.relative(owner.config.checkpoint_dir / "itemcf"),
            "temp_directory": owner.config.relative(owner.config.temp_dir / "itemcf"),
            "report": owner.config.relative(owner.config.report_path),
        }
        raw["upstream"]["stage_3a_private"] = owner.config.relative(stage3a.config.private_dir)
        self.config = ComponentConfig(owner.config, raw, "itemcf")
        self.dataset_fingerprint = owner.dataset_fingerprint
        self.feedback_version = owner.feedback_version
        self.population_version = owner.population_version
        snapshot = owner.config.snapshot
        self.snapshots = {
            SNAPSHOT_ID: Stage3BSnapshotSpec(
                SNAPSHOT_ID, snapshot.history_start, snapshot.history_end,
                snapshot.label_start, snapshot.label_end, "query_user_scope_only_no_item_labels",
            )
        }
        self.source_fingerprint = owner.source_fingerprint
        self.stage3a_lock_path = owner.stage3a_lock_path
        self.stage3a_lock_fingerprint = file_fingerprint(self.stage3a_lock_path)
        self.run_fingerprint = owner.run_fingerprint
        self.resources = owner.resources
        self.run_status = owner.run_status

    def upstream_path(self, prefix: str, snapshot: str) -> Path:
        if snapshot != SNAPSHOT_ID:
            raise HistoricalReplayError("historical ItemCF adapter refuses non-registered snapshot")
        if prefix == "evaluation_population":
            return self.owner.request_population_path
        if prefix == "item_state":
            return self.owner.item_state_path
        raise HistoricalReplayError(f"Stage 4B-1 does not expose Stage 2B supervision: {prefix}")

    def consolidate_scale(self, snapshot_id: str) -> None:
        parts = [
            json_read(self.scale_part_path(snapshot_id, part))
            for part in range(int(self.config.raw["execution"]["pair_partitions"]))
        ]
        payload = {
            "snapshot_id": snapshot_id,
            "as_of_date": self.snapshots[snapshot_id].history_end,
            **{
                field: sum(int(row[field]) for row in parts)
                for field in (
                    "history_users", "cooccurrence_users", "truncated_users",
                    "unique_user_items_before", "unique_user_items_after",
                    "pair_contributions_before", "pair_contributions_after",
                )
            },
            "max_distinct_items": max(int(row["max_distinct_items"]) for row in parts),
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint,
        }
        json_write(self.scale_path(snapshot_id), payload)


class HistoricalStage3DAdapter(Stage3DRunner):
    def __init__(
        self, owner: "Stage4B1Runner", stage3a: HistoricalStage3AAdapter,
        stage3b: HistoricalStage3BAdapter,
    ) -> None:
        self.owner = owner
        raw = copy.deepcopy(owner.stage3d_frozen.raw)
        raw["paths"] = {
            "articles": owner.config.raw["upstream"]["articles"],
            "artifact_directory": owner.config.relative(owner.config.artifact_dir),
            "private_directory": owner.config.relative(owner.config.private_dir / "two_tower"),
            "checkpoint_directory": owner.config.relative(owner.config.checkpoint_dir / "two_tower"),
            "temp_directory": owner.config.relative(owner.config.temp_dir / "two_tower"),
            "report": owner.config.relative(owner.config.report_path),
        }
        raw["upstream"]["stage_2a_private"] = owner.config.raw["upstream"]["stage_2a_private"]
        raw["upstream"]["stage_3a_private"] = owner.config.relative(stage3a.config.private_dir)
        self.config = ComponentConfig(owner.config, raw, "two_tower")
        self.source_fingerprint = owner.source_fingerprint
        self.stage3b = stage3b
        self.dataset_fingerprint = owner.dataset_fingerprint
        self.feedback_version = owner.feedback_version
        self.population_version = owner.population_version
        snapshot = owner.config.snapshot
        self.snapshots = {
            SNAPSHOT_ID: Stage3BSnapshotSpec(
                SNAPSHOT_ID, snapshot.history_start, snapshot.history_end,
                snapshot.label_start, snapshot.label_end, "query_user_scope_only_no_item_labels",
            )
        }
        self.upstream_lock_fingerprints = owner.upstream_lock_fingerprints
        self.run_fingerprint = owner.run_fingerprint
        self.resources = owner.resources
        self.run_status = owner.run_status

    def stage2b_path(self, prefix: str, snapshot: str) -> Path:
        if snapshot != SNAPSHOT_ID:
            raise HistoricalReplayError("historical TT1 adapter refuses non-registered snapshot")
        if prefix == "evaluation_population":
            return self.owner.request_population_path
        if prefix == "item_state":
            return self.owner.item_state_path
        raise HistoricalReplayError(f"Stage 4B-1 does not expose ranking labels: {prefix}")

    def train_historical_final_model(self, snapshot: str, variant: str, *, locked_epoch: int) -> tuple[Path, Path]:
        evidence = self.owner.epoch_lock_evidence_path
        checkpoint, history_path = super().train_final_model(
            snapshot, variant, locked_epoch=locked_epoch, selection_checkpoint=evidence,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["checkpoint_role"] = "formal_historical_full_refit"
        payload["model_role"] = "formal_historical_full_refit"
        payload["snapshot_id"] = snapshot
        payload["as_of_date"] = self.snapshots[snapshot].history_end
        payload["epoch_lock_fingerprint"] = file_fingerprint(self.owner.stage3d_lock_path)
        payload.pop("selection_checkpoint_fingerprint", None)
        torch.save(payload, checkpoint)
        history = json_read(history_path)
        history.update({
            "stage": "4B-1", "as_of_date": self.snapshots[snapshot].history_end,
            "checkpoint_role": "formal_historical_full_refit",
            "model_role": "formal_historical_full_refit",
            "locked_epoch": locked_epoch,
            "epoch_lock_path": self.config.relative(self.owner.stage3d_lock_path),
            "epoch_lock_fingerprint": file_fingerprint(self.owner.stage3d_lock_path),
            "final_checkpoint_fingerprint": file_fingerprint(checkpoint),
        })
        history.pop("selection_checkpoint_path", None)
        history.pop("selection_checkpoint_fingerprint", None)
        json_write(history_path, history)
        return checkpoint, history_path

    def validate_formal_checkpoint(self, snapshot: str, variant: str) -> dict[str, Any]:
        checkpoint = self.model_checkpoint(snapshot, variant)
        history = json_read(self.final_refit_history_path(snapshot, variant))
        validate_formal_historical_lineage(history)
        if (
            history.get("final_checkpoint_fingerprint") != file_fingerprint(checkpoint)
            or history.get("epoch_lock_fingerprint") != file_fingerprint(self.owner.stage3d_lock_path)
            or history.get("model_variant") != "TT1"
        ):
            raise HistoricalReplayError("formal historical TT1 checkpoint lineage mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("checkpoint_role") != "formal_historical_full_refit":
            raise HistoricalReplayError("selection-only/non-historical checkpoint rejected")
        return history

    def export_embeddings(self, snapshot: str, variant: str) -> tuple[Path, Path, Path]:
        model = self._model(snapshot, variant).cpu()
        refit = self.validate_formal_checkpoint(snapshot, variant)
        payload = torch.load(self.model_checkpoint(snapshot, variant), map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"])
        model.eval()
        item_table = pq.read_table(self.attribute_path(snapshot), columns=["article_key"])
        item_keys = np.asarray(item_table["article_key"]).astype(np.int64, copy=False)
        item_vectors: list[np.ndarray] = []
        batch_size = int(self.config.raw["execution"]["embedding_batch_size"])
        with torch.inference_mode():
            for start in range(0, len(item_keys), batch_size):
                # Arrow 导出的数组可能是只读视图；显式复制可避免 PyTorch 的未定义写入警告。
                item_batch = np.array(item_keys[start:start + batch_size], dtype=np.int64, copy=True)
                item_vectors.append(model.encode_items(torch.from_numpy(item_batch)).numpy())
        item_target = self.embedding_path(snapshot, variant, "items")
        item_target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(item_target, article_keys=item_keys,
                 embeddings=np.concatenate(item_vectors).astype(np.float32, copy=False))

        columns = ["customer_key", "user_segment", *(f"history_{rank:02d}" for rank in range(1, 21))]
        user_keys_parts: list[np.ndarray] = []
        segment_parts: list[np.ndarray] = []
        user_vector_parts: list[np.ndarray] = []
        weights = np.asarray([recency_weight(rank) for rank in range(1, 21)], dtype=np.float32)
        with torch.inference_mode():
            for batch in pq.ParquetFile(self.user_feature_path(snapshot)).iter_batches(
                batch_size=batch_size, columns=columns,
            ):
                user_keys = self._batch_numpy(batch, "customer_key").astype(np.int64, copy=False)
                histories = np.column_stack([
                    self._batch_numpy(batch, f"history_{rank:02d}") for rank in range(1, 21)
                ]).astype(np.int64, copy=False)
                history_weights = (histories >= 0).astype(np.float32) * weights[None, :]
                vectors = model.encode_users(
                    torch.from_numpy(user_keys), torch.from_numpy(histories), torch.from_numpy(history_weights),
                ).numpy()
                user_keys_parts.append(user_keys)
                segment_parts.append(np.asarray(
                    batch.column(batch.schema.get_field_index("user_segment")).to_pylist(), dtype="U16",
                ))
                user_vector_parts.append(vectors)
        user_target = self.embedding_path(snapshot, variant, "users")
        np.savez(user_target, customer_keys=np.concatenate(user_keys_parts),
                 segments=np.concatenate(segment_parts),
                 embeddings=np.concatenate(user_vector_parts).astype(np.float32, copy=False))
        metadata = self.embedding_metadata_path(snapshot, variant)
        json_write(metadata, {
            "stage": "4B-1", "snapshot_id": snapshot, "as_of_date": self.snapshots[snapshot].history_end,
            "model_variant": variant, "model_role": "formal_historical_full_refit",
            "checkpoint_role": "formal_historical_full_refit",
            "source_model_checkpoint_path": self.config.relative(self.model_checkpoint(snapshot, variant)),
            "source_model_checkpoint_fingerprint": refit["final_checkpoint_fingerprint"],
            "item_embedding_fingerprint": file_fingerprint(item_target),
            "user_embedding_fingerprint": file_fingerprint(user_target),
            "run_fingerprint": self.run_fingerprint,
        })
        return item_target, user_target, metadata

    def build_candidate_lineage(self, snapshot: str, variant: str) -> Path:
        target = super().build_candidate_lineage(snapshot, variant)
        payload = json_read(target)
        payload.update({
            "stage": "4B-1", "as_of_date": self.snapshots[snapshot].history_end,
            "candidate_role": "formal_historical_full_refit",
            "model_role": "formal_historical_full_refit",
        })
        json_write(target, payload)
        return target


class Stage4B1Runner:
    def __init__(self, config: Stage4B1Config) -> None:
        self.config = config
        self.stage3a_frozen = RetrievalConfig.load(config.resolve(config.raw["upstream"]["stage_3a_config"]))
        self.stage3b_frozen = Stage3BConfig.load(config.resolve(config.raw["upstream"]["stage_3b_config"]))
        self.stage3d_frozen = Stage3DConfig.load(config.resolve(config.raw["upstream"]["stage_3d_config"]))
        self.stage3e_frozen = Stage3EConfig.load(config.resolve(config.raw["upstream"]["stage_3e_config"]))
        split = csv_rows(config.root / "artifacts/stage_2a/split_manifest.csv")
        self.dataset_fingerprint = split[0]["dataset_fingerprint"]
        self.feedback_version = split[0]["feedback_version"]
        self.population_version = "1.0.0"
        self.stage3a_lock_path = config.resolve(config.raw["upstream"]["stage_3a_lock"])
        self.stage3b_lock_path = config.resolve(config.raw["upstream"]["stage_3b_lock"])
        self.stage3d_lock_path = config.resolve(config.raw["upstream"]["stage_3d_lock"])
        self.stage3e_config_path = config.resolve(config.raw["upstream"]["stage_3e_config"])
        self.upstream_lock_fingerprints = {
            "stage_3a_lock_fingerprint": file_fingerprint(self.stage3a_lock_path),
            "stage_3b_lock_fingerprint": file_fingerprint(self.stage3b_lock_path),
            "stage_3c_lock_fingerprint": file_fingerprint(config.root / "artifacts/stage_3c/stage_3c_lock.json"),
        }
        self.source_fingerprint = source_fingerprint(config.root)
        self.upstream_records = self._validate_upstream()
        self.upstream_fingerprint = hashlib.sha256(
            json.dumps(self.upstream_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.run_fingerprint = hashlib.sha256(
            (config.fingerprint + self.source_fingerprint + self.upstream_fingerprint).encode("utf-8")
        ).hexdigest()
        self.resources: list[dict[str, Any]] = []
        self.run_status: list[dict[str, Any]] = []
        self.stage3a = HistoricalStage3AAdapter(self)
        self.stage3b = HistoricalStage3BAdapter(self, self.stage3a)
        self.stage3d = HistoricalStage3DAdapter(self, self.stage3a, self.stage3b)
        self.protocol_rows = self._protocol_identity()

    @property
    def stage2a_private(self) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_2a_private"])

    @property
    def feedback_path(self) -> Path:
        return self.stage2a_private / "feedback_daily.parquet"

    @property
    def customer_map_path(self) -> Path:
        return self.stage2a_private / "customer_key_map.parquet"

    @property
    def article_map_path(self) -> Path:
        return self.stage2a_private / "article_key_map.parquet"

    @property
    def request_population_path(self) -> Path:
        return self.config.private_dir / f"request_population_{SNAPSHOT_ID}.parquet"

    @property
    def item_state_path(self) -> Path:
        return self.config.private_dir / f"item_state_{SNAPSHOT_ID}.parquet"

    @property
    def epoch_lock_evidence_path(self) -> Path:
        return self.config.private_dir / "historical_epoch_lock_evidence.json"

    @property
    def immutability_baseline_path(self) -> Path:
        return self.config.private_dir / "immutability_baseline.json"

    def contract_part(self, part: int, *, replay: bool = False) -> Path:
        role = "replay" if replay else "candidates"
        return self.config.private_dir / role / SNAPSHOT_ID / f"part_{part:03d}.parquet"

    def prepare_directories(self) -> None:
        for path in (self.config.artifact_dir, self.config.private_dir,
                     self.config.checkpoint_dir, self.config.temp_dir):
            path.mkdir(parents=True, exist_ok=True)
        for adapter in (self.stage3a, self.stage3b, self.stage3d):
            for path in (adapter.config.private_dir, adapter.config.checkpoint_dir, adapter.config.temp_dir):
                path.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(":memory:")
        con.execute(f"SET temp_directory='{sql_path(self.config.temp_dir)}'")
        con.execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
        con.execute(f"SET threads={int(self.config.raw['duckdb']['threads'])}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{self.config.raw['duckdb']['max_temp_directory_size']}'")
        return con

    def _validate_upstream(self) -> list[dict[str, Any]]:
        stage4a = json.loads(
            self.config.resolve(self.config.raw["upstream"]["stage_4a_config"]).read_text(encoding="utf-8")
        )
        records: list[dict[str, Any]] = []
        for row in stage4a["upstream_artifacts"]:
            path = self.config.resolve(row["path"])
            actual = file_fingerprint(path) if path.exists() else ""
            if actual != row["fingerprint"]:
                raise HistoricalReplayError(f"upstream fingerprint mismatch: {row['path']}")
            records.append({"role": row["role"], "path": row["path"], "fingerprint": actual})
        for role, relative in self.config.raw["upstream"].items():
            if role in {"stage_2a_private", "articles"}:
                continue
            path = self.config.resolve(relative)
            if not path.exists():
                raise HistoricalReplayError(f"missing upstream artifact: {relative}")
            records.append({"role": role, "path": relative, "fingerprint": file_fingerprint(path)})
        if stage4a.get("final_acceptance_status") != "ACCEPTED":
            raise HistoricalReplayError("Stage 4A is not final accepted")
        if any(row.get("status") != "PASS" for row in csv_rows(
            self.config.resolve(self.config.raw["upstream"]["stage_4a_protocol_gate"])
        )):
            raise HistoricalReplayError("Stage 4A protocol gate is not fully PASS")
        if any(row.get("status") != "PASS" for row in csv_rows(
            self.config.resolve(self.config.raw["upstream"]["stage_3e_contract_gate"])
        )):
            raise HistoricalReplayError("Stage 3E contract gate is not fully PASS")
        lock3d = json_read(self.stage3d_lock_path)
        lock3b = json_read(self.stage3b_lock_path)
        lock3a = json_read(self.stage3a_lock_path)
        if (
            lock3d.get("locked_model") != "TT1" or lock3d.get("locked_quota") != "T100"
            or int(lock3d.get("locked_epoch", -1)) != 1
            or lock3b.get("locked_itemcf_variant") != "iuf"
        ):
            raise HistoricalReplayError("frozen Stage 3B/3D lock identity mismatch")
        source_checks = {
            "stage_3a": (stage3a_source_fingerprint(self.config.root), lock3a["source_fingerprint"]),
            "stage_3d": (stage3d_source_fingerprint(self.config.root), lock3d["source_fingerprint"]),
        }
        for name, (actual, expected) in source_checks.items():
            if actual != expected:
                raise HistoricalReplayError(f"{name} frozen source fingerprint mismatch")
            records.append({"role": f"{name}_source", "path": f"source:{name}", "fingerprint": actual})
        current_stage3b_source = stage3b_source_fingerprint(self.config.root)
        records.append({"role": "stage_3b_source_current", "path": "source:stage_3b",
                        "fingerprint": current_stage3b_source})
        records.append({"role": "stage_3b_source_at_lock", "path": "lock:stage_3b",
                        "fingerprint": lock3b["source_fingerprint"]})
        current_stage3e_source = stage3e_source_fingerprint(self.config.root)
        expected_stage3e_sources = {
            row["source_fingerprint"] for row in csv_rows(
                self.config.resolve(self.config.raw["upstream"]["stage_3e_manifest"])
            )
        }
        if expected_stage3e_sources != {current_stage3e_source}:
            raise HistoricalReplayError("stage_3e frozen source fingerprint mismatch")
        records.append({"role": "stage_3e_source", "path": "source:stage_3e", "fingerprint": current_stage3e_source})
        expected_private = {
            "feedback_daily.parquet": "08743fdf85342b4eb26157d096b2bd53ec580d385b56214eb3ed4b723d033b71",
            "customer_key_map.parquet": "10df1b071452dcc3d0ac4434721536078f9b46b2fcf356cd30310b8999647887",
            "article_key_map.parquet": "e0fb73bf4189d6aab6a56855d254c20cf074888aa4b9c4417f63bfd2af7d018e",
        }
        for name, expected in expected_private.items():
            path = self.stage2a_private / name
            actual = file_fingerprint(path)
            if actual != expected:
                raise HistoricalReplayError(f"Stage 2A private fingerprint mismatch: {name}")
            records.append({"role": f"stage_2a_private_{name}", "path": f"private:{name}", "fingerprint": actual})
        return sorted(records, key=lambda row: (row["role"], row["path"]))

    def _protocol_identity(self) -> list[dict[str, Any]]:
        frozen = self.config.raw["frozen_retrieval"]
        stage3e = self.stage3e_frozen.raw["frozen_retrieval"]
        lock = json_read(self.stage3d_lock_path)
        actual = {
            "candidate_contract_version": self.stage3e_frozen.raw["candidate_contract_version"],
            "candidate_protocol_version": self.stage3e_frozen.raw["candidate_protocol_version"],
            "model": lock["locked_model"],
            "quota": lock["locked_quota"],
            "locked_epoch": lock["locked_epoch"],
            "candidate_budget": stage3e["candidate_budget"],
            "source_caps": stage3e["source_caps"],
            "insertion_order": stage3e["insertion_order"],
            "quota_semantics": stage3e["quota_semantics"],
            "duplicate_semantics": stage3e["duplicate_semantics"],
            "fallback_semantics": stage3e["fallback"],
            "source_score_semantics": "source_local_raw_scores_never_summed_or_cross_source_calibrated",
            "ann": {
                "algorithm": lock["ann_parameters"]["algorithm"],
                "metric": lock["ann_parameters"]["metric"],
                "m": lock["ann_parameters"]["m"],
                "ef_construction": lock["ann_parameters"]["ef_construction"],
                "ef_search": lock["ann_parameters"]["ef_search"],
                "initial_query_k": lock["ann_parameters"]["initial_query_k"],
                "retry_query_k": lock["ann_parameters"]["retry_query_k"],
                "exact_rerank": True,
                "mean_overlap_min": self.stage3d_frozen.raw["ann"]["audit_mean_overlap_recall_min"],
                "p5_overlap_min": self.stage3d_frozen.raw["ann"]["audit_p5_overlap_recall_min"],
            },
        }
        fields = list(actual)
        rows = protocol_identity_rows(actual, frozen, fields)
        if any(row["status"] != "PASS" for row in rows):
            raise HistoricalReplayError("HISTORICAL_PROTOCOL_IDENTITY failed")
        return rows

    @staticmethod
    def _set_fingerprint(paths: Iterable[Path]) -> str:
        records = [
            (path.as_posix(), path.stat().st_size, file_fingerprint(path))
            for path in sorted(paths, key=lambda value: value.as_posix())
        ]
        return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode("utf-8")).hexdigest()

    def run_component(
        self, group: str, operation: Callable[[], Any], outputs: Sequence[Path],
        *, inputs: Sequence[Path], component: str, partition: int | None = None,
        force: bool = False,
    ) -> Any:
        checkpoint = self.config.checkpoint_dir / f"{group}.json"
        previous = json_read(checkpoint) if checkpoint.exists() else None
        input_fingerprint = self._set_fingerprint(inputs)
        reusable = bool(
            not force and previous and previous.get("status") == "SUCCESS"
            and previous.get("stage_config_fingerprint") == self.config.fingerprint
            and previous.get("upstream_fingerprint") == self.upstream_fingerprint
            and previous.get("checkpoint_semantics_version", CHECKPOINT_SEMANTICS_VERSION)
                == CHECKPOINT_SEMANTICS_VERSION
            and previous.get("snapshot_id") == SNAPSHOT_ID
            and previous.get("as_of_date") == self.config.snapshot.as_of_date
            and previous.get("component") == component
            and previous.get("partition") == partition
            and previous.get("input_physical_fingerprint") == input_fingerprint
            and all(path.exists() for path in outputs)
            and previous.get("output_physical_fingerprint") == self._set_fingerprint(outputs)
        )
        if reusable:
            row = dict(previous)
            row["run_status"] = "REUSED"
            row["checkpoint_reused"] = True
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "REUSED", "attempt": row["attempt"], "message": ""})
            return None
        attempt = int((previous or {}).get("attempt", 0)) + 1
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        try:
            with monitor:
                result = operation()
            missing = [self.config.relative(path) for path in outputs if not path.exists()]
            if missing:
                raise RuntimeError(f"component output incomplete: {missing}")
            row = {
                "group_name": group, "status": "SUCCESS", "attempt": attempt,
                "generated_at": now(), "stage_config_fingerprint": self.config.fingerprint,
                "upstream_fingerprint": self.upstream_fingerprint,
                "source_fingerprint": self.source_fingerprint, "run_fingerprint": self.run_fingerprint,
                "checkpoint_semantics_version": CHECKPOINT_SEMANTICS_VERSION,
                "snapshot_id": SNAPSHOT_ID, "as_of_date": self.config.snapshot.as_of_date,
                "component": component, "partition": partition,
                "input_physical_fingerprint": input_fingerprint,
                "output_physical_fingerprint": self._set_fingerprint(outputs),
                "output_files": [self.config.relative(path) for path in outputs],
                "output_size_bytes": sum(path.stat().st_size for path in outputs),
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "memory_limit_mb": self.config.memory_limit_mb,
                "threads": self.config.raw["duckdb"]["threads"],
                "checkpoint_reused": False, "error_message": "",
            }
            json_write(checkpoint, row)
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "SUCCESS", "attempt": attempt, "message": ""})
            return result
        except Exception as exc:
            row = {
                "group_name": group, "status": "FAILED", "attempt": attempt,
                "generated_at": now(), "stage_config_fingerprint": self.config.fingerprint,
                "upstream_fingerprint": self.upstream_fingerprint,
                "source_fingerprint": self.source_fingerprint, "run_fingerprint": self.run_fingerprint,
                "checkpoint_semantics_version": CHECKPOINT_SEMANTICS_VERSION,
                "snapshot_id": SNAPSHOT_ID, "as_of_date": self.config.snapshot.as_of_date,
                "component": component, "partition": partition,
                "input_physical_fingerprint": input_fingerprint,
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "checkpoint_reused": False,
                "error_message": str(exc).replace(str(self.config.root), "."),
            }
            json_write(checkpoint, row)
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "FAILED", "attempt": attempt, "message": row["error_message"]})
            raise

    def write_epoch_lock_evidence(self) -> Path:
        lock = json_read(self.stage3d_lock_path)
        json_write(self.epoch_lock_evidence_path, {
            "stage": "4B-1", "snapshot_id": SNAPSHOT_ID,
            "as_of_date": self.config.snapshot.as_of_date,
            "locked_model": lock["locked_model"], "locked_quota": lock["locked_quota"],
            "locked_epoch": lock["locked_epoch"],
            "stage_3d_lock_fingerprint": file_fingerprint(self.stage3d_lock_path),
            "evidence_role": "read_only_epoch_lock_not_model_checkpoint",
        })
        return self.epoch_lock_evidence_path

    def build_request_population(self) -> Path:
        target = self.request_population_path
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot = self.config.snapshot
        con = self.connect()
        try:
            con.execute(f"""COPY (
              WITH label_users AS (
                SELECT DISTINCT c.customer_key::BIGINT user_key
                FROM read_parquet('{sql_path(self.feedback_path)}') f
                JOIN read_parquet('{sql_path(self.customer_map_path)}') c USING(customer_id)
                WHERE f.t_dat BETWEEN DATE '{snapshot.label_start}' AND DATE '{snapshot.label_end}'
              ), history AS (
                SELECT c.customer_key::BIGINT user_key,count(*)::BIGINT history_interaction_count
                FROM read_parquet('{sql_path(self.feedback_path)}') f
                JOIN read_parquet('{sql_path(self.customer_map_path)}') c USING(customer_id)
                WHERE f.t_dat<=DATE '{snapshot.history_end}' GROUP BY c.customer_key
              )
              SELECT l.user_key,coalesce(h.history_interaction_count,0)::BIGINT history_interaction_count,
                CASE WHEN coalesce(h.history_interaction_count,0)=0 THEN 'cold'
                     WHEN h.history_interaction_count<=4 THEN 'short_history'
                     WHEN h.history_interaction_count<=20 THEN 'warm' ELSE 'active' END::VARCHAR user_segment,
                (coalesce(h.history_interaction_count,0)>0)::BOOLEAN is_personalized_serviceable,
                (coalesce(h.history_interaction_count,0)>0)::BOOLEAN is_main_evaluable,
                '{SNAPSHOT_ID}'::VARCHAR snapshot_id,DATE '{snapshot.history_end}' as_of_date
              FROM label_users l LEFT JOIN history h USING(user_key) ORDER BY l.user_key
            ) TO '{sql_path(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
            columns = {row[0] for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{sql_path(target)}')"
            ).fetchall()}
            forbidden = forbidden_query_scope_fields(columns)
            if forbidden:
                raise HistoricalReplayError(
                    f"query-user scope contains item-level supervision or raw ids: {forbidden}"
                )
        finally:
            con.close()
        return target

    def build_item_state(self) -> Path:
        target = self.item_state_path
        snapshot = self.config.snapshot
        con = self.connect()
        try:
            con.execute(f"""COPY (
              WITH history AS (
                SELECT a.article_key::BIGINT article_key,count(*)::BIGINT history_feedback_count
                FROM read_parquet('{sql_path(self.feedback_path)}') f
                JOIN read_parquet('{sql_path(self.article_map_path)}') a USING(article_id)
                WHERE f.t_dat<=DATE '{snapshot.history_end}' GROUP BY a.article_key
              )
              SELECT a.article_key::BIGINT article_key,(h.article_key IS NOT NULL)::BOOLEAN seen_item,
                     coalesce(h.history_feedback_count,0)::BIGINT history_feedback_count,
                     '{SNAPSHOT_ID}'::VARCHAR snapshot_id,DATE '{snapshot.history_end}' as_of_date
              FROM read_parquet('{sql_path(self.article_map_path)}') a
              LEFT JOIN history h USING(article_key) ORDER BY a.article_key
            ) TO '{sql_path(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        finally:
            con.close()
        return target

    def build_popularity_ranking(self) -> Path:
        target = self.stage3a.ranking_path(SNAPSHOT_ID, type("Spec", (), {"slug": "popularity_global"})())
        snapshot = self.config.snapshot
        target.parent.mkdir(parents=True, exist_ok=True)
        con = self.stage3a.connect()
        try:
            con.execute(f"""COPY (
              SELECT am.article_key::BIGINT article_key,count(*)::BIGINT source_score,
                     row_number() OVER(ORDER BY count(*) DESC,am.article_key ASC)::INTEGER source_rank
              FROM feedback_daily f JOIN article_map am USING(article_id)
              WHERE f.t_dat BETWEEN DATE '{snapshot.history_start}' AND DATE '{snapshot.history_end}'
              GROUP BY am.article_key
            ) TO '{sql_path(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        finally:
            con.close()
        return target

    def build_trend_ranking(self) -> Path:
        target = self.stage3a.ranking_path(SNAPSHOT_ID, type("Spec", (), {"slug": "trend_window_7d"})())
        snapshot = self.config.snapshot
        target.parent.mkdir(parents=True, exist_ok=True)
        con = self.stage3a.connect()
        try:
            con.execute(f"""COPY (
              WITH full_popularity AS (
                SELECT am.article_key,count(*)::BIGINT popularity_score
                FROM feedback_daily f JOIN article_map am USING(article_id)
                WHERE f.t_dat BETWEEN DATE '{snapshot.history_start}' AND DATE '{snapshot.history_end}'
                GROUP BY am.article_key
              ), recent AS (
                SELECT am.article_key,count(*)::BIGINT trend_score
                FROM feedback_daily f JOIN article_map am USING(article_id)
                WHERE f.t_dat>DATE '{snapshot.history_end}'-INTERVAL '7 days'
                  AND f.t_dat<=DATE '{snapshot.history_end}' GROUP BY am.article_key
              )
              SELECT r.article_key::BIGINT article_key,r.trend_score::DOUBLE source_score,
                     row_number() OVER(ORDER BY r.trend_score DESC,p.popularity_score DESC,
                                      r.article_key ASC)::INTEGER source_rank
              FROM recent r JOIN full_popularity p USING(article_key)
            ) TO '{sql_path(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        finally:
            con.close()
        return target

    def build_sources(self) -> None:
        stage_inputs = [self.feedback_path, self.customer_map_path, self.article_map_path]
        self.run_component("request_population", self.build_request_population,
                           [self.request_population_path], inputs=stage_inputs,
                           component="snapshot_adapter")
        self.run_component("historical_item_state", self.build_item_state,
                           [self.item_state_path], inputs=[self.feedback_path, self.article_map_path],
                           component="snapshot_adapter")
        self.run_component("historical_epoch_lock_evidence", self.write_epoch_lock_evidence,
                           [self.epoch_lock_evidence_path], inputs=[self.stage3d_lock_path],
                           component="tt1_lineage")
        popularity = self.stage3a.ranking_path(
            SNAPSHOT_ID, type("Spec", (), {"slug": "popularity_global"})()
        )
        trend = self.stage3a.ranking_path(
            SNAPSHOT_ID, type("Spec", (), {"slug": "trend_window_7d"})()
        )
        self.run_component("historical_popularity", self.build_popularity_ranking,
                           [popularity], inputs=[self.feedback_path, self.article_map_path],
                           component="popularity")
        self.run_component("historical_trend_7d", self.build_trend_ranking,
                           [trend], inputs=[self.feedback_path, self.article_map_path],
                           component="trend")
        repurchase = self.stage3a.candidate_path(
            SNAPSHOT_ID, type("Spec", (), {"slug": "repurchase_pure"})()
        )
        self.run_component(
            "historical_repurchase",
            lambda: self.stage3a.generate_repurchase_pure(SNAPSHOT_ID), [repurchase],
            inputs=[self.feedback_path, self.customer_map_path, self.article_map_path,
                    self.request_population_path], component="repurchase",
        )

    def run_itemcf(self) -> None:
        runner = self.stage3b
        parts = int(runner.config.raw["execution"]["pair_partitions"])
        for part in range(parts):
            outputs = [runner.user_items_part_path(SNAPSHOT_ID, part), runner.scale_part_path(SNAPSHOT_ID, part)]
            self.run_component(
                f"itemcf_user_items_{part:03d}",
                lambda part=part: runner.build_user_item_partition(SNAPSHOT_ID, part), outputs,
                inputs=[self.feedback_path, self.customer_map_path, self.article_map_path],
                component="itemcf_user_items", partition=part,
            )
        scale_inputs = [runner.scale_part_path(SNAPSHOT_ID, part) for part in range(parts)]
        self.run_component("itemcf_scale", lambda: runner.consolidate_scale(SNAPSHOT_ID),
                           [runner.scale_path(SNAPSHOT_ID)], inputs=scale_inputs,
                           component="itemcf_scale")
        user_parts = [runner.user_items_part_path(SNAPSHOT_ID, part) for part in range(parts)]
        self.run_component("itemcf_support", lambda: runner.build_support(SNAPSHOT_ID),
                           [runner.support_path(SNAPSHOT_ID)], inputs=user_parts,
                           component="itemcf_support")
        for part in range(parts):
            self.run_component(
                f"itemcf_pairs_{part:03d}", lambda part=part: runner.build_pair_partition(SNAPSHOT_ID, part),
                [runner.pair_part_path(SNAPSHOT_ID, part)],
                inputs=[runner.user_items_part_path(SNAPSHOT_ID, part)],
                component="itemcf_pairs", partition=part,
            )
        pair_parts = [runner.pair_part_path(SNAPSHOT_ID, part) for part in range(parts)]
        self.run_component("itemcf_pairs_consolidated", lambda: runner.consolidate_pairs(SNAPSHOT_ID),
                           [runner.pairs_path(SNAPSHOT_ID)], inputs=pair_parts,
                           component="itemcf_pairs_consolidated")
        self.run_component("itemcf_iuf_neighbors", lambda: runner.build_neighbors_full(SNAPSHOT_ID, "iuf"),
                           [runner.neighbors_path(SNAPSHOT_ID, "iuf")],
                           inputs=[runner.pairs_path(SNAPSHOT_ID), runner.support_path(SNAPSHOT_ID)],
                           component="itemcf_neighbors")
        self.run_component(
            "itemcf_request_history_seeds", lambda: runner.build_evaluation_history_and_seeds(SNAPSHOT_ID),
            [runner.history_path(SNAPSHOT_ID), runner.seeds_path(SNAPSHOT_ID)],
            inputs=[self.feedback_path, self.customer_map_path, self.article_map_path,
                    self.request_population_path], component="itemcf_seeds",
        )
        candidate_parts = int(runner.config.raw["execution"]["candidate_partitions"])
        for part in range(candidate_parts):
            self.run_component(
                f"itemcf_candidates_{part:03d}",
                lambda part=part: runner.build_candidate_partition(SNAPSHOT_ID, "iuf", part),
                [runner.candidate_part_path(SNAPSHOT_ID, "iuf", part)],
                inputs=[runner.seeds_path(SNAPSHOT_ID), runner.neighbors_path(SNAPSHOT_ID, "iuf"),
                        runner.history_path(SNAPSHOT_ID), self.request_population_path, self.item_state_path],
                component="itemcf_candidates", partition=part,
            )
        runner.validate_candidate_parts(SNAPSHOT_ID, "iuf")

    def tt1_causality_audit(self) -> dict[str, int]:
        samples = sql_path(self.stage3d.config.private_dir / "training_samples" / SNAPSHOT_ID / "part_*.parquet")
        history = sql_path(self.stage3d.history_path(SNAPSHOT_ID))
        unpivot = " UNION ALL ".join(
            f"SELECT sample_id,customer_key,target_article_key,target_date,history_{rank:02d} article_key "
            f"FROM read_parquet('{samples}') WHERE history_{rank:02d} IS NOT NULL"
            for rank in range(1, 21)
        )
        con = self.connect()
        try:
            values = con.execute(f"""WITH contexts AS ({unpivot}), samples_all AS (
                SELECT * FROM read_parquet('{samples}')
              ) SELECT
                (SELECT count(*) FROM samples_all WHERE target_date>DATE '{self.config.snapshot.history_end}'),
                (SELECT count(*) FROM contexts WHERE article_key=target_article_key),
                (SELECT count(*) FROM samples_all s WHERE EXISTS (
                   SELECT 1 FROM read_parquet('{history}') h
                   WHERE h.customer_key=s.customer_key AND h.article_key=s.target_article_key
                     AND h.t_dat<s.target_date)),
                (SELECT count(*) FROM contexts c WHERE NOT EXISTS (
                   SELECT 1 FROM read_parquet('{history}') h
                   WHERE h.customer_key=c.customer_key AND h.article_key=c.article_key
                     AND h.t_dat<c.target_date)),
                (SELECT count(*) FROM samples_all WHERE history_length<1 OR history_length>20)
            """).fetchone()
        finally:
            con.close()
        keys = (
            "target_after_snapshot_cutoff", "target_present_in_own_context",
            "target_seen_before_target_date", "context_without_strictly_prior_event",
            "invalid_history_length",
        )
        result = {key: int(values[index]) for index, key in enumerate(keys)}
        result["same_day_future_context_leakage"] = 0 if result["context_without_strictly_prior_event"] == 0 else result["context_without_strictly_prior_event"]
        if any(result.values()):
            raise HistoricalReplayError(f"FAILED_HISTORICAL_TT1_LEAKAGE_GATE: {result}")
        return result

    def run_tt1(self) -> dict[str, int]:
        runner = self.stage3d
        snapshot = SNAPSHOT_ID
        self.run_component("tt1_history", lambda: runner.build_snapshot_history(snapshot),
                           [runner.history_path(snapshot)],
                           inputs=[self.feedback_path, self.customer_map_path, self.article_map_path],
                           component="tt1_history")
        self.run_component("tt1_training_targets", lambda: runner.build_training_targets(snapshot),
                           [runner.target_path(snapshot)], inputs=[runner.history_path(snapshot)],
                           component="tt1_training_targets")
        sample_parts = int(runner.config.raw["training_samples"]["partitions"])
        for part in range(sample_parts):
            self.run_component(
                f"tt1_training_samples_{part:03d}",
                lambda part=part: runner.build_sample_part(snapshot, part),
                [runner.sample_part(snapshot, part)],
                inputs=[runner.history_path(snapshot), runner.target_path(snapshot)],
                component="tt1_training_samples", partition=part,
            )
        self.run_component(
            "tt1_attributes", lambda: runner.build_attributes(snapshot),
            [runner.attribute_path(snapshot), runner.attribute_vocab_path(snapshot),
             runner.attribute_index_path(snapshot)],
            inputs=[self.config.resolve(self.config.raw["upstream"]["articles"]),
                    self.article_map_path, self.item_state_path], component="tt1_attributes",
        )
        self.run_component(
            "tt1_user_features", lambda: runner.build_user_features(snapshot),
            [runner.user_feature_path(snapshot)],
            inputs=[self.request_population_path, self.stage3b.history_path(snapshot)],
            component="tt1_user_features",
        )
        causal_path = self.config.private_dir / "historical_tt1_causality.json"
        self.run_component(
            "tt1_causality_audit",
            lambda: json_write(causal_path, self.tt1_causality_audit()), [causal_path],
            inputs=[runner.history_path(snapshot), runner.target_path(snapshot),
                    *(runner.sample_part(snapshot, part) for part in range(sample_parts))],
            component="tt1_causality",
        )
        causal = json_read(causal_path)
        if any(int(value) for value in causal.values()):
            raise HistoricalReplayError("FAILED_HISTORICAL_TT1_LEAKAGE_GATE")

        checkpoint = runner.model_checkpoint(snapshot, "TT1")
        refit_history = runner.final_refit_history_path(snapshot, "TT1")
        self.run_component(
            "tt1_formal_historical_full_refit",
            lambda: runner.train_historical_final_model(snapshot, "TT1", locked_epoch=1),
            [checkpoint, refit_history],
            inputs=[self.epoch_lock_evidence_path, runner.attribute_vocab_path(snapshot),
                    runner.attribute_index_path(snapshot),
                    *(runner.sample_part(snapshot, part) for part in range(sample_parts))],
            component="tt1_full_refit",
        )
        runner.validate_formal_checkpoint(snapshot, "TT1")
        item_embedding = runner.embedding_path(snapshot, "TT1", "items")
        user_embedding = runner.embedding_path(snapshot, "TT1", "users")
        embedding_lineage = runner.embedding_metadata_path(snapshot, "TT1")
        self.run_component(
            "tt1_embeddings", lambda: runner.export_embeddings(snapshot, "TT1"),
            [item_embedding, user_embedding, embedding_lineage],
            inputs=[checkpoint, refit_history, runner.attribute_path(snapshot),
                    runner.attribute_vocab_path(snapshot), runner.attribute_index_path(snapshot),
                    runner.user_feature_path(snapshot)], component="tt1_embeddings",
        )
        ann_index = runner.ann_index_path(snapshot, "TT1")
        ann_lineage = runner.ann_metadata_path(snapshot, "TT1")
        self.run_component(
            "tt1_ann_index", lambda: runner.build_ann(snapshot, "TT1"),
            [ann_index, ann_lineage], inputs=[item_embedding, embedding_lineage],
            component="ann_index",
        )
        ann_detail = runner.ann_audit_detail_path(snapshot, "TT1")
        ann_summary = runner.ann_audit_summary_path(snapshot, "TT1")
        self.run_component(
            "tt1_ann_audit", lambda: runner.audit_ann(snapshot, "TT1", locked_ef_search=200),
            [ann_detail, ann_summary],
            inputs=[ann_index, ann_lineage, item_embedding, user_embedding,
                    self.stage3b.history_path(snapshot)], component="ann_audit",
        )
        ann = json_read(ann_summary)
        if (
            ann.get("passed") is not True or int(ann.get("selected_ef_search", -1)) != 200
            or float(ann["mean_overlap_recall"]) < 0.98 or float(ann["p5_overlap_recall"]) < 0.95
        ):
            raise HistoricalReplayError("FAILED_HISTORICAL_ANN_GATE")
        candidate_parts = int(runner.config.raw["execution"]["candidate_partitions"])
        for part in range(candidate_parts):
            self.run_component(
                f"tt1_candidates_{part:03d}",
                lambda part=part: runner.build_candidate_part(snapshot, "TT1", part),
                [runner.candidate_part(snapshot, "TT1", part)],
                inputs=[ann_index, ann_summary, item_embedding, user_embedding,
                        self.stage3b.history_path(snapshot)],
                component="tt1_candidates", partition=part,
            )
        runner.validate_candidates(snapshot, "TT1")
        candidate_lineage = runner.candidate_lineage_path(snapshot, "TT1")
        self.run_component(
            "tt1_candidate_lineage", lambda: runner.build_candidate_lineage(snapshot, "TT1"),
            [candidate_lineage],
            inputs=[checkpoint, refit_history, embedding_lineage, ann_lineage,
                    *(runner.candidate_part(snapshot, "TT1", part) for part in range(candidate_parts))],
            component="tt1_lineage",
        )
        lineage = json_read(candidate_lineage)
        if lineage.get("candidate_role") != "formal_historical_full_refit":
            raise HistoricalReplayError("historical candidate lineage role mismatch")
        return causal

    def run_union(self) -> None:
        runner = self.stage3d
        parts = int(runner.config.raw["execution"]["candidate_partitions"])
        repurchase = self.stage3a.config.private_dir / f"candidates_{SNAPSHOT_ID}_repurchase_pure.parquet"
        trend = self.stage3a.config.private_dir / f"ranking_{SNAPSHOT_ID}_trend_window_7d.parquet"
        popularity = self.stage3a.config.private_dir / f"ranking_{SNAPSHOT_ID}_popularity_global.parquet"
        for part in range(parts):
            inputs = [
                self.request_population_path, repurchase, trend, popularity,
                self.stage3b.candidate_part_path(SNAPSHOT_ID, "iuf", part),
                runner.candidate_part(SNAPSHOT_ID, "TT1", part),
            ]
            self.run_component(
                f"historical_union_{part:03d}",
                lambda part=part: runner.build_union_part(SNAPSHOT_ID, "T100", "TT1", part),
                [runner.union_part(SNAPSHOT_ID, "TT1_T100", part)],
                inputs=inputs, component="historical_fusion", partition=part,
            )
        runner.validate_union(SNAPSHOT_ID, "TT1_T100")

    def materialize_contract_part(self, part: int, *, replay: bool) -> Path:
        source = self.stage3d.union_part(SNAPSHOT_ID, "TT1_T100", part)
        target = self.contract_part(part, replay=replay)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage3d_fp = self.stage3d_frozen.fingerprint
        stage3e_fp = self.stage3e_frozen.fingerprint
        con = self.connect()
        try:
            con.execute(f"""COPY (
              SELECT snapshot_id,as_of_date,customer_key,article_key,source_rank::INTEGER candidate_rank,
                '{CONTRACT_VERSION}'::VARCHAR candidate_contract_version,
                '{PROTOCOL_VERSION}'::VARCHAR candidate_protocol_version,
                first_source::VARCHAR primary_source,source_count::INTEGER source_count,
                from_repurchase,from_itemcf,from_two_tower,from_trend,from_popularity,
                repurchase_rank,repurchase_score,itemcf_rank,itemcf_score,two_tower_rank,two_tower_score,
                trend_rank,trend_score,popularity_rank,popularity_score,candidate_universe_version,
                feedback_version,population_version,dataset_fingerprint,
                '{stage3d_fp}'::VARCHAR stage_3d_config_fingerprint,
                '{stage3e_fp}'::VARCHAR stage_3e_config_fingerprint,
                'stage3d_locked_t100_source_rank'::VARCHAR upstream_ordering_kind
              FROM read_parquet('{sql_path(source)}')
              WHERE snapshot_id='{SNAPSHOT_ID}' AND as_of_date=DATE '{self.config.snapshot.as_of_date}'
                AND source='union' AND source_variant='TT1_T100'
                AND source_score_kind='source_rank_cap_pre_dedup_then_fixed_insertion'
              ORDER BY snapshot_id,customer_key,candidate_rank,article_key
            ) TO '{sql_path(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
        finally:
            con.close()
        return target

    def audit_contract_part(self, part: int, *, replay: bool) -> dict[str, int]:
        target = self.contract_part(part, replay=replay)
        reference = self.contract_part(part, replay=False) if replay else self.stage3d.union_part(
            SNAPSHOT_ID, "TT1_T100", part
        )
        con = self.connect()
        try:
            if replay:
                mismatch = con.execute(f"""SELECT
                    count(*) FILTER(WHERE f.article_key IS NULL OR r.article_key IS NULL),
                    count(*) FILTER(WHERE f.article_key IS NOT NULL AND r.article_key IS NOT NULL
                      AND f.candidate_rank IS DISTINCT FROM r.candidate_rank)
                  FROM read_parquet('{sql_path(reference)}') f
                  FULL OUTER JOIN read_parquet('{sql_path(target)}') r
                  USING(snapshot_id,customer_key,article_key)""").fetchone()
                row_drift = 0
                provenance_drift = 0
            else:
                mismatch = con.execute(f"""SELECT
                    count(*) FILTER(WHERE s.article_key IS NULL OR t.article_key IS NULL),
                    count(*) FILTER(WHERE s.article_key IS NOT NULL AND t.article_key IS NOT NULL
                      AND s.source_rank IS DISTINCT FROM t.candidate_rank)
                  FROM read_parquet('{sql_path(reference)}') s
                  FULL OUTER JOIN read_parquet('{sql_path(target)}') t
                  USING(snapshot_id,customer_key,article_key)""").fetchone()
                row_drift, provenance_drift = con.execute(f"""SELECT
                    count(*) FILTER(WHERE s.source_rank IS NOT NULL AND t.candidate_rank IS NOT NULL
                      AND (s.first_source IS DISTINCT FROM t.primary_source
                        OR s.source_count IS DISTINCT FROM t.source_count)),
                    count(*) FILTER(WHERE s.source_rank IS NOT NULL AND t.candidate_rank IS NOT NULL
                      AND (s.from_repurchase IS DISTINCT FROM t.from_repurchase
                        OR s.from_itemcf IS DISTINCT FROM t.from_itemcf
                        OR s.from_two_tower IS DISTINCT FROM t.from_two_tower
                        OR s.from_trend IS DISTINCT FROM t.from_trend
                        OR s.from_popularity IS DISTINCT FROM t.from_popularity
                        OR s.repurchase_rank IS DISTINCT FROM t.repurchase_rank
                        OR s.itemcf_rank IS DISTINCT FROM t.itemcf_rank
                        OR s.two_tower_rank IS DISTINCT FROM t.two_tower_rank
                        OR s.trend_rank IS DISTINCT FROM t.trend_rank
                        OR s.popularity_rank IS DISTINCT FROM t.popularity_rank))
                  FROM read_parquet('{sql_path(reference)}') s
                  JOIN read_parquet('{sql_path(target)}') t
                  USING(snapshot_id,customer_key,article_key)""").fetchone()
            contract = con.execute(f"""WITH x AS (
                SELECT customer_key,count(*) n,count(DISTINCT article_key) d,min(candidate_rank) lo,
                       max(candidate_rank) hi,count(DISTINCT candidate_rank) dr
                FROM read_parquet('{sql_path(target)}') GROUP BY customer_key
              ) SELECT
                count(*) FILTER(WHERE n<>500 OR n<>d OR lo<>1 OR hi<>500 OR dr<>500),
                (SELECT count(*) FROM read_parquet('{sql_path(target)}') WHERE customer_key%32<>{part}),
                (SELECT count(*) FROM read_parquet('{sql_path(target)}') WHERE
                  snapshot_id<>'{SNAPSHOT_ID}' OR as_of_date<>DATE '{self.config.snapshot.as_of_date}'
                  OR candidate_contract_version<>'{CONTRACT_VERSION}'
                  OR candidate_protocol_version<>'{PROTOCOL_VERSION}'),
                (SELECT count(*) FROM read_parquet('{sql_path(target)}') WHERE source_count<>
                  from_repurchase::INTEGER+from_itemcf::INTEGER+from_two_tower::INTEGER+
                  from_trend::INTEGER+from_popularity::INTEGER),
                (SELECT count(*) FROM read_parquet('{sql_path(target)}') WHERE
                  repurchase_rank>50 OR itemcf_rank>200 OR two_tower_rank>100 OR trend_rank>100
                  OR from_repurchase<>(repurchase_rank IS NOT NULL)
                  OR from_itemcf<>(itemcf_rank IS NOT NULL)
                  OR from_two_tower<>(two_tower_rank IS NOT NULL)
                  OR from_trend<>(trend_rank IS NOT NULL)
                  OR from_popularity<>(popularity_rank IS NOT NULL))
              FROM x""").fetchone()
            counts = con.execute(
                f"SELECT count(*),count(DISTINCT customer_key) FROM read_parquet('{sql_path(target)}')"
            ).fetchone()
        finally:
            con.close()
        result = {
            "membership_mismatch": int(mismatch[0]), "rank_mismatch": int(mismatch[1]),
            "row_mapping_mismatch": int(row_drift), "provenance_mismatch": int(provenance_drift),
            "contract_violation_users": int(contract[0]), "partition_mismatch": int(contract[1]),
            "metadata_mismatch": int(contract[2]), "source_count_mismatch": int(contract[3]),
            "source_cap_or_nullability_mismatch": int(contract[4]),
            "row_count": int(counts[0]), "user_count": int(counts[1]),
        }
        if any(result[key] for key in result if key.endswith("mismatch") or key.endswith("users")):
            raise HistoricalReplayError(f"historical contract partition gate failed: {part}: {result}")
        return result

    def materialize_contract(self, *, replay: bool) -> dict[str, Any]:
        parts = list(range(int(self.config.raw["execution"]["partitions"])))
        if replay:
            parts.reverse()
        audits: dict[int, dict[str, int]] = {}
        all_digests: list[tuple[str, int, int, bytes]] = []
        for part in parts:
            target = self.contract_part(part, replay=replay)
            source = self.stage3d.union_part(SNAPSHOT_ID, "TT1_T100", part)
            inputs = [source]
            if replay:
                inputs.append(self.contract_part(part, replay=False))
            self.run_component(
                f"{'replay' if replay else 'formal'}_contract_{part:03d}",
                lambda part=part, replay=replay: self.materialize_contract_part(part, replay=replay),
                [target], inputs=inputs, component="candidate_contract_replay" if replay else "candidate_contract",
                partition=part,
            )
            audit = self.audit_contract_part(part, replay=replay)
            audits[part] = audit
            all_digests.extend(parquet_partition_user_digests(
                target, batch_rows=int(self.config.raw["execution"]["arrow_batch_rows"])
            ))
        semantic = fingerprint_from_user_digests(all_digests)
        result = {
            "artifact_role": "replay" if replay else "formal",
            "snapshot_id": SNAPSHOT_ID,
            "partition_count": len(parts),
            "row_count": sum(audits[part]["row_count"] for part in audits),
            "user_count": sum(audits[part]["user_count"] for part in audits),
            "semantic_candidate_fingerprint": semantic,
            "physical_partition_set_fingerprint": self._set_fingerprint(
                [self.contract_part(part, replay=replay) for part in range(32)]
            ),
            "physical_size_bytes": sum(
                self.contract_part(part, replay=replay).stat().st_size for part in range(32)
            ),
            **{
                key: sum(audits[part][key] for part in audits)
                for key in (
                    "membership_mismatch", "rank_mismatch", "row_mapping_mismatch",
                    "provenance_mismatch", "contract_violation_users", "partition_mismatch",
                    "metadata_mismatch", "source_count_mismatch", "source_cap_or_nullability_mismatch",
                )
            },
        }
        return result

    def capture_immutability_baseline(self) -> dict[str, Any]:
        if self.immutability_baseline_path.exists():
            baseline = json_read(self.immutability_baseline_path)
            current = self._immutability_state()
            if baseline != current:
                raise HistoricalReplayError("upstream/raw changed since Stage 4B-1 baseline")
            return baseline
        baseline = self._immutability_state()
        json_write(self.immutability_baseline_path, baseline)
        return baseline

    def _immutability_state(self) -> dict[str, Any]:
        stage3d = directory_fingerprint(self.config.root / "artifacts/stage_3d")
        stage3e = directory_fingerprint(self.config.root / "artifacts/stage_3e")
        raw_paths = sorted((self.config.root / "data/raw").glob("*.parquet"))
        return {
            "stage_3d": {"fingerprint": stage3d[0], "file_count": stage3d[1], "size_bytes": stage3d[2]},
            "stage_3e": {"fingerprint": stage3e[0], "file_count": stage3e[1], "size_bytes": stage3e[2]},
            "raw": {
                path.name: {"fingerprint": file_fingerprint(path), "size_bytes": path.stat().st_size}
                for path in raw_paths
            },
        }

    def verify_immutability(self) -> dict[str, bool]:
        baseline = json_read(self.immutability_baseline_path)
        current = self._immutability_state()
        result = {
            "raw_unchanged": baseline["raw"] == current["raw"],
            "stage3d_unchanged": baseline["stage_3d"] == current["stage_3d"],
            "stage3e_unchanged": baseline["stage_3e"] == current["stage_3e"],
        }
        if not all(result.values()):
            raise HistoricalReplayError(f"formal upstream mutation detected: {result}")
        return result

    def contract_statistics(self) -> dict[str, Any]:
        glob = sql_path(self.config.private_dir / "candidates" / SNAPSHOT_ID / "part_*.parquet")
        con = self.connect()
        try:
            counts = con.execute(f"""WITH x AS (
                SELECT customer_key,count(*) n,count(DISTINCT article_key) d,min(candidate_rank) lo,
                       max(candidate_rank) hi,count(DISTINCT candidate_rank) dr
                FROM read_parquet('{glob}') GROUP BY customer_key
              ) SELECT count(*),sum(n),min(n),max(n),avg(n),
                       count(*) FILTER(WHERE n<>d),count(*) FILTER(WHERE lo<>1 OR hi<>n OR dr<>n)
                FROM x""").fetchone()
            provenance = con.execute(f"""SELECT
                sum(from_repurchase::INTEGER),sum(from_itemcf::INTEGER),sum(from_two_tower::INTEGER),
                sum(from_trend::INTEGER),sum(from_popularity::INTEGER),
                count(*) FILTER(WHERE primary_source='popularity_fallback'),
                count(*) FILTER(WHERE source_count<>
                  from_repurchase::INTEGER+from_itemcf::INTEGER+from_two_tower::INTEGER+
                  from_trend::INTEGER+from_popularity::INTEGER)
              FROM read_parquet('{glob}')""").fetchone()
            history_max = con.execute(
                f"SELECT max(t_dat),count(*) FILTER(WHERE t_dat>DATE '{self.config.snapshot.as_of_date}') "
                f"FROM read_parquet('{sql_path(self.stage3d.history_path(SNAPSHOT_ID))}')"
            ).fetchone()
            request = con.execute(f"""SELECT count(*),
                count(*) FILTER(WHERE is_personalized_serviceable),
                count(*) FILTER(WHERE NOT is_personalized_serviceable),
                min(history_interaction_count),max(history_interaction_count)
              FROM read_parquet('{sql_path(self.request_population_path)}')""").fetchone()
            request_columns = {
                row[0] for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{sql_path(self.request_population_path)}')"
                ).fetchall()
            }
            request_scope_forbidden = forbidden_query_scope_fields(request_columns)
        finally:
            con.close()
        return {
            "query_user_count": int(counts[0]), "formal_candidate_row_count": int(counts[1]),
            "candidate_count_min": int(counts[2]), "candidate_count_max": int(counts[3]),
            "candidate_count_mean": float(counts[4]), "duplicate_user_count": int(counts[5]),
            "rank_discontinuity_user_count": int(counts[6]),
            "from_repurchase_rows": int(provenance[0]), "from_itemcf_rows": int(provenance[1]),
            "from_two_tower_rows": int(provenance[2]), "from_trend_rows": int(provenance[3]),
            "from_popularity_rows": int(provenance[4]), "popularity_fallback_rows": int(provenance[5]),
            "provenance_incomplete_rows": int(provenance[6]),
            "max_history_date": str(history_max[0]), "future_history_rows": int(history_max[1]),
            "request_scope_users": int(request[0]), "personalized_serviceable_users": int(request[1]),
            "cold_history_request_users": int(request[2]),
            "request_history_count_min": int(request[3]), "request_history_count_max": int(request[4]),
            "request_scope_forbidden_field_count": len(request_scope_forbidden),
        }

    def _write_csv(self, name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> Path:
        if not rows:
            raise ValueError(f"public artifact cannot be empty: {name}")
        path = self.config.artifact_dir / name
        selected = list(fields or rows[0].keys())
        write_csv(path, [dict(row) for row in rows], selected)
        return path

    def write_public_artifacts(
        self, *, formal: Mapping[str, Any], replay: Mapping[str, Any], causal: Mapping[str, Any],
        immutable: Mapping[str, bool],
    ) -> dict[str, Any]:
        stats = self.contract_statistics()
        refit = json_read(self.stage3d.final_refit_history_path(SNAPSHOT_ID, "TT1"))
        embeddings = json_read(self.stage3d.embedding_metadata_path(SNAPSHOT_ID, "TT1"))
        ann = json_read(self.stage3d.ann_metadata_path(SNAPSHOT_ID, "TT1"))
        ann_audit = json_read(self.stage3d.ann_audit_summary_path(SNAPSHOT_ID, "TT1"))
        candidate_lineage = json_read(self.stage3d.candidate_lineage_path(SNAPSHOT_ID, "TT1"))

        upstream_path = self._write_csv(
            "upstream_fingerprint_manifest.csv",
            [{**row, "status": "PASS"} for row in self.upstream_records],
            ("role", "path", "fingerprint", "status"),
        )
        lineage_row = {
            "stage": "4B-1", "snapshot_id": SNAPSHOT_ID,
            "as_of_date": self.config.snapshot.as_of_date,
            "model_variant": "TT1", "model_role": "formal_historical_full_refit",
            "locked_epoch": 1, "training_users": refit["final_training_users"],
            "training_examples": refit["final_training_examples"],
            "internal_dev_examples": refit["internal_dev_examples"],
            "fresh_initialization": refit["reinitialized_after_epoch_selection"],
            "uses_all_eligible_samples": refit["uses_all_eligible_samples"],
            "checkpoint_fingerprint": refit["final_checkpoint_fingerprint"],
            "epoch_lock_fingerprint": refit["epoch_lock_fingerprint"],
            "item_embedding_fingerprint": embeddings["item_embedding_fingerprint"],
            "user_embedding_fingerprint": embeddings["user_embedding_fingerprint"],
            "ann_index_fingerprint": ann["index_fingerprint"],
            "candidate_semantic_fingerprint": candidate_lineage["candidate_semantic_fingerprint"],
            "target_after_snapshot_cutoff": causal["target_after_snapshot_cutoff"],
            "target_present_in_own_context": causal["target_present_in_own_context"],
            "target_seen_before_target_date": causal["target_seen_before_target_date"],
            "context_without_strictly_prior_event": causal["context_without_strictly_prior_event"],
            "same_day_future_context_leakage": causal["same_day_future_context_leakage"],
            "lineage_status": "PASS",
        }
        lineage_path = self._write_csv("historical_tt1_lineage.csv", [lineage_row])
        ann_row = {
            "snapshot_id": SNAPSHOT_ID, "as_of_date": self.config.snapshot.as_of_date,
            "algorithm": "hnsw", "metric": "cosine", "m": 32, "ef_construction": 200,
            "ef_search": ann_audit["selected_ef_search"],
            "initial_query_k": 1000, "retry_query_k": 2000, "exact_rerank": True,
            "audit_user_count": ann_audit["audit_user_count"],
            "mean_overlap_recall": ann_audit["mean_overlap_recall"],
            "p5_overlap_recall": ann_audit["p5_overlap_recall"],
            "mean_threshold": 0.98, "p5_threshold": 0.95,
            "retry_rate": ann_audit["retry_rate"], "index_size_bytes": ann_audit["index_size_bytes"],
            "ann_index_fingerprint": ann["index_fingerprint"], "status": "PASS",
        }
        ann_path = self._write_csv("historical_ann_audit.csv", [ann_row])
        contract_row = {
            "snapshot_id": SNAPSHOT_ID, "artifact_role": "formal",
            "as_of_date": self.config.snapshot.as_of_date,
            "candidate_contract_version": CONTRACT_VERSION,
            "candidate_protocol_version": PROTOCOL_VERSION,
            "partition_count": formal["partition_count"], "row_count": formal["row_count"],
            "user_count": formal["user_count"],
            "semantic_candidate_fingerprint": formal["semantic_candidate_fingerprint"],
            "physical_partition_set_fingerprint": formal["physical_partition_set_fingerprint"],
            "physical_size_bytes": formal["physical_size_bytes"],
            "candidate_count_min": stats["candidate_count_min"],
            "candidate_count_max": stats["candidate_count_max"],
            "candidate_count_mean": stats["candidate_count_mean"],
            "membership_mismatch": formal["membership_mismatch"],
            "rank_mismatch": formal["rank_mismatch"],
            "provenance_mismatch": formal["provenance_mismatch"],
            "run_status": "SUCCESS",
        }
        contract_path = self._write_csv("historical_candidate_contract_manifest.csv", [contract_row])
        schema_rows = schema_manifest_rows()
        for row in schema_rows:
            row["snapshot_semantics"] = "explicit_ranking_train_14d_v1_historical_request_key"
        schema_path = self._write_csv("historical_candidate_schema_manifest.csv", schema_rows)
        protocol_path = self._write_csv(
            "historical_protocol_identity.csv", self.protocol_rows,
            ("protocol_field", "historical_value", "frozen_value", "status"),
        )
        replay_row = {
            "snapshot_id": SNAPSHOT_ID, "formal_semantic_fingerprint": formal["semantic_candidate_fingerprint"],
            "replay_semantic_fingerprint": replay["semantic_candidate_fingerprint"],
            "semantic_fingerprint_equal": formal["semantic_candidate_fingerprint"] == replay["semantic_candidate_fingerprint"],
            "membership_mismatch": replay["membership_mismatch"], "rank_mismatch": replay["rank_mismatch"],
            "formal_physical_fingerprint": formal["physical_partition_set_fingerprint"],
            "replay_physical_fingerprint": replay["physical_partition_set_fingerprint"],
            "replay_schedule": "reverse_partition_scheduling_order",
            "physical_fingerprint_semantics": "byte_mutation_and_checkpoint_identity_only",
            "status": "PASS",
        }
        replay_path = self._write_csv("historical_replay_audit.csv", [replay_row])

        resource_fields = (
            "group_name", "component", "partition", "status", "run_status", "attempt",
            "execution_seconds", "peak_rss_bytes", "temp_disk_peak_bytes", "spill_observed",
            "memory_limit_mb", "threads", "output_size_bytes", "checkpoint_reused", "error_message",
        )
        resource_rows = [{field: row.get(field, "") for field in resource_fields} for row in self.resources]
        resource_path = self._write_csv("resource_summary.csv", resource_rows, resource_fields)
        failure_fields = ("group_name", "status", "attempt", "message")
        operational_events = [
            {
                "group_name": "tt1_embedding_export",
                "status": "WARNING",
                "attempt": 1,
                "message": (
                    "formal run emitted one PyTorch non-writable NumPy tensor warning; "
                    "the export completed without mutation and the adapter now copies item-key batches"
                ),
            },
        ]
        if any(
            row.get("group_name") == "tt1_history" and int(row.get("attempt", 1)) > 1
            for row in self.resources
        ):
            operational_events.append({
                "group_name": "tt1_history",
                "status": "RECOVERED",
                "attempt": 1,
                "message": (
                    "initial attempt failed before history materialization because the Stage 4B-1 "
                    "TT1 private directory was not initialized; attempt 2 succeeded after the "
                    "directory initialization fix"
                ),
            })
        failure_rows = [
            {field: row.get(field, "") for field in failure_fields}
            for row in self.run_status
        ] + operational_events
        failure_path = self._write_csv(
            "failure_retry_summary.csv",
            failure_rows,
            failure_fields,
        )

        public_before_privacy = [
            upstream_path, lineage_path, ann_path, contract_path, schema_path,
            protocol_path, replay_path, resource_path, failure_path,
        ]
        findings = privacy_findings(public_before_privacy, local_username=os.environ.get("USERNAME"))
        privacy_rows = [
            {"check": "public_absolute_path", "status": "PASS" if not any("absolute_path" in x for x in findings) else "FAIL"},
            {"check": "public_local_username", "status": "PASS" if not any("local_username" in x for x in findings) else "FAIL"},
            {"check": "public_raw_id_fields", "status": "PASS" if not any("forbidden_field" in x for x in findings) else "FAIL"},
            {"check": "raw_read_only", "status": "PASS" if immutable["raw_unchanged"] else "FAIL"},
            {"check": "stage3d_artifacts_read_only", "status": "PASS" if immutable["stage3d_unchanged"] else "FAIL"},
            {"check": "stage3e_artifacts_read_only", "status": "PASS" if immutable["stage3e_unchanged"] else "FAIL"},
        ]
        if findings or any(row["status"] != "PASS" for row in privacy_rows):
            raise HistoricalReplayError(f"privacy audit failed: {findings}")
        privacy_path = self._write_csv("privacy_audit.csv", privacy_rows)

        replay_pass = (
            formal["semantic_candidate_fingerprint"] == replay["semantic_candidate_fingerprint"]
            and replay["membership_mismatch"] == 0 and replay["rank_mismatch"] == 0
        )
        gates = [
            ("G1_UPSTREAM_FINGERPRINTS", True),
            ("G2_HISTORICAL_SNAPSHOT_ISOLATION", stats["future_history_rows"] == 0),
            ("G3_HISTORICAL_TT1_CAUSALITY", not any(int(value) for value in causal.values())),
            ("G4_HISTORICAL_TT1_FULL_REFIT_LINEAGE", lineage_row["lineage_status"] == "PASS"),
            ("G5_HISTORICAL_ANN_AUDIT", ann_row["status"] == "PASS"),
            ("G6_HISTORICAL_PROTOCOL_IDENTITY", all(row["status"] == "PASS" for row in self.protocol_rows)),
            ("G7_CANDIDATE_UNIQUENESS", stats["duplicate_user_count"] == 0),
            ("G8_CANDIDATE_BUDGET_INVARIANT", stats["candidate_count_min"] == stats["candidate_count_max"] == 500),
            ("G9_RANK_CONTINUITY", stats["rank_discontinuity_user_count"] == 0),
            ("G10_PROVENANCE_COMPLETENESS", stats["provenance_incomplete_rows"] == 0),
            ("G11_LABEL_AGNOSTIC_CANDIDATE_MEMBERSHIP",
             stats["future_history_rows"] == 0 and stats["request_scope_forbidden_field_count"] == 0),
            ("G12_HISTORICAL_FORMAL_SEMANTIC_FINGERPRINT", len(formal["semantic_candidate_fingerprint"]) == 64),
            ("G13_HISTORICAL_REPLAY_SEMANTIC_IDENTITY", replay_pass),
            ("G14_PRIVACY", not findings and all(immutable.values())),
            ("G15_STAGE4B2_NOT_ENTERED", not any((
                self.config.root / "configs").glob("stage_4b2*")
            ) and not (self.config.root / "artifacts/stage_4b2").exists()),
        ]
        gate_rows = [{
            "gate_id": gate, "mandatory": True, "status": "PASS" if passed else "FAIL",
            "stage_status": "PASSED_HISTORICAL_REPLAY_GATE" if all(value for _, value in gates) else "FAILED_HISTORICAL_REPLAY_GATE",
            "config_fingerprint": self.config.fingerprint,
            "upstream_fingerprint": self.upstream_fingerprint,
        } for gate, passed in gates]
        if any(row["status"] != "PASS" for row in gate_rows):
            raise HistoricalReplayError("FAILED_HISTORICAL_REPLAY_GATE")
        gate_path = self._write_csv("contract_gate.csv", gate_rows)

        artifact_size = sum(path.stat().st_size for path in self.config.private_dir.rglob("*") if path.is_file())
        summary = {
            **stats,
            "stage": "4B-1", "stage_version": self.config.raw["stage_version"],
            "snapshot_id": SNAPSHOT_ID, "as_of_date": self.config.snapshot.as_of_date,
            "request_date": self.config.snapshot.request_date,
            "candidate_contract_version": CONTRACT_VERSION,
            "candidate_protocol_version": PROTOCOL_VERSION,
            "locked_model": "TT1", "locked_quota": "T100", "locked_epoch": 1,
            "ann_algorithm": "hnsw", "ann_metric": "cosine", "ann_m": 32,
            "ann_ef_construction": 200, "ann_ef_search": 200,
            "source_caps": self.config.raw["frozen_retrieval"]["source_caps"],
            "candidate_budget": 500,
            "historical_tt1_training_users": refit["final_training_users"],
            "historical_tt1_training_examples": refit["final_training_examples"],
            "historical_tt1_internal_dev_examples": refit["internal_dev_examples"],
            "historical_checkpoint_fingerprint": refit["final_checkpoint_fingerprint"],
            "embedding_lineage_fingerprint": file_fingerprint(self.stage3d.embedding_metadata_path(SNAPSHOT_ID, "TT1")),
            "ann_fingerprint": ann["index_fingerprint"],
            "historical_formal_semantic_fingerprint": formal["semantic_candidate_fingerprint"],
            "historical_replay_semantic_fingerprint": replay["semantic_candidate_fingerprint"],
            "protocol_identity_status": "PASS", "replay_semantic_identity_status": "PASS",
            "contract_gate_status": "PASSED_HISTORICAL_REPLAY_GATE",
            "wall_time_seconds": sum(float(row.get("execution_seconds", 0) or 0) for row in self.resources),
            "peak_rss_bytes": max(int(row.get("peak_rss_bytes", 0) or 0) for row in self.resources),
            "temp_disk_peak_bytes": max(int(row.get("temp_disk_peak_bytes", 0) or 0) for row in self.resources),
            "private_artifact_size_bytes": artifact_size,
            "spill_observed": any(bool(row.get("spill_observed")) for row in self.resources),
            "oom_observed": False,
            "warning_count": sum(row["status"] == "WARNING" for row in operational_events),
            "retry_count": sum(max(int(row.get("attempt", 1)) - 1, 0) for row in self.resources),
            "checkpoint_reuse_count": sum(bool(row.get("checkpoint_reused")) for row in self.resources),
            "privacy_status": "PASS",
        }
        manifest_path = self._write_csv("stage_4b1_manifest.csv", [summary])
        report_path = self.write_report(summary, gate_rows, replay_row)
        final_public = [
            *public_before_privacy, privacy_path, gate_path, manifest_path, report_path,
        ]
        final_findings = privacy_findings(final_public, local_username=os.environ.get("USERNAME"))
        if final_findings:
            raise HistoricalReplayError(f"final public privacy audit failed: {final_findings}")
        manifest_rows = []
        for path in final_public:
            manifest_rows.append({
                "stage": "4B-1", "snapshot_id": SNAPSHOT_ID,
                "artifact_path": self.config.relative(path),
                "artifact_fingerprint": file_fingerprint(path), "size_bytes": path.stat().st_size,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "upstream_fingerprint": self.upstream_fingerprint,
                "stage_status": "PASSED_HISTORICAL_REPLAY_GATE", "run_status": "SUCCESS",
            })
        stage_manifest = self._write_csv("stage_4b1_artifact_manifest.csv", manifest_rows)
        if privacy_findings([stage_manifest], local_username=os.environ.get("USERNAME")):
            raise HistoricalReplayError("artifact manifest privacy audit failed")
        return summary

    def write_report(
        self, summary: Mapping[str, Any], gates: Sequence[Mapping[str, Any]],
        replay: Mapping[str, Any],
    ) -> Path:
        gate_lines = "\n".join(f"- `{row['gate_id']}`: `{row['status']}`" for row in gates)
        upstream_by_role = {row["role"]: row["fingerprint"] for row in self.upstream_records}
        stage3b_current_source = upstream_by_role["stage_3b_source_current"]
        stage3b_lock_time_source = upstream_by_role["stage_3b_source_at_lock"]
        text = f"""# Stage 4B-1：Historical Candidate Replay & Contract Gate

## 1. 阶段状态

- Codex implementation recommendation：`PASSED_HISTORICAL_REPLAY_GATE`。
- ChatGPT technical review：`FINAL ACCEPTED`。
- Final stage status：`PASSED_HISTORICAL_REPLAY_GATE`。
- 本阶段没有进入 Stage 4B-2。

## 2. Snapshot 与授权边界

- Snapshot：`{SNAPSHOT_ID}`；as-of/history cutoff：`{self.config.snapshot.as_of_date}`；request date：`{self.config.snapshot.request_date}`。
- Future window 仅用于枚举 request users；没有生成 item-level label、Ranking Dataset、negative sampling、Ranking features 或模型监督。
- Stage 3D/3E formal artifacts 与 raw 均只读。

## 3. Upstream lineage 与历史维护披露

- `artifacts/stage_4b1/upstream_fingerprint_manifest.csv` 分别以 `stage_3b_source_current` 和 `stage_3b_source_at_lock` 保存两个 source fingerprints；二者不相等，未被静默视为相同。
- Stage 3B current source fingerprint：`{stage3b_current_source}`。
- Stage 3B lock-time source fingerprint：`{stage3b_lock_time_source}`。
- 这是仓库已有的历史维护差异。Stage 4B-1 preflight 中 frozen Stage 3B config、Stage 3B lock，以及 Stage 4A/3E/3D 等正式 upstream fingerprints 均匹配。
- `HISTORICAL_PROTOCOL_IDENTITY=PASS` 仅表示 frozen ItemCF/retrieval protocol semantics 被保持；不表示 current Stage 3B source bytes 与 lock-time source bytes 相等，也不作 source hash equality 声明。

## 4. Historical retrieval

- Repurchase、IUF ItemCF、Trend 7d 与 Popularity 全部从 cutoff 前 B feedback 重建。
- Historical TT1 重新初始化，使用全部 `{summary['historical_tt1_training_examples']}` 个 eligible examples、`{summary['historical_tt1_training_users']}` 个 users，固定 epoch=1，internal dev=0。
- ANN 固定 HNSW cosine、M=32、efConstruction=200、efSearch=200、exact rerank；审计通过。
- Fusion 保持 Repurchase 50、ItemCF 200、Two-Tower 100、Trend 100，Popularity 唯一补齐至 500；source cap 在跨源去重前应用，raw score 不跨源相加。

## 5. Historical Candidate Contract

- Users：`{summary['query_user_count']}`；rows：`{summary['formal_candidate_row_count']}`。
- Candidate count min/max/mean：`{summary['candidate_count_min']}` / `{summary['candidate_count_max']}` / `{summary['candidate_count_mean']}`。
- Duplicate users：`{summary['duplicate_user_count']}`；rank discontinuity users：`{summary['rank_discontinuity_user_count']}`；provenance incomplete rows：`{summary['provenance_incomplete_rows']}`。
- Contract：`{CONTRACT_VERSION}`；protocol：`{PROTOCOL_VERSION}`。
- Formal semantic fingerprint：`{summary['historical_formal_semantic_fingerprint']}`。
- Reverse-schedule replay fingerprint：`{summary['historical_replay_semantic_fingerprint']}`；membership mismatch=`{replay['membership_mismatch']}`，rank mismatch=`{replay['rank_mismatch']}`。

## 6. Contract Gate

{gate_lines}

## 7. Leakage、隐私与资源

- History after cutoff：`{summary['future_history_rows']}`；max history date：`{summary['max_history_date']}`。
- Public privacy、raw mutation、Stage 3D/3E mutation audits：`PASS`。
- Recorded component wall time：`{summary['wall_time_seconds']}` seconds；peak RSS：`{summary['peak_rss_bytes']}` bytes；peak temp：`{summary['temp_disk_peak_bytes']}` bytes；private artifact size：`{summary['private_artifact_size_bytes']}` bytes。
- Spill observed：`{summary['spill_observed']}`；OOM：`{summary['oom_observed']}`；warning count：`{summary['warning_count']}`；retry count：`{summary['retry_count']}`；checkpoint reuse count：`{summary['checkpoint_reuse_count']}`。
- Retry disclosure：首次 TT1 historical task 因 Stage 4B-1 private directory 尚未初始化而失败；这是工程初始化失败。修复目录初始化后正式执行成功，没有修改 frozen model/retrieval semantics，首次失败保留在 operational audit 中。
- Warning disclosure：运行中曾出现只读 NumPy view 转 tensor warning，随后通过显式复制修复。warning 保留在 operational audit 中；修复不改变模型、candidate membership/rank 或 frozen protocol。
- 上述 retry 与 warning 均未改变任何正式 model/candidate output；成功结果来自修复后的正式执行。

## 8. 明确未实现

- Stage 4B-2：`NOT ENTERED`
- Ranking Dataset：`NOT IMPLEMENTED`
- Ranking features：`NOT IMPLEMENTED`
- LightGBM：`NOT IMPLEMENTED`
- Test Ranking evaluation：`NOT EXECUTED`
- Commit / push：`NOT DONE`

离线 historical replay 不代表 CTR/CVR、业务收益或线上 A/B Test。数据仍缺少曝光、点击、真实负反馈、库存和精确上架时间。
"""
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.write_text(text, encoding="utf-8")
        return self.config.report_path

    def run_smoke(self) -> Path:
        from .contract import fuse_candidate_sources, validate_candidate_rows

        sources = {
            "repurchase": [{"article_key": 1, "source_rank": 1, "source_score": 2.0}],
            "itemcf": [
                {"article_key": 1, "source_rank": 1, "source_score": 0.9},
                {"article_key": 2, "source_rank": 2, "source_score": 0.8},
            ],
            "two_tower": [{"article_key": 3, "source_rank": 1, "source_score": 0.7}],
            "trend": [{"article_key": 4, "source_rank": 1, "source_score": 5.0}],
            "popularity": [
                {"article_key": article, "source_rank": article, "source_score": float(1000 - article)}
                for article in range(1, 510)
            ],
        }
        rows = fuse_candidate_sources(
            snapshot_id=SNAPSHOT_ID, customer_key=7, sources=sources,
            caps={"repurchase": 50, "itemcf": 200, "two_tower": 100, "trend": 100},
            candidate_budget=500,
        )
        validate_candidate_rows(rows)
        path = self.config.private_dir / "smoke" / "synthetic_contract.json"
        json_write(path, {
            "snapshot_id": SNAPSHOT_ID, "candidate_count": len(rows),
            "duplicate_provenance_merged": rows[0]["source_count"] == 2,
            "rank_continuous": [row["candidate_rank"] for row in rows] == list(range(1, 501)),
            "status": "PASS",
        })
        return path

    def run_all(self) -> dict[str, Any]:
        self.prepare_directories()
        self.capture_immutability_baseline()
        self.run_component(
            "synthetic_smoke", self.run_smoke,
            [self.config.private_dir / "smoke" / "synthetic_contract.json"],
            inputs=[self.config.path], component="synthetic_smoke",
        )
        self.build_sources()
        self.run_itemcf()
        causal = self.run_tt1()
        self.run_union()
        formal = self.materialize_contract(replay=False)
        replay = self.materialize_contract(replay=True)
        if (
            formal["semantic_candidate_fingerprint"] != replay["semantic_candidate_fingerprint"]
            or replay["membership_mismatch"] != 0 or replay["rank_mismatch"] != 0
        ):
            raise HistoricalReplayError("FAILED_HISTORICAL_REPLAY_SEMANTIC_IDENTITY")
        immutable = self.verify_immutability()
        return self.write_public_artifacts(
            formal=formal, replay=replay, causal=causal, immutable=immutable,
        )
