from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.stage3d.ann import build_hnsw_index
from retrieval.stage3d.config import Stage3DConfig
from retrieval.stage3d.data import recency_weight
from retrieval.stage3d.runner import Stage3DRunner, json_write, semantic_partition_fingerprint


class SelectionMetricRecoveryRunner(Stage3DRunner):
    @property
    def recovery_root(self) -> Path:
        return self.config.temp_dir / "selection_checkpoint_metric_recovery"

    def embedding_path(self, snapshot: str, variant: str, kind: str) -> Path:
        return self.recovery_root / "embeddings" / f"{variant}_{kind}.npz"

    def embedding_metadata_path(self, snapshot: str, variant: str) -> Path:
        return self.recovery_root / "embeddings" / f"{variant}_lineage.json"

    def ann_index_path(self, snapshot: str, variant: str) -> Path:
        return self.recovery_root / "ann" / f"{variant}.hnsw"

    def ann_metadata_path(self, snapshot: str, variant: str) -> Path:
        return self.recovery_root / "ann" / f"{variant}.json"

    def ann_audit_summary_path(self, snapshot: str, variant: str) -> Path:
        return self.recovery_root / "ann" / f"{variant}_audit.json"

    def candidate_part(self, snapshot: str, variant: str, part: int) -> Path:
        return self.recovery_root / "candidates" / variant / f"part_{part:03d}.parquet"

    def candidate_glob(self, snapshot: str, variant: str) -> str:
        return (self.recovery_root / "candidates" / variant / "part_*.parquet").as_posix()

    def user_metric_path(self, snapshot: str, source: str, variant: str) -> Path:
        return self.recovery_root / f"user_metrics_{source}_{variant}.parquet"


def export_selection_embeddings(runner: SelectionMetricRecoveryRunner) -> tuple[Path, Path]:
    snapshot, variant = "validation_14d", "TT1"
    model = runner._model(snapshot, variant).cpu()
    checkpoint = runner.selection_model_checkpoint(snapshot, variant)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    item_keys = np.asarray(
        pq.read_table(runner.attribute_path(snapshot), columns=["article_key"])["article_key"]
    ).astype(np.int64, copy=True)
    batch_size = int(runner.config.raw["execution"]["embedding_batch_size"])
    item_vectors = []
    with torch.inference_mode():
        for start in range(0, len(item_keys), batch_size):
            item_vectors.append(model.encode_items(torch.from_numpy(item_keys[start:start + batch_size])).numpy())
    item_target = runner.embedding_path(snapshot, variant, "items")
    item_target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(item_target, article_keys=item_keys,
             embeddings=np.concatenate(item_vectors).astype(np.float32, copy=False))

    columns = ["customer_key", "user_segment", *(f"history_{rank:02d}" for rank in range(1, 21))]
    weights = np.asarray([recency_weight(rank) for rank in range(1, 21)], dtype=np.float32)
    user_keys_parts, segment_parts, vector_parts = [], [], []
    with torch.inference_mode():
        for batch in pq.ParquetFile(runner.user_feature_path(snapshot)).iter_batches(
                batch_size=batch_size, columns=columns):
            user_keys = runner._batch_numpy(batch, "customer_key").astype(np.int64, copy=True)
            histories = np.column_stack([
                runner._batch_numpy(batch, f"history_{rank:02d}") for rank in range(1, 21)
            ]).astype(np.int64, copy=True)
            history_weights = (histories >= 0).astype(np.float32) * weights[None, :]
            vectors = model.encode_users(
                torch.from_numpy(user_keys), torch.from_numpy(histories),
                torch.from_numpy(history_weights),
            ).numpy()
            user_keys_parts.append(user_keys)
            segment_parts.append(np.asarray(
                batch.column(batch.schema.get_field_index("user_segment")).to_pylist(), dtype="U16"
            ))
            vector_parts.append(vectors)
    user_target = runner.embedding_path(snapshot, variant, "users")
    np.savez(user_target, customer_keys=np.concatenate(user_keys_parts),
             segments=np.concatenate(segment_parts),
             embeddings=np.concatenate(vector_parts).astype(np.float32))
    json_write(runner.embedding_metadata_path(snapshot, variant), {
        "audit_role": "selection_only_metric_recovery",
        "source_model_checkpoint_fingerprint": file_fingerprint(checkpoint),
        "item_embedding_fingerprint": file_fingerprint(item_target),
        "user_embedding_fingerprint": file_fingerprint(user_target),
    })
    return item_target, user_target


def main() -> int:
    runner = SelectionMetricRecoveryRunner(
        Stage3DConfig.load(Path("configs/stage_3d_two_tower.yaml"))
    )
    snapshot, variant = "validation_14d", "TT1"
    runner.prepare_directories()
    export_selection_embeddings(runner)
    items = np.load(runner.embedding_path(snapshot, variant, "items"))
    index_path = runner.ann_index_path(snapshot, variant)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    build_hnsw_index(
        items["embeddings"], index_path,
        m=int(runner.config.raw["ann"]["m"]),
        ef_construction=int(runner.config.raw["ann"]["ef_construction"]),
        seed=int(runner.config.raw["training"]["seed"]),
    )
    json_write(runner.ann_audit_summary_path(snapshot, variant), {
        "selected_ef_search": 200,
    })
    for part in range(int(runner.config.raw["execution"]["candidate_partitions"])):
        runner.build_candidate_part(snapshot, variant, part)
    runner.validate_candidates(snapshot, variant)
    metric_path = runner.evaluate(
        snapshot, candidate_glob=runner.candidate_glob(snapshot, variant),
        source="two_tower_selection_recovery", variant=variant,
    )
    summary = runner.metric_summary(snapshot, "two_tower_selection_recovery", variant, metric_path)
    recall = next(row["metric_value"] for row in summary
                  if row["segment"] == "evaluable_warm_users" and row["metric"] == "recall_at_500")
    output = runner.config.artifact_dir / "selection_checkpoint_metric_recovery.csv"
    row = {
        "stage": "3D", "phase": "validation", "snapshot_id": snapshot,
        "model_variant": variant, "metric": "recall_at_500", "metric_value": recall,
        "checkpoint_role": "model_selection_only",
        "checkpoint_fingerprint": file_fingerprint(runner.selection_model_checkpoint(snapshot, variant)),
        "candidate_semantic_fingerprint": semantic_partition_fingerprint([
            runner.candidate_part(snapshot, variant, part)
            for part in range(int(runner.config.raw["execution"]["candidate_partitions"]))
        ]),
        "audit_scope": "recovered_for_before_after_reporting_not_formal_candidate_source",
        "run_status": "SUCCESS",
    }
    write_csv(output, [row], row.keys())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
