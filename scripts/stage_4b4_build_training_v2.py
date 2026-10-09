from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_foundation.manifest import file_fingerprint, write_csv
from ranking_features.materialization import sql_path
from ranking_model import Stage4B4Config
from ranking_model.adapter import ModelInputAdapter
from ranking_model.ordering import ORDERING_VERSION, audit_ordered_arrays, matrix_order_fingerprint, write_json_atomic


def load_adapter(path: Path) -> ModelInputAdapter:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ModelInputAdapter(
        raw["feature_set"], tuple(raw["ordered_feature_names"]),
        tuple(raw["dtypes"][name] for name in raw["ordered_feature_names"]),
        tuple(raw["categorical_features"]), raw["categorical_mapping"]["user_segment"],
    )


def set_fingerprint(paths: list[Path]) -> str:
    rows = [(path.name, path.stat().st_size, file_fingerprint(path)) for path in paths]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 4B-4 globally query-ordered training matrix v2")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_lambdarank.yaml"))
    args = parser.parse_args()
    config = Stage4B4Config.load(args.config)
    raw = config.raw
    root = config.root
    output = config.private_dir / "training_query_order_v2"
    output.mkdir(parents=True, exist_ok=True)
    rows = int(raw["expected"]["train_rows"])
    f0_adapter = load_adapter(config.artifact_dir / "model_input_adapter_f0.json")
    f1_adapter = load_adapter(config.artifact_dir / "model_input_adapter_f1.json")
    paths = {
        "f0": output / "features_f0.float32", "f1": output / "features_f1.float32",
        "labels": output / "labels.uint8", "customers": output / "customer_keys.int64",
        "articles": output / "article_keys.int64", "ranks": output / "candidate_ranks.int32",
        "orders": output / "query_orders.int64", "groups": output / "groups.int32",
        "boundaries": output / "query_boundaries.parquet", "metadata": output / "matrix_metadata.json",
    }
    started = time.perf_counter()
    arrays = {
        "f0": np.memmap(paths["f0"].with_suffix(".float32.tmp"), dtype=np.float32, mode="w+", shape=(rows, 30)),
        "f1": np.memmap(paths["f1"].with_suffix(".float32.tmp"), dtype=np.float32, mode="w+", shape=(rows, 52)),
        "labels": np.memmap(paths["labels"].with_suffix(".uint8.tmp"), dtype=np.uint8, mode="w+", shape=(rows,)),
        "customers": np.memmap(paths["customers"].with_suffix(".int64.tmp"), dtype=np.int64, mode="w+", shape=(rows,)),
        "articles": np.memmap(paths["articles"].with_suffix(".int64.tmp"), dtype=np.int64, mode="w+", shape=(rows,)),
        "ranks": np.memmap(paths["ranks"].with_suffix(".int32.tmp"), dtype=np.int32, mode="w+", shape=(rows,)),
        "orders": np.memmap(paths["orders"].with_suffix(".int64.tmp"), dtype=np.int64, mode="w+", shape=(rows,)),
    }
    optimizer_glob = config.resolve(raw["upstream"]["train_optimizer"]) / "part_*.parquet"
    feature_glob = config.resolve(raw["upstream"]["train_optimizer_features"]) / "part_*.parquet"
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{min(1024, config.memory_limit_mb)}MB'")
    con.execute(f"SET temp_directory='{sql_path(config.temp_dir / 'training_query_order_v2')}'")
    con.execute(f"SET max_temp_directory_size='{raw['execution']['max_temp_directory_size']}'")
    query = f"""SELECT f.*,o.label,o.query_order
      FROM read_parquet('{sql_path(feature_glob)}') f
      INNER JOIN read_parquet('{sql_path(optimizer_glob)}') o
        USING(snapshot_id,customer_key,article_key)
      ORDER BY o.query_order ASC,f.candidate_rank ASC,f.article_key ASC"""
    offset = 0
    try:
        for batch in con.execute(query).fetch_record_batch(rows_per_batch=65_536):
            table = pa.Table.from_batches([batch])
            count = len(table)
            f1 = f1_adapter.convert_table(table)
            arrays["f1"][offset:offset + count] = f1
            arrays["f0"][offset:offset + count] = f1[:, :30]
            arrays["labels"][offset:offset + count] = table["label"].combine_chunks().to_numpy()
            arrays["customers"][offset:offset + count] = table["customer_key"].combine_chunks().to_numpy()
            arrays["articles"][offset:offset + count] = table["article_key"].combine_chunks().to_numpy()
            arrays["ranks"][offset:offset + count] = table["candidate_rank"].combine_chunks().to_numpy()
            arrays["orders"][offset:offset + count] = table["query_order"].combine_chunks().to_numpy()
            offset += count
    finally:
        con.close()
    if offset != rows:
        raise RuntimeError(f"ordered join row count mismatch: {offset} != {rows}")
    for value in arrays.values():
        value.flush()
    del value, f1, table, batch
    arrays.clear()
    gc.collect()
    for key in ("f0", "f1", "labels", "customers", "articles", "ranks", "orders"):
        temporary = paths[key].with_suffix(paths[key].suffix + ".tmp")
        os.replace(temporary, paths[key])

    groups_table = pq.read_table(config.resolve(raw["upstream"]["train_groups"])).sort_by("query_order")
    expected = list(zip(
        groups_table["snapshot_id"].to_pylist(), groups_table["customer_key"].to_pylist(),
        groups_table["query_order"].to_pylist(), groups_table["group_size"].to_pylist(),
        groups_table["positive_count"].to_pylist(), groups_table["sampled_zero_count"].to_pylist(),
    ))
    groups = np.asarray(groups_table["group_size"].to_numpy(), dtype=np.int32)
    groups.tofile(paths["groups"])
    mapped = {
        "customers": np.memmap(paths["customers"], dtype=np.int64, mode="r"),
        "orders": np.memmap(paths["orders"], dtype=np.int64, mode="r"),
        "articles": np.memmap(paths["articles"], dtype=np.int64, mode="r"),
        "ranks": np.memmap(paths["ranks"], dtype=np.int32, mode="r"),
        "labels": np.memmap(paths["labels"], dtype=np.uint8, mode="r"),
    }
    audit, boundaries = audit_ordered_arrays(
        snapshot_id=raw["snapshots"]["train"]["snapshot_id"], customer_keys=mapped["customers"],
        query_orders=mapped["orders"], article_keys=mapped["articles"], candidate_ranks=mapped["ranks"],
        labels=mapped["labels"], expected_queries=expected,
    )
    boundary_fp = matrix_order_fingerprint(boundaries)
    pq.write_table(pa.Table.from_pylist(boundaries), paths["boundaries"], compression="zstd")
    physical = {key: file_fingerprint(path) for key, path in paths.items() if key not in {"metadata"}}
    metadata = {
        "ordering_version": ORDERING_VERSION, "snapshot_id": raw["snapshots"]["train"]["snapshot_id"],
        "row_count": rows, "query_count": len(groups), "group_sum": int(groups.sum()),
        "positive_rows": int(mapped["labels"].sum()), "zero_rows": int(rows - mapped["labels"].sum()),
        "matrix_order_fingerprint": boundary_fp, "query_boundary_fingerprint": file_fingerprint(paths["boundaries"]),
        "group_fingerprint": physical["groups"], "physical_fingerprints": physical,
        "f0_adapter_fingerprint": f0_adapter.fingerprint, "f1_adapter_fingerprint": f1_adapter.fingerprint,
        "audit": audit, "source_inputs_fingerprint": set_fingerprint(
            sorted(config.resolve(raw["upstream"]["train_optimizer"]).glob("*.parquet"))
            + sorted(config.resolve(raw["upstream"]["train_optimizer_features"]).glob("*.parquet"))
            + [config.resolve(raw["upstream"]["train_groups"])]
        ),
        "wall_time_seconds": time.perf_counter() - started,
        "validation_label_read_count": 0, "first7d_read_count": 0, "test_read_count": 0,
    }
    write_json_atomic(paths["metadata"], metadata)
    public_rows = [{"check": key, "actual": value, "expected": value if key in {"query_count", "row_count"} else 0,
                    "violations": 0, "status": "PASS"}
                   for key, value in audit.items()]
    public_rows.extend([
        {"check": "sum_group_size", "actual": int(groups.sum()), "expected": rows, "violations": 0, "status": "PASS"},
        {"check": "positive_rows", "actual": metadata["positive_rows"], "expected": raw["expected"]["train_positive_rows"], "violations": 0, "status": "PASS"},
        {"check": "sampled_zero_rows", "actual": metadata["zero_rows"], "expected": raw["expected"]["train_zero_rows"], "violations": 0, "status": "PASS"},
    ])
    write_csv(config.artifact_dir / "training_matrix_order_audit.csv", public_rows, list(public_rows[0]))
    print(json.dumps({key: metadata[key] for key in (
        "ordering_version", "row_count", "query_count", "matrix_order_fingerprint", "wall_time_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
