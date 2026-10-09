from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from data_foundation.item_state import popularity_bucket, popularity_thresholds
from ranking_features.materialization import sql_path
from ranking_model.validation_selection import canonical_fingerprint, file_fingerprint

from .config import Config


def build_metadata(config: Config) -> tuple[pa.Table, dict]:
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit='256MB'")
    con.execute("SET threads=1")
    con.execute("SET temp_directory=''")
    try:
        support_path = sql_path(config.upstream("support"))
        invalid = con.execute(f"SELECT count(*) FROM read_parquet('{support_path}') WHERE interaction_count IS NULL OR interaction_count<=0 OR latest_date>DATE '2020-08-25'").fetchone()[0]
        if invalid:
            raise ValueError("support cutoff/count contract violation")
        support = con.execute(f"SELECT article_key,interaction_count FROM read_parquet('{support_path}') ORDER BY article_key").fetchall()
        if len({key for key, _ in support}) != len(support):
            raise ValueError("duplicate history support keys")
        p50, p90 = popularity_thresholds(count for _, count in support)
        table = con.execute(f"""SELECT m.article_key,a.product_code,g.article_garment_group_code,
            coalesce(s.interaction_count,0)::BIGINT history_support
            FROM read_parquet('{sql_path(config.upstream('article_key_map'))}') m
            LEFT JOIN read_parquet('{sql_path(config.upstream('articles'))}') a USING(article_id)
            LEFT JOIN read_parquet('{sql_path(config.upstream('canonical_articles'))}') g USING(article_key)
            LEFT JOIN read_parquet('{support_path}') s USING(article_key)
            ORDER BY m.article_key""").fetch_arrow_table()
    finally:
        con.close()
    keys = table["article_key"].to_pylist()
    if len(set(keys)) != len(table) or table["product_code"].null_count or table["article_garment_group_code"].null_count:
        raise ValueError("metadata join expansion/missing attributes")
    if len(table) != pq.ParquetFile(config.upstream("article_key_map")).metadata.num_rows:
        raise ValueError("metadata join lost rows")
    buckets = [popularity_bucket(int(count), p50, p90) for count in table["history_support"].to_pylist()]
    table = table.append_column("popularity_bucket", pa.array(buckets))
    table = table.append_column("is_tail_item", pa.array([bucket == "tail" for bucket in buckets]))
    sources = {name: file_fingerprint(config.upstream(name)) for name in ("support", "articles", "article_key_map", "canonical_articles")}
    manifest = {"snapshot_id": config.raw["snapshot_id"], "as_of_date": config.raw["as_of_date"],
                "row_count": len(table), "seen_item_count": len(support), "p50": p50, "p90": p90,
                "source_fingerprints": sources, "tail_helper_fingerprint": file_fingerprint(config.root / "src/data_foundation/item_state.py"),
                "missing_metadata_rows": 0, "join_row_expansion": 0,
                "tail_semantics": "Stage 5 tail semantics are identical to the existing project-level history-safe item-state definition.",
                "static_attribute_effective_timestamp_verified": False}
    manifest["metadata_binding_fingerprint"] = canonical_fingerprint(manifest)
    return table, manifest


def metadata_lookup(path: Path) -> dict[int, tuple]:
    t = pq.read_table(path)
    return {key: (product, garment, support, tail) for key, product, garment, support, tail in zip(
        *(t[name].to_pylist() for name in ("article_key", "product_code", "article_garment_group_code", "history_support", "is_tail_item"))
    )}
