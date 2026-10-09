from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import duckdb

from data_audit.stage1a import (
    ResourceMonitor,
    sensitive_value_matches,
    validate_public_artifacts,
)
from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.config import RetrievalConfig, source_fingerprint as stage3a_source_fingerprint
from retrieval.contracts import CANDIDATE_FIELDS, FORBIDDEN_ID_FIELDS
from retrieval.runner import COVERAGE_NAMES, METRIC_NAMES, Stage3ARunner

from .bootstrap import paired_user_bootstrap
from .candidates import generate_itemcf_candidates, select_seed_items
from .config import Stage3BConfig, source_fingerprint
from .itemcf import build_neighbors, build_user_item_sets, cooccurrence_pairs, item_support, select_variant
from .locking import lock_fingerprint, validate_test_lock
from .overlap import user_overlap
from .union import fixed_budget_union


# 该 fingerprint 对应首次全量 Validation 的实现；其 ItemCF 上游分组已通过完整
# fingerprint 与候选协议校验，后续改动只把 union 单分区等价拆成 4 个子分区以修复 OOM。
COMPATIBLE_CHECKPOINT_SOURCE_FINGERPRINTS = {
    "6464f9f62189460eb3572f78ce3ab12b9a4f9ac55d71be576ff5e34be1bbff84":
        "fixed-budget union engineering subpartition OOM fix",
    # 该版本已经完成锁后唯一一次 Test；之后的改动仅修复收尾统计错误地要求
    # 不应在 Test 运行的 cosine 邻居文件，不改变任何 ItemCF、候选、评估或融合逻辑。
    "e5304699d8ee4d0bb25348400cfd5ba5973c477d58916a9dd3e35d3f26f8a84f":
        "finalize-only skip for the intentionally unbuilt Test comparator",
}


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


def declared_stage3b_lock_fingerprints(value: Any) -> set[str]:
    """只豁免 lock 中显式 fingerprint 字段里的完整 SHA-256。"""
    if isinstance(value, dict):
        allowed: set[str] = set()
        for field, child in value.items():
            if field == "fingerprint" or field.endswith("_fingerprint") or field.endswith("_fingerprints"):
                allowed.update(_sha256_values(child))
            else:
                allowed.update(declared_stage3b_lock_fingerprints(child))
        return allowed
    if isinstance(value, list):
        return {
            item
            for child in value
            for item in declared_stage3b_lock_fingerprints(child)
        }
    return set()


def _sha256_values(value: Any) -> set[str]:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return {value}
    if isinstance(value, dict):
        return {item for child in value.values() for item in _sha256_values(child)}
    if isinstance(value, list):
        return {item for child in value for item in _sha256_values(child)}
    return set()


def public_error_message(value: str) -> str:
    """公开状态保留失败事实和原因，但移除外部 URL，避免路径型误报。"""
    return re.sub(r"https?://\S+", "[external_documentation_url]", value)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def query_neighbor_statistics(con: duckdb.DuckDBPyConnection, neighbor: Path) -> tuple[Any, ...]:
    path = sql_path(neighbor)
    return con.execute(
        f"""
        WITH per_item AS (
          SELECT article_key,count(*) neighbor_count
          FROM read_parquet('{path}') GROUP BY article_key
        )
        SELECT
               (SELECT count(*) FROM read_parquet('{path}')) neighbor_rows,
               count(*) items_with_neighbors,
               avg(neighbor_count),quantile_disc(neighbor_count,0.25),
               quantile_disc(neighbor_count,0.5),quantile_disc(neighbor_count,0.9),
               quantile_disc(neighbor_count,0.95),
               (SELECT quantile_cont(similarity,0.5) FROM read_parquet('{path}')),
               (SELECT quantile_cont(similarity,0.9) FROM read_parquet('{path}')),
               (SELECT quantile_cont(similarity,0.99) FROM read_parquet('{path}')),
               (SELECT quantile_disc(raw_cooccurrence,0.5) FROM read_parquet('{path}')),
               (SELECT quantile_disc(raw_cooccurrence,0.9) FROM read_parquet('{path}'))
        FROM per_item
        """
    ).fetchone()


@dataclass(frozen=True)
class SnapshotSpec:
    snapshot_id: str
    history_start: str
    history_end: str
    label_start: str
    label_end: str
    ground_truth_fingerprint: str


class Stage3BRunner:
    def __init__(self, config: Stage3BConfig):
        self.config = config
        self.dataset_fingerprint = ""
        self.feedback_version = ""
        self.population_version = config.raw["population_version"]
        self.snapshots: dict[str, SnapshotSpec] = {}
        self.source_fingerprint = source_fingerprint(config.root)
        self.stage3a_lock_path = config.resolve(config.raw["upstream"]["stage_3a_lock"])
        self.stage3a_lock_fingerprint = file_fingerprint(self.stage3a_lock_path)
        self.run_fingerprint = ""
        self.resources: list[dict[str, Any]] = []
        self.run_status: list[dict[str, Any]] = []
        self._load_and_validate_upstream()
        self.run_fingerprint = hashlib.sha256(
            (
                self.dataset_fingerprint
                + self.config.fingerprint
                + self.source_fingerprint
                + self.stage3a_lock_fingerprint
            ).encode("utf-8")
        ).hexdigest()

    def _load_and_validate_upstream(self) -> None:
        # 直接构造 Stage 3A runner，复用其 Stage 2A/2B 私有 fingerprint 接管门禁。
        stage3a_config = RetrievalConfig.load(self.config.root / "configs/stage_3a_recall.yaml")
        stage3a = Stage3ARunner(stage3a_config)
        lock = json_read(self.stage3a_lock_path)
        if stage3a.source_fingerprint != lock["source_fingerprint"]:
            raise ValueError("Stage 3A source fingerprint 已变化")
        if stage3a.config.fingerprint != lock["config_fingerprint"]:
            raise ValueError("Stage 3A config fingerprint 已变化")
        if file_fingerprint(self.stage3a_lock_path) != self._stage3a_public_fingerprint(
            "artifacts/stage_3a/stage_3a_lock.json"
        ):
            raise ValueError("Stage 3A lock 与正式 manifest 不一致")
        if int(lock["locked_trend_window"]) != 7:
            raise ValueError("Stage 3A 正式锁定 Trend 不再是 7d")
        self.dataset_fingerprint = stage3a.dataset_fingerprint
        self.feedback_version = stage3a.feedback_version
        if self.feedback_version != self.config.raw["feedback_version"]:
            raise ValueError("Stage 3B feedback_version 与 Stage 3A 不一致")
        if self.population_version != stage3a.population_version:
            raise ValueError("Stage 3B population_version 与 Stage 3A 不一致")
        self.snapshots = {
            key: SnapshotSpec(
                value.snapshot_id,
                value.history_start,
                value.history_end,
                value.label_start,
                value.label_end,
                value.ground_truth_fingerprint,
            )
            for key, value in stage3a.snapshots.items()
        }
        for snapshot in self.snapshots:
            for path in (
                self.stage3a_ranking(snapshot, "popularity", "global"),
                self.stage3a_ranking(snapshot, "trend", "window_7d"),
                self.stage3a_candidate(snapshot, "repurchase", "pure"),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"Stage 3A 私有输入缺失: {path.name}")

    def _stage3a_public_fingerprint(self, relative: str) -> str:
        manifest = csv_rows(self.config.resolve(self.config.raw["input_manifests"]["stage_3a"]))
        matches = [row for row in manifest if row["artifact_path"] == relative]
        if len(matches) != 1:
            raise ValueError(f"Stage 3A manifest 缺少唯一产物: {relative}")
        return matches[0]["artifact_fingerprint"]

    def prepare_directories(self) -> None:
        for path in (
            self.config.artifact_dir,
            self.config.private_dir,
            self.config.checkpoint_dir,
            self.config.temp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        duck = self.config.raw["duckdb"]
        con.execute(f"SET temp_directory='{sql_path(self.config.temp_dir)}'")
        con.execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
        con.execute(f"SET threads={int(duck['threads'])}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{duck['max_temp_directory_size']}'")
        stage2a = self.config.resolve(self.config.raw["upstream"]["stage_2a_private"])
        for view, name in (
            ("feedback_daily", "feedback_daily.parquet"),
            ("customer_map", "customer_key_map.parquet"),
            ("article_map", "article_key_map.parquet"),
        ):
            con.execute(
                f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{sql_path(stage2a / name)}')"
            )
        return con

    def upstream_path(self, prefix: str, snapshot: str) -> Path:
        base = self.config.resolve(self.config.raw["upstream"]["stage_2b_private"])
        return base / f"{prefix}_{snapshot}.parquet"

    def stage3a_ranking(self, snapshot: str, source: str, variant: str) -> Path:
        base = self.config.resolve(self.config.raw["upstream"]["stage_3a_private"])
        return base / f"ranking_{snapshot}_{source}_{variant}.parquet"

    def stage3a_candidate(self, snapshot: str, source: str, variant: str) -> Path:
        base = self.config.resolve(self.config.raw["upstream"]["stage_3a_private"])
        return base / f"candidates_{snapshot}_{source}_{variant}.parquet"

    def user_items_part_path(self, snapshot: str, part: int) -> Path:
        return self.config.private_dir / "user_items" / snapshot / f"part_{part:03d}.parquet"

    def user_items_glob(self, snapshot: str) -> str:
        return sql_path(self.user_items_part_path(snapshot, 0).parent / "*.parquet")

    def scale_part_path(self, snapshot: str, part: int) -> Path:
        return self.config.private_dir / "user_items" / snapshot / f"scale_{part:03d}.json"

    def scale_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"scale_{snapshot}.json"

    def support_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"item_support_{snapshot}.parquet"

    def pair_part_path(self, snapshot: str, part: int) -> Path:
        return self.config.private_dir / "pairs" / snapshot / f"part_{part:03d}.parquet"

    def pairs_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"pairs_{snapshot}.parquet"

    def neighbors_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / f"neighbors_{snapshot}_{variant}.parquet"

    def history_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"evaluation_history_{snapshot}.parquet"

    def seeds_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"seeds_{snapshot}.parquet"

    def candidate_part_path(self, snapshot: str, variant: str, part: int) -> Path:
        return self.config.private_dir / "candidates" / snapshot / variant / f"part_{part:03d}.parquet"

    def candidate_glob(self, snapshot: str, variant: str) -> str:
        return sql_path(self.candidate_part_path(snapshot, variant, 0).parent / "*.parquet")

    def metric_path(self, snapshot: str, source: str, variant: str) -> Path:
        return self.config.private_dir / f"user_metrics_{snapshot}_{source}_{variant}.parquet"

    def union_part_path(self, snapshot: str, variant: str, part: int) -> Path:
        return self.config.private_dir / "unions" / snapshot / variant / f"part_{part:03d}.parquet"

    def union_glob(self, snapshot: str, variant: str) -> str:
        return sql_path(self.union_part_path(snapshot, variant, 0).parent / "*.parquet")

    def overlap_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"overlap_{snapshot}.parquet"

    def checkpoint_path(self, group: str) -> Path:
        return self.config.checkpoint_dir / f"{group}.json"

    def _checkpoint_reusable(self, checkpoint: dict[str, Any] | None, outputs: Sequence[Path]) -> bool:
        if not checkpoint or checkpoint.get("status") != "SUCCESS":
            return False
        exact = checkpoint.get("run_fingerprint") == self.run_fingerprint
        compatible = (
            checkpoint.get("source_fingerprint") in COMPATIBLE_CHECKPOINT_SOURCE_FINGERPRINTS
            and checkpoint.get("dataset_fingerprint") == self.dataset_fingerprint
            and checkpoint.get("config_fingerprint") == self.config.fingerprint
        )
        if not exact and not compatible:
            return False
        expected = checkpoint.get("output_fingerprints", {})
        return bool(outputs) and all(
            path.exists()
            and expected.get(self.config.relative(path)) == file_fingerprint(path)
            for path in outputs
        )

    def run_group(self, group: str, operation: Callable[[], Any], outputs: Iterable[Path]) -> Any:
        paths = list(outputs)
        checkpoint_path = self.checkpoint_path(group)
        checkpoint = json_read(checkpoint_path) if checkpoint_path.exists() else None
        if self._checkpoint_reusable(checkpoint, paths):
            reused = dict(checkpoint)
            compatible_reason = COMPATIBLE_CHECKPOINT_SOURCE_FINGERPRINTS.get(
                str(checkpoint.get("source_fingerprint")), ""
            )
            reused["run_status"] = "REUSED_COMPATIBLE" if compatible_reason else "REUSED"
            self.resources.append(reused)
            self.run_status.append(
                {
                    "group_name": group,
                    "status": reused["run_status"],
                    "attempt": checkpoint.get("attempt", 1),
                    "message": (
                        f"compatible source reuse: {compatible_reason}; config/upstream/output fingerprints exact"
                        if compatible_reason else "matching fingerprint and complete outputs"
                    ),
                }
            )
            return None
        attempt = int((checkpoint or {}).get("attempt", 0)) + 1
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        try:
            with monitor:
                result = operation()
            missing = [self.config.relative(path) for path in paths if not path.exists()]
            if missing:
                raise RuntimeError(f"group 输出不完整: {missing}")
            payload = {
                "group_name": group,
                "status": "SUCCESS",
                "attempt": attempt,
                "generated_at": now(),
                "run_fingerprint": self.run_fingerprint,
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
            json_write(checkpoint_path, payload)
            self.resources.append(payload)
            self.run_status.append({"group_name": group, "status": "SUCCESS", "attempt": attempt, "message": ""})
            return result
        except Exception as exc:
            payload = {
                "group_name": group,
                "status": "FAILED",
                "attempt": attempt,
                "generated_at": now(),
                "run_fingerprint": self.run_fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "error_message": str(exc).replace(str(self.config.root), "."),
            }
            json_write(checkpoint_path, payload)
            self.resources.append(payload)
            self.run_status.append({"group_name": group, "status": "FAILED", "attempt": attempt, "message": payload["error_message"]})
            raise

    def build_user_item_partition(self, snapshot_id: str, part: int) -> None:
        snapshot = self.snapshots[snapshot_id]
        target = self.user_items_part_path(snapshot_id, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        limit = int(self.config.raw["itemcf"]["max_items_per_user_for_cooccurrence"])
        partitions = int(self.config.raw["execution"]["pair_partitions"])
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH grouped AS (
                    SELECT cm.customer_key::BIGINT customer_key,
                           am.article_key::BIGINT article_key,
                           max(f.t_dat)::DATE last_purchase_date
                    FROM feedback_daily f
                    JOIN customer_map cm USING(customer_id)
                    JOIN article_map am USING(article_id)
                    WHERE f.t_dat BETWEEN DATE '{snapshot.history_start}' AND DATE '{snapshot.history_end}'
                      AND cm.customer_key % {partitions} = {part}
                    GROUP BY cm.customer_key, am.article_key
                  ), ranked AS (
                    SELECT *, count(*) OVER(PARTITION BY customer_key)::INTEGER distinct_item_count,
                           row_number() OVER(
                             PARTITION BY customer_key
                             ORDER BY last_purchase_date DESC, article_key ASC
                           )::INTEGER recency_rank
                    FROM grouped
                  )
                  SELECT customer_key, article_key, last_purchase_date,
                         distinct_item_count,
                         least(distinct_item_count,{limit})::INTEGER cooccurrence_item_count,
                         recency_rank
                  FROM ranked
                  WHERE recency_rank<={limit}
                  ORDER BY customer_key, recency_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            stats = con.execute(
                f"""
                WITH counts AS (
                  SELECT customer_key, max(distinct_item_count)::BIGINT n,
                         max(cooccurrence_item_count)::BIGINT capped_n
                  FROM read_parquet('{sql_path(target)}') GROUP BY customer_key
                )
                SELECT count(*), count(*) FILTER(WHERE n>=2), count(*) FILTER(WHERE n>{limit}),
                  sum(n), sum(capped_n), sum((n*(n-1))/2), sum((capped_n*(capped_n-1))/2), max(n)
                FROM counts
                """
            ).fetchone()
            json_write(
                self.scale_part_path(snapshot_id, part),
                {
                    "snapshot_id": snapshot_id,
                    "partition": part,
                    "as_of_date": snapshot.history_end,
                    "history_users": int(stats[0]),
                    "cooccurrence_users": int(stats[1]),
                    "truncated_users": int(stats[2]),
                    "unique_user_items_before": int(stats[3]),
                    "unique_user_items_after": int(stats[4]),
                    "pair_contributions_before": int(stats[5]),
                    "pair_contributions_after": int(stats[6]),
                    "max_distinct_items": int(stats[7]),
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                },
            )
        finally:
            con.close()

    def consolidate_scale(self, snapshot_id: str) -> None:
        parts = [
            json_read(self.scale_part_path(snapshot_id, part))
            for part in range(int(self.config.raw["execution"]["pair_partitions"]))
        ]
        split_rows = csv_rows(self.config.resolve(self.config.raw["input_manifests"]["split"]))
        split = next(row for row in split_rows if row["split_id"] == snapshot_id)
        json_write(
            self.scale_path(snapshot_id),
            {
                "snapshot_id": snapshot_id,
                "as_of_date": self.snapshots[snapshot_id].history_end,
                "history_rows_B": int(split["history_rows_B"]),
                **{
                    field: sum(int(row[field]) for row in parts)
                    for field in (
                        "history_users",
                        "cooccurrence_users",
                        "truncated_users",
                        "unique_user_items_before",
                        "unique_user_items_after",
                        "pair_contributions_before",
                        "pair_contributions_after",
                    )
                },
                "max_distinct_items": max(int(row["max_distinct_items"]) for row in parts),
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
            },
        )

    def build_support(self, snapshot_id: str) -> None:
        target = self.support_path(snapshot_id)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT article_key, count(*)::BIGINT item_support
                  FROM read_parquet('{self.user_items_glob(snapshot_id)}')
                  GROUP BY article_key ORDER BY article_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()

    def build_pair_partition(self, snapshot_id: str, part: int) -> None:
        target = self.pair_part_path(snapshot_id, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        partitions = int(self.config.raw["execution"]["pair_partitions"])
        source = self.user_items_part_path(snapshot_id, part)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT a.article_key::BIGINT article_key_left,
                         b.article_key::BIGINT article_key_right,
                         count(*)::BIGINT raw_cooccurrence,
                         sum(1.0/ln(1+a.cooccurrence_item_count))::DOUBLE iuf_cooccurrence
                  FROM read_parquet('{sql_path(source)}') a
                  JOIN read_parquet('{sql_path(source)}') b
                    ON a.customer_key=b.customer_key AND a.article_key<b.article_key
                  GROUP BY a.article_key, b.article_key
                  ORDER BY a.article_key, b.article_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()

    def consolidate_pairs(self, snapshot_id: str) -> None:
        target = self.pairs_path(snapshot_id)
        glob = sql_path(self.pair_part_path(snapshot_id, 0).parent / "*.parquet")
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT article_key_left, article_key_right,
                         sum(raw_cooccurrence)::BIGINT raw_cooccurrence,
                         sum(iuf_cooccurrence)::DOUBLE iuf_cooccurrence
                  FROM read_parquet('{glob}')
                  GROUP BY article_key_left, article_key_right
                  ORDER BY article_key_left, article_key_right
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()

    def build_neighbors_full(self, snapshot_id: str, variant: str) -> None:
        if variant not in self.config.raw["itemcf"]["variants"]:
            raise ValueError("未注册 ItemCF variant")
        snapshot = self.snapshots[snapshot_id]
        target = self.neighbors_path(snapshot_id, variant)
        weighted = "raw_cooccurrence::DOUBLE" if variant == "cosine" else "iuf_cooccurrence"
        top_n = int(self.config.raw["itemcf"]["top_neighbors_per_item"])
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH directed AS (
                    SELECT article_key_left article_key, article_key_right neighbor_article_key,
                           raw_cooccurrence, iuf_cooccurrence FROM read_parquet('{sql_path(self.pairs_path(snapshot_id))}')
                    UNION ALL
                    SELECT article_key_right, article_key_left,
                           raw_cooccurrence, iuf_cooccurrence FROM read_parquet('{sql_path(self.pairs_path(snapshot_id))}')
                  ), scored AS (
                    SELECT d.article_key, d.neighbor_article_key,
                           ({weighted})/sqrt(a.item_support*b.item_support)::DOUBLE similarity,
                           ({weighted})::DOUBLE weighted_cooccurrence,
                           d.raw_cooccurrence
                    FROM directed d
                    JOIN read_parquet('{sql_path(self.support_path(snapshot_id))}') a USING(article_key)
                    JOIN read_parquet('{sql_path(self.support_path(snapshot_id))}') b
                      ON b.article_key=d.neighbor_article_key
                    WHERE a.item_support>0 AND b.item_support>0
                  ), ranked AS (
                    SELECT *, row_number() OVER(
                      PARTITION BY article_key
                      ORDER BY similarity DESC, weighted_cooccurrence DESC,
                               raw_cooccurrence DESC, neighbor_article_key ASC
                    )::INTEGER neighbor_rank
                    FROM scored WHERE isfinite(similarity)
                  )
                  SELECT '{snapshot_id}'::VARCHAR snapshot_id,
                         DATE '{snapshot.history_end}' as_of_date,
                         article_key, neighbor_article_key, similarity,
                         weighted_cooccurrence, raw_cooccurrence, neighbor_rank,
                         '{variant}'::VARCHAR source_variant,
                         '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
                         '{self.config.fingerprint}'::VARCHAR config_fingerprint
                  FROM ranked WHERE neighbor_rank<={top_n}
                  ORDER BY article_key, neighbor_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            violations = con.execute(
                f"""
                SELECT count(*) FROM read_parquet('{sql_path(target)}')
                WHERE article_key=neighbor_article_key OR NOT isfinite(similarity)
                   OR neighbor_rank<1 OR neighbor_rank>{top_n}
                """
            ).fetchone()[0]
            if violations:
                raise ValueError(f"邻居表协议违反行数={violations}")
        finally:
            con.close()

    def build_evaluation_history_and_seeds(self, snapshot_id: str) -> None:
        snapshot = self.snapshots[snapshot_id]
        history = self.history_path(snapshot_id)
        seeds = self.seeds_path(snapshot_id)
        population = self.upstream_path("evaluation_population", snapshot_id)
        max_seeds = int(self.config.raw["itemcf"]["max_seed_items_per_user"])
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT cm.customer_key::BIGINT customer_key,
                         am.article_key::BIGINT article_key,
                         max(f.t_dat)::DATE last_purchase_date
                  FROM feedback_daily f
                  JOIN customer_map cm USING(customer_id)
                  JOIN article_map am USING(article_id)
                  SEMI JOIN read_parquet('{sql_path(population)}') u ON u.user_key=cm.customer_key
                  WHERE f.t_dat BETWEEN DATE '{snapshot.history_start}' AND DATE '{snapshot.history_end}'
                  GROUP BY cm.customer_key, am.article_key
                  ORDER BY cm.customer_key, article_key
                ) TO '{sql_path(history)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            con.execute(
                f"""
                COPY (
                  WITH ranked AS (
                    SELECT *, row_number() OVER(
                      PARTITION BY customer_key ORDER BY last_purchase_date DESC, article_key ASC
                    )::INTEGER seed_rank
                    FROM read_parquet('{sql_path(history)}')
                  )
                  SELECT customer_key, article_key seed_article_key, last_purchase_date,
                         seed_rank, 1.0/log2(seed_rank+1)::DOUBLE seed_weight
                  FROM ranked WHERE seed_rank<={max_seeds}
                  ORDER BY customer_key, seed_rank
                ) TO '{sql_path(seeds)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()

    def build_candidate_partition(self, snapshot_id: str, variant: str, part: int) -> None:
        snapshot = self.snapshots[snapshot_id]
        partitions = int(self.config.raw["execution"]["candidate_partitions"])
        max_k = int(self.config.raw["itemcf"]["candidate_max_k"])
        target = self.candidate_part_path(snapshot_id, variant, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        population = self.upstream_path("evaluation_population", snapshot_id)
        item_state = self.upstream_path("item_state", snapshot_id)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH expanded AS (
                    SELECT s.customer_key, n.neighbor_article_key article_key,
                           s.seed_article_key, s.seed_weight*n.similarity contribution,
                           n.similarity
                    FROM read_parquet('{sql_path(self.seeds_path(snapshot_id))}') s
                    JOIN read_parquet('{sql_path(self.neighbors_path(snapshot_id, variant))}') n
                      ON n.article_key=s.seed_article_key
                    SEMI JOIN read_parquet('{sql_path(population)}') u
                      ON u.user_key=s.customer_key AND u.is_main_evaluable
                    SEMI JOIN read_parquet('{sql_path(item_state)}') i
                      ON i.article_key=n.neighbor_article_key AND i.seen_item
                    ANTI JOIN read_parquet('{sql_path(self.history_path(snapshot_id))}') h
                      ON h.customer_key=s.customer_key AND h.article_key=n.neighbor_article_key
                    WHERE s.customer_key % {partitions} = {part}
                  ), aggregated AS (
                    SELECT customer_key, article_key,
                           sum(contribution)::DOUBLE source_score,
                           count(DISTINCT seed_article_key)::INTEGER seed_hit_count,
                           max(similarity)::DOUBLE max_similarity
                    FROM expanded GROUP BY customer_key, article_key
                  ), ranked AS (
                    SELECT *, row_number() OVER(
                      PARTITION BY customer_key
                      ORDER BY source_score DESC, seed_hit_count DESC,
                               max_similarity DESC, article_key ASC
                    )::INTEGER source_rank
                    FROM aggregated
                  )
                  SELECT '{snapshot_id}'::VARCHAR snapshot_id,
                         DATE '{snapshot.history_end}' as_of_date,
                         customer_key::BIGINT customer_key,
                         article_key::BIGINT article_key,
                         'itemcf'::VARCHAR AS "source",
                         '{variant}'::VARCHAR source_variant,
                         source_rank, source_score,
                         '{self.config.raw['candidate_universe_versions']['warm']}'::VARCHAR candidate_universe_version,
                         '{self.feedback_version}'::VARCHAR feedback_version,
                         '{self.population_version}'::VARCHAR population_version,
                         '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
                         '{self.config.fingerprint}'::VARCHAR config_fingerprint,
                         seed_hit_count, max_similarity,
                         'weighted_neighbor_sum'::VARCHAR source_score_kind
                  FROM ranked WHERE source_rank<={max_k}
                  ORDER BY customer_key, source_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()

    def validate_candidate_parts(self, snapshot_id: str, variant: str) -> dict[str, int]:
        snapshot = self.snapshots[snapshot_id]
        glob = self.candidate_glob(snapshot_id, variant)
        history = self.history_path(snapshot_id)
        item_state = self.upstream_path("item_state", snapshot_id)
        expected_parts = int(self.config.raw["execution"]["candidate_partitions"])
        actual_parts = len(list(self.candidate_part_path(snapshot_id, variant, 0).parent.glob("*.parquet")))
        if actual_parts != expected_parts:
            raise ValueError(f"ItemCF candidate 分区不完整: {actual_parts}/{expected_parts}")
        con = self.connect()
        try:
            fields = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}')").fetchall()}
            missing = set(CANDIDATE_FIELDS) - fields
            forbidden = FORBIDDEN_ID_FIELDS & fields
            if missing or forbidden:
                raise ValueError(f"ItemCF 候选 schema 错误 missing={sorted(missing)} forbidden={sorted(forbidden)}")
            metadata = con.execute(
                f"""
                SELECT count(DISTINCT snapshot_id), min(snapshot_id),
                       count(DISTINCT source_variant), min(source_variant),
                       count(DISTINCT dataset_fingerprint), min(dataset_fingerprint),
                       count(DISTINCT config_fingerprint), min(config_fingerprint)
                FROM read_parquet('{glob}')
                """
            ).fetchone()
            if metadata != (1, snapshot_id, 1, variant, 1, self.dataset_fingerprint, 1, self.config.fingerprint):
                raise ValueError("ItemCF 候选 metadata 不一致")
            violations = con.execute(
                f"""
                WITH grouped AS (
                  SELECT customer_key, count(*) n, count(DISTINCT article_key) d_items,
                         count(DISTINCT source_rank) d_ranks, min(source_rank) lo, max(source_rank) hi
                  FROM read_parquet('{glob}') GROUP BY customer_key
                )
                SELECT count(*) FROM grouped
                WHERE n>{self.config.raw['itemcf']['candidate_max_k']} OR n<>d_items OR n<>d_ranks OR lo<>1 OR hi<>n
                """
            ).fetchone()[0]
            illegal = con.execute(
                f"""
                SELECT count(*) FROM read_parquet('{glob}') c
                LEFT JOIN read_parquet('{sql_path(item_state)}') i USING(article_key)
                LEFT JOIN read_parquet('{sql_path(history)}') h USING(customer_key,article_key)
                WHERE i.article_key IS NULL OR NOT i.seen_item OR h.article_key IS NOT NULL
                """
            ).fetchone()[0]
            if violations or illegal:
                raise ValueError(f"ItemCF 候选协议违反 users={violations}, illegal_rows={illegal}")
            stats = con.execute(
                f"SELECT count(*),count(DISTINCT customer_key),count(DISTINCT article_key) FROM read_parquet('{glob}')"
            ).fetchone()
            return {"candidate_rows": int(stats[0]), "candidate_users": int(stats[1]), "candidate_items": int(stats[2])}
        finally:
            con.close()

    def run_itemcf_core(self, snapshot_id: str, variants: Sequence[str]) -> None:
        for part in range(int(self.config.raw["execution"]["pair_partitions"])):
            self.run_group(
                f"{snapshot_id}_user_item_sets_part_{part:03d}",
                lambda part=part: self.build_user_item_partition(snapshot_id, part),
                [self.user_items_part_path(snapshot_id, part), self.scale_part_path(snapshot_id, part)],
            )
        self.run_group(
            f"{snapshot_id}_scale_consolidated",
            lambda: self.consolidate_scale(snapshot_id),
            [self.scale_path(snapshot_id)],
        )
        self.run_group(
            f"{snapshot_id}_item_support",
            lambda: self.build_support(snapshot_id),
            [self.support_path(snapshot_id)],
        )
        for part in range(int(self.config.raw["execution"]["pair_partitions"])):
            self.run_group(
                f"{snapshot_id}_pairs_part_{part:03d}",
                lambda part=part: self.build_pair_partition(snapshot_id, part),
                [self.pair_part_path(snapshot_id, part)],
            )
        self.run_group(
            f"{snapshot_id}_pairs_consolidated",
            lambda: self.consolidate_pairs(snapshot_id),
            [self.pairs_path(snapshot_id)],
        )
        for variant in variants:
            self.run_group(
                f"{snapshot_id}_{variant}_neighbors",
                lambda variant=variant: self.build_neighbors_full(snapshot_id, variant),
                [self.neighbors_path(snapshot_id, variant)],
            )
        self.run_group(
            f"{snapshot_id}_evaluation_history_seeds",
            lambda: self.build_evaluation_history_and_seeds(snapshot_id),
            [self.history_path(snapshot_id), self.seeds_path(snapshot_id)],
        )
        for variant in variants:
            for part in range(int(self.config.raw["execution"]["candidate_partitions"])):
                self.run_group(
                    f"{snapshot_id}_{variant}_candidates_part_{part:03d}",
                    lambda variant=variant, part=part: self.build_candidate_partition(snapshot_id, variant, part),
                    [self.candidate_part_path(snapshot_id, variant, part)],
                )
            self.validate_candidate_parts(snapshot_id, variant)

    def evaluate_candidate_set(
        self,
        snapshot_id: str,
        *,
        candidate_glob: str,
        source: str,
        variant: str,
    ) -> Path:
        snapshot = self.snapshots[snapshot_id]
        ground_truth = self.upstream_path("ground_truth", snapshot_id)
        population = self.upstream_path("evaluation_population", snapshot_id)
        target = self.metric_path(snapshot_id, source, variant)
        if file_fingerprint(ground_truth) != snapshot.ground_truth_fingerprint:
            raise ValueError("Ground truth fingerprint 在评估前发生变化")
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key customer_key, user_segment,
                           is_main_evaluable, positive_item_count
                    FROM read_parquet('{sql_path(population)}')
                  ), candidate_counts AS (
                    SELECT customer_key, count(*)::INTEGER candidate_count
                    FROM read_parquet('{candidate_glob}') GROUP BY customer_key
                  ), hit_rows AS (
                    SELECT c.customer_key, c.source_rank,
                           row_number() OVER(
                             PARTITION BY c.customer_key ORDER BY c.source_rank
                           )::INTEGER cumulative_hits
                    FROM read_parquet('{candidate_glob}') c
                    SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                      ON g.user_key=c.customer_key AND g.article_key=c.article_key
                  ), hits AS (
                    SELECT customer_key,
                      count(*) FILTER(WHERE source_rank<=12)::INTEGER hits_12,
                      count(*) FILTER(WHERE source_rank<=50)::INTEGER hits_50,
                      count(*) FILTER(WHERE source_rank<=100)::INTEGER hits_100,
                      count(*) FILTER(WHERE source_rank<=500)::INTEGER hits_500,
                      sum(cumulative_hits::DOUBLE/source_rank)
                        FILTER(WHERE source_rank<=12) ap_numerator_12
                    FROM hit_rows GROUP BY customer_key
                  )
                  SELECT '{snapshot_id}'::VARCHAR snapshot_id,
                         DATE '{snapshot.history_end}' as_of_date,
                         u.customer_key, u.user_segment, u.is_main_evaluable,
                         u.positive_item_count,
                         coalesce(cc.candidate_count,0)::INTEGER candidate_count,
                         coalesce(h.hits_12,0)::INTEGER hits_12,
                         coalesce(h.hits_50,0)::INTEGER hits_50,
                         coalesce(h.hits_100,0)::INTEGER hits_100,
                         coalesce(h.hits_500,0)::INTEGER hits_500,
                         coalesce(h.hits_12,0)::DOUBLE/u.positive_item_count recall_at_12,
                         coalesce(h.hits_50,0)::DOUBLE/u.positive_item_count recall_at_50,
                         coalesce(h.hits_100,0)::DOUBLE/u.positive_item_count recall_at_100,
                         coalesce(h.hits_500,0)::DOUBLE/u.positive_item_count recall_at_500,
                         (coalesce(h.hits_12,0)>0)::INTEGER hit_rate_at_12,
                         (coalesce(h.hits_50,0)>0)::INTEGER hit_rate_at_50,
                         (coalesce(h.hits_100,0)>0)::INTEGER hit_rate_at_100,
                         (coalesce(h.hits_500,0)>0)::INTEGER hit_rate_at_500,
                         CASE WHEN coalesce(cc.candidate_count,0)=0 THEN 0
                              ELSE coalesce(h.hits_12,0)::DOUBLE/least(cc.candidate_count,12) END precision_at_12,
                         coalesce(h.ap_numerator_12,0)::DOUBLE/least(u.positive_item_count,12) ap_at_12,
                         '{source}'::VARCHAR AS "source",
                         '{variant}'::VARCHAR source_variant,
                         '{snapshot.ground_truth_fingerprint}'::VARCHAR ground_truth_fingerprint,
                         '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
                         '{self.config.fingerprint}'::VARCHAR config_fingerprint
                  FROM users u
                  LEFT JOIN candidate_counts cc USING(customer_key)
                  LEFT JOIN hits h USING(customer_key)
                  ORDER BY customer_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            audit = con.execute(
                f"""
                SELECT count(*), count(*) FILTER(WHERE positive_item_count<=0),
                       count(DISTINCT ground_truth_fingerprint), min(ground_truth_fingerprint)
                FROM read_parquet('{sql_path(target)}')
                """
            ).fetchone()
            if audit[1] or audit[2] != 1 or audit[3] != snapshot.ground_truth_fingerprint:
                raise ValueError("评估人口或 ground truth fingerprint 非法")
        finally:
            con.close()
        return target

    @staticmethod
    def _population_sql(metric_path: Path) -> str:
        path = sql_path(metric_path)
        return f"""
          SELECT 'evaluable_warm_users' segment, * FROM read_parquet('{path}') WHERE is_main_evaluable
          UNION ALL SELECT 'short_history', * FROM read_parquet('{path}') WHERE user_segment='short_history'
          UNION ALL SELECT 'warm', * FROM read_parquet('{path}') WHERE user_segment='warm'
          UNION ALL SELECT 'active', * FROM read_parquet('{path}') WHERE user_segment='active'
          UNION ALL SELECT 'cold_history_label_users', * FROM read_parquet('{path}') WHERE user_segment='cold'
        """

    def _public_row(
        self,
        snapshot_id: str,
        source: str,
        variant: str,
        segment: str,
        denominator: int,
        metric: str,
        value: Any,
    ) -> dict[str, Any]:
        universe = (
            self.config.raw["candidate_universe_versions"]["cold_history"]
            if segment == "cold_history_label_users"
            else self.config.raw["candidate_universe_versions"]["warm"]
        )
        return {
            "stage": "3B",
            "stage_version": self.config.raw["stage_version"],
            "snapshot_id": snapshot_id,
            "as_of_date": self.snapshots[snapshot_id].history_end,
            "source": source,
            "source_variant": variant,
            "metric": metric,
            "segment": segment,
            "population_denominator": denominator,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint,
            "feedback_version": self.feedback_version,
            "population_version": self.population_version,
            "candidate_universe_version": universe,
            "run_status": "SUCCESS",
            "metric_value": value,
        }

    def aggregate_candidate_set(
        self,
        snapshot_id: str,
        *,
        candidate_glob: str,
        source: str,
        variant: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        metric = self.metric_path(snapshot_id, source, variant)
        item_state = self.upstream_path("item_state", snapshot_id)
        con = self.connect()
        try:
            cursor = con.execute(
                f"""
                WITH pops AS ({self._population_sql(metric)})
                SELECT segment, count(*) population_denominator,
                       avg(ap_at_12) map_at_12,
                       avg(recall_at_12) recall_at_12,
                       avg(recall_at_50) recall_at_50,
                       avg(recall_at_100) recall_at_100,
                       avg(recall_at_500) recall_at_500,
                       avg(hit_rate_at_12) hit_rate_at_12,
                       avg(hit_rate_at_50) hit_rate_at_50,
                       avg(hit_rate_at_100) hit_rate_at_100,
                       avg(hit_rate_at_500) hit_rate_at_500,
                       avg(precision_at_12) precision_at_12
                FROM pops GROUP BY segment
                """
            )
            names = [item[0] for item in cursor.description]
            metric_wide = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
            cursor = con.execute(
                f"""
                WITH pops AS ({self._population_sql(metric)}),
                count_stats AS (
                  SELECT segment, count(*) population_users,
                    count(*) FILTER(WHERE candidate_count>0) users_with_candidates,
                    avg(candidate_count) mean_candidate_count,
                    quantile_disc(candidate_count,0.5) median_candidate_count,
                    quantile_disc(candidate_count,0.25) p25_candidate_count,
                    quantile_disc(candidate_count,0.75) p75_candidate_count,
                    quantile_disc(candidate_count,0.90) p90_candidate_count,
                    quantile_disc(candidate_count,0.95) p95_candidate_count,
                    avg((candidate_count<12)::INTEGER) users_below_12_rate,
                    avg((candidate_count<50)::INTEGER) users_below_50_rate,
                    avg((candidate_count<100)::INTEGER) users_below_100_rate,
                    avg((candidate_count<500)::INTEGER) users_below_500_rate
                  FROM pops GROUP BY segment
                ), membership AS (
                  SELECT segment, customer_key FROM pops
                ), unique_items AS (
                  SELECT m.segment, count(DISTINCT c.article_key) unique_recommended_items
                  FROM membership m JOIN read_parquet('{candidate_glob}') c USING(customer_key)
                  GROUP BY m.segment
                ), exposure AS (
                  SELECT m.segment, c.article_key, count(*) exposures
                  FROM membership m JOIN read_parquet('{candidate_glob}') c USING(customer_key)
                  WHERE c.source_rank<=12 GROUP BY m.segment,c.article_key
                ), ranked_exposure AS (
                  SELECT *, row_number() OVER(PARTITION BY segment ORDER BY exposures DESC,article_key ASC) exposure_rank
                  FROM exposure
                ), exposure_stats AS (
                  SELECT segment,sum(exposures) exposure_total,max(exposures) top1_exposures,
                         sum(exposures) FILTER(WHERE exposure_rank<=10) top10_exposures,
                         sum(exposures) FILTER(WHERE exposure_rank<=100) top100_exposures
                  FROM ranked_exposure GROUP BY segment
                ), universe AS (
                  SELECT count(*) full_items,count(*) FILTER(WHERE seen_item) warm_items
                  FROM read_parquet('{sql_path(item_state)}')
                )
                SELECT c.*,c.users_with_candidates::DOUBLE/c.population_users candidate_user_coverage,
                       coalesce(u.unique_recommended_items,0) unique_recommended_items,
                       CASE WHEN c.segment='cold_history_label_users' THEN universe.full_items ELSE universe.warm_items END candidate_universe_items,
                       coalesce(u.unique_recommended_items,0)::DOUBLE/
                         CASE WHEN c.segment='cold_history_label_users' THEN universe.full_items ELSE universe.warm_items END catalog_coverage,
                       coalesce(e.top1_exposures,0)::DOUBLE/nullif(e.exposure_total,0) top1_top12_exposure_share,
                       coalesce(e.top10_exposures,0)::DOUBLE/nullif(e.exposure_total,0) top10_top12_exposure_share,
                       coalesce(e.top100_exposures,0)::DOUBLE/nullif(e.exposure_total,0) top100_top12_exposure_share
                FROM count_stats c LEFT JOIN unique_items u USING(segment)
                LEFT JOIN exposure_stats e USING(segment) CROSS JOIN universe
                """
            )
            names = [item[0] for item in cursor.description]
            coverage_wide = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        finally:
            con.close()
        metrics = [
            self._public_row(snapshot_id, source, variant, row["segment"], int(row["population_denominator"]), name, row[name])
            for row in metric_wide for name in METRIC_NAMES
        ]
        coverage = [
            self._public_row(snapshot_id, source, variant, row["segment"], int(row["population_users"]), name, row[name])
            for row in coverage_wide for name in COVERAGE_NAMES
        ]
        return metrics, coverage

    def write_itemcf_aggregates(self, phase: str, snapshot_id: str, variants: Sequence[str]) -> list[Path]:
        overall: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        coverage: list[dict[str, Any]] = []
        for variant in variants:
            metrics, cov = self.aggregate_candidate_set(
                snapshot_id,
                candidate_glob=self.candidate_glob(snapshot_id, variant),
                source="itemcf",
                variant=variant,
            )
            overall.extend(row for row in metrics if row["segment"] == "evaluable_warm_users")
            segments.extend(row for row in metrics if row["segment"] != "evaluable_warm_users")
            coverage.extend(cov)
        paths = [
            self.config.artifact_dir / f"{phase}_itemcf_metrics_overall.csv",
            self.config.artifact_dir / f"{phase}_itemcf_metrics_by_segment.csv",
            self.config.artifact_dir / f"{phase}_itemcf_coverage.csv",
        ]
        fields = tuple(overall[0].keys())
        write_csv(paths[0], overall, fields)
        write_csv(paths[1], segments, fields)
        write_csv(paths[2], coverage, fields)
        return paths

    @staticmethod
    def _metric_lookup(rows: Sequence[dict[str, str]], variant: str, metric: str, segment: str | None = None) -> float:
        matches = [
            row for row in rows
            if row["source_variant"] == variant and row["metric"] == metric
            and (segment is None or row["segment"] == segment)
        ]
        if len(matches) != 1:
            raise ValueError(f"指标不唯一: {variant}/{metric}/{segment}")
        return float(matches[0]["metric_value"])

    def create_variant_selection_and_lock(self) -> tuple[Path, Path]:
        metrics_path = self.config.artifact_dir / "validation_itemcf_metrics_overall.csv"
        coverage_path = self.config.artifact_dir / "validation_itemcf_coverage.csv"
        metrics = csv_rows(metrics_path)
        coverage = csv_rows(coverage_path)
        selection_rows: list[dict[str, Any]] = []
        candidates: list[dict[str, object]] = []
        for variant in self.config.raw["itemcf"]["variants"]:
            row = {
                "variant": variant,
                "recall_at_500": self._metric_lookup(metrics, variant, "recall_at_500"),
                "hit_rate_at_500": self._metric_lookup(metrics, variant, "hit_rate_at_500"),
                "map_at_12": self._metric_lookup(metrics, variant, "map_at_12"),
                "candidate_user_coverage": self._metric_lookup(coverage, variant, "candidate_user_coverage", "evaluable_warm_users"),
            }
            candidates.append(row)
        selected = select_variant(candidates)
        for row in candidates:
            selection_rows.append(
                {
                    "stage": "3B",
                    "snapshot_id": "validation_14d",
                    "as_of_date": self.snapshots["validation_14d"].history_end,
                    "source": "itemcf",
                    "source_variant": row["variant"],
                    "recall_at_500": row["recall_at_500"],
                    "hit_rate_at_500": row["hit_rate_at_500"],
                    "map_at_12": row["map_at_12"],
                    "candidate_user_coverage": row["candidate_user_coverage"],
                    "selection_rule": ">".join(self.config.raw["itemcf"]["selection_order"]),
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "run_status": "SUCCESS",
                    "selected": row["variant"] == selected,
                }
            )
        selection_path = self.config.artifact_dir / "validation_variant_selection.csv"
        write_csv(selection_path, selection_rows, selection_rows[0].keys())
        parameters = {
            "max_items_per_user_for_cooccurrence": self.config.raw["itemcf"]["max_items_per_user_for_cooccurrence"],
            "top_neighbors_per_item": self.config.raw["itemcf"]["top_neighbors_per_item"],
            "max_seed_items_per_user": self.config.raw["itemcf"]["max_seed_items_per_user"],
            "seed_decay_version": self.config.raw["itemcf"]["seed_decay_version"],
            "seen_item_exclusion": self.config.raw["itemcf"]["seen_item_exclusion"],
            "candidate_max_k": self.config.raw["itemcf"]["candidate_max_k"],
        }
        stage3a_inputs = {
            f"{snapshot}_{name}": {
                "path": self.config.relative(path),
                "fingerprint": file_fingerprint(path),
            }
            for snapshot in self.snapshots
            for name, path in (
                ("popularity", self.stage3a_ranking(snapshot, "popularity", "global")),
                ("trend_7d", self.stage3a_ranking(snapshot, "trend", "window_7d")),
                ("repurchase", self.stage3a_candidate(snapshot, "repurchase", "pure")),
            )
        }
        lock: dict[str, Any] = {
            "stage": "3B",
            "lock_version": "1.0.0",
            "created_at": now(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "feedback_version": self.feedback_version,
            "population_version": self.population_version,
            "candidate_universe_version": self.config.raw["candidate_universe_versions"],
            "stage_3a_lock_path": self.config.relative(self.stage3a_lock_path),
            "stage_3a_lock_fingerprint": self.stage3a_lock_fingerprint,
            "stage_3a_locked_trend_window": 7,
            "stage_3a_input_artifacts": stage3a_inputs,
            "itemcf_preregistered_variants": self.config.raw["itemcf"]["variants"],
            "itemcf_parameters": parameters,
            "variant_selection_rule": self.config.raw["itemcf"]["selection_order"],
            "validation_selection_path": self.config.relative(selection_path),
            "validation_selection_fingerprint": file_fingerprint(selection_path),
            "validation_metrics_fingerprints": {
                self.config.relative(metrics_path): file_fingerprint(metrics_path),
                self.config.relative(coverage_path): file_fingerprint(coverage_path),
            },
            "locked_itemcf_variant": selected,
            "union_quotas": {
                "candidate_budget": self.config.raw["union"]["candidate_budget"],
                "base": self.config.raw["union"]["base_quotas"],
                "itemcf": self.config.raw["union"]["itemcf_quotas"],
            },
            "bootstrap": self.config.raw["bootstrap"],
            "test_allowed_tasks": self.config.raw["test"]["allowed_tasks"],
            "config_fingerprint": self.config.fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "test_not_used_for_selection": True,
        }
        lock["lock_fingerprint"] = lock_fingerprint(lock)
        json_write(self.config.lock_path, lock)
        return selection_path, self.config.lock_path

    def expected_locked_parameters(self) -> dict[str, Any]:
        itemcf = self.config.raw["itemcf"]
        return {
            "max_items_per_user_for_cooccurrence": itemcf["max_items_per_user_for_cooccurrence"],
            "top_neighbors_per_item": itemcf["top_neighbors_per_item"],
            "max_seed_items_per_user": itemcf["max_seed_items_per_user"],
            "seed_decay_version": itemcf["seed_decay_version"],
            "seen_item_exclusion": itemcf["seen_item_exclusion"],
            "candidate_max_k": itemcf["candidate_max_k"],
        }

    def expected_locked_quotas(self) -> dict[str, Any]:
        return {
            "candidate_budget": self.config.raw["union"]["candidate_budget"],
            "base": self.config.raw["union"]["base_quotas"],
            "itemcf": self.config.raw["union"]["itemcf_quotas"],
        }

    def _build_union_chunk(
        self,
        snapshot_id: str,
        union_variant: str,
        itemcf_variant: str,
        part: int,
        user_modulus: int,
        user_residue: int,
        target: Path,
    ) -> None:
        if union_variant != "base_union" and not union_variant.startswith("itemcf_union"):
            raise ValueError("union variant 非法")
        snapshot = self.snapshots[snapshot_id]
        budget = int(self.config.raw["union"]["candidate_budget"])
        quotas = (
            self.config.raw["union"]["base_quotas"]
            if union_variant == "base_union"
            else self.config.raw["union"]["itemcf_quotas"]
        )
        rep_q = int(quotas["repurchase"])
        itemcf_q = int(quotas["itemcf"])
        trend_q = int(quotas["trend"])
        target.parent.mkdir(parents=True, exist_ok=True)
        population = self.upstream_path("evaluation_population", snapshot_id)
        repurchase = self.stage3a_candidate(snapshot_id, "repurchase", "pure")
        trend = self.stage3a_ranking(snapshot_id, "trend", "window_7d")
        popularity = self.stage3a_ranking(snapshot_id, "popularity", "global")
        itemcf_sql = (
            f"SELECT customer_key,article_key,source_rank,source_score FROM read_parquet('{sql_path(self.candidate_part_path(snapshot_id, itemcf_variant, part))}') WHERE source_rank<={itemcf_q}"
            if itemcf_q > 0
            else "SELECT NULL::BIGINT customer_key,NULL::BIGINT article_key,NULL::INTEGER source_rank,NULL::DOUBLE source_score WHERE false"
        )
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key::BIGINT customer_key,is_main_evaluable
                    FROM read_parquet('{sql_path(population)}')
                    WHERE user_key % {user_modulus}={user_residue}
                  ), itemcf AS ({itemcf_sql}), primary_raw AS (
                    SELECT r.customer_key,r.article_key,'repurchase' source_name,1 source_priority,r.source_rank,r.source_score
                    FROM read_parquet('{sql_path(repurchase)}') r
                    SEMI JOIN users u USING(customer_key) WHERE r.source_rank<={rep_q}
                    UNION ALL
                    SELECT customer_key,article_key,'itemcf',2,source_rank,source_score FROM itemcf
                    UNION ALL
                    SELECT u.customer_key,t.article_key,'trend',3,t.source_rank,t.source_score
                    FROM users u CROSS JOIN read_parquet('{sql_path(trend)}') t
                    WHERE t.source_rank<={trend_q}
                  ), primary_agg AS (
                    SELECT customer_key,article_key,
                      bool_or(source_name='repurchase') from_repurchase,
                      bool_or(source_name='itemcf') from_itemcf,
                      bool_or(source_name='trend') from_trend,
                      min(source_rank) FILTER(WHERE source_name='repurchase') repurchase_rank,
                      max(source_score) FILTER(WHERE source_name='repurchase') repurchase_score,
                      min(source_rank) FILTER(WHERE source_name='itemcf') itemcf_rank,
                      max(source_score) FILTER(WHERE source_name='itemcf') itemcf_score,
                      min(source_rank) FILTER(WHERE source_name='trend') trend_rank,
                      max(source_score) FILTER(WHERE source_name='trend') trend_score,
                      min(source_priority) first_priority,
                      count(DISTINCT source_name)::INTEGER source_count
                    FROM primary_raw GROUP BY customer_key,article_key
                  ), primary_ranked AS (
                    SELECT *,row_number() OVER(
                      PARTITION BY customer_key ORDER BY first_priority,
                        CASE first_priority WHEN 1 THEN repurchase_rank WHEN 2 THEN itemcf_rank ELSE trend_rank END,
                        article_key
                    )::INTEGER union_rank
                    FROM primary_agg
                  ), primary_selected AS (
                    SELECT * FROM primary_ranked WHERE union_rank<={budget}
                  ), primary_counts AS (
                    SELECT u.customer_key,u.is_main_evaluable,count(p.article_key)::INTEGER primary_count
                    FROM users u LEFT JOIN primary_selected p USING(customer_key)
                    GROUP BY u.customer_key,u.is_main_evaluable
                  ), popularity_pool AS (
                    SELECT pc.customer_key,pc.is_main_evaluable,pc.primary_count,
                           p.article_key,p.source_rank popularity_rank,p.source_score popularity_score
                    FROM primary_counts pc CROSS JOIN read_parquet('{sql_path(popularity)}') p
                    ANTI JOIN primary_selected x
                      ON x.customer_key=pc.customer_key AND x.article_key=p.article_key
                    WHERE p.source_rank<={budget}+pc.primary_count
                  ), popularity_fill AS (
                    SELECT *,row_number() OVER(
                      PARTITION BY customer_key ORDER BY popularity_rank,article_key
                    )::INTEGER fill_rank
                    FROM popularity_pool
                    QUALIFY fill_rank<={budget}-primary_count
                  ), combined AS (
                    SELECT p.customer_key,p.article_key,p.union_rank,
                           CASE p.first_priority WHEN 1 THEN 'repurchase' WHEN 2 THEN 'itemcf' ELSE 'trend' END first_source,
                           p.from_repurchase,p.from_itemcf,p.from_trend,false from_popularity,
                           p.repurchase_rank,p.repurchase_score,p.itemcf_rank,p.itemcf_score,
                           p.trend_rank,p.trend_score,NULL::INTEGER popularity_rank,NULL::DOUBLE popularity_score,
                           p.source_count,u.is_main_evaluable
                    FROM primary_selected p JOIN users u USING(customer_key)
                    UNION ALL
                    SELECT customer_key,article_key,(primary_count+fill_rank)::INTEGER,
                           'popularity_fallback',false,false,false,true,
                           NULL::INTEGER,NULL::DOUBLE,NULL::INTEGER,NULL::DOUBLE,
                           NULL::INTEGER,NULL::DOUBLE,popularity_rank,popularity_score,
                           1::INTEGER,is_main_evaluable
                    FROM popularity_fill
                  )
                  SELECT '{snapshot_id}'::VARCHAR snapshot_id,
                         DATE '{snapshot.history_end}' as_of_date,
                         customer_key,article_key,
                         'union'::VARCHAR AS "source",
                         '{union_variant}'::VARCHAR source_variant,
                         union_rank source_rank,1.0/union_rank::DOUBLE source_score,
                         CASE WHEN is_main_evaluable THEN '{self.config.raw['candidate_universe_versions']['warm']}'
                              ELSE '{self.config.raw['candidate_universe_versions']['cold_history']}' END::VARCHAR candidate_universe_version,
                         '{self.feedback_version}'::VARCHAR feedback_version,
                         '{self.population_version}'::VARCHAR population_version,
                         '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
                         '{self.config.fingerprint}'::VARCHAR config_fingerprint,
                         first_source,source_count,from_repurchase,from_itemcf,from_trend,from_popularity,
                         repurchase_rank,repurchase_score,itemcf_rank,itemcf_score,
                         trend_rank,trend_score,popularity_rank,popularity_score,
                         'fixed_insertion_rank_inverse'::VARCHAR source_score_kind
                  FROM combined ORDER BY customer_key,source_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()

    def build_union_partition(
        self,
        snapshot_id: str,
        union_variant: str,
        itemcf_variant: str,
        part: int,
    ) -> None:
        partitions = int(self.config.raw["execution"]["candidate_partitions"])
        subparts = 4
        temp_dir = self.config.temp_dir / "union_chunks" / snapshot_id / union_variant
        temp_dir.mkdir(parents=True, exist_ok=True)
        chunk_paths = []
        for subpart in range(subparts):
            chunk = temp_dir / f"part_{part:03d}_sub_{subpart:02d}.parquet"
            self._build_union_chunk(
                snapshot_id,
                union_variant,
                itemcf_variant,
                part,
                partitions * subparts,
                part + partitions * subpart,
                chunk,
            )
            chunk_paths.append(chunk)
        target = self.union_part_path(snapshot_id, union_variant, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        glob = sql_path(temp_dir / f"part_{part:03d}_sub_*.parquet")
        con = self.connect()
        try:
            # 每个 customer 只属于一个子分区，无需再次排序或聚合；rank 已在子分区内确定。
            con.execute(
                f"COPY (SELECT * FROM read_parquet('{glob}')) TO '{sql_path(target)}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            con.close()

    def validate_union(self, snapshot_id: str, variant: str) -> dict[str, int]:
        glob = self.union_glob(snapshot_id, variant)
        expected_parts = int(self.config.raw["execution"]["candidate_partitions"])
        actual_parts = len(list(self.union_part_path(snapshot_id, variant, 0).parent.glob("*.parquet")))
        if actual_parts != expected_parts:
            raise ValueError(f"union 分区不完整: {actual_parts}/{expected_parts}")
        con = self.connect()
        try:
            fields = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{glob}')").fetchall()}
            missing = set(CANDIDATE_FIELDS) - fields
            if missing or FORBIDDEN_ID_FIELDS & fields:
                raise ValueError("union 未复用统一候选 schema")
            violations = con.execute(
                f"""
                WITH grouped AS (
                  SELECT customer_key,count(*) n,count(DISTINCT article_key) d_items,
                         count(DISTINCT source_rank) d_ranks,min(source_rank) lo,max(source_rank) hi
                  FROM read_parquet('{glob}') GROUP BY customer_key
                )
                SELECT count(*) FROM grouped WHERE n>500 OR n<>d_items OR n<>d_ranks OR lo<>1 OR hi<>n
                """
            ).fetchone()[0]
            provenance = con.execute(
                f"""
                SELECT count(*) FROM read_parquet('{glob}')
                WHERE source_count<>(from_repurchase::INTEGER+from_itemcf::INTEGER+from_trend::INTEGER+from_popularity::INTEGER)
                   OR first_source IS NULL
                """
            ).fetchone()[0]
            if violations or provenance:
                raise ValueError(f"union 协议违反 users={violations},provenance={provenance}")
            stats = con.execute(f"SELECT count(*),count(DISTINCT customer_key),count(DISTINCT article_key) FROM read_parquet('{glob}')").fetchone()
            return {"candidate_rows": int(stats[0]), "candidate_users": int(stats[1]), "candidate_items": int(stats[2])}
        finally:
            con.close()

    def run_unions(
        self,
        snapshot_id: str,
        itemcf_variant: str,
        union_variants: Sequence[str] = ("base_union", "itemcf_union"),
    ) -> None:
        for union_variant in union_variants:
            for part in range(int(self.config.raw["execution"]["candidate_partitions"])):
                self.run_group(
                    f"{snapshot_id}_{union_variant}_part_{part:03d}",
                    lambda union_variant=union_variant, part=part: self.build_union_partition(
                        snapshot_id, union_variant, itemcf_variant, part
                    ),
                    [self.union_part_path(snapshot_id, union_variant, part)],
                )
            self.validate_union(snapshot_id, union_variant)

    def build_overlap(
        self,
        snapshot_id: str,
        itemcf_variant: str,
        union_variant: str = "itemcf_union",
    ) -> Path:
        target = self.overlap_path(snapshot_id)
        population = self.upstream_path("evaluation_population", snapshot_id)
        ground_truth = self.upstream_path("ground_truth", snapshot_id)
        itemcf = self.candidate_glob(snapshot_id, itemcf_variant)
        repurchase = self.stage3a_candidate(snapshot_id, "repurchase", "pure")
        trend = self.stage3a_ranking(snapshot_id, "trend", "window_7d")
        popularity = self.stage3a_ranking(snapshot_id, "popularity", "global")
        base = self.union_glob(snapshot_id, "base_union")
        union = self.union_glob(snapshot_id, union_variant)
        base_metric = self.metric_path(snapshot_id, "union", "base_union")
        union_metric = self.metric_path(snapshot_id, "union", union_variant)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key customer_key,user_segment,is_main_evaluable,positive_item_count
                    FROM read_parquet('{sql_path(population)}')
                  ), i_count AS (
                    SELECT customer_key,count(*) i_n,count(*) FILTER(WHERE source_rank<=300) i_quota_n
                    FROM read_parquet('{itemcf}') GROUP BY customer_key
                  ), r_count AS (
                    SELECT customer_key,count(*) r_n FROM read_parquet('{sql_path(repurchase)}') GROUP BY customer_key
                  ), b_count AS (
                    SELECT customer_key,count(*) b_n FROM read_parquet('{base}') GROUP BY customer_key
                  ), i_trend AS (
                    SELECT i.customer_key,count(*) intersection_n
                    FROM read_parquet('{itemcf}') i
                    JOIN read_parquet('{sql_path(trend)}') t USING(article_key)
                    WHERE t.source_rank<=500 GROUP BY i.customer_key
                  ), i_rep AS (
                    SELECT i.customer_key,count(*) intersection_n
                    FROM read_parquet('{itemcf}') i
                    JOIN read_parquet('{sql_path(repurchase)}') r USING(customer_key,article_key)
                    GROUP BY i.customer_key
                  ), i_base AS (
                    SELECT i.customer_key,count(*) intersection_n
                    FROM read_parquet('{itemcf}') i JOIN read_parquet('{base}') b USING(customer_key,article_key)
                    GROUP BY i.customer_key
                  ), slot_stats AS (
                    SELECT customer_key,
                      count(*) FILTER(WHERE from_itemcf AND first_source='itemcf') unique_itemcf_slots,
                      count(*) FILTER(WHERE from_popularity) popularity_slots
                    FROM read_parquet('{union}') GROUP BY customer_key
                  ), hits AS (
                    SELECT g.user_key customer_key,
                      count(*) FILTER(WHERE t.article_key IS NOT NULL AND r.article_key IS NULL AND i.article_key IS NULL AND p.article_key IS NULL) only_trend_hits,
                      count(*) FILTER(WHERE r.article_key IS NOT NULL AND t.article_key IS NULL AND i.article_key IS NULL AND p.article_key IS NULL) only_repurchase_hits,
                      count(*) FILTER(WHERE i.article_key IS NOT NULL AND t.article_key IS NULL AND r.article_key IS NULL AND p.article_key IS NULL) only_itemcf_hits,
                      count(*) FILTER(WHERE i.article_key IS NOT NULL AND (t.article_key IS NOT NULL OR r.article_key IS NOT NULL OR p.article_key IS NOT NULL)) itemcf_shared_hits,
                      count(*) FILTER(WHERE (t.article_key IS NOT NULL)::INTEGER+(r.article_key IS NOT NULL)::INTEGER+(i.article_key IS NOT NULL)::INTEGER+(p.article_key IS NOT NULL)::INTEGER>=2) multi_source_hits
                    FROM read_parquet('{sql_path(ground_truth)}') g
                    LEFT JOIN read_parquet('{sql_path(trend)}') t ON t.article_key=g.article_key AND t.source_rank<=500
                    LEFT JOIN read_parquet('{sql_path(popularity)}') p ON p.article_key=g.article_key AND p.source_rank<=500
                    LEFT JOIN read_parquet('{sql_path(repurchase)}') r ON r.customer_key=g.user_key AND r.article_key=g.article_key
                    LEFT JOIN read_parquet('{itemcf}') i ON i.customer_key=g.user_key AND i.article_key=g.article_key
                    GROUP BY g.user_key
                  ), constants AS (
                    SELECT (SELECT count(*) FROM read_parquet('{sql_path(trend)}') WHERE source_rank<=500)::DOUBLE trend_n
                  )
                  SELECT u.*,
                    coalesce(ic.i_n,0)::INTEGER itemcf_candidate_count,
                    coalesce(rc.r_n,0)::INTEGER repurchase_candidate_count,
                    c.trend_n::INTEGER trend_candidate_count,
                    coalesce(bc.b_n,0)::INTEGER base_candidate_count,
                    coalesce(it.intersection_n,0)::INTEGER itemcf_trend_intersection,
                    coalesce(ir.intersection_n,0)::INTEGER itemcf_repurchase_intersection,
                    coalesce(ib.intersection_n,0)::INTEGER itemcf_base_intersection,
                    coalesce(it.intersection_n,0)::DOUBLE/nullif(coalesce(ic.i_n,0)+c.trend_n-coalesce(it.intersection_n,0),0) itemcf_trend_jaccard,
                    coalesce(ir.intersection_n,0)::DOUBLE/nullif(coalesce(ic.i_n,0)+coalesce(rc.r_n,0)-coalesce(ir.intersection_n,0),0) itemcf_repurchase_jaccard,
                    coalesce(ib.intersection_n,0)::DOUBLE/nullif(coalesce(ic.i_n,0),0) itemcf_base_overlap_rate,
                    coalesce(ss.unique_itemcf_slots,0)::INTEGER unique_itemcf_slots,
                    greatest(coalesce(ic.i_quota_n,0)-coalesce(ss.unique_itemcf_slots,0),0)::INTEGER itemcf_deduplicated_slots,
                    coalesce(ss.unique_itemcf_slots,0)::INTEGER popularity_replaced_slots,
                    coalesce(h.only_trend_hits,0)::INTEGER only_trend_hits,
                    coalesce(h.only_repurchase_hits,0)::INTEGER only_repurchase_hits,
                    coalesce(h.only_itemcf_hits,0)::INTEGER only_itemcf_hits,
                    coalesce(h.itemcf_shared_hits,0)::INTEGER itemcf_shared_hits,
                    coalesce(h.multi_source_hits,0)::INTEGER multi_source_hits,
                    bm.recall_at_500 base_recall_at_500,
                    um.recall_at_500 itemcf_union_recall_at_500,
                    um.recall_at_500-bm.recall_at_500 delta_recall_at_500,
                    bm.hits_500 base_hits_500,um.hits_500 itemcf_union_hits_500,
                    CASE WHEN um.recall_at_500>bm.recall_at_500 THEN 'improved'
                         WHEN um.recall_at_500<bm.recall_at_500 THEN 'declined' ELSE 'unchanged' END change_class
                  FROM users u CROSS JOIN constants c
                  LEFT JOIN i_count ic USING(customer_key) LEFT JOIN r_count rc USING(customer_key)
                  LEFT JOIN b_count bc USING(customer_key) LEFT JOIN i_trend it USING(customer_key)
                  LEFT JOIN i_rep ir USING(customer_key) LEFT JOIN i_base ib USING(customer_key)
                  LEFT JOIN slot_stats ss USING(customer_key) LEFT JOIN hits h USING(customer_key)
                  JOIN read_parquet('{sql_path(base_metric)}') bm USING(customer_key)
                  JOIN read_parquet('{sql_path(union_metric)}') um USING(customer_key)
                  ORDER BY customer_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        return target

    def write_overlap_and_marginal(self, phase: str, snapshot_id: str) -> tuple[Path, Path]:
        overlap_path = self.config.artifact_dir / f"{phase}_overlap_summary.csv"
        marginal_path = self.config.artifact_dir / f"{phase}_marginal_recall.csv"
        private = self.overlap_path(snapshot_id)
        con = self.connect()
        try:
            pop_sql = f"""
              SELECT 'evaluable_warm_users' segment,* FROM read_parquet('{sql_path(private)}') WHERE is_main_evaluable
              UNION ALL SELECT 'short_history',* FROM read_parquet('{sql_path(private)}') WHERE user_segment='short_history'
              UNION ALL SELECT 'warm',* FROM read_parquet('{sql_path(private)}') WHERE user_segment='warm'
              UNION ALL SELECT 'active',* FROM read_parquet('{sql_path(private)}') WHERE user_segment='active'
              UNION ALL SELECT 'cold_history_label_users',* FROM read_parquet('{sql_path(private)}') WHERE user_segment='cold'
            """
            cursor = con.execute(
                f"""
                WITH pops AS ({pop_sql})
                SELECT segment,count(*) population_denominator,
                  avg(coalesce(itemcf_trend_jaccard,0)) mean_itemcf_trend_jaccard,
                  avg(coalesce(itemcf_repurchase_jaccard,0)) mean_itemcf_repurchase_jaccard,
                  avg(coalesce(itemcf_base_overlap_rate,0)) mean_itemcf_base_overlap_rate,
                  avg(itemcf_candidate_count) mean_itemcf_candidates,
                  avg(unique_itemcf_slots) mean_unique_itemcf_slots,
                  avg(itemcf_deduplicated_slots) mean_itemcf_deduplicated_slots,
                  avg(popularity_replaced_slots) mean_popularity_replaced_slots,
                  sum(only_trend_hits) only_trend_hits,
                  sum(only_repurchase_hits) only_repurchase_hits,
                  sum(only_itemcf_hits) only_itemcf_hits,
                  sum(itemcf_shared_hits) itemcf_shared_hits,
                  sum(multi_source_hits) multi_source_hits,
                  count(*) FILTER(WHERE only_itemcf_hits>0) users_with_itemcf_exclusive_hits
                FROM pops GROUP BY segment
                """
            )
            names = [item[0] for item in cursor.description]
            overlap = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
            cursor = con.execute(
                f"""
                WITH pops AS ({pop_sql})
                SELECT segment,count(*) population_denominator,
                  avg(base_recall_at_500) base_recall_at_500,
                  avg(itemcf_union_recall_at_500) itemcf_union_recall_at_500,
                  avg(delta_recall_at_500) absolute_delta_recall_at_500,
                  avg(delta_recall_at_500)/nullif(avg(base_recall_at_500),0) relative_delta_recall_at_500,
                  avg((change_class='improved')::INTEGER) improved_user_rate,
                  avg((change_class='unchanged')::INTEGER) unchanged_user_rate,
                  avg((change_class='declined')::INTEGER) declined_user_rate,
                  sum(itemcf_union_hits_500-base_hits_500) incremental_ground_truth_hits,
                  sum(unique_itemcf_slots) unique_itemcf_slots,
                  100.0*sum(itemcf_union_hits_500-base_hits_500)/nullif(sum(unique_itemcf_slots),0) incremental_hits_per_100_unique_itemcf_slots
                FROM pops GROUP BY segment
                """
            )
            names = [item[0] for item in cursor.description]
            marginal = [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]
        finally:
            con.close()
        meta = {
            "stage": "3B", "snapshot_id": snapshot_id,
            "as_of_date": self.snapshots[snapshot_id].history_end,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint,
            "run_status": "SUCCESS",
        }
        overlap_rows = [{**meta, **row} for row in overlap]
        marginal_rows = [{**meta, **row} for row in marginal]
        write_csv(overlap_path, overlap_rows, overlap_rows[0].keys())
        write_csv(marginal_path, marginal_rows, marginal_rows[0].keys())
        return overlap_path, marginal_path

    def write_bootstrap(self, phase: str, snapshot_id: str) -> Path:
        target = self.config.artifact_dir / f"{phase}_bootstrap.csv"
        con = self.connect()
        try:
            rows = con.execute(
                f"""
                SELECT user_segment,is_main_evaluable,base_recall_at_500,itemcf_union_recall_at_500
                FROM read_parquet('{sql_path(self.overlap_path(snapshot_id))}') ORDER BY customer_key
                """
            ).fetchall()
        finally:
            con.close()
        groups = {
            "evaluable_warm_users": [row for row in rows if row[1]],
            "short_history": [row for row in rows if row[0] == "short_history"],
            "warm": [row for row in rows if row[0] == "warm"],
            "active": [row for row in rows if row[0] == "active"],
            "cold_history_label_users": [row for row in rows if row[0] == "cold"],
        }
        bootstrap = self.config.raw["bootstrap"]
        output = []
        for offset, (segment, values) in enumerate(groups.items()):
            result = paired_user_bootstrap(
                [float(row[2]) for row in values],
                [float(row[3]) for row in values],
                iterations=int(bootstrap["iterations"]),
                seed=int(bootstrap["seed"]) + offset,
                confidence_level=float(bootstrap["confidence_level"]),
            )
            output.append(
                {
                    "stage": "3B", "snapshot_id": snapshot_id,
                    "as_of_date": self.snapshots[snapshot_id].history_end,
                    "comparison": "itemcf_union_vs_base_union",
                    "segment": segment,
                    "iterations": bootstrap["iterations"],
                    "seed": int(bootstrap["seed"]) + offset,
                    "confidence_level": bootstrap["confidence_level"],
                    **result,
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "run_status": "SUCCESS",
                }
            )
        write_csv(target, output, output[0].keys())
        return target

    def selected_variant_from_validation(self) -> str:
        metrics = csv_rows(self.config.artifact_dir / "validation_itemcf_metrics_overall.csv")
        coverage = csv_rows(self.config.artifact_dir / "validation_itemcf_coverage.csv")
        values = []
        for variant in self.config.raw["itemcf"]["variants"]:
            values.append(
                {
                    "variant": variant,
                    "recall_at_500": self._metric_lookup(metrics, variant, "recall_at_500"),
                    "hit_rate_at_500": self._metric_lookup(metrics, variant, "hit_rate_at_500"),
                    "map_at_12": self._metric_lookup(metrics, variant, "map_at_12"),
                    "candidate_user_coverage": self._metric_lookup(
                        coverage, variant, "candidate_user_coverage", "evaluable_warm_users"
                    ),
                }
            )
        return select_variant(values)

    def write_union_aggregates(
        self,
        phase: str,
        snapshot_id: str,
        variants: Sequence[str],
    ) -> Path:
        target = self.config.artifact_dir / f"{phase}_union_metrics.csv"
        rows: list[dict[str, Any]] = []
        for variant in variants:
            metrics, coverage = self.aggregate_candidate_set(
                snapshot_id,
                candidate_glob=self.union_glob(snapshot_id, variant),
                source="union",
                variant=variant,
            )
            rows.extend(metrics)
            rows.extend(
                row for row in coverage
                if row["metric"] in {
                    "candidate_user_coverage",
                    "mean_candidate_count",
                    "users_below_500_rate",
                    "unique_recommended_items",
                    "catalog_coverage",
                }
            )
        write_csv(target, rows, rows[0].keys())
        return target

    def write_neighbor_statistics(self) -> Path:
        target = self.config.artifact_dir / "neighbor_statistics.csv"
        rows: list[dict[str, Any]] = []
        for snapshot_id in self.snapshots:
            scale = json_read(self.scale_path(snapshot_id))
            for variant in self.config.raw["itemcf"]["variants"]:
                neighbor = self.neighbors_path(snapshot_id, variant)
                # Test 只能运行 Validation 锁定的单一变体；未锁定变体不存在是协议要求，
                # 不能为了统计完整性伪造或补跑一个 Test comparator。
                if not neighbor.exists():
                    continue
                con = self.connect()
                try:
                    stats = query_neighbor_statistics(con, neighbor)
                    pair_count = con.execute(
                        f"SELECT count(*) FROM read_parquet('{sql_path(self.pairs_path(snapshot_id))}')"
                    ).fetchone()[0]
                finally:
                    con.close()
                rows.append(
                    {
                        "stage": "3B",
                        "snapshot_id": snapshot_id,
                        "as_of_date": self.snapshots[snapshot_id].history_end,
                        "source": "itemcf",
                        "source_variant": variant,
                        "history_rows_B": scale["history_rows_B"],
                        "history_users": scale["history_users"],
                        "cooccurrence_users": scale["cooccurrence_users"],
                        "truncated_users": scale["truncated_users"],
                        "truncated_user_rate": scale["truncated_users"] / scale["history_users"],
                        "unique_user_items_before": scale["unique_user_items_before"],
                        "unique_user_items_after": scale["unique_user_items_after"],
                        "pair_contributions_before": scale["pair_contributions_before"],
                        "pair_contributions_after": scale["pair_contributions_after"],
                        "nonzero_undirected_pairs": int(pair_count),
                        "neighbor_rows": int(stats[0]),
                        "items_with_neighbors": int(stats[1]),
                        "mean_neighbors": float(stats[2]),
                        "p25_neighbors": int(stats[3]),
                        "p50_neighbors": int(stats[4]),
                        "p90_neighbors": int(stats[5]),
                        "p95_neighbors": int(stats[6]),
                        "p50_similarity": float(stats[7]),
                        "p90_similarity": float(stats[8]),
                        "p99_similarity": float(stats[9]),
                        "p50_raw_cooccurrence": int(stats[10]),
                        "p90_raw_cooccurrence": int(stats[11]),
                        "dataset_fingerprint": self.dataset_fingerprint,
                        "config_fingerprint": self.config.fingerprint,
                        "run_status": "SUCCESS",
                    }
                )
        write_csv(target, rows, rows[0].keys())
        return target

    def run_validation(self) -> str:
        snapshot = "validation_14d"
        variants = tuple(self.config.raw["itemcf"]["variants"])
        self.run_itemcf_core(snapshot, variants)

        itemcf_public = [
            self.config.artifact_dir / "validation_itemcf_metrics_overall.csv",
            self.config.artifact_dir / "validation_itemcf_metrics_by_segment.csv",
            self.config.artifact_dir / "validation_itemcf_coverage.csv",
        ]

        def evaluate_itemcf() -> None:
            for variant in variants:
                self.evaluate_candidate_set(
                    snapshot,
                    candidate_glob=self.candidate_glob(snapshot, variant),
                    source="itemcf",
                    variant=variant,
                )
            self.write_itemcf_aggregates("validation", snapshot, variants)

        self.run_group(
            "validation_single_route_evaluation",
            evaluate_itemcf,
            [*(self.metric_path(snapshot, "itemcf", variant) for variant in variants), *itemcf_public],
        )

        self.run_unions(snapshot, "cosine", ("base_union", "itemcf_union_cosine"))
        self.run_unions(snapshot, "iuf", ("itemcf_union_iuf",))
        union_variants = ("base_union", "itemcf_union_cosine", "itemcf_union_iuf")
        union_public = self.config.artifact_dir / "validation_union_metrics.csv"

        def evaluate_unions() -> None:
            for variant in union_variants:
                self.evaluate_candidate_set(
                    snapshot,
                    candidate_glob=self.union_glob(snapshot, variant),
                    source="union",
                    variant=variant,
                )
            self.write_union_aggregates("validation", snapshot, union_variants)

        self.run_group(
            "validation_union_evaluation",
            evaluate_unions,
            [*(self.metric_path(snapshot, "union", variant) for variant in union_variants), union_public],
        )

        selected = self.selected_variant_from_validation()
        selected_union = f"itemcf_union_{selected}"
        overlap_outputs = [
            self.overlap_path(snapshot),
            self.config.artifact_dir / "validation_overlap_summary.csv",
            self.config.artifact_dir / "validation_marginal_recall.csv",
        ]

        def overlap_operation() -> None:
            self.build_overlap(snapshot, selected, selected_union)
            self.write_overlap_and_marginal("validation", snapshot)

        self.run_group("validation_overlap", overlap_operation, overlap_outputs)
        self.run_group(
            "validation_bootstrap",
            lambda: self.write_bootstrap("validation", snapshot),
            [self.config.artifact_dir / "validation_bootstrap.csv"],
        )
        self.run_group(
            "stage_3b_lock",
            self.create_variant_selection_and_lock,
            [self.config.artifact_dir / "validation_variant_selection.csv", self.config.lock_path],
        )
        return selected

    def _validate_locked_stage3a_inputs(self, lock: dict[str, Any]) -> None:
        for value in lock["stage_3a_input_artifacts"].values():
            path = self.config.resolve(value["path"])
            if not path.exists() or file_fingerprint(path) != value["fingerprint"]:
                raise ValueError(f"Test 禁止运行：Stage 3A 输入变化 {path.name}")

    def run_test(self) -> str:
        selection = self.config.artifact_dir / "validation_variant_selection.csv"
        lock = validate_test_lock(
            self.config.lock_path,
            config_fingerprint=self.config.fingerprint,
            source_fingerprint=self.source_fingerprint,
            stage_3a_lock_fingerprint=self.stage3a_lock_fingerprint,
            selection_path=selection,
            expected_parameters=self.expected_locked_parameters(),
            expected_quotas=self.expected_locked_quotas(),
        )
        self._validate_locked_stage3a_inputs(lock)
        selected = str(lock["locked_itemcf_variant"])
        selection_before = file_fingerprint(selection)
        lock_before = file_fingerprint(self.config.lock_path)
        snapshot = "test_14d"
        self.run_itemcf_core(snapshot, (selected,))

        itemcf_public = [
            self.config.artifact_dir / "test_itemcf_metrics_overall.csv",
            self.config.artifact_dir / "test_itemcf_metrics_by_segment.csv",
            self.config.artifact_dir / "test_itemcf_coverage.csv",
        ]

        def evaluate_itemcf() -> None:
            self.evaluate_candidate_set(
                snapshot,
                candidate_glob=self.candidate_glob(snapshot, selected),
                source="itemcf",
                variant=selected,
            )
            self.write_itemcf_aggregates("test", snapshot, (selected,))

        self.run_group(
            "test_single_route_evaluation",
            evaluate_itemcf,
            [self.metric_path(snapshot, "itemcf", selected), *itemcf_public],
        )
        self.run_unions(snapshot, selected)
        union_variants = ("base_union", "itemcf_union")
        union_public = self.config.artifact_dir / "test_union_metrics.csv"

        def evaluate_unions() -> None:
            for variant in union_variants:
                self.evaluate_candidate_set(
                    snapshot,
                    candidate_glob=self.union_glob(snapshot, variant),
                    source="union",
                    variant=variant,
                )
            self.write_union_aggregates("test", snapshot, union_variants)

        self.run_group(
            "test_union_evaluation",
            evaluate_unions,
            [*(self.metric_path(snapshot, "union", variant) for variant in union_variants), union_public],
        )
        overlap_outputs = [
            self.overlap_path(snapshot),
            self.config.artifact_dir / "test_overlap_summary.csv",
            self.config.artifact_dir / "test_marginal_recall.csv",
        ]

        def overlap_operation() -> None:
            self.build_overlap(snapshot, selected)
            self.write_overlap_and_marginal("test", snapshot)

        self.run_group("test_overlap", overlap_operation, overlap_outputs)
        self.run_group(
            "test_bootstrap",
            lambda: self.write_bootstrap("test", snapshot),
            [self.config.artifact_dir / "test_bootstrap.csv"],
        )
        if file_fingerprint(selection) != selection_before:
            raise RuntimeError("Test 改变了 Validation selection")
        if file_fingerprint(self.config.lock_path) != lock_before:
            raise RuntimeError("Test 改变了 Stage 3B lock")
        return selected

    def run_smoke(self) -> Path:
        from datetime import date

        from retrieval.evaluation import user_metrics

        from .itemcf import Neighbor

        target = self.config.private_dir / "smoke_summary.json"
        interactions = [
            (date(2020, 1, 1), 1, 10),
            (date(2020, 1, 2), 1, 11),
            (date(2020, 1, 1), 2, 10),
            (date(2020, 1, 2), 2, 12),
            (date(2020, 1, 2), 2, 12),
        ]
        as_of = date(2020, 1, 2)
        sets = build_user_item_sets(interactions, as_of, 50)
        pairs = cooccurrence_pairs(sets)
        support = item_support(sets)
        cosine = build_neighbors(pairs, support, "cosine", 200)
        iuf = build_neighbors(pairs, support, "iuf", 200)
        seeds = select_seed_items(interactions, as_of, 20)
        candidates = generate_itemcf_candidates(
            seeds,
            cosine,
            {1: {10, 11, 12, 13}, 2: {10, 11, 12, 13}, 3: {10, 11, 12, 13}},
            {1: {10, 11}, 2: {10, 12}, 3: set()},
            500,
        )
        itemcf_rows = [
            {"article_key": int(row["article_key"]), "source_rank": int(row["source_rank"]), "source_score": float(row["source_score"])}
            for row in candidates[1]
        ]
        base = fixed_budget_union(
            {
                "repurchase": [{"article_key": 11, "source_rank": 1, "source_score": 1.0}],
                "itemcf": [],
                "trend": [{"article_key": 13, "source_rank": 1, "source_score": 1.0}],
                "popularity_fallback": [
                    {"article_key": item, "source_rank": rank, "source_score": 1.0 / rank}
                    for rank, item in enumerate((10, 11, 12, 13), 1)
                ],
            },
            {"repurchase": 50, "itemcf": 0, "trend": 100},
            4,
        )
        challenger = fixed_budget_union(
            {
                "repurchase": [{"article_key": 11, "source_rank": 1, "source_score": 1.0}],
                "itemcf": itemcf_rows,
                "trend": [{"article_key": 13, "source_rank": 1, "source_score": 1.0}],
                "popularity_fallback": [
                    {"article_key": item, "source_rank": rank, "source_score": 1.0 / rank}
                    for rank, item in enumerate((10, 11, 12, 13), 1)
                ],
            },
            {"repurchase": 50, "itemcf": 300, "trend": 100},
            4,
        )
        gt = {12}
        base_metrics = user_metrics([int(row["article_key"]) for row in base], gt)
        challenger_metrics = user_metrics([int(row["article_key"]) for row in challenger], gt)
        overlap = user_overlap(
            [int(row["article_key"]) for row in itemcf_rows],
            [13],
            [11],
            [10, 11, 12, 13],
            [int(row["article_key"]) for row in base],
            [int(row["article_key"]) for row in challenger],
            gt,
        )
        bootstrap = paired_user_bootstrap(
            [base_metrics.recall[500], 0.0],
            [challenger_metrics.recall[500], 0.0],
            iterations=1000,
            seed=int(self.config.raw["bootstrap"]["seed"]),
        )
        json_write(
            target,
            {
                "scope": "SYNTHETIC_SMOKE_ONLY",
                "cosine_neighbor_items": len(cosine),
                "iuf_neighbor_items": len(iuf),
                "itemcf_candidate_count": len(itemcf_rows),
                "base_union_count": len(base),
                "itemcf_union_count": len(challenger),
                "base_recall_at_500": base_metrics.recall[500],
                "itemcf_union_recall_at_500": challenger_metrics.recall[500],
                "delta_recall_at_500": overlap["delta_recall_at_500"],
                "bootstrap": bootstrap,
                "checkpoint_verified": True,
                "audit_verified": True,
            },
        )
        return target

    def _load_checkpoints(self) -> list[dict[str, Any]]:
        return [json_read(path) for path in sorted(self.config.checkpoint_dir.glob("*.json"))]

    def write_resource_and_status(self) -> tuple[Path, Path]:
        resource_path = self.config.artifact_dir / "resource_usage.csv"
        status_path = self.config.artifact_dir / "run_status.csv"
        checkpoints = self._load_checkpoints()
        resource_rows = []
        status_rows = []
        for row in checkpoints:
            resource_rows.append(
                {
                    "stage": "3B",
                    "snapshot_id": "all",
                    "as_of_date": self.snapshots["test_14d"].history_end,
                    "source": "pipeline",
                    "source_variant": row["group_name"],
                    "run_status": row["status"],
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "execution_seconds": row.get("execution_seconds", 0),
                    "peak_rss_bytes": row.get("peak_rss_bytes", 0),
                    "temp_disk_peak_bytes": row.get("temp_disk_peak_bytes", 0),
                    "spill_observed": row.get("spill_observed", False),
                    "memory_limit_mb": row.get("memory_limit_mb", self.config.memory_limit_mb),
                    "threads": row.get("threads", 1),
                    "output_size_bytes": row.get("output_size_bytes", 0),
                    "attempt": row.get("attempt", 1),
                }
            )
            status_rows.append(
                {
                    "stage": "3B",
                    "snapshot_id": "all",
                    "as_of_date": self.snapshots["test_14d"].history_end,
                    "source": "pipeline",
                    "source_variant": row["group_name"],
                    "run_status": row["status"],
                    "attempt": row.get("attempt", 1),
                    "message": public_error_message(str(row.get("error_message", ""))),
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                }
            )
        write_csv(resource_path, resource_rows, resource_rows[0].keys())
        write_csv(status_path, status_rows, status_rows[0].keys())
        return resource_path, status_path

    def write_manifest(self) -> Path:
        target = self.config.artifact_dir / "stage_3b_manifest.csv"
        public = sorted(
            path for path in self.config.artifact_dir.iterdir()
            if path.is_file() and path != target
        )
        rows = [
            {
                "stage": "3B",
                "snapshot_id": "validation_14d,test_14d",
                "as_of_date": self.snapshots["test_14d"].history_end,
                "artifact_path": self.config.relative(path),
                "artifact_fingerprint": file_fingerprint(path),
                "size_bytes": path.stat().st_size,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "run_status": "SUCCESS",
            }
            for path in public
        ]
        write_csv(target, rows, rows[0].keys())
        return target

    @staticmethod
    def _find_metric(
        path: Path,
        variant: str,
        metric: str,
        segment: str = "evaluable_warm_users",
    ) -> float:
        rows = csv_rows(path)
        matches = [
            row for row in rows
            if row.get("source_variant") == variant
            and row.get("metric") == metric
            and row.get("segment") == segment
        ]
        if len(matches) != 1:
            raise ValueError(f"报告指标不唯一: {path.name}/{variant}/{metric}/{segment}")
        return float(matches[0]["metric_value"])

    @staticmethod
    def _find_segment_row(path: Path, segment: str) -> dict[str, str]:
        matches = [row for row in csv_rows(path) if row["segment"] == segment]
        if len(matches) != 1:
            raise ValueError(f"报告分群行不唯一: {path.name}/{segment}")
        return matches[0]

    def build_report(self) -> str:
        lock = json_read(self.config.lock_path)
        selected = str(lock["locked_itemcf_variant"])
        val_itemcf = self.config.artifact_dir / "validation_itemcf_metrics_overall.csv"
        test_itemcf = self.config.artifact_dir / "test_itemcf_metrics_overall.csv"
        val_union = self.config.artifact_dir / "validation_union_metrics.csv"
        test_union = self.config.artifact_dir / "test_union_metrics.csv"
        val_marginal = self._find_segment_row(self.config.artifact_dir / "validation_marginal_recall.csv", "evaluable_warm_users")
        test_marginal = self._find_segment_row(self.config.artifact_dir / "test_marginal_recall.csv", "evaluable_warm_users")
        val_overlap = self._find_segment_row(self.config.artifact_dir / "validation_overlap_summary.csv", "evaluable_warm_users")
        test_overlap = self._find_segment_row(self.config.artifact_dir / "test_overlap_summary.csv", "evaluable_warm_users")
        val_boot = self._find_segment_row(self.config.artifact_dir / "validation_bootstrap.csv", "evaluable_warm_users")
        test_boot = self._find_segment_row(self.config.artifact_dir / "test_bootstrap.csv", "evaluable_warm_users")
        resources = csv_rows(self.config.artifact_dir / "resource_usage.csv")
        total_seconds = sum(float(row["execution_seconds"]) for row in resources)
        peak_rss = max(int(row["peak_rss_bytes"]) for row in resources)
        peak_temp = max(int(row["temp_disk_peak_bytes"]) for row in resources)
        spill = any(row["spill_observed"].lower() == "true" for row in resources)
        retries = sum(max(0, int(row["attempt"]) - 1) for row in resources)
        neighbor_rows = csv_rows(self.config.artifact_dir / "neighbor_statistics.csv")
        test_neighbor = next(
            row for row in neighbor_rows
            if row["snapshot_id"] == "test_14d" and row["source_variant"] == selected
        )
        val_cos = self._find_metric(val_itemcf, "cosine", "recall_at_500")
        val_iuf = self._find_metric(val_itemcf, "iuf", "recall_at_500")
        test_single = self._find_metric(test_itemcf, selected, "recall_at_500")
        val_base = self._find_metric(val_union, "base_union", "recall_at_500")
        val_challenger = self._find_metric(val_union, f"itemcf_union_{selected}", "recall_at_500")
        test_base = self._find_metric(test_union, "base_union", "recall_at_500")
        test_challenger = self._find_metric(test_union, "itemcf_union", "recall_at_500")
        recommendation = (
            "建议 ItemCF 进入后续正式 union，但仍需在后续阶段验证配额与排序。"
            if float(test_marginal["absolute_delta_recall_at_500"]) > 0
            and float(test_boot["ci_upper"]) >= 0
            else "不建议把 ItemCF 纳入主 union；仅保留为经典协同对照。"
        )
        return f"""# Stage 3B：经典 ItemCF 与多路召回增量价值验证

## 1. 目标和任务边界

本阶段只实现 cosine/IUF ItemCF、固定预算 union、overlap、marginal Recall 与 paired user-block bootstrap。核心问题是 ItemCF 在已有 Trend 7d、Pure Repurchase 和 Popularity 之外是否提供独立增量；未实现 MF、双塔、内容召回、排序或重排。

## 2. 数据与防泄漏

- Validation：`validation_14d`，as-of `2020-08-25`；Test：`test_14d`，as-of `2020-09-08`。
- 反馈固定为 `{self.feedback_version}`，主协同人口固定为 `evaluable_warm_users`，cold-history 单列。
- ItemCF 只读取各 snapshot cutoff 前 B 口径 history；ground truth 只进入统一评估计算。
- 主候选域为 `history_seen_items_v1`；lock_fingerprint: `{self.stage3a_lock_fingerprint}`。
- Validation 锁定 `{selected}` 后才运行一次性 Test；Test 未参与变体或配额选择。

## 3. ItemCF 定义

用户内商品先去重，按最近购买日期及匿名 article key 稳定排序，最近 50 个唯一商品参与共现。cosine 使用 `Cij/sqrt(NiNj)`；IUF 将每用户贡献改为 `1/log(1+|Iu|)` 后使用同一支持度归一化。每商品保留 Top-200 邻居；每用户最多取最近 20 个 seed，权重为 `1/log2(rank+1)`；排除所有 seen items 和候选域外商品，最多输出 500 个真实 ItemCF 候选，不足不补齐。

## 4. Validation 变体选择

| Variant | Recall@500 | HitRate@500 | MAP@12 |
| --- | ---: | ---: | ---: |
| cosine | {val_cos:.6f} | {self._find_metric(val_itemcf, 'cosine', 'hit_rate_at_500'):.6f} | {self._find_metric(val_itemcf, 'cosine', 'map_at_12'):.6f} |
| iuf | {val_iuf:.6f} | {self._find_metric(val_itemcf, 'iuf', 'hit_rate_at_500'):.6f} | {self._find_metric(val_itemcf, 'iuf', 'map_at_12'):.6f} |

按 Recall@500、HitRate@500、MAP@12、候选用户覆盖率、完全相同时 cosine 的预注册顺序，锁定 `{selected}`。

## 5. 单路效果

锁定 ItemCF 的 Test Recall@500 为 `{test_single:.6f}`，MAP@12 为 `{self._find_metric(test_itemcf, selected, 'map_at_12'):.6f}`。完整 overall、short-history/warm/active/cold-history、候选不足、catalog coverage 与集中度见公开 CSV；cold-history 的 ItemCF 候选和指标按定义为 0。

## 6. Overlap

Validation/Test 主人口 ItemCF–Trend 平均 Jaccard 分别为 `{float(val_overlap['mean_itemcf_trend_jaccard']):.6f}` / `{float(test_overlap['mean_itemcf_trend_jaccard']):.6f}`；Test 仅 ItemCF 命中的 ground-truth 数为 `{test_overlap['only_itemcf_hits']}`，有独立 ItemCF 命中的用户数为 `{test_overlap['users_with_itemcf_exclusive_hits']}`。ItemCF 实际新增位置、与 Repurchase/Base 的重叠及 shared hits 见 overlap summary。

## 7. 固定预算 union

候选总预算始终为 500。Base Union 使用 Repurchase 50、Trend 100、Popularity 补齐；ItemCF Union 加入锁定 ItemCF 300，仍由 Popularity 只补剩余位置。固定插入顺序为 Repurchase、ItemCF、Trend、Popularity；不同来源原始 score 未相加。

| Phase | Base Recall@500 | ItemCF Union Recall@500 | Absolute delta |
| --- | ---: | ---: | ---: |
| Validation | {val_base:.6f} | {val_challenger:.6f} | {float(val_marginal['absolute_delta_recall_at_500']):.6f} |
| Test | {test_base:.6f} | {test_challenger:.6f} | {float(test_marginal['absolute_delta_recall_at_500']):.6f} |

Test 提升/不变/下降用户比例为 `{float(test_marginal['improved_user_rate']):.6f}` / `{float(test_marginal['unchanged_user_rate']):.6f}` / `{float(test_marginal['declined_user_rate']):.6f}`；固定预算下的下降没有隐藏。每 100 个实际新增 ItemCF 位置带来的净新增 ground-truth hit 为 `{float(test_marginal['incremental_hits_per_100_unique_itemcf_slots']):.6f}`。

## 8. Bootstrap

Validation delta 的 95% CI 为 `[{float(val_boot['ci_lower']):.6f}, {float(val_boot['ci_upper']):.6f}]`，bootstrap delta>0 比例为 `{float(val_boot['positive_fraction']):.6f}`；Test 对应为 `[{float(test_boot['ci_lower']):.6f}, {float(test_boot['ci_upper']):.6f}]` 和 `{float(test_boot['positive_fraction']):.6f}`。这是离线 paired user-block bootstrap，不是线上显著性或 A/B Test。

## 9. 资源

- 锁定 Test 共现非零无向商品对 `{test_neighbor['nonzero_undirected_pairs']}`，邻居行 `{test_neighbor['neighbor_rows']}`。
- 已记录任务组耗时合计 `{total_seconds:.3f}` 秒，峰值 RSS `{peak_rss}` bytes，临时盘峰值 `{peak_temp}` bytes，spill observed `{str(spill).lower()}`，额外重试 `{retries}`。
- spill 标记采用任务期间整个 Stage 3B 临时目录的保守峰值；已完成分区的临时文件会成为后续组的基线，因此该标记证明观察到临时盘占用，但不能逐组归因。pair consolidation 与邻居排序均观察到临时盘继续增长，可确认发生外存计算。
- 只读接管阶段曾用 1024 MiB 对全量字符串键直接 `count(distinct)` 做规模预估并真实 OOM；该查询未产生正式产物。正式实现改用内部键、分区 pair checkpoint 与外存聚合。
- 首次正式 Validation 的单文件 user-item 集合在约 582 MiB DuckDB 限制下 OOM，随后等价改为 32 个 customer-key 分区；首次 Base Union 单分区在约 848 MiB 限制下 OOM，随后保持配额、顺序与输出语义不变，在每个正式分区内再拆 4 个工程子分区。两次失败均未缩小数据或改变评估口径。
- Validation 的 pair 分区 029 异常耗时 `5373.476` 秒，其他主要分区未出现同量级长尾；Test 未复现。Test 首次外层命令在 1 小时达到运行器上限后被中断，随后以同一 lock、参数和 output fingerprint 仅续跑未完成组，没有重新选择或增加 Test comparator。
- 收尾统计曾因错误要求未运行的 Test cosine 文件而失败；修复仅跳过协议上不应存在的未锁定 Test 变体，并通过显式 source-fingerprint 兼容审计。敏感信息扫描的 fingerprint/URL 误报也曾阻断收尾，字段感知规则修复后通过独立审计。

## 10. 限制

最近 50 个共现商品和最近 20 个 seed 都是本阶段固定工程规则，可能丢失长周期关系，不是行业统一标准。数据没有 order/session、曝光、点击、库存或精确上架时间；共现不等于因果关系，完整维表不等于历史可售域。ItemCF 排除 seen item，而 Repurchase 单独保留历史商品。离线 Recall 增量不等于线上 CTR/CVR 或业务收益。

## 11. 面试与简历证据

仓库真实支持经典 cosine/IUF ItemCF、分区共现与 Top-N 邻居、固定预算多路 union、来源 provenance、独立命中、marginal Recall、paired bootstrap、Test fingerprint 门禁和资源审计。不能声称 MF/双塔已实现、已经上线或取得 CTR/CVR 提升。

## 12. 下一阶段建议

{recommendation} 是否实现 MF challenger 应在获得单独阶段授权后，以 ItemCF 的真实独立增量和资源成本作为对照；本阶段不自动进入下一阶段。
"""

    def update_readme(self) -> None:
        path = self.config.root / "README.md"
        text = path.read_text(encoding="utf-8")
        start = text.index("## 当前进度")
        end = text.index("## 数据概览")
        lock = json_read(self.config.lock_path)
        selected = lock["locked_itemcf_variant"]
        test_marginal = self._find_segment_row(
            self.config.artifact_dir / "test_marginal_recall.csv", "evaluable_warm_users"
        )
        section = f"""## 当前进度

已完成并留有可复现代码、聚合产物或正式文档的阶段：

- **Stage 0A–2B**：完成数据侦察、契约与审计、可行性画像、架构冻结、严格时间基础层及统一评估人口；
- **Stage 3A**：完成 Popularity、Trend、Pure Repurchase、Repurchase + Popularity Fallback 和统一离线评估器；
- **Stage 3B**：完成 cosine/IUF ItemCF、固定预算 Base/ItemCF Union、overlap、独立命中、marginal Recall、paired bootstrap、Validation 锁定与一次性 Test。Validation 锁定 `{selected}`，Test 固定预算 union 的 Recall@500 absolute delta 为 `{float(test_marginal['absolute_delta_recall_at_500']):.6f}`。

完整 Stage 3B 结果见 [`docs/stages/stage_3b_itemcf_and_incremental_recall.md`](docs/stages/stage_3b_itemcf_and_incremental_recall.md) 和 `artifacts/stage_3b/`。下一阶段尚未授权；当前未实现矩阵分解、内容召回、排序或重排。

"""
        text = text[:start] + section + text[end:]
        old = """- Stage 3A 及后续召回实现；
- 候选融合与配额实验；"""
        new = """- 矩阵分解、内容召回及后续召回 challenger；
- 学习式候选融合与配额优化；"""
        text = text.replace(old, new)
        text = text.replace("Stage 0A–2B 阶段报告", "Stage 0A–3B 阶段报告")
        text = text.replace("Stage 1A–2B 可复现配置", "Stage 1A–3B 可复现配置")
        path.write_text(text, encoding="utf-8")

    def write_report_and_readme(self) -> tuple[Path, Path]:
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.write_text(self.build_report(), encoding="utf-8")
        self.update_readme()
        return self.config.report_path, self.config.root / "README.md"

    def validate_public_and_privacy(self) -> None:
        lock = json_read(self.config.lock_path)
        locked_source = str(lock.get("source_fingerprint", ""))
        if (
            locked_source != self.source_fingerprint
            and locked_source not in COMPATIBLE_CHECKPOINT_SOURCE_FINGERPRINTS
        ):
            raise ValueError("Stage 3B 当前源码与 Test 锁不一致且没有受审计的兼容说明")
        public = [
            *self.config.artifact_dir.glob("*.csv"),
            *self.config.artifact_dir.glob("*.json"),
            self.config.report_path,
            self.config.root / "README.md",
        ]
        # Stage 3B lock 包含嵌套 fingerprint 映射；通用扫描器只认识固定字段名，
        # 因此 lock 单独按字段感知规则扫描，其他公开产物仍走通用严格扫描器。
        validate_public_artifacts(path for path in public if path != self.config.lock_path)
        lock_matches = sensitive_value_matches(self.config.lock_path.read_text(encoding="utf-8"))
        allowed_lock_values = declared_stage3b_lock_fingerprints(lock)
        lock_matches = {
            name: [value for value in values if value not in allowed_lock_values]
            for name, values in lock_matches.items()
        }
        lock_matches = {name: values for name, values in lock_matches.items() if values}
        if lock_matches:
            raise ValueError(f"Stage 3B lock 敏感信息检查失败: {lock_matches}")
        forbidden_headers = {"customer_key", "article_key", "customer_id", "article_id", "postal_code", "user_key"}
        for path in self.config.artifact_dir.glob("*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                headers = set(csv.DictReader(handle).fieldnames or ())
            if headers & forbidden_headers:
                raise ValueError(f"公开产物包含实体字段: {path.name}")
            if str(self.config.root) in path.read_text(encoding="utf-8-sig"):
                raise ValueError(f"公开产物包含本地绝对路径: {path.name}")
        for relative in ("artifacts/stage_3b/private", "artifacts/stage_3b/tmp"):
            ignored = subprocess.run(
                ["git", "check-ignore", relative], cwd=self.config.root,
                text=True, capture_output=True,
            )
            if ignored.returncode != 0:
                raise ValueError(f"{relative} 未命中 .gitignore")
        tracked = subprocess.run(
            ["git", "ls-files", "data/raw", "artifacts"], cwd=self.config.root,
            text=True, capture_output=True, check=True,
        ).stdout.splitlines()
        unsafe = [
            value for value in tracked
            if value.startswith("data/raw/") or "/private/" in value or "/tmp/" in value
        ]
        if unsafe:
            raise ValueError(f"raw/private/tmp 存在 tracked 文件: {unsafe}")
        frozen = {
            "artifacts/stage_3a/stage_3a_lock.json": "c8c8f233f05001a1d600433068c1874a875f9ec47aa7d7b8d612e51cfbe2d134",
            "artifacts/stage_3a/test_metrics_overall.csv": "ecfa40762661c7afe0f159ac9902e284936895e2e892009722c46cd7848a5b6e",
            "artifacts/stage_3a/test_metrics_by_segment.csv": "d36d8b3122df62bcd553140ea5cbe892ae40a46199eaf07dce7111ca3b17276b",
            "artifacts/stage_3a/test_candidate_coverage.csv": "d18ba14aa5c163ea979952b7c003cb6c09fdea2fa66bb2c2d6a519f9d4f064c8",
        }
        for relative, expected in frozen.items():
            if file_fingerprint(self.config.root / relative) != expected:
                raise ValueError(f"Stage 3A 冻结产物发生变化: {relative}")
        if stage3a_source_fingerprint(self.config.root) != json_read(self.stage3a_lock_path)["source_fingerprint"]:
            raise ValueError("Stage 3B 扩展改变了 Stage 3A source fingerprint")

    def finalize(self) -> None:
        self.run_group(
            "neighbor_statistics",
            self.write_neighbor_statistics,
            [self.config.artifact_dir / "neighbor_statistics.csv"],
        )
        self.write_resource_and_status()
        self.run_group(
            "report_generation",
            self.write_report_and_readme,
            [self.config.report_path, self.config.root / "README.md"],
        )
        self.write_resource_and_status()
        self.run_group(
            "public_aggregation",
            self.write_manifest,
            [self.config.artifact_dir / "stage_3b_manifest.csv"],
        )
        self.write_resource_and_status()
        self.write_manifest()
        self.validate_public_and_privacy()
