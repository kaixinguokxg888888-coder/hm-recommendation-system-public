from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import duckdb

from data_audit.stage1a import ResourceMonitor, validate_public_artifacts
from data_foundation.manifest import file_fingerprint, write_csv

from .config import RetrievalConfig, source_fingerprint
from .contracts import CANDIDATE_FIELDS, FORBIDDEN_ID_FIELDS
from .locking import (
    lock_fingerprint,
    select_trend_window,
    validate_test_lock,
)


METRIC_NAMES = (
    "map_at_12",
    "recall_at_12",
    "recall_at_50",
    "recall_at_100",
    "recall_at_500",
    "hit_rate_at_12",
    "hit_rate_at_50",
    "hit_rate_at_100",
    "hit_rate_at_500",
    "precision_at_12",
)
COVERAGE_NAMES = (
    "population_users",
    "users_with_candidates",
    "candidate_user_coverage",
    "mean_candidate_count",
    "median_candidate_count",
    "p25_candidate_count",
    "p75_candidate_count",
    "p90_candidate_count",
    "p95_candidate_count",
    "users_below_12_rate",
    "users_below_50_rate",
    "users_below_100_rate",
    "users_below_500_rate",
    "unique_recommended_items",
    "candidate_universe_items",
    "catalog_coverage",
    "top1_top12_exposure_share",
    "top10_top12_exposure_share",
    "top100_top12_exposure_share",
)
PUBLIC_META_FIELDS = (
    "stage",
    "stage_version",
    "snapshot_id",
    "as_of_date",
    "source",
    "source_variant",
    "metric",
    "segment",
    "population_denominator",
    "dataset_fingerprint",
    "config_fingerprint",
    "feedback_version",
    "population_version",
    "candidate_universe_version",
    "run_status",
)


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


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@dataclass(frozen=True)
class SnapshotSpec:
    snapshot_id: str
    history_start: str
    history_end: str
    label_start: str
    label_end: str
    ground_truth_fingerprint: str


@dataclass(frozen=True)
class SourceSpec:
    source: str
    variant: str

    @property
    def slug(self) -> str:
        return f"{self.source}_{self.variant}".replace("-", "_")


class Stage3ARunner:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.dataset_fingerprint = ""
        self.feedback_version = ""
        self.population_version = config.raw["population_version"]
        self.snapshots: dict[str, SnapshotSpec] = {}
        self.source_fingerprint = source_fingerprint(config.root)
        self.run_fingerprint = ""
        self.resources: list[dict[str, Any]] = []
        self.run_status: list[dict[str, Any]] = []
        self._load_and_validate_upstream()
        self.run_fingerprint = hashlib.sha256(
            (
                self.dataset_fingerprint
                + self.config.fingerprint
                + self.source_fingerprint
            ).encode("utf-8")
        ).hexdigest()

    def _load_and_validate_upstream(self) -> None:
        split_path = self.config.resolve(
            self.config.raw["input_manifests"]["split"]
        )
        population_path = self.config.resolve(
            self.config.raw["input_manifests"]["population"]
        )
        split_rows = csv_rows(split_path)
        population_rows = csv_rows(population_path)
        datasets = {row["dataset_fingerprint"] for row in split_rows + population_rows}
        feedback = {row["feedback_version"] for row in split_rows + population_rows}
        populations = {row["population_version"] for row in population_rows}
        if len(datasets) != 1 or len(feedback) != 1 or len(populations) != 1:
            raise ValueError("Stage 2A/2B manifest 版本不一致")
        self.dataset_fingerprint = datasets.pop()
        self.feedback_version = feedback.pop()
        if self.feedback_version != self.config.raw["feedback_version"]:
            raise ValueError("Stage 3A feedback_version 与上游 manifest 不匹配")
        if populations.pop() != self.population_version:
            raise ValueError("Stage 3A population_version 与上游 manifest 不匹配")

        population_lookup = {
            (row["snapshot_id"], row["population"]): row
            for row in population_rows
        }
        for row in split_rows:
            split_id = row["split_id"]
            if split_id not in self.config.raw["snapshots"]:
                continue
            pop = population_lookup[(split_id, "evaluable_warm_users")]
            if row["history_end"] != pop["as_of_date"]:
                raise ValueError(f"{split_id} as_of_date 在 Stage 2A/2B 不一致")
            self.snapshots[split_id] = SnapshotSpec(
                split_id,
                row["history_start"],
                row["history_end"],
                row["label_start"],
                row["label_end"],
                pop["ground_truth_fingerprint"],
            )
        if set(self.snapshots) != set(self.config.raw["snapshots"]):
            raise ValueError("Stage 3A 所需 snapshot 不完整")

        stage2a_private = self.config.resolve(
            self.config.raw["upstream"]["stage_2a_private"]
        )
        expected_stage2a = json.loads(split_rows[0]["output_fingerprints"])
        for name in (
            "feedback_daily.parquet",
            "customer_key_map.parquet",
            "article_key_map.parquet",
        ):
            path = stage2a_private / name
            expected = expected_stage2a[self.config.relative(path)]
            if not path.exists() or file_fingerprint(path) != expected:
                raise ValueError(f"Stage 2A private fingerprint 不匹配: {name}")

        stage2b_private = self.config.resolve(
            self.config.raw["upstream"]["stage_2b_private"]
        )
        group_dir = stage2b_private.parent / "groups"
        group_payloads = {
            name: json_read(group_dir / name)["payload"]
            for name in (
                "group_a_ground_truth.json",
                "group_c_user_segments.json",
                "group_d_item_state.json",
                "group_e_evaluation_protocol.json",
            )
        }
        prefixes = {
            "group_a_ground_truth.json": ("ground_truth", "ground_truth_fingerprint"),
            "group_c_user_segments.json": ("user_segments", "fingerprint"),
            "group_d_item_state.json": ("item_state", "fingerprint"),
            "group_e_evaluation_protocol.json": ("evaluation_population", "fingerprint"),
        }
        for group, (prefix, fp_field) in prefixes.items():
            for split_id in self.snapshots:
                path = stage2b_private / f"{prefix}_{split_id}.parquet"
                expected = group_payloads[group][split_id][fp_field]
                if not path.exists() or file_fingerprint(path) != expected:
                    raise ValueError(
                        f"Stage 2B private fingerprint 不匹配: {path.name}"
                    )

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
        con.execute(
            f"SET max_temp_directory_size='{duck['max_temp_directory_size']}'"
        )
        stage2a = self.config.resolve(self.config.raw["upstream"]["stage_2a_private"])
        for view, name in (
            ("feedback_daily", "feedback_daily.parquet"),
            ("customer_map", "customer_key_map.parquet"),
            ("article_map", "article_key_map.parquet"),
        ):
            con.execute(
                f"CREATE VIEW {view} AS SELECT * FROM read_parquet("
                f"'{sql_path(stage2a / name)}')"
            )
        return con

    def upstream_path(self, prefix: str, snapshot: str) -> Path:
        directory = self.config.resolve(self.config.raw["upstream"]["stage_2b_private"])
        return directory / f"{prefix}_{snapshot}.parquet"

    def candidate_path(self, snapshot: str, spec: SourceSpec) -> Path:
        return self.config.private_dir / f"candidates_{snapshot}_{spec.slug}.parquet"

    def metric_path(self, snapshot: str, spec: SourceSpec) -> Path:
        return self.config.private_dir / f"user_metrics_{snapshot}_{spec.slug}.parquet"

    def ranking_path(self, snapshot: str, spec: SourceSpec) -> Path:
        return self.config.private_dir / f"ranking_{snapshot}_{spec.slug}.parquet"

    def checkpoint_path(self, group: str) -> Path:
        return self.config.checkpoint_dir / f"{group}.json"

    def _checkpoint_reusable(
        self, checkpoint: dict[str, Any] | None, outputs: Iterable[Path]
    ) -> bool:
        paths = list(outputs)
        if not checkpoint or checkpoint.get("status") != "SUCCESS":
            return False
        if checkpoint.get("run_fingerprint") != self.run_fingerprint:
            return False
        expected = checkpoint.get("output_fingerprints", {})
        return bool(paths) and all(
            path.exists()
            and expected.get(self.config.relative(path)) == file_fingerprint(path)
            for path in paths
        )

    def run_group(
        self,
        group: str,
        operation: Callable[[], Any],
        outputs: Iterable[Path],
    ) -> Any:
        paths = list(outputs)
        checkpoint_path = self.checkpoint_path(group)
        checkpoint = json_read(checkpoint_path) if checkpoint_path.exists() else None
        if self._checkpoint_reusable(checkpoint, paths):
            row = dict(checkpoint)
            row["run_status"] = "REUSED"
            self.resources.append(row)
            self.run_status.append(
                {
                    "group_name": group,
                    "status": "REUSED",
                    "attempt": checkpoint.get("attempt", 1),
                    "message": "matching fingerprint and complete outputs",
                }
            )
            return None

        attempt = int((checkpoint or {}).get("attempt", 0)) + 1
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        try:
            with monitor:
                result = operation()
            elapsed = time.perf_counter() - started
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
                "execution_seconds": elapsed,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "memory_limit_mb": self.config.memory_limit_mb,
                "threads": self.config.raw["duckdb"]["threads"],
                "output_files": [self.config.relative(path) for path in paths],
                "output_fingerprints": {
                    self.config.relative(path): file_fingerprint(path)
                    for path in paths
                },
                "output_size_bytes": sum(path.stat().st_size for path in paths),
                "error_message": "",
            }
            json_write(checkpoint_path, payload)
            self.resources.append(payload)
            self.run_status.append(
                {
                    "group_name": group,
                    "status": "SUCCESS",
                    "attempt": attempt,
                    "message": "",
                }
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - started
            payload = {
                "group_name": group,
                "status": "FAILED",
                "attempt": attempt,
                "generated_at": now(),
                "run_fingerprint": self.run_fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "execution_seconds": elapsed,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "error_message": str(exc).replace(str(self.config.root), "."),
            }
            json_write(checkpoint_path, payload)
            self.resources.append(payload)
            self.run_status.append(
                {
                    "group_name": group,
                    "status": "FAILED",
                    "attempt": attempt,
                    "message": payload["error_message"],
                }
            )
            raise

    def _candidate_select_metadata(
        self, snapshot: SnapshotSpec, spec: SourceSpec
    ) -> str:
        warm = self.config.raw["candidate_universe_versions"]["warm"]
        cold = self.config.raw["candidate_universe_versions"]["cold_history"]
        return f"""
            '{snapshot.snapshot_id}'::VARCHAR snapshot_id,
            DATE '{snapshot.history_end}' as_of_date,
            u.user_key::BIGINT customer_key,
            r.article_key::BIGINT article_key,
            '{spec.source}'::VARCHAR AS "source",
            '{spec.variant}'::VARCHAR AS source_variant,
            r.source_rank::INTEGER source_rank,
            r.source_score::DOUBLE source_score,
            CASE WHEN u.is_main_evaluable THEN '{warm}' ELSE '{cold}' END::VARCHAR
              candidate_universe_version,
            '{self.feedback_version}'::VARCHAR feedback_version,
            '{self.population_version}'::VARCHAR population_version,
            '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
            '{self.config.fingerprint}'::VARCHAR config_fingerprint
        """

    def generate_popularity(self, snapshot_id: str) -> SourceSpec:
        snapshot = self.snapshots[snapshot_id]
        spec = SourceSpec("popularity", "global")
        ranking = self.ranking_path(snapshot_id, spec)
        candidate = self.candidate_path(snapshot_id, spec)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT am.article_key::BIGINT article_key,
                         count(*)::BIGINT source_score,
                         row_number() OVER (
                           ORDER BY count(*) DESC, am.article_key ASC
                         )::INTEGER source_rank
                  FROM feedback_daily f JOIN article_map am USING(article_id)
                  WHERE f.t_dat BETWEEN DATE '{snapshot.history_start}'
                                    AND DATE '{snapshot.history_end}'
                  GROUP BY am.article_key
                ) TO '{sql_path(ranking)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            population = self.upstream_path("evaluation_population", snapshot_id)
            con.execute(
                f"""
                COPY (
                  WITH u AS (
                    SELECT user_key, is_main_evaluable
                    FROM read_parquet('{sql_path(population)}')
                  ), r AS (
                    SELECT article_key, source_score, source_rank
                    FROM read_parquet('{sql_path(ranking)}')
                    WHERE source_rank <= {self.config.raw['candidate_max_k']}
                  )
                  SELECT {self._candidate_select_metadata(snapshot, spec)},
                         'global_popularity'::VARCHAR candidate_origin,
                         'history_user_item_day_count'::VARCHAR source_score_kind
                  FROM u CROSS JOIN r
                  ORDER BY customer_key, source_rank
                ) TO '{sql_path(candidate)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        return spec

    def generate_trend(self, snapshot_id: str, window_days: int) -> SourceSpec:
        if window_days not in self.config.raw["trend"]["windows_days"]:
            raise ValueError("Trend window 未预注册")
        snapshot = self.snapshots[snapshot_id]
        spec = SourceSpec("trend", f"window_{window_days}d")
        ranking = self.ranking_path(snapshot_id, spec)
        candidate = self.candidate_path(snapshot_id, spec)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH full_popularity AS (
                    SELECT am.article_key, count(*)::BIGINT popularity_score
                    FROM feedback_daily f JOIN article_map am USING(article_id)
                    WHERE f.t_dat BETWEEN DATE '{snapshot.history_start}'
                                      AND DATE '{snapshot.history_end}'
                    GROUP BY am.article_key
                  ), recent AS (
                    SELECT am.article_key, count(*)::BIGINT trend_score
                    FROM feedback_daily f JOIN article_map am USING(article_id)
                    WHERE f.t_dat > DATE '{snapshot.history_end}'
                                      - INTERVAL '{window_days} days'
                      AND f.t_dat <= DATE '{snapshot.history_end}'
                    GROUP BY am.article_key
                  )
                  SELECT r.article_key::BIGINT article_key,
                         r.trend_score::DOUBLE source_score,
                         row_number() OVER (
                           ORDER BY r.trend_score DESC,
                                    p.popularity_score DESC,
                                    r.article_key ASC
                         )::INTEGER source_rank
                  FROM recent r JOIN full_popularity p USING(article_key)
                ) TO '{sql_path(ranking)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            population = self.upstream_path("evaluation_population", snapshot_id)
            con.execute(
                f"""
                COPY (
                  WITH u AS (
                    SELECT user_key, is_main_evaluable
                    FROM read_parquet('{sql_path(population)}')
                  ), r AS (
                    SELECT article_key, source_score, source_rank
                    FROM read_parquet('{sql_path(ranking)}')
                    WHERE source_rank <= {self.config.raw['candidate_max_k']}
                  )
                  SELECT {self._candidate_select_metadata(snapshot, spec)},
                         'recent_popularity'::VARCHAR candidate_origin,
                         'window_user_item_day_count'::VARCHAR source_score_kind
                  FROM u CROSS JOIN r
                  ORDER BY customer_key, source_rank
                ) TO '{sql_path(candidate)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        return spec

    def generate_repurchase_pure(self, snapshot_id: str) -> SourceSpec:
        snapshot = self.snapshots[snapshot_id]
        spec = SourceSpec("repurchase", "pure")
        candidate = self.candidate_path(snapshot_id, spec)
        population = self.upstream_path("evaluation_population", snapshot_id)
        warm = self.config.raw["candidate_universe_versions"]["warm"]
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH grouped AS (
                    SELECT cm.customer_key, am.article_key,
                           count(*)::BIGINT purchase_day_count,
                           max(f.t_dat) last_purchase_date
                    FROM feedback_daily f
                    JOIN customer_map cm USING(customer_id)
                    JOIN article_map am USING(article_id)
                    SEMI JOIN read_parquet('{sql_path(population)}') u
                      ON u.user_key=cm.customer_key
                    WHERE f.t_dat BETWEEN DATE '{snapshot.history_start}'
                                      AND DATE '{snapshot.history_end}'
                    GROUP BY cm.customer_key, am.article_key
                  ), ranked AS (
                    SELECT *,
                      row_number() OVER (
                        PARTITION BY customer_key
                        ORDER BY purchase_day_count DESC,
                                 last_purchase_date DESC,
                                 article_key ASC
                      )::INTEGER source_rank
                    FROM grouped
                  )
                  SELECT '{snapshot.snapshot_id}'::VARCHAR snapshot_id,
                         DATE '{snapshot.history_end}' as_of_date,
                         customer_key::BIGINT customer_key,
                         article_key::BIGINT article_key,
                         'repurchase'::VARCHAR AS "source",
                         'pure'::VARCHAR AS source_variant,
                         source_rank,
                         purchase_day_count::DOUBLE source_score,
                         '{warm}'::VARCHAR candidate_universe_version,
                         '{self.feedback_version}'::VARCHAR feedback_version,
                         '{self.population_version}'::VARCHAR population_version,
                         '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
                         '{self.config.fingerprint}'::VARCHAR config_fingerprint,
                         'pure'::VARCHAR candidate_origin,
                         'purchase_day_count'::VARCHAR source_score_kind,
                         purchase_day_count,
                         last_purchase_date
                  FROM ranked
                  WHERE source_rank <= {self.config.raw['candidate_max_k']}
                  ORDER BY customer_key, source_rank
                ) TO '{sql_path(candidate)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        return spec

    def generate_repurchase_fallback(self, snapshot_id: str) -> SourceSpec:
        snapshot = self.snapshots[snapshot_id]
        pure_spec = SourceSpec("repurchase", "pure")
        pop_spec = SourceSpec("popularity", "global")
        spec = SourceSpec("repurchase", "popularity_fallback")
        pure = self.candidate_path(snapshot_id, pure_spec)
        ranking = self.ranking_path(snapshot_id, pop_spec)
        target = self.candidate_path(snapshot_id, spec)
        population = self.upstream_path("evaluation_population", snapshot_id)
        warm = self.config.raw["candidate_universe_versions"]["warm"]
        cold = self.config.raw["candidate_universe_versions"]["cold_history"]
        max_k = int(self.config.raw["candidate_max_k"])
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key customer_key, is_main_evaluable
                    FROM read_parquet('{sql_path(population)}')
                  ), pure AS (
                    SELECT customer_key, article_key, source_rank,
                           source_score, purchase_day_count, last_purchase_date
                    FROM read_parquet('{sql_path(pure)}')
                  ), pure_counts AS (
                    SELECT u.customer_key, u.is_main_evaluable,
                           count(p.article_key)::INTEGER pure_count
                    FROM users u LEFT JOIN pure p USING(customer_key)
                    GROUP BY u.customer_key, u.is_main_evaluable
                  ), fill_pool AS (
                    SELECT pc.customer_key, pc.is_main_evaluable, pc.pure_count,
                           r.article_key, r.source_rank popularity_rank,
                           r.source_score popularity_score
                    FROM pure_counts pc
                    CROSS JOIN read_parquet('{sql_path(ranking)}') r
                    ANTI JOIN pure p
                      ON p.customer_key=pc.customer_key
                     AND p.article_key=r.article_key
                    WHERE pc.pure_count < {max_k}
                      AND r.source_rank <= {max_k} + pc.pure_count
                  ), fill AS (
                    SELECT *,
                      row_number() OVER (
                        PARTITION BY customer_key ORDER BY popularity_rank
                      )::INTEGER fill_rank
                    FROM fill_pool
                    QUALIFY fill_rank <= {max_k} - pure_count
                  ), combined AS (
                    SELECT p.customer_key, p.article_key, p.source_rank,
                           p.source_score, 'pure' candidate_origin,
                           'purchase_day_count' source_score_kind,
                           p.purchase_day_count, p.last_purchase_date,
                           u.is_main_evaluable
                    FROM pure p JOIN users u USING(customer_key)
                    UNION ALL
                    SELECT customer_key, article_key,
                           (pure_count + fill_rank)::INTEGER source_rank,
                           popularity_score::DOUBLE source_score,
                           'popularity_fallback' candidate_origin,
                           'history_user_item_day_count' source_score_kind,
                           NULL::BIGINT purchase_day_count,
                           NULL::DATE last_purchase_date,
                           is_main_evaluable
                    FROM fill
                  )
                  SELECT '{snapshot.snapshot_id}'::VARCHAR snapshot_id,
                         DATE '{snapshot.history_end}' as_of_date,
                         customer_key::BIGINT customer_key,
                         article_key::BIGINT article_key,
                         'repurchase'::VARCHAR AS "source",
                         'popularity_fallback'::VARCHAR AS source_variant,
                         source_rank,
                         source_score,
                         CASE WHEN is_main_evaluable THEN '{warm}' ELSE '{cold}' END::VARCHAR
                           candidate_universe_version,
                         '{self.feedback_version}'::VARCHAR feedback_version,
                         '{self.population_version}'::VARCHAR population_version,
                         '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
                         '{self.config.fingerprint}'::VARCHAR config_fingerprint,
                         candidate_origin::VARCHAR candidate_origin,
                         source_score_kind::VARCHAR source_score_kind,
                         purchase_day_count,
                         last_purchase_date
                  FROM combined
                  ORDER BY customer_key, source_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        return spec

    def validate_candidate_file(
        self, snapshot_id: str, spec: SourceSpec
    ) -> dict[str, int]:
        snapshot = self.snapshots[snapshot_id]
        candidate = self.candidate_path(snapshot_id, spec)
        population = self.upstream_path("evaluation_population", snapshot_id)
        item_state = self.upstream_path("item_state", snapshot_id)
        con = self.connect()
        try:
            fields = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{sql_path(candidate)}')"
                ).fetchall()
            }
            missing = set(CANDIDATE_FIELDS) - fields
            forbidden = FORBIDDEN_ID_FIELDS & fields
            if missing or forbidden:
                raise ValueError(
                    f"候选 schema 错误 missing={sorted(missing)} "
                    f"forbidden={sorted(forbidden)}"
                )
            null_predicate = " OR ".join(
                f"{field} IS NULL" for field in CANDIDATE_FIELDS
            )
            null_count = con.execute(
                f"SELECT count(*) FROM read_parquet('{sql_path(candidate)}') "
                f"WHERE {null_predicate}"
            ).fetchone()[0]
            if null_count:
                raise ValueError(f"候选协议字段存在 {null_count} 个空值行")

            metadata = con.execute(
                f"""
                SELECT count(DISTINCT snapshot_id),
                       min(snapshot_id), count(DISTINCT as_of_date), min(as_of_date),
                       count(DISTINCT "source"), min("source"),
                       count(DISTINCT source_variant), min(source_variant),
                       count(DISTINCT feedback_version), min(feedback_version),
                       count(DISTINCT population_version), min(population_version),
                       count(DISTINCT dataset_fingerprint), min(dataset_fingerprint),
                       count(DISTINCT config_fingerprint), min(config_fingerprint)
                FROM read_parquet('{sql_path(candidate)}')
                """
            ).fetchone()
            expected = (
                1,
                snapshot.snapshot_id,
                1,
                snapshot.history_end,
                1,
                spec.source,
                1,
                spec.variant,
                1,
                self.feedback_version,
                1,
                self.population_version,
                1,
                self.dataset_fingerprint,
                1,
                self.config.fingerprint,
            )
            actual = tuple(
                str(value) if index in (3,) else value
                for index, value in enumerate(metadata)
            )
            if actual != expected:
                raise ValueError(
                    f"候选 metadata 与上下文不匹配: actual={actual}"
                )

            violations = con.execute(
                f"""
                WITH c AS (
                  SELECT * FROM read_parquet('{sql_path(candidate)}')
                ), grouped AS (
                  SELECT customer_key, count(*) n,
                         count(DISTINCT article_key) distinct_items,
                         count(DISTINCT source_rank) distinct_ranks,
                         min(source_rank) min_rank, max(source_rank) max_rank
                  FROM c GROUP BY customer_key
                )
                SELECT
                  count(*) FILTER (
                    WHERE n>{self.config.raw['candidate_max_k']}
                       OR n<>distinct_items OR n<>distinct_ranks
                       OR min_rank<>1 OR max_rank<>n
                  )
                FROM grouped
                """
            ).fetchone()[0]
            if violations:
                raise ValueError(f"候选唯一性/rank/上限协议违反用户数={violations}")

            illegal = con.execute(
                f"""
                SELECT count(*)
                FROM read_parquet('{sql_path(candidate)}') c
                JOIN read_parquet('{sql_path(population)}') u
                  ON u.user_key=c.customer_key
                LEFT JOIN read_parquet('{sql_path(item_state)}') i
                  USING(article_key)
                WHERE i.article_key IS NULL
                   OR (u.is_main_evaluable AND NOT i.seen_item)
                """
            ).fetchone()[0]
            if illegal:
                raise ValueError(f"候选域外商品行数={illegal}")

            stats = con.execute(
                f"""
                SELECT count(*) candidate_rows,
                       count(DISTINCT customer_key) candidate_users,
                       count(DISTINCT article_key) candidate_items
                FROM read_parquet('{sql_path(candidate)}')
                """
            ).fetchone()
            return {
                "candidate_rows": int(stats[0]),
                "candidate_users": int(stats[1]),
                "candidate_items": int(stats[2]),
            }
        finally:
            con.close()

    def evaluate_source(self, snapshot_id: str, spec: SourceSpec) -> Path:
        self.validate_candidate_file(snapshot_id, spec)
        snapshot = self.snapshots[snapshot_id]
        candidate = self.candidate_path(snapshot_id, spec)
        ground_truth = self.upstream_path("ground_truth", snapshot_id)
        population = self.upstream_path("evaluation_population", snapshot_id)
        target = self.metric_path(snapshot_id, spec)
        con = self.connect()
        try:
            gt_fp = file_fingerprint(ground_truth)
            if gt_fp != snapshot.ground_truth_fingerprint:
                raise ValueError("Ground truth fingerprint 在评估前发生变化")
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key customer_key, user_segment,
                           is_main_evaluable, positive_item_count
                    FROM read_parquet('{sql_path(population)}')
                  ), candidate_counts AS (
                    SELECT customer_key, count(*)::INTEGER candidate_count
                    FROM read_parquet('{sql_path(candidate)}')
                    GROUP BY customer_key
                  ), hit_rows AS (
                    SELECT c.customer_key, c.source_rank,
                           row_number() OVER (
                             PARTITION BY c.customer_key ORDER BY c.source_rank
                           )::INTEGER cumulative_hits
                    FROM read_parquet('{sql_path(candidate)}') c
                    SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                      ON g.user_key=c.customer_key
                     AND g.article_key=c.article_key
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
                  SELECT '{snapshot.snapshot_id}'::VARCHAR snapshot_id,
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
                              ELSE coalesce(h.hits_12,0)::DOUBLE
                                   / least(cc.candidate_count,12) END precision_at_12,
                         coalesce(h.ap_numerator_12,0)::DOUBLE
                           / least(u.positive_item_count,12) ap_at_12,
                         '{spec.source}'::VARCHAR AS "source",
                         '{spec.variant}'::VARCHAR AS source_variant,
                         '{snapshot.ground_truth_fingerprint}'::VARCHAR
                           ground_truth_fingerprint,
                         '{self.dataset_fingerprint}'::VARCHAR dataset_fingerprint,
                         '{self.config.fingerprint}'::VARCHAR config_fingerprint
                  FROM users u
                  LEFT JOIN candidate_counts cc USING(customer_key)
                  LEFT JOIN hits h USING(customer_key)
                  ORDER BY customer_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            row = con.execute(
                f"""
                SELECT count(*), count(*) FILTER(WHERE positive_item_count<=0),
                       count(DISTINCT ground_truth_fingerprint),
                       min(ground_truth_fingerprint)
                FROM read_parquet('{sql_path(target)}')
                """
            ).fetchone()
            if row[1] or row[2] != 1 or row[3] != snapshot.ground_truth_fingerprint:
                raise ValueError("用户级评估分母或 ground truth fingerprint 非法")
        finally:
            con.close()
        return target

    def _population_sql(self, metric_path: Path) -> str:
        path = sql_path(metric_path)
        return f"""
          SELECT 'evaluable_warm_users' segment, * FROM read_parquet('{path}')
          WHERE is_main_evaluable
          UNION ALL SELECT 'short_history', * FROM read_parquet('{path}')
          WHERE user_segment='short_history'
          UNION ALL SELECT 'warm', * FROM read_parquet('{path}')
          WHERE user_segment='warm'
          UNION ALL SELECT 'active', * FROM read_parquet('{path}')
          WHERE user_segment='active'
          UNION ALL SELECT 'cold_history_label_users', * FROM read_parquet('{path}')
          WHERE user_segment='cold'
        """

    def aggregate_source(
        self, snapshot_id: str, spec: SourceSpec
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        snapshot = self.snapshots[snapshot_id]
        candidate = self.candidate_path(snapshot_id, spec)
        metric = self.metric_path(snapshot_id, spec)
        item_state = self.upstream_path("item_state", snapshot_id)
        warm_universe = self.config.raw["candidate_universe_versions"]["warm"]
        cold_universe = self.config.raw["candidate_universe_versions"]["cold_history"]
        con = self.connect()
        try:
            metric_rows = self._dict_rows(
                con,
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
                ORDER BY CASE segment
                  WHEN 'evaluable_warm_users' THEN 0
                  WHEN 'short_history' THEN 1 WHEN 'warm' THEN 2
                  WHEN 'active' THEN 3 ELSE 4 END
                """,
            )
            coverage_wide = self._dict_rows(
                con,
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
                  FROM membership m
                  JOIN read_parquet('{sql_path(candidate)}') c USING(customer_key)
                  GROUP BY m.segment
                ), exposure AS (
                  SELECT m.segment, c.article_key, count(*) exposures
                  FROM membership m
                  JOIN read_parquet('{sql_path(candidate)}') c USING(customer_key)
                  WHERE c.source_rank<=12 GROUP BY m.segment, c.article_key
                ), ranked_exposure AS (
                  SELECT *, row_number() OVER (
                    PARTITION BY segment ORDER BY exposures DESC, article_key ASC
                  ) exposure_rank
                  FROM exposure
                ), exposure_stats AS (
                  SELECT segment, sum(exposures) exposure_total,
                    max(exposures) top1_exposures,
                    sum(exposures) FILTER(WHERE exposure_rank<=10) top10_exposures,
                    sum(exposures) FILTER(WHERE exposure_rank<=100) top100_exposures
                  FROM ranked_exposure GROUP BY segment
                ), universe AS (
                  SELECT count(*) full_items,
                         count(*) FILTER(WHERE seen_item) warm_items
                  FROM read_parquet('{sql_path(item_state)}')
                )
                SELECT c.*, c.users_with_candidates::DOUBLE/c.population_users
                         candidate_user_coverage,
                       coalesce(u.unique_recommended_items,0) unique_recommended_items,
                       CASE WHEN c.segment='cold_history_label_users'
                            THEN universe.full_items ELSE universe.warm_items END
                         candidate_universe_items,
                       coalesce(u.unique_recommended_items,0)::DOUBLE/
                         CASE WHEN c.segment='cold_history_label_users'
                              THEN universe.full_items ELSE universe.warm_items END
                         catalog_coverage,
                       coalesce(e.top1_exposures,0)::DOUBLE/
                         nullif(e.exposure_total,0) top1_top12_exposure_share,
                       coalesce(e.top10_exposures,0)::DOUBLE/
                         nullif(e.exposure_total,0) top10_top12_exposure_share,
                       coalesce(e.top100_exposures,0)::DOUBLE/
                         nullif(e.exposure_total,0) top100_top12_exposure_share
                FROM count_stats c
                LEFT JOIN unique_items u USING(segment)
                LEFT JOIN exposure_stats e USING(segment)
                CROSS JOIN universe
                """,
            )
        finally:
            con.close()

        metrics_long = []
        for row in metric_rows:
            for name in METRIC_NAMES:
                metrics_long.append(
                    self._public_row(
                        snapshot,
                        spec,
                        row["segment"],
                        int(row["population_denominator"]),
                        name,
                        row[name],
                        (
                            cold_universe
                            if row["segment"] == "cold_history_label_users"
                            else warm_universe
                        ),
                    )
                )
        coverage_long = []
        for row in coverage_wide:
            for name in COVERAGE_NAMES:
                coverage_long.append(
                    self._public_row(
                        snapshot,
                        spec,
                        row["segment"],
                        int(row["population_users"]),
                        name,
                        row[name],
                        (
                            cold_universe
                            if row["segment"] == "cold_history_label_users"
                            else warm_universe
                        ),
                    )
                )
        return metrics_long, coverage_long

    @staticmethod
    def _dict_rows(
        con: duckdb.DuckDBPyConnection, query: str
    ) -> list[dict[str, Any]]:
        cursor = con.execute(query)
        fields = [item[0] for item in cursor.description]
        return [dict(zip(fields, row, strict=True)) for row in cursor.fetchall()]

    def _public_row(
        self,
        snapshot: SnapshotSpec,
        spec: SourceSpec,
        segment: str,
        denominator: int,
        metric: str,
        value: Any,
        universe_version: str,
    ) -> dict[str, Any]:
        return {
            "stage": "3A",
            "stage_version": self.config.raw["stage_version"],
            "snapshot_id": snapshot.snapshot_id,
            "as_of_date": snapshot.history_end,
            "source": spec.source,
            "source_variant": spec.variant,
            "metric": metric,
            "metric_value": value,
            "segment": segment,
            "population_denominator": denominator,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint,
            "feedback_version": self.feedback_version,
            "population_version": self.population_version,
            "candidate_universe_version": universe_version,
            "run_status": "SUCCESS",
        }

    def write_aggregates(
        self, phase: str, snapshot_id: str, specs: Iterable[SourceSpec]
    ) -> list[Path]:
        overall: list[dict[str, Any]] = []
        by_segment: list[dict[str, Any]] = []
        coverage: list[dict[str, Any]] = []
        for spec in specs:
            metric_rows, coverage_rows = self.aggregate_source(snapshot_id, spec)
            overall.extend(
                row for row in metric_rows
                if row["segment"] == "evaluable_warm_users"
            )
            by_segment.extend(
                row for row in metric_rows
                if row["segment"] != "evaluable_warm_users"
            )
            coverage.extend(coverage_rows)
        paths = [
            self.config.artifact_dir / f"{phase}_metrics_overall.csv",
            self.config.artifact_dir / f"{phase}_metrics_by_segment.csv",
            self.config.artifact_dir / f"{phase}_candidate_coverage.csv",
        ]
        fields = (*PUBLIC_META_FIELDS, "metric_value")
        write_csv(paths[0], overall, fields)
        write_csv(paths[1], by_segment, fields)
        write_csv(paths[2], coverage, fields)
        return paths

    def validation_specs(self) -> list[SourceSpec]:
        return [
            SourceSpec("popularity", "global"),
            *(SourceSpec("trend", f"window_{days}d") for days in (7, 14, 28)),
            SourceSpec("repurchase", "pure"),
            SourceSpec("repurchase", "popularity_fallback"),
        ]

    def test_specs(self, locked_window: int) -> list[SourceSpec]:
        return [
            SourceSpec("popularity", "global"),
            SourceSpec("trend", f"window_{locked_window}d"),
            SourceSpec("repurchase", "pure"),
            SourceSpec("repurchase", "popularity_fallback"),
        ]

    def create_trend_selection_and_lock(self) -> tuple[Path, Path]:
        metrics = csv_rows(
            self.config.artifact_dir / "validation_metrics_overall.csv"
        )
        coverage = csv_rows(
            self.config.artifact_dir / "validation_candidate_coverage.csv"
        )
        selection_rows = []
        for window in self.config.raw["trend"]["windows_days"]:
            variant = f"window_{window}d"

            def metric_value(name: str) -> float:
                matches = [
                    row for row in metrics
                    if row["source"] == "trend"
                    and row["source_variant"] == variant
                    and row["metric"] == name
                ]
                if len(matches) != 1:
                    raise ValueError(f"Trend selection 缺少唯一指标 {variant}/{name}")
                return float(matches[0]["metric_value"])

            matches = [
                row for row in coverage
                if row["source"] == "trend"
                and row["source_variant"] == variant
                and row["segment"] == "evaluable_warm_users"
                and row["metric"] == "candidate_user_coverage"
            ]
            if len(matches) != 1:
                raise ValueError(f"Trend selection 缺少候选覆盖率 {variant}")
            selection_rows.append(
                {
                    "stage": "3A",
                    "snapshot_id": "validation_14d",
                    "as_of_date": self.snapshots["validation_14d"].history_end,
                    "source": "trend",
                    "source_variant": variant,
                    "window_days": window,
                    "recall_at_500": metric_value("recall_at_500"),
                    "map_at_12": metric_value("map_at_12"),
                    "candidate_user_coverage": float(matches[0]["metric_value"]),
                    "selection_rule_version": self.config.raw[
                        "trend_selection_rule_version"
                    ],
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "feedback_version": self.feedback_version,
                    "population_version": self.population_version,
                    "candidate_universe_version": self.config.raw[
                        "candidate_universe_versions"
                    ]["warm"],
                    "run_status": "SUCCESS",
                }
            )
        selected = select_trend_window(selection_rows)
        for row in selection_rows:
            row["selected"] = int(row["window_days"]) == selected
        selection_path = self.config.artifact_dir / "trend_window_selection.csv"
        write_csv(selection_path, selection_rows, selection_rows[0].keys())

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.config.root,
            text=True,
            capture_output=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.config.root,
            text=True,
            capture_output=True,
        )
        lock: dict[str, Any] = {
            "stage": "3A",
            "lock_version": "1.0.0",
            "created_at": now(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "feedback_version": self.feedback_version,
            "population_version": self.population_version,
            "candidate_universe_version": self.config.raw[
                "candidate_universe_versions"
            ],
            "validation_snapshot": "validation_14d",
            "test_snapshot": "test_14d",
            "trend_candidate_windows": self.config.raw["trend"]["windows_days"],
            "trend_selection_rule": self.config.raw["trend"]["selection_order"],
            "trend_selection_rule_version": self.config.raw[
                "trend_selection_rule_version"
            ],
            "validation_selection_path": self.config.relative(selection_path),
            "validation_selection_fingerprint": file_fingerprint(selection_path),
            "validation_metrics_fingerprint": file_fingerprint(
                self.config.artifact_dir / "validation_metrics_overall.csv"
            ),
            "locked_trend_window": selected,
            "test_allowed_variants": [
                "popularity:global",
                f"trend:window_{selected}d",
                "repurchase:pure",
                "repurchase:popularity_fallback",
            ],
            "config_fingerprint": self.config.fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "git_head": head.stdout.strip() if head.returncode == 0 else "UNAVAILABLE",
            "working_tree_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "test_not_used_for_selection": True,
        }
        lock["lock_fingerprint"] = lock_fingerprint(lock)
        json_write(self.config.lock_path, lock)
        return selection_path, self.config.lock_path

    def run_validation(self) -> list[SourceSpec]:
        snapshot = "validation_14d"
        popularity = SourceSpec("popularity", "global")
        pure = SourceSpec("repurchase", "pure")
        fallback = SourceSpec("repurchase", "popularity_fallback")
        self.run_group(
            "popularity_validation",
            lambda: self.generate_popularity(snapshot),
            [
                self.ranking_path(snapshot, popularity),
                self.candidate_path(snapshot, popularity),
            ],
        )
        for window in (7, 14, 28):
            spec = SourceSpec("trend", f"window_{window}d")
            self.run_group(
                f"trend_{window}_validation",
                lambda window=window: self.generate_trend(snapshot, window),
                [
                    self.ranking_path(snapshot, spec),
                    self.candidate_path(snapshot, spec),
                ],
            )
        self.run_group(
            "repurchase_pure_validation",
            lambda: self.generate_repurchase_pure(snapshot),
            [self.candidate_path(snapshot, pure)],
        )
        self.run_group(
            "repurchase_fallback_validation",
            lambda: self.generate_repurchase_fallback(snapshot),
            [self.candidate_path(snapshot, fallback)],
        )
        specs = self.validation_specs()
        validation_public = [
            self.config.artifact_dir / "validation_metrics_overall.csv",
            self.config.artifact_dir / "validation_metrics_by_segment.csv",
            self.config.artifact_dir / "validation_candidate_coverage.csv",
        ]

        def evaluate() -> None:
            for spec in specs:
                self.evaluate_source(snapshot, spec)
            self.write_aggregates("validation", snapshot, specs)

        self.run_group(
            "validation_evaluation",
            evaluate,
            [
                *(self.metric_path(snapshot, spec) for spec in specs),
                *validation_public,
            ],
        )
        self.run_group(
            "stage_3a_lock",
            self.create_trend_selection_and_lock,
            [
                self.config.artifact_dir / "trend_window_selection.csv",
                self.config.lock_path,
            ],
        )
        return specs

    def run_test(self) -> list[SourceSpec]:
        selection_path = self.config.artifact_dir / "trend_window_selection.csv"
        lock = validate_test_lock(
            self.config.lock_path,
            config_fingerprint=self.config.fingerprint,
            dataset_fingerprint=self.dataset_fingerprint,
            source_fingerprint=self.source_fingerprint,
            allowed_windows=self.config.raw["trend"]["windows_days"],
            expected_selection_path=selection_path,
        )
        selection_before = file_fingerprint(selection_path)
        snapshot = "test_14d"
        locked = int(lock["locked_trend_window"])
        popularity = SourceSpec("popularity", "global")
        trend = SourceSpec("trend", f"window_{locked}d")
        pure = SourceSpec("repurchase", "pure")
        fallback = SourceSpec("repurchase", "popularity_fallback")
        self.run_group(
            "popularity_test",
            lambda: self.generate_popularity(snapshot),
            [
                self.ranking_path(snapshot, popularity),
                self.candidate_path(snapshot, popularity),
            ],
        )
        self.run_group(
            "trend_locked_test",
            lambda: self.generate_trend(snapshot, locked),
            [
                self.ranking_path(snapshot, trend),
                self.candidate_path(snapshot, trend),
            ],
        )
        self.run_group(
            "repurchase_pure_test",
            lambda: self.generate_repurchase_pure(snapshot),
            [self.candidate_path(snapshot, pure)],
        )
        self.run_group(
            "repurchase_fallback_test",
            lambda: self.generate_repurchase_fallback(snapshot),
            [self.candidate_path(snapshot, fallback)],
        )
        specs = self.test_specs(locked)
        public = [
            self.config.artifact_dir / "test_metrics_overall.csv",
            self.config.artifact_dir / "test_metrics_by_segment.csv",
            self.config.artifact_dir / "test_candidate_coverage.csv",
        ]

        def evaluate() -> None:
            for spec in specs:
                self.evaluate_source(snapshot, spec)
            self.write_aggregates("test", snapshot, specs)

        self.run_group(
            "test_evaluation",
            evaluate,
            [*(self.metric_path(snapshot, spec) for spec in specs), *public],
        )
        if file_fingerprint(selection_path) != selection_before:
            raise RuntimeError("Test 运行改变了 Validation selection artifact")
        return specs

    def run_smoke(self) -> Path:
        from datetime import date

        from .contracts import CandidateContext, validate_candidates
        from .evaluation import macro_average, user_metrics
        from .popularity import popularity_ranking
        from .repurchase import popularity_fallback, pure_repurchase

        target = self.config.private_dir / "smoke_summary.json"
        checkpoint = self.config.checkpoint_dir / "smoke.json"
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        with monitor:
            interactions = [
                (date(2020, 1, 1), 1, 10),
                (date(2020, 1, 2), 1, 10),
                (date(2020, 1, 2), 2, 11),
            ]
            universe = {1: {10, 11}, 2: {10, 11}, 3: {10, 11}}
            ranking = popularity_ranking(
                interactions, date(2020, 1, 2), {10, 11}
            )
            pure = pure_repurchase(
                interactions, date(2020, 1, 2), universe, universe, 2
            )
            fallback = popularity_fallback(pure, ranking, universe, 2)
            context = CandidateContext(
                "smoke",
                date(2020, 1, 2),
                "repurchase",
                "popularity_fallback",
                self.feedback_version,
                self.population_version,
                self.dataset_fingerprint,
                self.config.fingerprint,
                2,
            )
            rows = []
            for user, values in fallback.items():
                for rank, (item, origin, score) in enumerate(values, start=1):
                    rows.append(
                        {
                            "snapshot_id": "smoke",
                            "as_of_date": date(2020, 1, 2),
                            "customer_key": user,
                            "article_key": item,
                            "source": "repurchase",
                            "source_variant": "popularity_fallback",
                            "source_rank": rank,
                            "source_score": float(score or 0),
                            "candidate_universe_version": "smoke",
                            "feedback_version": self.feedback_version,
                            "population_version": self.population_version,
                            "dataset_fingerprint": self.dataset_fingerprint,
                            "config_fingerprint": self.config.fingerprint,
                            "candidate_origin": origin,
                        }
                    )
            validate_candidates(rows, context, universe)
            metrics = []
            for user in sorted(universe):
                ranked = [
                    row["article_key"]
                    for row in rows
                    if row["customer_key"] == user
                ]
                metrics.append(user_metrics(ranked, {10}))
            aggregate = macro_average(metrics)
            json_write(
                target,
                {
                    "scope": "SYNTHETIC_SMOKE_ONLY",
                    "candidate_rows": len(rows),
                    "aggregate": aggregate,
                    "checkpoint_verified": True,
                },
            )
        json_write(
            checkpoint,
            {
                "group_name": "smoke",
                "status": "SUCCESS",
                "run_fingerprint": self.run_fingerprint,
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "output_fingerprint": file_fingerprint(target),
                "scope": "SYNTHETIC_SMOKE_ONLY",
            },
        )
        return target

    def write_public_operations(self) -> list[Path]:
        resource_path = self.config.artifact_dir / "resource_usage.csv"
        status_path = self.config.artifact_dir / "run_status.csv"
        manifest_path = self.config.artifact_dir / "stage_3a_manifest.csv"
        resource_rows = []
        for row in self.resources:
            resource_rows.append(
                {
                    "stage": "3A",
                    "snapshot_id": "all",
                    "as_of_date": self.snapshots["test_14d"].history_end,
                    "source": "pipeline",
                    "source_variant": row["group_name"],
                    "metric": "resource_usage",
                    "segment": "not_applicable",
                    "population_denominator": 0,
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "feedback_version": self.feedback_version,
                    "population_version": self.population_version,
                    "candidate_universe_version": "mixed_by_population",
                    "run_status": row.get("run_status", row["status"]),
                    "execution_seconds": row.get("execution_seconds", 0),
                    "peak_rss_bytes": row.get("peak_rss_bytes", 0),
                    "temp_disk_peak_bytes": row.get("temp_disk_peak_bytes", 0),
                    "spill_observed": row.get("spill_observed", False),
                    "memory_limit_mb": row.get(
                        "memory_limit_mb", self.config.memory_limit_mb
                    ),
                    "threads": row.get(
                        "threads", self.config.raw["duckdb"]["threads"]
                    ),
                    "output_size_bytes": row.get("output_size_bytes", 0),
                    "attempt": row.get("attempt", 1),
                }
            )
        write_csv(resource_path, resource_rows, resource_rows[0].keys())

        status_rows = []
        for row in self.run_status:
            status_rows.append(
                {
                    "stage": "3A",
                    "snapshot_id": "all",
                    "as_of_date": self.snapshots["test_14d"].history_end,
                    "source": "pipeline",
                    "source_variant": row["group_name"],
                    "metric": "run_status",
                    "segment": "not_applicable",
                    "population_denominator": 0,
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "feedback_version": self.feedback_version,
                    "population_version": self.population_version,
                    "candidate_universe_version": "mixed_by_population",
                    "run_status": row["status"],
                    "attempt": row["attempt"],
                    "message": row["message"],
                }
            )
        write_csv(status_path, status_rows, status_rows[0].keys())

        public_files = sorted(
            path
            for path in self.config.artifact_dir.iterdir()
            if path.is_file() and path.name != manifest_path.name
        )
        manifest_rows = [
            {
                "stage": "3A",
                "snapshot_id": "validation_14d,test_14d",
                "as_of_date": self.snapshots["test_14d"].history_end,
                "source": "pipeline",
                "source_variant": "public_artifact",
                "metric": "artifact_manifest",
                "segment": "not_applicable",
                "population_denominator": 0,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "feedback_version": self.feedback_version,
                "population_version": self.population_version,
                "candidate_universe_version": "mixed_by_population",
                "run_status": "SUCCESS",
                "artifact_path": self.config.relative(path),
                "artifact_fingerprint": file_fingerprint(path),
                "size_bytes": path.stat().st_size,
            }
            for path in public_files
        ]
        write_csv(manifest_path, manifest_rows, manifest_rows[0].keys())
        return [manifest_path, resource_path, status_path]

    @staticmethod
    def _metric_lookup(
        rows: list[dict[str, str]],
        source: str,
        variant: str,
        metric: str,
        segment: str | None = None,
    ) -> float:
        values = [
            row for row in rows
            if row["source"] == source
            and row["source_variant"] == variant
            and row["metric"] == metric
            and (segment is None or row["segment"] == segment)
        ]
        if len(values) != 1:
            raise ValueError(
                f"公开指标不唯一: {source}/{variant}/{metric}/{segment}"
            )
        return float(values[0]["metric_value"])

    def build_report(self) -> str:
        val = csv_rows(self.config.artifact_dir / "validation_metrics_overall.csv")
        test = csv_rows(self.config.artifact_dir / "test_metrics_overall.csv")
        test_segments = csv_rows(
            self.config.artifact_dir / "test_metrics_by_segment.csv"
        )
        coverage = csv_rows(self.config.artifact_dir / "test_candidate_coverage.csv")
        selection = csv_rows(
            self.config.artifact_dir / "trend_window_selection.csv"
        )
        lock = json_read(self.config.lock_path)
        locked = int(lock["locked_trend_window"])
        val_specs = self.validation_specs()
        test_specs = self.test_specs(locked)

        def metric_table(rows: list[dict[str, str]], specs: list[SourceSpec]) -> str:
            lines = []
            for spec in specs:
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            spec.source,
                            spec.variant,
                            f"{self._metric_lookup(rows, spec.source, spec.variant, 'map_at_12'):.6f}",
                            f"{self._metric_lookup(rows, spec.source, spec.variant, 'recall_at_12'):.6f}",
                            f"{self._metric_lookup(rows, spec.source, spec.variant, 'recall_at_500'):.6f}",
                            f"{self._metric_lookup(rows, spec.source, spec.variant, 'hit_rate_at_12'):.6f}",
                        )
                    )
                    + " |"
                )
            return "\n".join(lines)

        trend_lines = "\n".join(
            f"| {row['window_days']} | {float(row['recall_at_500']):.6f} | "
            f"{float(row['map_at_12']):.6f} | "
            f"{float(row['candidate_user_coverage']):.6f} | {row['selected']} |"
            for row in selection
        )
        segment_lines = []
        for spec in test_specs:
            for segment in (
                "short_history",
                "warm",
                "active",
                "cold_history_label_users",
            ):
                segment_lines.append(
                    f"| {spec.source} | {spec.variant} | {segment} | "
                    f"{self._metric_lookup(test_segments, spec.source, spec.variant, 'map_at_12', segment):.6f} | "
                    f"{self._metric_lookup(test_segments, spec.source, spec.variant, 'recall_at_500', segment):.6f} |"
                )
        coverage_lines = []
        for spec in test_specs:
            segment = "evaluable_warm_users"
            coverage_lines.append(
                f"| {spec.source} | {spec.variant} | "
                f"{self._metric_lookup(coverage, spec.source, spec.variant, 'candidate_user_coverage', segment):.6f} | "
                f"{self._metric_lookup(coverage, spec.source, spec.variant, 'mean_candidate_count', segment):.2f} | "
                f"{self._metric_lookup(coverage, spec.source, spec.variant, 'users_below_500_rate', segment):.6f} | "
                f"{self._metric_lookup(coverage, spec.source, spec.variant, 'catalog_coverage', segment):.6f} | "
                f"{self._metric_lookup(coverage, spec.source, spec.variant, 'top10_top12_exposure_share', segment):.6f} |"
            )
        resources = csv_rows(self.config.artifact_dir / "resource_usage.csv")
        total_seconds = sum(float(row["execution_seconds"]) for row in resources)
        peak_rss = max(int(row["peak_rss_bytes"]) for row in resources)
        peak_temp = max(int(row["temp_disk_peak_bytes"]) for row in resources)
        spill = any(row["spill_observed"].lower() == "true" for row in resources)
        attempts = sum(max(0, int(row["attempt"]) - 1) for row in resources)

        return f"""# Stage 3A：Baseline 召回与统一离线评估

## 1. 目标与范围

本阶段实际实现并全量运行 Popularity、7/14/28 天 Trend、Pure Repurchase、Repurchase + Popularity Fallback，以及与具体 baseline 解耦的统一候选协议和离线评估器。选择这些简单基线，是为了先建立可复现、可审计的推荐效果下界和 Test 封存机制。未实现 ItemCF、MF、内容召回、特征工程、排序、重排或 Stage 3B 能力。

## 2. 数据与协议

- Validation snapshot：`validation_14d`，as-of `{self.snapshots['validation_14d'].history_end}`，标签 `{self.snapshots['validation_14d'].label_start}` 至 `{self.snapshots['validation_14d'].label_end}`。
- Test snapshot：`test_14d`，as-of `{self.snapshots['test_14d'].history_end}`，标签 `{self.snapshots['test_14d'].label_start}` 至 `{self.snapshots['test_14d'].label_end}`。
- feedback：`{self.feedback_version}`；population：`{self.population_version}`。
- 主协同候选域：history seen items；cold-history fallback 使用完整 articles 身份域并单列。
- dataset fingerprint：`{self.dataset_fingerprint}`；Stage 3A config fingerprint：`{self.config.fingerprint}`。
- Validation 只选择 Trend window；Test 在 lock 后只运行固定版本，未参与选择。

## 3. 算法定义

- Popularity：snapshot history 中每个商品的 B 口径 user-item-day 记录数降序，同分按匿名 `article_key` 升序；不删除历史已购。
- Trend：区间 `(as_of_date-window_days, as_of_date]` 内的 B 口径计数降序，再按同 snapshot 全历史 Popularity 降序、`article_key` 升序。
- Pure Repurchase：每用户按不同购买日数降序、最近购买日降序、`article_key` 升序；不足 500 不补造。
- Fallback：完整保留 pure 顺序，按 Popularity 顺序去重补齐至最多 500；补齐分数不与复购分数作跨层比较。

## 4. Validation 结果

| Source | Variant | MAP@12 | Recall@12 | Recall@500 | HitRate@12 |
| --- | --- | ---: | ---: | ---: | ---: |
{metric_table(val, val_specs)}

Trend 预注册选择：

| Window | Recall@500 | MAP@12 | Candidate user coverage | Selected |
| ---: | ---: | ---: | ---: | --- |
{trend_lines}

按 Recall@500、MAP@12、用户候选覆盖率、较短窗口的确定性顺序，锁定 `{locked}` 天。

## 5. Test 结果

Test 仅运行 lock 允许的四个固定版本：

| Source | Variant | MAP@12 | Recall@12 | Recall@500 | HitRate@12 |
| --- | --- | ---: | ---: | ---: | ---: |
{metric_table(test, test_specs)}

## 6. 分群分析

| Source | Variant | Segment | MAP@12 | Recall@500 |
| --- | --- | --- | ---: | ---: |
{chr(10).join(segment_lines)}

`evaluable_warm_users` 是所有 history>0 的主人口，不等同于 5–20 的 `warm` 分组；cold-history 始终单列，未混入主协同宏平均。分群差异仅描述本次离线购买标签，不预设任一路必然适合某一人群。

## 7. 候选覆盖和集中度

以下为 Test 主协同人口：

| Source | Variant | User coverage | Mean candidates | Below 500 rate | Catalog coverage | Top-10 Top-12 share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(coverage_lines)}

Pure Repurchase 的候选不足是真实覆盖限制；fallback 的作用是保持复购头部顺序并改善候选完整性。集中度指标只是离线推荐列表统计，不代表线上曝光或收益。

## 8. 防泄漏验证

- Popularity、Trend、Repurchase 统计都限定在各 snapshot history 与 as-of 之前。
- ground truth 只在解耦评估器中读取，不参与候选分数、排序或补齐。
- Validation/Test 使用各自独立 history；Test 入口校验 config、dataset、source 和 selection fingerprints。
- 主协同候选逐项验证为 history seen item；cold-history 使用明确的完整身份域版本。
- Test 前后 Validation selection artifact fingerprint 保持不变。

## 9. 资源与工程

- 记录任务组总耗时：{total_seconds:.3f} 秒；峰值 RSS：{peak_rss} bytes。
- DuckDB 内存上限：{self.config.memory_limit_mb} MiB，单线程；临时盘峰值：{peak_temp} bytes；spill observed：{str(spill).lower()}。
- 失败后额外尝试次数：{attempts}；每组 checkpoint 记录输入、配置、源码与输出 fingerprint。
- 候选和用户级指标位于 Git 忽略的 private；公开产物只有聚合。

## 10. 真实限制

数据无曝光、点击、真实负反馈、库存/可售状态和精确上架时间。完整 articles 维表只表示身份域，不等于历史可售域；未购买不等于真实不喜欢；离线 Recall/MAP 不等于 CTR/CVR 或线上收益；本项目未做线上 A/B Test。

## 11. 可用于简历和面试的真实证据

仓库可以支持：严格相邻时间快照、B 口径基础层、统一候选协议、宏平均评估器、三类 baseline 与分群结果、Validation 选择和 Test fingerprint 封存、DuckDB 大规模执行、checkpoint/资源/隐私测试。不能声称已实现复杂协同或深度模型，也不能将离线购买指标扩写为线上业务提升。

## 12. 下一阶段建议

后续是否进入 Stage 3B，应以本报告中各路 Validation/Test Recall、MAP、覆盖与分群事实判断。Popularity 适合作为固定对照和兜底；fallback 是否进入后续 candidate union、Pure Repurchase 是否保留为独立来源，应依据其真实独立覆盖价值。Stage 3B 如获单独授权，优先验证冻结路线中的经典协同召回，不应自动扩展到排序或深度模型。本报告不授权也未实施 Stage 3B。
"""

    def update_readme(self) -> None:
        path = self.config.root / "README.md"
        text = path.read_text(encoding="utf-8")
        start = text.index("## 当前进度")
        end = text.index("## 数据概览")
        test = csv_rows(self.config.artifact_dir / "test_metrics_overall.csv")
        lock = json_read(self.config.lock_path)
        popularity_recall = self._metric_lookup(
            test, "popularity", "global", "recall_at_500"
        )
        fallback_map = self._metric_lookup(
            test, "repurchase", "popularity_fallback", "map_at_12"
        )
        section = f"""## 当前进度

已完成并留有可复现代码、聚合产物或正式文档的阶段：

- **Stage 0A–2B**：完成数据侦察、契约与审计、可行性画像、架构冻结、严格时间基础层及统一评估人口；
- **Stage 3A**：已实现并全量运行 Popularity、Trend、Pure Repurchase、Repurchase + Popularity Fallback 和统一离线评估器；Validation 在预注册的 7/14/28 天窗口中锁定 {lock['locked_trend_window']} 天 Trend，Test 只运行锁定配置。

Stage 3A Test 主协同人口的示例事实：Popularity Recall@500={popularity_recall:.6f}，Repurchase + Popularity Fallback MAP@12={fallback_map:.6f}。完整整体、分群、覆盖与资源结果见 [`docs/stages/stage_3a_baseline_retrieval_and_evaluation.md`](docs/stages/stage_3a_baseline_retrieval_and_evaluation.md) 和 `artifacts/stage_3a/`。

下一阶段是 **Stage 3B**，尚未开始且需要单独授权。当前仍未实现 ItemCF、矩阵分解、内容召回、排序或重排。

"""
        path.write_text(text[:start] + section + text[end:], encoding="utf-8")

    def write_report_and_readme(self) -> list[Path]:
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.write_text(self.build_report(), encoding="utf-8")
        self.update_readme()
        return [self.config.report_path, self.config.root / "README.md"]

    def validate_public_and_privacy(self) -> None:
        public = [
            *self.config.artifact_dir.glob("*.csv"),
            self.config.lock_path,
            self.config.report_path,
            self.config.root / "README.md",
        ]
        allowed = {
            self.dataset_fingerprint,
            self.config.fingerprint,
            self.source_fingerprint,
            *(
                row["ground_truth_fingerprint"]
                for row in csv_rows(
                    self.config.resolve(
                        self.config.raw["input_manifests"]["population"]
                    )
                )
            ),
        }
        validate_public_artifacts(public, allowed_tokens=allowed)
        forbidden_headers = {
            "customer_key",
            "article_key",
            "customer_id",
            "article_id",
            "postal_code",
        }
        for path in self.config.artifact_dir.glob("*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                headers = set(csv.DictReader(handle).fieldnames or ())
            if headers & forbidden_headers:
                raise ValueError(f"公开产物包含实体字段: {path.name}")
            text = path.read_text(encoding="utf-8-sig")
            if str(self.config.root) in text:
                raise ValueError(f"公开产物包含本地绝对路径: {path.name}")

        for relative in (
            "artifacts/stage_3a/private",
            "artifacts/stage_3a/tmp",
        ):
            result = subprocess.run(
                ["git", "check-ignore", relative],
                cwd=self.config.root,
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                raise ValueError(f"{relative} 未命中 .gitignore")
        tracked = subprocess.run(
            ["git", "ls-files", "data/raw", "artifacts"],
            cwd=self.config.root,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.splitlines()
        unsafe = [
            value for value in tracked
            if value.startswith("data/raw/")
            or "/private/" in value
            or "/tmp/" in value
        ]
        if unsafe:
            raise ValueError(f"raw/private/tmp 存在 tracked 文件: {unsafe}")

    def finalize_resource_files(self) -> None:
        checkpoints = []
        for path in sorted(self.config.checkpoint_dir.glob("*.json")):
            value = json_read(path)
            if value.get("group_name") == "smoke":
                continue
            checkpoints.append(value)
        self.resources = checkpoints
        self.run_status = [
            {
                "group_name": row["group_name"],
                "status": row["status"],
                "attempt": row.get("attempt", 1),
                "message": row.get("error_message", ""),
            }
            for row in checkpoints
        ]
        self.write_public_operations()
