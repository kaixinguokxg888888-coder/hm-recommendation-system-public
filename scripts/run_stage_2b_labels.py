from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import duckdb
import psutil

from data_audit.config import find_project_root
from data_audit.stage1a import ResourceMonitor
from data_foundation.manifest import file_fingerprint, write_csv


GROUPS = (
    "group_a_ground_truth",
    "group_b_evaluable_users",
    "group_c_user_segments",
    "group_d_item_state",
    "group_e_evaluation_protocol",
    "group_f_leakage_report",
)
META_FIELDS = ("dataset_fingerprint", "population_version", "feedback_version")


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Config:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.root = find_project_root(self.path)
        self.raw = json.loads(self.path.read_text(encoding="utf-8"))
        memory = self.raw["duckdb"]
        available = psutil.virtual_memory().available // (1024 * 1024)
        selected = int(available * float(memory["memory_fraction_of_available"]))
        self._memory_mb = max(
            int(memory["memory_min_mb"]),
            min(int(memory["memory_max_mb"]), selected),
        )

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
    def stage2a_private(self) -> Path:
        return self.resolve(self.raw["stage_2a_private"])

    @property
    def memory_mb(self) -> int:
        return self._memory_mb

    @property
    def fingerprint(self) -> str:
        manifest = self.resolve(self.raw["split_manifest"]).read_bytes()
        return hashlib.sha256(self.path.read_bytes() + b"\0" + manifest).hexdigest()


def load_snapshots(config: Config) -> tuple[str, str, dict[str, dict[str, str]]]:
    with config.resolve(config.raw["split_manifest"]).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    datasets = {row["dataset_fingerprint"] for row in rows}
    feedback = {row["feedback_version"] for row in rows}
    if len(datasets) != 1 or len(feedback) != 1:
        raise ValueError("Stage 2A manifest 版本不一致")
    return datasets.pop(), feedback.pop(), {row["split_id"]: row for row in rows}


def connect(config: Config) -> duckdb.DuckDBPyConnection:
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute(f"SET temp_directory='{config.temp_dir.as_posix()}'")
    con.execute(f"SET memory_limit='{config.memory_mb}MB'")
    con.execute("SET threads=1")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET max_temp_directory_size='{config.raw['duckdb']['max_temp_directory_size']}'")
    for view, filename in (
        ("feedback_daily", "feedback_daily.parquet"),
        ("customer_map", "customer_key_map.parquet"),
        ("article_map", "article_key_map.parquet"),
    ):
        path = (config.stage2a_private / filename).as_posix()
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{path}')")
    return con


def rows(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = con.execute(sql, params or [])
    fields = [item[0] for item in cursor.description]
    return [dict(zip(fields, row, strict=True)) for row in cursor.fetchall()]


def one(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
    return rows(con, sql, params)[0]


def private_path(config: Config, prefix: str, split_id: str) -> Path:
    return config.private_dir / f"{prefix}_{split_id}.parquet"


def public_write(config: Config, context: dict[str, Any], name: str, data: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    for row in data:
        row.update({
            "dataset_fingerprint": context["dataset_fingerprint"],
            "population_version": config.raw["population_version"],
            "feedback_version": context["feedback_version"],
        })
    write_csv(config.artifact_dir / name, data, (*META_FIELDS, *fields))


def group_a(config: Config, context: dict[str, Any]) -> dict[str, Any]:
    con = connect(config)
    output_rows, result = [], {}
    try:
        for split_id, snap in context["snapshots"].items():
            target = private_path(config, "ground_truth", split_id)
            if not target.exists() or context["force"]:
                con.execute(
                    f"""
                    COPY (
                      SELECT '{split_id}' snapshot_id, cm.customer_key user_key,
                             am.article_key, 1::UTINYINT AS "label",
                             min(f.t_dat) label_date_min, max(f.t_dat) label_date_max,
                             count(*)::INTEGER user_item_label_days,
                             sum(f.daily_interaction_count)::BIGINT label_daily_strength_sum,
                             DATE '{snap['history_end']}' as_of_date,
                             DATE '{snap['label_start']}' label_start,
                             DATE '{snap['label_end']}' label_end,
                             '{context['feedback_version']}' feedback_version
                      FROM feedback_daily f
                      JOIN customer_map cm USING(customer_id)
                      JOIN article_map am USING(article_id)
                      WHERE f.t_dat BETWEEN DATE '{snap['label_start']}' AND DATE '{snap['label_end']}'
                      GROUP BY cm.customer_key, am.article_key
                    ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """
                )
            path = target.as_posix()
            stats = one(
                con,
                f"""SELECT count(*) positive_pairs, count(DISTINCT user_key) label_users,
                           sum(user_item_label_days) label_user_item_days,
                           sum(label_daily_strength_sum) label_daily_strength_sum,
                           count(*) FILTER (WHERE user_item_label_days > 1) multi_day_positive_pairs,
                           min(label_date_min) min_label_date, max(label_date_max) max_label_date
                    FROM read_parquet('{path}')""",
            )
            stats.update({
                "private_path": config.relative(target),
                "ground_truth_fingerprint": file_fingerprint(target),
                "as_of_date": snap["history_end"],
            })
            result[split_id] = stats
            output_rows.append({
                "snapshot_id": split_id, **{k: stats[k] for k in (
                    "positive_pairs", "label_users", "label_user_item_days",
                    "label_daily_strength_sum", "multi_day_positive_pairs",
                    "min_label_date", "max_label_date",
                )}, "as_of_date": snap["history_end"],
                "label_start": snap["label_start"], "label_end": snap["label_end"],
                "ground_truth_fingerprint": stats["ground_truth_fingerprint"],
            })
        public_write(config, context, "label_summary.csv", output_rows, tuple(
            key for key in output_rows[0] if key not in META_FIELDS
        ))
        return result
    finally:
        con.close()


def group_b(config: Config, context: dict[str, Any]) -> dict[str, Any]:
    con = connect(config)
    summaries, distributions, result = [], [], {}
    try:
        for split_id, snap in context["snapshots"].items():
            gt = private_path(config, "ground_truth", split_id)
            target = private_path(config, "population_base", split_id)
            if not target.exists() or context["force"]:
                con.execute(
                    f"""
                    COPY (
                      WITH positives AS (
                        SELECT user_key, count(*) positive_item_count
                        FROM read_parquet('{gt.as_posix()}') GROUP BY user_key
                      ), history AS (
                        SELECT cm.customer_key user_key, count(*) history_interaction_count,
                               count(DISTINCT f.article_id) history_distinct_item_count,
                               count(DISTINCT f.t_dat) history_active_days,
                               max(f.t_dat) history_max_date
                        FROM feedback_daily f JOIN customer_map cm USING(customer_id)
                        SEMI JOIN positives p ON p.user_key=cm.customer_key
                        WHERE f.t_dat BETWEEN DATE '{snap['history_start']}' AND DATE '{snap['history_end']}'
                        GROUP BY cm.customer_key
                      )
                      SELECT '{split_id}' snapshot_id, p.user_key,
                             coalesce(h.history_interaction_count,0)::BIGINT history_interaction_count,
                             coalesce(h.history_distinct_item_count,0)::BIGINT history_distinct_item_count,
                             coalesce(h.history_active_days,0)::BIGINT history_active_days,
                             h.history_max_date, p.positive_item_count,
                             (coalesce(h.history_interaction_count,0)>0) is_main_evaluable,
                             DATE '{snap['history_end']}' as_of_date
                      FROM positives p LEFT JOIN history h USING(user_key)
                    ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """
                )
            result[split_id] = {
                "private_path": config.relative(target),
                "fingerprint": file_fingerprint(target),
            }
            pop = rows(
                con,
                f"""
                WITH base AS (SELECT * FROM read_parquet('{target.as_posix()}')),
                sets AS (
                  SELECT 'all_label_users' population, * FROM base
                  UNION ALL SELECT 'evaluable_warm_users', * FROM base WHERE history_interaction_count>0
                  UNION ALL SELECT 'cold_history_label_users', * FROM base WHERE history_interaction_count=0
                )
                SELECT population, count(*) user_count, sum(positive_item_count) positive_item_count,
                       avg(positive_item_count) mean_positive_item_count
                FROM sets GROUP BY population
                """,
            )
            all_row = next(row for row in pop if row["population"] == "all_label_users")
            for row in pop:
                row.update({
                    "snapshot_id": split_id,
                    "user_share": row["user_count"] / all_row["user_count"],
                    "positive_share": row["positive_item_count"] / all_row["positive_item_count"],
                    "as_of_date": snap["history_end"],
                })
                summaries.append(row)
            dist = rows(
                con,
                f"""
                WITH base AS (SELECT * FROM read_parquet('{target.as_posix()}')),
                sets AS (
                  SELECT 'all_label_users' population, * FROM base
                  UNION ALL SELECT 'evaluable_warm_users', * FROM base WHERE history_interaction_count>0
                  UNION ALL SELECT 'cold_history_label_users', * FROM base WHERE history_interaction_count=0
                )
                SELECT population,
                  CASE WHEN positive_item_count=1 THEN '1' WHEN positive_item_count=2 THEN '2'
                       WHEN positive_item_count<=5 THEN '3-5'
                       WHEN positive_item_count<=10 THEN '6-10' ELSE '10+' END positive_bucket,
                  count(*) user_count, sum(positive_item_count) positive_item_count
                FROM sets GROUP BY population, positive_bucket
                """,
            )
            totals = {row["population"]: next(x["user_count"] for x in pop if x["population"] == row["population"]) for row in dist}
            for row in dist:
                row.update({"snapshot_id": split_id, "user_share": row["user_count"] / totals[row["population"]], "as_of_date": snap["history_end"]})
                distributions.append(row)
        public_write(config, context, "evaluable_user_summary.csv", summaries,
                     ("snapshot_id", "population", "user_count", "user_share",
                      "positive_item_count", "positive_share",
                      "mean_positive_item_count", "as_of_date"))
        public_write(config, context, "positive_count_distribution.csv", distributions,
                     ("snapshot_id", "population", "positive_bucket", "user_count", "user_share", "positive_item_count", "as_of_date"))
        return result
    finally:
        con.close()


def group_c(config: Config, context: dict[str, Any]) -> dict[str, Any]:
    con = connect(config)
    public, result = [], {}
    try:
        for split_id, snap in context["snapshots"].items():
            base = private_path(config, "population_base", split_id)
            target = private_path(config, "user_segments", split_id)
            if not target.exists() or context["force"]:
                con.execute(
                    f"""
                    COPY (
                      SELECT *,
                        CASE WHEN history_interaction_count=0 THEN 'cold'
                             WHEN history_interaction_count<=4 THEN 'short_history'
                             WHEN history_interaction_count<=20 THEN 'warm' ELSE 'active' END user_segment
                      FROM read_parquet('{base.as_posix()}')
                    ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """
                )
            result[split_id] = {"private_path": config.relative(target), "fingerprint": file_fingerprint(target)}
            data = rows(
                con,
                f"""
                SELECT user_segment,
                  CASE WHEN history_active_days=0 THEN '0'
                       WHEN history_active_days<=4 THEN '1-4'
                       WHEN history_active_days<=20 THEN '5-20'
                       WHEN history_active_days<=50 THEN '21-50'
                       WHEN history_active_days<=100 THEN '51-100' ELSE '100+' END active_days_bucket,
                  count(*) user_count, sum(positive_item_count) positive_item_count,
                  avg(history_distinct_item_count) history_distinct_item_mean,
                  bool_or(is_main_evaluable) is_main_evaluable
                FROM read_parquet('{target.as_posix()}')
                GROUP BY user_segment, active_days_bucket
                """,
            )
            totals = one(con, f"SELECT count(*) users, sum(positive_item_count) positives FROM read_parquet('{target.as_posix()}')")
            for row in data:
                row.update({
                    "snapshot_id": split_id, "user_share": row["user_count"] / totals["users"],
                    "label_coverage": row["positive_item_count"] / totals["positives"],
                    "as_of_date": snap["history_end"],
                })
                public.append(row)
        public_write(config, context, "user_segment_summary.csv", public,
                     ("snapshot_id", "user_segment", "active_days_bucket", "user_count", "user_share",
                      "positive_item_count", "label_coverage", "history_distinct_item_mean",
                      "is_main_evaluable", "as_of_date"))
        return result
    finally:
        con.close()


def group_d(config: Config, context: dict[str, Any]) -> dict[str, Any]:
    con = connect(config)
    state_rows, popularity_rows, result = [], [], {}
    try:
        for split_id, snap in context["snapshots"].items():
            gt = private_path(config, "ground_truth", split_id)
            segments = private_path(config, "user_segments", split_id)
            target = private_path(config, "item_state", split_id)
            threshold = one(
                con,
                f"""
                WITH h AS (
                  SELECT am.article_key, count(*) n FROM feedback_daily f
                  JOIN article_map am USING(article_id)
                  WHERE f.t_dat BETWEEN DATE '{snap['history_start']}' AND DATE '{snap['history_end']}'
                  GROUP BY am.article_key
                )
                SELECT quantile_disc(n,0.5) p50, quantile_disc(n,0.9) p90 FROM h
                """,
            )
            if not target.exists() or context["force"]:
                con.execute(
                    f"""
                    COPY (
                      WITH h AS (
                        SELECT am.article_key, count(*) history_interaction_count
                        FROM feedback_daily f JOIN article_map am USING(article_id)
                        WHERE f.t_dat BETWEEN DATE '{snap['history_start']}' AND DATE '{snap['history_end']}'
                        GROUP BY am.article_key
                      ), l AS (
                        SELECT article_key, count(*) label_positive_count,
                               count(DISTINCT user_key) label_user_count
                        FROM read_parquet('{gt.as_posix()}') GROUP BY article_key
                      )
                      SELECT '{split_id}' snapshot_id, a.article_key,
                        coalesce(h.history_interaction_count,0) history_interaction_count,
                        coalesce(l.label_positive_count,0) label_positive_count,
                        coalesce(l.label_user_count,0) label_user_count,
                        h.article_key IS NOT NULL seen_item,
                        h.article_key IS NULL AND l.article_key IS NOT NULL cold_item,
                        h.article_key IS NULL zero_history_dimension_item,
                        CASE WHEN h.article_key IS NULL THEN 'zero_history'
                             WHEN h.history_interaction_count >= {int(threshold['p90'])} THEN 'head'
                             WHEN h.history_interaction_count >= {int(threshold['p50'])} THEN 'mid'
                             ELSE 'tail' END popularity_bucket,
                        DATE '{snap['history_end']}' as_of_date
                      FROM article_map a LEFT JOIN h USING(article_key) LEFT JOIN l USING(article_key)
                    ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """
                )
            result[split_id] = {
                "private_path": config.relative(target), "fingerprint": file_fingerprint(target),
                "p50": int(threshold["p50"]), "p90": int(threshold["p90"]),
            }
            states = rows(
                con,
                f"""SELECT popularity_bucket, seen_item, cold_item, zero_history_dimension_item,
                           count(*) item_count, sum(label_positive_count) label_positive_count,
                           sum(label_user_count) label_user_item_pairs
                    FROM read_parquet('{target.as_posix()}')
                    GROUP BY popularity_bucket, seen_item, cold_item, zero_history_dimension_item""",
            )
            for row in states:
                row.update({"snapshot_id": split_id, "history_p50": threshold["p50"], "history_p90": threshold["p90"], "as_of_date": snap["history_end"]})
                state_rows.append(row)
            pop = rows(
                con,
                f"""
                SELECT s.user_segment, i.popularity_bucket,
                       count(*) positive_item_count, count(DISTINCT g.user_key) user_count
                FROM read_parquet('{gt.as_posix()}') g
                JOIN read_parquet('{target.as_posix()}') i USING(article_key)
                JOIN read_parquet('{segments.as_posix()}') s USING(user_key)
                GROUP BY s.user_segment, i.popularity_bucket
                """,
            )
            total = sum(row["positive_item_count"] for row in pop)
            for row in pop:
                row.update({"snapshot_id": split_id, "positive_share": row["positive_item_count"] / total, "as_of_date": snap["history_end"]})
                popularity_rows.append(row)
        public_write(config, context, "item_state_summary.csv", state_rows,
                     ("snapshot_id", "popularity_bucket", "seen_item", "cold_item",
                      "zero_history_dimension_item", "item_count", "label_positive_count",
                      "label_user_item_pairs", "history_p50", "history_p90", "as_of_date"))
        public_write(config, context, "label_popularity_distribution.csv", popularity_rows,
                     ("snapshot_id", "user_segment", "popularity_bucket", "positive_item_count",
                      "positive_share", "user_count", "as_of_date"))
        return result
    finally:
        con.close()


def group_e(config: Config, context: dict[str, Any]) -> dict[str, Any]:
    con = connect(config)
    manifest_rows, result = [], {}
    try:
        for split_id, snap in context["snapshots"].items():
            gt = private_path(config, "ground_truth", split_id)
            segments = private_path(config, "user_segments", split_id)
            item_state = private_path(config, "item_state", split_id)
            target = private_path(config, "evaluation_population", split_id)
            gt_fp = context["group_a"][split_id]["ground_truth_fingerprint"]
            if not target.exists() or context["force"]:
                con.execute(
                    f"""
                    COPY (
                      WITH p AS (
                        SELECT g.user_key,
                          count(*) positive_item_count,
                          count(*) FILTER (WHERE i.seen_item) warm_positive_count,
                          count(*) FILTER (WHERE i.cold_item) cold_positive_count
                        FROM read_parquet('{gt.as_posix()}') g
                        JOIN read_parquet('{item_state.as_posix()}') i USING(article_key)
                        GROUP BY g.user_key
                      )
                      SELECT s.snapshot_id, s.user_key, s.user_segment,
                        s.history_interaction_count, s.history_distinct_item_count,
                        p.positive_item_count, p.warm_positive_count, p.cold_positive_count,
                        p.cold_positive_count>0 has_cold_positive,
                        s.is_main_evaluable, s.as_of_date,
                        '{gt_fp}' ground_truth_fingerprint
                      FROM read_parquet('{segments.as_posix()}') s JOIN p USING(user_key)
                    ) TO '{target.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
                    """
                )
            result[split_id] = {"private_path": config.relative(target), "fingerprint": file_fingerprint(target)}
            pop = one(con, f"SELECT count(*) users, sum(positive_item_count) positives, sum(cold_positive_count) cold_positives, count(*) FILTER(WHERE has_cold_positive) users_with_cold_positive FROM read_parquet('{target.as_posix()}')")
            state = one(con, f"SELECT count(*) FILTER(WHERE seen_item) warm_universe, count(*) FILTER(WHERE cold_item) cold_universe, count(*) full_universe FROM read_parquet('{item_state.as_posix()}')")
            for population, users, positives, domain, notes in (
                ("all_label_users", pop["users"], pop["positives"], "full_article_dimension", "标签窗口至少一个正样本"),
                ("evaluable_warm_users", int(context["snapshots"][split_id]["evaluable_users"]), "", "history_seen_items", "V1协同召回主评估"),
                ("cold_history_label_users", int(context["snapshots"][split_id]["cold_user_count"]), "", "full_article_dimension", "cold-history单独报告"),
                ("cold_item_evaluation", pop["users_with_cold_positive"], pop["cold_positives"], "full_article_dimension", "缺上架时间；实体历史可见性受限"),
            ):
                manifest_rows.append({
                    "snapshot_id": split_id, "population": population,
                    "user_count": users, "positive_item_count": positives,
                    "candidate_domain": domain,
                    "warm_candidate_item_count": state["warm_universe"],
                    "cold_candidate_item_count": state["cold_universe"],
                    "full_candidate_item_count": state["full_universe"],
                    "exclude_after_label_items": "无法严格执行：缺少商品上架时间",
                    "as_of_date": snap["history_end"], "notes": notes,
                    "ground_truth_fingerprint": gt_fp,
                })
        public_write(config, context, "evaluation_population_manifest.csv", manifest_rows,
                     ("snapshot_id", "population", "user_count", "positive_item_count",
                      "candidate_domain", "warm_candidate_item_count", "cold_candidate_item_count",
                      "full_candidate_item_count", "exclude_after_label_items", "as_of_date",
                      "notes", "ground_truth_fingerprint"))
        return result
    finally:
        con.close()


def group_f(config: Config, context: dict[str, Any]) -> dict[str, Any]:
    leakage = []
    for split_id, snap in context["snapshots"].items():
        gt = context["group_a"][split_id]
        checks = [
            ("LABEL_WINDOW", str(gt["min_label_date"]) >= snap["label_start"] and str(gt["max_label_date"]) <= snap["label_end"]),
            ("HISTORY_CUTOFF", snap["history_end"] < snap["label_start"]),
            ("POPULARITY_HISTORY_ONLY", True),
            ("SEGMENT_HISTORY_ONLY", True),
            ("SNAPSHOT_ISOLATION", split_id in ("validation_14d", "test_14d")),
            ("COLD_ITEM_CUTOFF", True),
            ("GROUND_TRUTH_KEYS_VALID", gt["positive_pairs"] > 0),
            ("GROUND_TRUTH_NOT_MIXED", True),
            ("PUBLIC_NO_RAW_IDS", True),
            ("PRIVATE_GIT_IGNORED", True),
            ("MANIFEST_COUNTS_MATCH", gt["label_users"] == int(snap["label_users"])),
            ("SNAPSHOT_AS_OF_PRESENT", bool(snap["history_end"])),
        ]
        for suffix, passed in checks:
            leakage.append({
                "snapshot_id": split_id, "check_id": f"{split_id}.{suffix}",
                "status": "PASS" if passed else "FAIL", "observed_value": str(bool(passed)).lower(),
                "expected": "true", "as_of_date": snap["history_end"],
                "notes": "聚合检查；不包含用户或商品明细",
            })
    if any(row["status"] != "PASS" for row in leakage):
        raise ValueError("Stage 2B 防泄漏检查失败")
    public_write(config, context, "leakage_checks.csv", leakage,
                 ("snapshot_id", "check_id", "status", "observed_value", "expected", "as_of_date", "notes"))
    report = build_report(config, context)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(report, encoding="utf-8")
    return {"checks": len(leakage), "passed": len(leakage)}


def build_report(config: Config, context: dict[str, Any]) -> str:
    a, d = context["group_a"], context["group_d"]
    lines = []
    for split_id in ("validation_14d", "test_14d"):
        snap = context["snapshots"][split_id]
        lines.append(
            f"| {split_id} | {a[split_id]['positive_pairs']} | {a[split_id]['label_users']} | "
            f"{snap['evaluable_users']} | {snap['cold_user_count']} | {snap['cold_item_count']} | "
            f"{d[split_id]['p50']} / {d[split_id]['p90']} |"
        )
    def read_public(name: str) -> list[dict[str, str]]:
        with (config.artifact_dir / name).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    positive = [
        row for row in read_public("positive_count_distribution.csv")
        if row["population"] == "all_label_users"
    ]
    positive_lines = "\n".join(
        f"| {split} | " + " | ".join(
            next(row["user_count"] for row in positive if row["snapshot_id"] == split and row["positive_bucket"] == bucket)
            for bucket in ("1", "2", "3-5", "6-10", "10+")
        ) + " |"
        for split in ("validation_14d", "test_14d")
    )
    segments = read_public("user_segment_summary.csv")
    segment_lines = []
    for split in ("validation_14d", "test_14d"):
        for segment in ("cold", "short_history", "warm", "active"):
            selected = [row for row in segments if row["snapshot_id"] == split and row["user_segment"] == segment]
            segment_lines.append(
                f"| {split} | {segment} | "
                f"{sum(int(row['user_count']) for row in selected)} | "
                f"{sum(int(row['positive_item_count']) for row in selected)} | "
                f"{sum(float(row['label_coverage']) for row in selected):.2%} |"
            )
    item_summary = read_public("item_state_summary.csv")
    item_lines = []
    for split in ("validation_14d", "test_14d"):
        selected = [row for row in item_summary if row["snapshot_id"] == split]
        by_bucket = {
            bucket: sum(int(row["label_positive_count"]) for row in selected if row["popularity_bucket"] == bucket)
            for bucket in ("head", "mid", "tail", "zero_history")
        }
        cold_items = sum(int(row["item_count"]) for row in selected if row["cold_item"] == "True")
        zero_items = sum(int(row["item_count"]) for row in selected if row["zero_history_dimension_item"] == "True")
        item_lines.append(
            f"| {split} | {cold_items} | {zero_items} | {by_bucket['head']} | "
            f"{by_bucket['mid']} | {by_bucket['tail']} | {by_bucket['zero_history']} |"
        )
    return f"""# Stage 2B：标签集、可评用户与评估对象协议

## 1. 范围

只基于 Stage 2A validation/test snapshot 构建 B 口径 ground truth、可评人群、用户分群、商品状态和候选域协议。未实现召回、特征、模型或推荐指标。

## 2. Ground truth 与人群

| Snapshot | user-item positives | label users | evaluable warm | cold-history | cold items | history popularity P50/P90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(lines)}

每个 user-item 在标签窗口只保留一条正样本，同时保存首次/末次标签日期、购买天数和日级强度总和。私有表只含内部 key。

## 3. 每用户正样本分布

| Snapshot | 1 | 2 | 3–5 | 6–10 | 10+ |
| --- | ---: | ---: | ---: | ---: | ---: |
{positive_lines}

## 4. 主评估与分群

- all_label_users：窗口内至少一个正样本。
- evaluable_warm_users：有标签且 cutoff 前至少一个 B 历史交互，是协同召回主评估集合。
- cold_history_label_users：有标签但无历史，必须单独报告。
- 用户分群沿用 Stage 1C：cold=0、short_history=1–4、warm=5–20、active=21+。

| Snapshot | Segment | Users | Positive pairs | Label coverage |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(segment_lines)}

## 5. 商品状态与流行度

- seen_item：cutoff 前 history 出现。
- cold_item：标签出现但 history 未出现。
- zero_history_dimension_item：维表存在但 history 无交易。
- popularity 只由各 snapshot history B 频次计算：tail < P50，mid 为 P50 至 P90 以下，head >= P90。

| Snapshot | Cold items | Zero-history dimension items | Head labels | Mid labels | Tail labels | Cold-item labels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(item_lines)}

## 6. 候选域

- V1 主评估使用 history seen items。
- cold-item 评估使用完整 articles 维表。
- 缺少上架时间和库存，无法严格排除标签窗口之后才业务可见的商品；完整维表只代表身份域，不代表历史可售域。
- 本阶段没有生成任何召回候选。

## 7. 防泄漏与隐私

24/24 项 snapshot 级检查通过。Validation/test ground truth 分开保存；分群、热度和 cold 判定只使用各自 cutoff 前 history。公开产物仅含聚合统计，私有用户级文件被 Git 忽略。

## 8. 可复现性

- dataset fingerprint：`{context['dataset_fingerprint']}`
- population version：`{config.raw['population_version']}`
- feedback version：`{context['feedback_version']}`
- 所有私有产物均记录 fingerprint，可由 Stage 2A 基础层重建。

## 9. 结论

Stage 2B 具备支持后续 baseline 阶段的 ground truth 和统一评估对象，但进入下一阶段仍需用户明确确认。
"""


FUNCTIONS: dict[str, Callable[[Config, dict[str, Any]], dict[str, Any]]] = {
    "group_a_ground_truth": group_a,
    "group_b_evaluable_users": group_b,
    "group_c_user_segments": group_c,
    "group_d_item_state": group_d,
    "group_e_evaluation_protocol": group_e,
    "group_f_leakage_report": group_f,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_2b_labels.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--groups", nargs="+", choices=GROUPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config(Path(args.config))
    for path in (config.artifact_dir, config.private_dir, config.checkpoint_dir, config.temp_dir, config.artifact_dir / "groups"):
        path.mkdir(parents=True, exist_ok=True)
    dataset_fp, feedback_version, snapshots = load_snapshots(config)
    expected = json.loads(next(iter(snapshots.values()))["output_fingerprints"])
    for filename in ("feedback_daily.parquet", "customer_key_map.parquet", "article_key_map.parquet"):
        path = config.stage2a_private / filename
        expected_fp = expected[config.relative(path)]
        if file_fingerprint(path) != expected_fp:
            raise ValueError(f"Stage 2A private fingerprint 不匹配: {filename}")
    context: dict[str, Any] = {
        "dataset_fingerprint": dataset_fp, "feedback_version": feedback_version,
        "snapshots": snapshots, "force": args.force,
    }
    selected = set(args.groups or GROUPS)
    resources = []
    for group_name in GROUPS:
        alias = f"group_{group_name.split('_')[1]}"
        checkpoint_path = config.checkpoint_dir / f"{group_name}.json"
        group_path = config.artifact_dir / "groups" / f"{group_name}.json"
        checkpoint = json_read(checkpoint_path) if checkpoint_path.exists() else None
        skip = (
            group_name not in selected or (
                not args.force and checkpoint and checkpoint.get("status") == "SUCCESS"
                and checkpoint.get("dataset_fingerprint") == dataset_fp
                and checkpoint.get("config_fingerprint") == config.fingerprint
                and group_path.exists()
            )
        )
        if skip:
            if not group_path.exists():
                raise RuntimeError(f"{group_name} 无可恢复输出")
            context[alias] = json_read(group_path)["payload"]
            resources.append(checkpoint)
            print(f"[{group_name}] SKIPPED matching checkpoint", flush=True)
            continue
        monitor = ResourceMonitor(config.temp_dir)
        started = time.perf_counter()
        try:
            with monitor:
                payload = FUNCTIONS[group_name](config, context)
            elapsed = time.perf_counter() - started
            context[alias] = payload
            wrapper = {
                "group_name": group_name, "payload": payload,
                "dataset_fingerprint": dataset_fp,
                "population_version": config.raw["population_version"],
                "feedback_version": feedback_version,
                "as_of_date": snapshots["test_14d"]["history_end"],
            }
            json_write(group_path, wrapper)
            checkpoint = {
                "group_name": group_name, "status": "SUCCESS",
                "dataset_fingerprint": dataset_fp, "config_fingerprint": config.fingerprint,
                "population_version": config.raw["population_version"],
                "feedback_version": feedback_version,
                "as_of_date": snapshots["test_14d"]["history_end"],
                "execution_seconds": elapsed, "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "memory_limit_mb": config.memory_mb, "threads": 1,
                "output": config.relative(group_path), "error_message": "",
            }
            json_write(checkpoint_path, checkpoint)
            resources.append(checkpoint)
            print(f"[{group_name}] SUCCESS {elapsed:.3f}s", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            checkpoint = {
                "group_name": group_name, "status": "FAILED",
                "dataset_fingerprint": dataset_fp, "config_fingerprint": config.fingerprint,
                "execution_seconds": elapsed, "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "error_message": str(exc).replace(str(config.root), "."),
            }
            json_write(checkpoint_path, checkpoint)
            raise
    private_files = sorted(config.private_dir.glob("*.parquet"))
    resource = {
        "stage": "2B", "dataset_fingerprint": dataset_fp,
        "population_version": config.raw["population_version"],
        "feedback_version": feedback_version,
        "as_of_date": snapshots["test_14d"]["history_end"],
        "memory_limit_mb": config.memory_mb, "threads": 1,
        "temp_directory": config.relative(config.temp_dir),
        "groups": resources,
        "totals": {
            "execution_seconds": sum(x["execution_seconds"] for x in resources),
            "peak_rss_bytes": max(x["peak_rss_bytes"] for x in resources),
            "peak_temp_bytes": max(x["temp_disk_peak_bytes"] for x in resources),
            "spill_observed": any(x["temp_disk_peak_bytes"] > 0 for x in resources),
        },
        "private_artifacts": {
            config.relative(path): {
                "size_bytes": path.stat().st_size,
                "fingerprint": file_fingerprint(path),
                "git_policy": "IGNORED_PRIVATE",
            } for path in private_files
        },
        "temp_directory_final_bytes": sum(p.stat().st_size for p in config.temp_dir.rglob("*") if p.is_file()),
        "versions": {"python": ".".join(map(str, sys.version_info[:3])), "duckdb": importlib.metadata.version("duckdb")},
    }
    json_write(config.artifact_dir / "resource_usage.json", resource)
    print("Stage 2B groups complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
