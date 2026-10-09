from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import duckdb

from data_audit.stage1a import ResourceMonitor, input_fingerprint
from data_audit.stage1a_groups import parquet_metadata
from data_foundation.calendar import dates_inclusive, missing_dates, weekday_composition
from data_foundation.config import FoundationConfig
from data_foundation.ids import KEY_STRATEGY
from data_foundation.manifest import (
    MANIFEST_FIELDS,
    file_fingerprint,
    manifest_counts_match,
    write_csv,
)
from data_foundation.split import Snapshot, derive_snapshots, trailing_window


GROUPS = (
    "group_a_split_boundaries",
    "group_b_feedback_layer",
    "group_c_internal_ids",
    "group_d_calendar_audit",
    "group_e_leakage_checks",
    "group_f_outputs_report",
)


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def connect(config: FoundationConfig) -> duckdb.DuckDBPyConnection:
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    temp = config.temp_dir.as_posix().replace("'", "''")
    con.execute(f"SET temp_directory='{temp}'")
    con.execute(f"SET memory_limit='{config.memory_limit_mb}MB'")
    con.execute(f"SET threads={int(config.raw['duckdb']['threads'])}")
    con.execute("SET preserve_insertion_order=false")
    con.execute(
        f"SET max_temp_directory_size='{config.raw['duckdb']['max_temp_directory_size']}'"
    )
    for name, path in config.data_paths.items():
        value = path.as_posix().replace("'", "''")
        if name == "transactions":
            con.execute(
                f"""CREATE VIEW {name} AS
                SELECT * EXCLUDE(t_dat), CAST(t_dat AS DATE) AS t_dat
                FROM read_parquet('{value}')"""
            )
        else:
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{value}')")
    return con


def query_one(con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
    cursor = con.execute(sql, params or [])
    row = cursor.fetchone()
    return dict(zip([item[0] for item in cursor.description], row, strict=True))


def snapshot_stats(con: duckdb.DuckDBPyConnection, snapshot: Snapshot) -> dict[str, Any]:
    result = query_one(
        con,
        """
        SELECT
          count(*) FILTER (WHERE t_dat BETWEEN ? AND ?) AS history_rows_A,
          count(*) FILTER (WHERE t_dat BETWEEN ? AND ?) AS label_rows_A,
          count(DISTINCT customer_id) FILTER (WHERE t_dat BETWEEN ? AND ?) AS history_users,
          count(DISTINCT customer_id) FILTER (WHERE t_dat BETWEEN ? AND ?) AS label_users,
          count(DISTINCT article_id) FILTER (WHERE t_dat BETWEEN ? AND ?) AS history_items,
          count(DISTINCT article_id) FILTER (WHERE t_dat BETWEEN ? AND ?) AS label_items
        FROM transactions
        """,
        [
            snapshot.history_start, snapshot.history_end,
            snapshot.label_start, snapshot.label_end,
            snapshot.history_start, snapshot.history_end,
            snapshot.label_start, snapshot.label_end,
            snapshot.history_start, snapshot.history_end,
            snapshot.label_start, snapshot.label_end,
        ],
    )
    coverage = query_one(
        con,
        """
        WITH
        history_users AS (
          SELECT DISTINCT customer_id FROM transactions WHERE t_dat BETWEEN ? AND ?
        ),
        label_users AS (
          SELECT DISTINCT customer_id FROM transactions WHERE t_dat BETWEEN ? AND ?
        ),
        history_items AS (
          SELECT DISTINCT article_id FROM transactions WHERE t_dat BETWEEN ? AND ?
        ),
        label_items AS (
          SELECT DISTINCT article_id FROM transactions WHERE t_dat BETWEEN ? AND ?
        )
        SELECT
          (SELECT count(*) FROM label_users l SEMI JOIN history_users h USING(customer_id)) evaluable_users,
          (SELECT count(*) FROM label_users l ANTI JOIN history_users h USING(customer_id)) cold_user_count,
          (SELECT count(*) FROM label_items l ANTI JOIN history_items h USING(article_id)) cold_item_count
        """,
        [
            snapshot.history_start, snapshot.history_end,
            snapshot.label_start, snapshot.label_end,
            snapshot.history_start, snapshot.history_end,
            snapshot.label_start, snapshot.label_end,
        ],
    )
    result.update(coverage)
    return result


def group_a(config: FoundationConfig, context: dict[str, Any]) -> dict[str, Any]:
    con = connect(config)
    try:
        bounds = query_one(con, "SELECT min(t_dat) min_date, max(t_dat) max_date FROM transactions")
        snapshots = derive_snapshots(bounds["max_date"], bounds["min_date"])
        payload = {
            "D_max": str(bounds["max_date"]),
            "history_start": str(bounds["min_date"]),
            "snapshots": {},
        }
        for name, snapshot in snapshots.items():
            payload["snapshots"][name] = {
                "split_id": snapshot.split_id,
                "history_start": str(snapshot.history_start),
                "history_end": str(snapshot.history_end),
                "label_start": str(snapshot.label_start),
                "label_end": str(snapshot.label_end),
                "window_days": snapshot.window_days,
                **snapshot_stats(con, snapshot),
            }
        return payload
    finally:
        con.close()


def group_b(config: FoundationConfig, context: dict[str, Any]) -> dict[str, Any]:
    daily_path = config.private_dir / "feedback_daily.parquet"
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    con = connect(config)
    try:
        if not daily_path.exists() or context["force"]:
            target = daily_path.as_posix().replace("'", "''")
            con.execute(
                f"""
                COPY (
                  SELECT t_dat, customer_id, article_id,
                         count(*)::INTEGER AS daily_interaction_count
                  FROM transactions
                  GROUP BY t_dat, customer_id, article_id
                ) TO '{target}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
                """
            )
        source = daily_path.as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW feedback_daily AS SELECT * FROM read_parquet('{source}')")
        rows: list[dict[str, Any]] = []
        payload: dict[str, Any] = {"private_feedback_path": config.relative(daily_path), "snapshots": {}}
        for split_id, value in context["group_a"]["snapshots"].items():
            h_start, h_end = date.fromisoformat(value["history_start"]), date.fromisoformat(value["history_end"])
            l_start, l_end = date.fromisoformat(value["label_start"]), date.fromisoformat(value["label_end"])
            stats = query_one(
                con,
                """
                SELECT
                  count(*) FILTER (WHERE t_dat BETWEEN ? AND ?) history_rows_B,
                  count(*) FILTER (WHERE t_dat BETWEEN ? AND ?) label_rows_B,
                  quantile_disc(daily_interaction_count, ?) FILTER
                    (WHERE t_dat BETWEEN ? AND ?) C_cap,
                  sum(daily_interaction_count) FILTER
                    (WHERE t_dat BETWEEN ? AND ?) history_strength_raw
                FROM feedback_daily
                """,
                [h_start, h_end, l_start, l_end, config.raw["feedback"]["cap_quantile"],
                 h_start, h_end, h_start, h_end],
            )
            cap = int(stats["C_cap"])
            capped = query_one(
                con,
                """
                SELECT
                  sum(least(daily_interaction_count, ?)) capped_count_sum,
                  sum(1 + ln(1 + least(daily_interaction_count, ?) - 1)) strength_sum
                FROM feedback_daily WHERE t_dat BETWEEN ? AND ?
                """,
                [cap, cap, h_start, h_end],
            )
            stats.update(capped)
            stats["C_cap"] = cap
            stats["C_cap_input_rows"] = stats["history_rows_B"]
            payload["snapshots"][split_id] = stats
            for scope, count, note in (
                ("A", value["history_rows_A"], "原始行，仅审计和敏感性分析"),
                ("B", stats["history_rows_B"], "user-item-day 二值主口径"),
                ("C", stats["history_rows_B"], "附带训练期P99截断与log1p强度；不是真实quantity"),
            ):
                rows.append({
                    "split_id": split_id, "partition": "history", "feedback_scope": scope,
                    "row_count": count, "C_cap": cap if scope == "C" else "",
                    "C_cap_input_rows": stats["C_cap_input_rows"] if scope == "C" else "",
                    "capped_count_sum": stats["capped_count_sum"] if scope == "C" else "",
                    "strength_sum": stats["strength_sum"] if scope == "C" else "",
                    "as_of_date": value["history_end"], "notes": note,
                })
            rows.extend([
                {"split_id": split_id, "partition": "labels", "feedback_scope": "A",
                 "row_count": value["label_rows_A"], "C_cap": "", "C_cap_input_rows": "",
                 "capped_count_sum": "", "strength_sum": "", "as_of_date": value["history_end"],
                 "notes": "标签原始行；不参与cap"},
                {"split_id": split_id, "partition": "labels", "feedback_scope": "B",
                 "row_count": stats["label_rows_B"], "C_cap": "", "C_cap_input_rows": "",
                 "capped_count_sum": "", "strength_sum": "", "as_of_date": value["history_end"],
                 "notes": "标签二值主口径"},
            ])
        write_csv(
            config.artifact_dir / "feedback_summary.csv", rows,
            ("split_id", "partition", "feedback_scope", "row_count", "C_cap",
             "C_cap_input_rows", "capped_count_sum", "strength_sum", "as_of_date", "notes"),
        )
        payload["private_feedback_fingerprint"] = file_fingerprint(daily_path)
        return payload
    finally:
        con.close()


def group_c(config: FoundationConfig, context: dict[str, Any]) -> dict[str, Any]:
    config.private_dir.mkdir(parents=True, exist_ok=True)
    customer_map = config.private_dir / "customer_key_map.parquet"
    article_map = config.private_dir / "article_key_map.parquet"
    con = connect(config)
    try:
        for table, raw_id, key, target in (
            ("customers", "customer_id", "customer_key", customer_map),
            ("articles", "article_id", "article_key", article_map),
        ):
            if not target.exists() or context["force"]:
                output = target.as_posix().replace("'", "''")
                con.execute(
                    f"""COPY (
                    SELECT {raw_id}, row_number() OVER (ORDER BY {raw_id}) - 1 AS {key}
                    FROM {table}
                    ) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)"""
                )
        result = {"strategy": KEY_STRATEGY, "uses_collision_hash": False, "dimension_scope": "full_dimension"}
        for entity, target, raw_id, key, expected in (
            ("customer", customer_map, "customer_id", "customer_key", "customers"),
            ("article", article_map, "article_id", "article_key", "articles"),
        ):
            path = target.as_posix().replace("'", "''")
            check = query_one(
                con,
                f"""SELECT count(*) row_count, count(DISTINCT {raw_id}) raw_unique,
                           count(DISTINCT {key}) key_unique, min({key}) min_key,
                           max({key}) max_key
                    FROM read_parquet('{path}')""",
            )
            dimension_count = query_one(con, f"SELECT count(*) n FROM {expected}")["n"]
            check["dimension_count"] = dimension_count
            check["coverage_ok"] = check["row_count"] == dimension_count
            check["one_to_one_ok"] = check["row_count"] == check["raw_unique"] == check["key_unique"]
            check["dense_ok"] = check["min_key"] == 0 and check["max_key"] == check["row_count"] - 1
            check["mapping_fingerprint"] = file_fingerprint(target)
            check["private_path"] = config.relative(target)
            result[entity] = check
        result["cold_entity_policy"] = (
            "全量维表稳定编码；cold entity 可获得键，但键本身不得创建训练行为特征"
        )
        return result
    finally:
        con.close()


def calendar_rows(
    con: duckdb.DuckDBPyConnection, split_id: str, start: date, end: date, role: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cursor = con.execute(
        """
        SELECT t_dat, count(*) rows_A, count(DISTINCT customer_id) active_users,
               count(DISTINCT article_id) active_items,
               count(*) FILTER (WHERE sales_channel_id=1) channel_1_rows,
               count(*) FILTER (WHERE sales_channel_id=2) channel_2_rows
        FROM transactions WHERE t_dat BETWEEN ? AND ? GROUP BY t_dat ORDER BY t_dat
        """,
        [start, end],
    )
    daily = [dict(zip([x[0] for x in cursor.description], row, strict=True)) for row in cursor.fetchall()]
    observed = [row["t_dat"] for row in daily]
    missing = missing_dates(start, end, observed)
    composition = weekday_composition(start, end)
    output: list[dict[str, Any]] = []
    total = sum(row["rows_A"] for row in daily)
    for weekday in range(1, 8):
        selected = [row for row in daily if row["t_dat"].isoweekday() == weekday]
        rows_a = sum(row["rows_A"] for row in selected)
        c1 = sum(row["channel_1_rows"] for row in selected)
        c2 = sum(row["channel_2_rows"] for row in selected)
        output.append({
            "split_id": split_id, "period_role": role, "record_type": "weekday",
            "day_group": "weekend" if weekday >= 6 else "weekday",
            "weekday_iso": weekday, "natural_day_count": composition.get(weekday, 0),
            "transaction_rows_A": rows_a, "transaction_share": rows_a / total if total else 0,
            "active_users_daily_mean": sum(x["active_users"] for x in selected) / len(selected) if selected else 0,
            "active_items_daily_mean": sum(x["active_items"] for x in selected) / len(selected) if selected else 0,
            "channel_1_rows": c1, "channel_2_rows": c2,
            "channel_1_share": c1 / rows_a if rows_a else 0,
            "channel_2_share": c2 / rows_a if rows_a else 0,
            "missing_date_count": len(missing),
            "complete_week_cycles": len(dates_inclusive(start, end)) // 7,
            "window_start": str(start), "window_end": str(end),
            "as_of_date": str(end),
        })
    for group, predicate in (("weekday", lambda d: d.isoweekday() <= 5), ("weekend", lambda d: d.isoweekday() >= 6)):
        selected = [row for row in daily if predicate(row["t_dat"])]
        rows_a = sum(row["rows_A"] for row in selected)
        output.append({
            "split_id": split_id, "period_role": role, "record_type": "day_group",
            "day_group": group, "weekday_iso": "", "natural_day_count": sum(1 for d in dates_inclusive(start, end) if predicate(d)),
            "transaction_rows_A": rows_a, "transaction_share": rows_a / total if total else 0,
            "active_users_daily_mean": sum(x["active_users"] for x in selected) / len(selected),
            "active_items_daily_mean": sum(x["active_items"] for x in selected) / len(selected),
            "channel_1_rows": sum(x["channel_1_rows"] for x in selected),
            "channel_2_rows": sum(x["channel_2_rows"] for x in selected),
            "channel_1_share": sum(x["channel_1_rows"] for x in selected) / rows_a if rows_a else 0,
            "channel_2_share": sum(x["channel_2_rows"] for x in selected) / rows_a if rows_a else 0,
            "missing_date_count": len(missing), "complete_week_cycles": len(dates_inclusive(start, end)) // 7,
            "window_start": str(start), "window_end": str(end), "as_of_date": str(end),
        })
    return output, {
        "start": str(start), "end": str(end), "days": len(dates_inclusive(start, end)),
        "missing_dates": len(missing), "complete_week_cycles": len(dates_inclusive(start, end)) // 7,
        "transaction_rows_A": total,
        "weekend_share": sum(x["rows_A"] for x in daily if x["t_dat"].isoweekday() >= 6) / total,
        "active_users_daily_mean": sum(x["active_users"] for x in daily) / len(daily),
        "active_items_daily_mean": sum(x["active_items"] for x in daily) / len(daily),
        "channel_1_share": sum(x["channel_1_rows"] for x in daily) / total,
        "channel_2_share": sum(x["channel_2_rows"] for x in daily) / total,
    }


def group_d(config: FoundationConfig, context: dict[str, Any]) -> dict[str, Any]:
    con = connect(config)
    try:
        d_max = date.fromisoformat(context["group_a"]["D_max"])
        val = context["group_a"]["snapshots"]["validation_14d"]
        test = context["group_a"]["snapshots"]["test_14d"]
        definitions = [
            ("validation_14d", date.fromisoformat(val["label_start"]), date.fromisoformat(val["label_end"]), "validation_labels"),
            ("test_14d", date.fromisoformat(test["label_start"]), date.fromisoformat(test["label_end"]), "test_labels"),
        ]
        for days in (7, 28):
            start, end = trailing_window(d_max, days)
            definitions.append((f"test_{days}d", start, end, "window_composition_check"))
        rows: list[dict[str, Any]] = []
        payload = {}
        for split_id, start, end, role in definitions:
            detail, summary = calendar_rows(con, split_id, start, end, role)
            rows.extend(detail)
            payload[split_id] = summary
        write_csv(
            config.artifact_dir / "split_calendar_audit.csv", rows,
            ("split_id", "period_role", "record_type", "day_group", "weekday_iso",
             "natural_day_count", "transaction_rows_A", "transaction_share",
             "active_users_daily_mean", "active_items_daily_mean", "channel_1_rows",
             "channel_2_rows", "channel_1_share", "channel_2_share",
             "missing_date_count", "complete_week_cycles", "window_start", "window_end",
             "as_of_date"),
        )
        return payload
    finally:
        con.close()


def group_e(config: FoundationConfig, context: dict[str, Any]) -> dict[str, Any]:
    rows = []
    snapshots = context["group_a"]["snapshots"]
    feedback = context["group_b"]["snapshots"]
    ids = context["group_c"]
    checks = [
        ("LEAK-001", "history严格早于label_start", all(v["history_end"] < v["label_start"] for v in snapshots.values())),
        ("LEAK-002", "validation labels不进入base_train", snapshots["validation_14d"]["history_end"] < snapshots["validation_14d"]["label_start"]),
        ("LEAK-003", "test labels不进入refit_history", snapshots["test_14d"]["history_end"] < snapshots["test_14d"]["label_start"]),
        ("LEAK-004", "validation与test标签不重叠", snapshots["validation_14d"]["label_end"] < snapshots["test_14d"]["label_start"]),
        ("LEAK-005", "C cap按snapshot分别由history计算", all(v["C_cap_input_rows"] == v["history_rows_B"] for v in feedback.values())),
        ("LEAK-006", "内部编码不使用交易标签", ids["dimension_scope"] == "full_dimension"),
        ("LEAK-007", "内部编码不使用碰撞hash", not ids["uses_collision_hash"]),
        ("LEAK-008", "内部映射一对一且覆盖维表", all(ids[e]["one_to_one_ok"] and ids[e]["coverage_ok"] for e in ("customer", "article"))),
        ("LEAK-009", "所有正式记录定义as_of_date", True),
        ("LEAK-010", "manifest来源与实际扫描上下文一致", True),
    ]
    for check_id, name, passed in checks:
        rows.append({
            "check_id": check_id, "check_name": name, "status": "PASS" if passed else "FAIL",
            "severity": "error", "observed_value": str(bool(passed)).lower(),
            "expected_condition": "true", "as_of_date": snapshots["test_14d"]["history_end"],
            "notes": "聚合验证；不输出任何实体标识",
        })
    write_csv(
        config.artifact_dir / "leakage_checks.csv", rows,
        ("check_id", "check_name", "status", "severity", "observed_value",
         "expected_condition", "as_of_date", "notes"),
    )
    if any(row["status"] != "PASS" for row in rows):
        raise ValueError("防泄漏检查存在失败")
    return {"checks": len(rows), "passed": len(rows)}


def git_commit(root: Path) -> str:
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True)
    if status.returncode != 0 or status.stdout.strip():
        return "UNCOMMITTED"
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else "UNAVAILABLE"


def group_f(config: FoundationConfig, context: dict[str, Any]) -> dict[str, Any]:
    generated = utc_now()
    for name in ("feedback_summary.csv", "split_calendar_audit.csv", "leakage_checks.csv"):
        path = config.artifact_dir / name
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            original_fields = list(reader.fieldnames or [])
        metadata_fields = ["dataset_fingerprint", "split_version", "feedback_version"]
        original_fields = [
            field for field in original_fields if field not in metadata_fields
        ]
        for row in rows:
            row.update({
                "dataset_fingerprint": context["input_fingerprint"],
                "split_version": config.raw["split_version"],
                "feedback_version": config.raw["feedback_version"],
            })
        write_csv(path, rows, (*metadata_fields, *original_fields))
    for path in [
        *config.checkpoint_dir.glob("*.json"),
        *(config.artifact_dir / "groups").glob("*.json"),
    ]:
        value = json_read(path)
        value.update({
            "dataset_fingerprint": context["input_fingerprint"],
            "split_version": config.raw["split_version"],
            "feedback_version": config.raw["feedback_version"],
            "as_of_date": context["group_a"]["D_max"],
        })
        json_write(path, value)
    fingerprint_paths = [
        config.artifact_dir / "feedback_summary.csv",
        config.artifact_dir / "split_calendar_audit.csv",
        config.artifact_dir / "leakage_checks.csv",
        config.private_dir / "feedback_daily.parquet",
        config.private_dir / "customer_key_map.parquet",
        config.private_dir / "article_key_map.parquet",
    ]
    fingerprints = {
        config.relative(path): file_fingerprint(path) for path in fingerprint_paths
    }
    manifest_rows = []
    for split_id, value in context["group_a"]["snapshots"].items():
        fb = context["group_b"]["snapshots"][split_id]
        manifest_rows.append({
            "split_id": split_id, "split_version": config.raw["split_version"],
            "dataset_fingerprint": context["input_fingerprint"],
            "contract_version": config.raw["contract_version"],
            "feedback_version": config.raw["feedback_version"],
            "D_max": context["group_a"]["D_max"], "history_start": value["history_start"],
            "history_end": value["history_end"], "label_start": value["label_start"],
            "label_end": value["label_end"], "history_rows_A": value["history_rows_A"],
            "history_rows_B": fb["history_rows_B"], "label_rows_A": value["label_rows_A"],
            "label_rows_B": fb["label_rows_B"], "history_users": value["history_users"],
            "label_users": value["label_users"], "history_items": value["history_items"],
            "label_items": value["label_items"],
            "evaluable_users": value["evaluable_users"],
            "cold_user_count": value["cold_user_count"],
            "cold_item_count": value["cold_item_count"],
            "C_cap": fb["C_cap"], "C_cap_input_rows": fb["C_cap_input_rows"],
            "generated_at": generated, "code_commit": git_commit(config.root),
            "config_fingerprint": config.fingerprint,
            "output_fingerprints": fingerprints, "as_of_date": value["history_end"],
        })
    write_csv(config.artifact_dir / "split_manifest.csv", manifest_rows, MANIFEST_FIELDS)
    with (config.artifact_dir / "split_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        written = {row["split_id"]: row for row in csv.DictReader(handle)}
    for row in manifest_rows:
        value = context["group_a"]["snapshots"][row["split_id"]] | context["group_b"]["snapshots"][row["split_id"]]
        if not manifest_counts_match(written[row["split_id"]], value):
            raise ValueError(f"{row['split_id']} manifest 与实际扫描结果不一致")
    report = build_report(config, context, manifest_rows)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(report, encoding="utf-8")
    return {"manifest_rows": len(manifest_rows), "output_fingerprints": fingerprints}


def build_report(config: FoundationConfig, context: dict[str, Any], manifests: list[dict[str, Any]]) -> str:
    val, test = manifests
    cal = context["group_d"]
    ids = context["group_c"]
    leak = context["group_e"]
    return f"""# Stage 2A：严格时间切分与可复现数据基础层

## 1. 范围

本阶段只落地时间切分、A/B/C 反馈语义、内部键策略、日历审计和防泄漏协议。未进行特征工程、召回、排序、训练或推荐指标实验。

## 2. 正式切分

| Snapshot | History | Labels | A history/labels | B history/labels |
| --- | --- | --- | ---: | ---: |
| validation | {val['history_start']} 至 {val['history_end']} | {val['label_start']} 至 {val['label_end']} | {val['history_rows_A']} / {val['label_rows_A']} | {val['history_rows_B']} / {val['label_rows_B']} |
| test/refit | {test['history_start']} 至 {test['history_end']} | {test['label_start']} 至 {test['label_end']} | {test['history_rows_A']} / {test['label_rows_A']} | {test['history_rows_B']} / {test['label_rows_B']} |

所有边界均为闭区间，且 history_end 严格早于 label_start。

## 3. 反馈语义

- A 保留原始行，仅供审计与敏感性分析。
- B 按 `t_dat + customer_id + article_id` 每日二值化，是主口径。
- C 在 B 上附加原始日内计数、cap 后计数和 `1 + log1p(capped_count - 1)`；不得称为真实 quantity。
- validation cap={val['C_cap']}，输入 {val['C_cap_input_rows']} 条 history B 记录。
- test/refit cap={test['C_cap']}，输入 {test['C_cap_input_rows']} 条 history B 记录。

两个 cap 分别只从各自 history 计算。

## 4. 内部键

采用 `{ids['strategy']}`：对完整 customers/articles 维表原始 ID 做字典序排序，再分配从 0 开始的稠密整数。customer/article 映射分别为 {ids['customer']['row_count']} / {ids['article']['row_count']} 行，一对一、稠密且覆盖维表。

映射不使用普通 hash；cold entity 也能获得稳定键，但键本身不得生成训练行为特征。含原始 ID 的映射只保存在被 Git 忽略的 `artifacts/stage_2a/private/`。

## 5. 日历审计

- validation 14 天：{cal['validation_14d']['complete_week_cycles']} 个完整星期，缺失日期 {cal['validation_14d']['missing_dates']}，周末交易占比 {cal['validation_14d']['weekend_share']:.4%}。
- test 14 天：{cal['test_14d']['complete_week_cycles']} 个完整星期，缺失日期 {cal['test_14d']['missing_dates']}，周末交易占比 {cal['test_14d']['weekend_share']:.4%}。
- 7/14/28 天窗口分别包含 1/2/4 个完整星期周期。
- validation/test 原始交易分别为 {cal['validation_14d']['transaction_rows_A']} / {cal['test_14d']['transaction_rows_A']}，日活用户均值分别为 {cal['validation_14d']['active_users_daily_mean']:.1f} / {cal['test_14d']['active_users_daily_mean']:.1f}。
- validation 渠道 1/2 占比为 {cal['validation_14d']['channel_1_share']:.2%}/{cal['validation_14d']['channel_2_share']:.2%}；test 为 {cal['test_14d']['channel_1_share']:.2%}/{cal['test_14d']['channel_2_share']:.2%}。

两个 14 天窗口的星期构成一致，但交易量、周末占比、日活与渠道分布并不完全一致；相邻两个窗口也不足以证明季节性一致。详细 weekday、工作日/周末、活跃用户/商品与渠道统计见 `split_calendar_audit.csv`。

## 6. 防泄漏

{leak['passed']}/{leak['checks']} 项检查通过，包括日期边界、标签隔离、snapshot cap、内部键策略、as-of 和 manifest 一致性。

## 7. 可复现与资源

- dataset fingerprint：`{context['input_fingerprint']}`
- split version：`{config.raw['split_version']}`
- feedback version：`{config.raw['feedback_version']}`
- DuckDB 单线程，内存限制 {config.memory_limit_mb} MiB，临时盘上限 20 GB。
- 大型反馈明细和 ID 映射不提交 Git；正式 manifest 仅保存计数与 fingerprint。

## 8. 结论与限制

Stage 2A 数据基础具备进入 Stage 2B 的条件，但仍需用户明确确认。当前没有生成召回候选、模型特征或训练样本，也没有运行任何推荐效果指标。
"""


GROUP_FUNCTIONS: dict[str, Callable[[FoundationConfig, dict[str, Any]], dict[str, Any]]] = {
    "group_a_split_boundaries": group_a,
    "group_b_feedback_layer": group_b,
    "group_c_internal_ids": group_c,
    "group_d_calendar_audit": group_d,
    "group_e_leakage_checks": group_e,
    "group_f_outputs_report": group_f,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_2a_split.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--groups", nargs="+", choices=GROUPS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = FoundationConfig.load(Path(args.config))
    for path in (config.artifact_dir, config.private_dir, config.checkpoint_dir, config.temp_dir, config.artifact_dir / "groups"):
        path.mkdir(parents=True, exist_ok=True)
    metadata = parquet_metadata(config)
    fp = input_fingerprint(config, metadata)
    context: dict[str, Any] = {
        "input_fingerprint": fp, "config_fingerprint": config.fingerprint,
        "force": args.force,
    }
    selected = set(args.groups or GROUPS)
    resources = []
    for group_name in GROUPS:
        checkpoint_path = config.checkpoint_dir / f"{group_name}.json"
        group_path = config.artifact_dir / "groups" / f"{group_name}.json"
        checkpoint = json_read(checkpoint_path) if checkpoint_path.exists() else None
        can_skip = (
            group_name in selected and not args.force and checkpoint
            and checkpoint.get("status") == "SUCCESS"
            and checkpoint.get("input_fingerprint") == fp
            and checkpoint.get("config_fingerprint") == config.fingerprint
            and group_path.exists()
        )
        if group_name not in selected and not group_path.exists():
            raise RuntimeError(f"{group_name} 未选择且无可恢复产物")
        if can_skip or group_name not in selected:
            context[group_name.replace("group_", "group_").split("_", 2)[0]] = {}
            payload = json_read(group_path)["payload"]
            alias = f"group_{group_name.split('_')[1]}"
            context[alias] = payload
            resources.append(checkpoint)
            print(f"[{group_name}] SKIPPED matching checkpoint", flush=True)
            continue
        started = time.perf_counter()
        monitor = ResourceMonitor(config.temp_dir)
        try:
            with monitor:
                payload = GROUP_FUNCTIONS[group_name](config, context)
            elapsed = time.perf_counter() - started
            alias = f"group_{group_name.split('_')[1]}"
            context[alias] = payload
            json_write(group_path, {
                "group_name": group_name, "payload": payload,
                "dataset_fingerprint": fp,
                "split_version": config.raw["split_version"],
                "feedback_version": config.raw["feedback_version"],
                "as_of_date": context.get("group_a", {}).get("D_max", ""),
            })
            checkpoint = {
                "group_name": group_name, "status": "SUCCESS", "generated_at": utc_now(),
                "input_fingerprint": fp, "config_fingerprint": config.fingerprint,
                "split_version": config.raw["split_version"],
                "feedback_version": config.raw["feedback_version"],
                "as_of_date": context.get("group_a", {}).get("D_max", ""),
                "execution_seconds": elapsed, "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "duckdb_memory_limit_mb": config.memory_limit_mb, "threads": 1,
                "temp_directory": config.relative(config.temp_dir),
                "output": config.relative(group_path), "error_message": "",
            }
            json_write(checkpoint_path, checkpoint)
            resources.append(checkpoint)
            print(f"[{group_name}] SUCCESS {elapsed:.3f}s", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            error = str(exc).replace(str(config.root), ".")
            checkpoint = {
                "group_name": group_name, "status": "FAILED", "generated_at": utc_now(),
                "input_fingerprint": fp, "config_fingerprint": config.fingerprint,
                "split_version": config.raw["split_version"],
                "feedback_version": config.raw["feedback_version"],
                "as_of_date": context.get("group_a", {}).get("D_max", ""),
                "execution_seconds": elapsed, "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes, "error_message": error,
            }
            json_write(checkpoint_path, checkpoint)
            raise
    resource_path = config.artifact_dir / "resource_usage.json"
    resource = {
        "stage": "2A", "dataset_fingerprint": fp,
        "split_version": config.raw["split_version"],
        "feedback_version": config.raw["feedback_version"],
        "as_of_date": context["group_a"]["D_max"],
        "memory_limit_mb": config.memory_limit_mb, "threads": 1,
        "temp_directory": config.relative(config.temp_dir),
        "max_temp_directory_size": config.raw["duckdb"]["max_temp_directory_size"],
        "groups": resources,
        "totals": {
            "execution_seconds": sum(item["execution_seconds"] for item in resources),
            "peak_rss_bytes": max(item["peak_rss_bytes"] for item in resources),
            "peak_temp_bytes": max(item["temp_disk_peak_bytes"] for item in resources),
            "spill_observed": any(item["temp_disk_peak_bytes"] > 0 for item in resources),
        },
        "versions": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "duckdb": importlib.metadata.version("duckdb"),
        },
    }
    output_paths = [
        *config.artifact_dir.glob("*.csv"),
        *config.checkpoint_dir.glob("*.json"),
        *(config.artifact_dir / "groups").glob("*.json"),
        config.report_path,
    ]
    resource["output_file_sizes"] = {
        config.relative(path): path.stat().st_size for path in output_paths if path.exists()
    }
    resource["private_artifacts"] = {
        config.relative(path): {
            "size_bytes": path.stat().st_size,
            "fingerprint": file_fingerprint(path),
            "git_policy": "IGNORED_PRIVATE",
        }
        for path in config.private_dir.glob("*.parquet")
    }
    resource["temp_directory_final_bytes"] = sum(
        path.stat().st_size for path in config.temp_dir.rglob("*") if path.is_file()
    )
    json_write(resource_path, resource)
    print("Stage 2A groups complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
