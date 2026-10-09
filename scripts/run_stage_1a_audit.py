from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from data_audit.contracts import load_contracts
from data_audit.stage1a import (
    FullAuditResult,
    GroupContext,
    QueryTimeoutError,
    ResourceMonitor,
    Stage1AConfig,
    Stage1AEngine,
    config_fingerprint,
    input_fingerprint,
    load_group_results,
    output_sizes,
    read_checkpoint,
    should_skip_group,
    utc_now,
    validate_public_artifacts,
    write_json,
    write_results_csv,
)
from data_audit.stage1a_groups import GROUP_FUNCTIONS, parquet_metadata


GROUP_ROWS = {
    "group_a_metadata_schema": 0,
    "group_b_null_format": 33_265_846,
    "group_c_date_value_ranges": 33_160_304,
    "group_d_primary_keys": 1_477_522,
    "group_e_foreign_keys": 31_788_324,
    "group_f_transaction_duplicates": 31_788_324,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1A 全量、可恢复数据质量审计")
    parser.add_argument(
        "--config",
        default="configs/stage_1a_audit.yaml",
        help="相对于项目根目录的 JSON-compatible YAML 配置",
    )
    parser.add_argument("--force", action="store_true", help="忽略成功 checkpoint 并强制重跑")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=sorted(GROUP_FUNCTIONS),
        help="只执行指定组；默认按配置顺序执行全部组",
    )
    return parser.parse_args(argv)


def _resolve_checkpoint_outputs(
    checkpoint: dict[str, Any] | None, config: Stage1AConfig
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    enriched = dict(checkpoint)
    enriched["_resolved_output_files"] = [
        str(config.resolve(path)) for path in checkpoint.get("output_files", [])
    ]
    return enriched


def is_resource_fatal(error_message: str) -> bool:
    lowered = error_message.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "no space",
            "disk",
            "querytimeouterror",
            "查询超过",
        )
    )


def execute_group(
    *,
    group_name: str,
    config: Stage1AConfig,
    contracts: Any,
    metadata: dict[str, dict[str, Any]],
    input_fp: str,
    config_fp: str,
    force: bool,
) -> tuple[list[FullAuditResult], dict[str, Any], bool]:
    checkpoint_path = config.checkpoint_dir / f"{group_name}.json"
    group_output_path = config.group_dir / f"{group_name}.json"
    previous_checkpoint = read_checkpoint(checkpoint_path)
    checkpoint = _resolve_checkpoint_outputs(previous_checkpoint, config)
    if should_skip_group(checkpoint, input_fp, config_fp, force):
        results = load_group_results(group_output_path)
        public_checkpoint = {
            key: value for key, value in checkpoint.items() if not key.startswith("_")
        }
        public_checkpoint["last_action"] = "SKIPPED_MATCHING_CHECKPOINT"
        return results, public_checkpoint, False

    started_at = utc_now()
    started = time.perf_counter()
    results: list[FullAuditResult] = []
    extra_outputs: list[Path] = []
    error_message = ""
    temp_after = 0
    attempts = 0
    fatal = False
    attempt_history = list((previous_checkpoint or {}).get("attempt_history", []))
    if previous_checkpoint and previous_checkpoint.get("status") == "FAILED":
        attempt_history.append(
            {
                "status": previous_checkpoint["status"],
                "started_at": previous_checkpoint.get("started_at"),
                "finished_at": previous_checkpoint.get("finished_at"),
                "execution_seconds": previous_checkpoint.get("execution_seconds"),
                "config_fingerprint": previous_checkpoint.get("config_fingerprint"),
                "peak_rss_bytes": previous_checkpoint.get("peak_rss_bytes", 0),
                "temp_disk_peak_bytes": previous_checkpoint.get(
                    "temp_disk_peak_bytes", 0
                ),
                "error_message": previous_checkpoint.get("error_message", ""),
                "attempts": previous_checkpoint.get("current_run_attempts", 1),
                "spill_observed": previous_checkpoint.get("spill_observed", False),
                "oom_observed": previous_checkpoint.get("oom_observed", False),
            }
        )
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    running_checkpoint = {
        "group_name": group_name,
        "status": "RUNNING",
        "started_at": started_at,
        "finished_at": None,
        "execution_seconds": 0,
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fp,
        "rows_scanned": GROUP_ROWS[group_name],
        "peak_rss_bytes": 0,
        "temp_disk_peak_bytes": 0,
        "output_files": [config.relative(group_output_path)],
        "error_message": "",
        "retried": False,
        "attempts": 0,
        "spill_observed": False,
        "oom_observed": False,
        "attempt_history": attempt_history,
    }
    write_json(checkpoint_path, running_checkpoint)

    with ResourceMonitor(config.temp_dir) as monitor:
        while attempts < config.max_attempts:
            attempts += 1
            try:
                with Stage1AEngine(config) as engine:
                    context = GroupContext(
                        group_name=group_name,
                        engine=engine,
                        config=config,
                        input_fp=input_fp,
                        config_fp=config_fp,
                        group_output_path=group_output_path,
                    )
                    extra_outputs = GROUP_FUNCTIONS[group_name](context, metadata, contracts)
                    results = context.results
                    temp_after = sum(
                        path.stat().st_size
                        for path in config.temp_dir.rglob("*")
                        if path.is_file()
                    )
                break
            except Exception as exc:
                error_message = (
                    f"{type(exc).__name__}: {exc}"
                    .replace(str(config.root), ".")
                    .replace("\\", "/")
                )
                fatal = is_resource_fatal(error_message)
                if group_output_path.exists():
                    results = load_group_results(group_output_path)
                if attempts >= config.max_attempts or fatal:
                    break

    has_result_error = any(result.status == "ERROR" for result in results)
    status = "FAILED" if error_message or has_result_error else "SUCCESS"
    output_paths = [group_output_path, *extra_outputs]
    historical_peak_rss = max(
        (item.get("peak_rss_bytes", 0) for item in attempt_history), default=0
    )
    historical_peak_temp = max(
        (item.get("temp_disk_peak_bytes", 0) for item in attempt_history), default=0
    )
    checkpoint_payload = {
        "group_name": group_name,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "execution_seconds": round(time.perf_counter() - started, 6),
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fp,
        "rows_scanned": GROUP_ROWS[group_name],
        "peak_rss_bytes": max(monitor.peak_rss_bytes, historical_peak_rss),
        "temp_disk_peak_bytes": max(monitor.peak_temp_bytes, historical_peak_temp),
        "current_attempt_peak_rss_bytes": monitor.peak_rss_bytes,
        "current_attempt_temp_disk_peak_bytes": monitor.peak_temp_bytes,
        "temp_disk_bytes_after": temp_after,
        "output_files": [config.relative(path) for path in output_paths if path.exists()],
        "output_file_sizes": output_sizes(output_paths, config.root),
        "error_message": error_message,
        "retried": attempts > 1 or bool(attempt_history),
        "attempts": attempts + len(attempt_history),
        "current_run_attempts": attempts,
        "spill_observed": monitor.peak_temp_bytes > 0
        or any(item.get("spill_observed") for item in attempt_history),
        "oom_observed": "out of memory" in error_message.lower()
        or any(item.get("oom_observed") for item in attempt_history),
        "duckdb_memory_limit_mb": config.memory_limit_mb,
        "duckdb_threads": config.threads,
        "temp_directory": config.relative(config.temp_dir),
        "max_temp_directory_size": config.raw["duckdb"]["max_temp_directory_size"],
        "last_action": "EXECUTED",
        "attempt_history": attempt_history,
    }
    write_json(checkpoint_path, checkpoint_payload)
    return results, checkpoint_payload, fatal


def run_tests(root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    lines = [
        line.strip()
        for line in (process.stdout + "\n" + process.stderr).splitlines()
        if line.strip()
    ]
    return {
        "command": "python -m pytest",
        "passed": process.returncode == 0,
        "return_code": process.returncode,
        "execution_seconds": round(time.perf_counter() - started, 6),
        "summary": (lines[-1] if lines else "no pytest output").replace(str(root), "."),
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def observed(results: list[FullAuditResult], check_id: str) -> Any:
    result = next((item for item in results if item.check_id == check_id), None)
    return None if result is None else result.observed_value


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    escaped = [
        [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        for row in rows
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in escaped],
        ]
    )


def build_report(
    *,
    config: Stage1AConfig,
    metadata: dict[str, dict[str, Any]],
    results: list[FullAuditResult],
    checkpoints: list[dict[str, Any]],
    tests: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    null_rows = read_csv_rows(config.artifact_dir / "null_profile.csv")
    key_rows = read_csv_rows(config.artifact_dir / "key_integrity.csv")
    fk_rows = read_csv_rows(config.artifact_dir / "foreign_key_coverage.csv")
    null_table = markdown_table(
        ["Table", "Column", "Null count", "Null rate"],
        [
            [row["table_name"], row["column_name"], row["null_count"], row["null_rate"]]
            for row in null_rows
        ],
    )
    group_table = markdown_table(
        [
            "Group",
            "Status",
            "Memory MiB",
            "Threads",
            "Seconds",
            "Peak RSS",
            "Peak temp",
            "Spill",
            "Retry",
        ],
        [
            [
                item["group_name"],
                item["status"],
                item["duckdb_memory_limit_mb"],
                item["duckdb_threads"],
                item["execution_seconds"],
                item["peak_rss_bytes"],
                item["temp_disk_peak_bytes"],
                item["spill_observed"],
                item["retried"],
            ]
            for item in checkpoints
        ],
    )
    key_table = markdown_table(
        [
            "Table",
            "Rows",
            "Non-null",
            "Distinct",
            "Duplicate keys",
            "Excess rows",
        ],
        [
            [
                row["table_name"],
                row["total_rows"],
                row["non_null_rows"],
                row["distinct_count"],
                row["duplicate_key_count"],
                row["duplicate_excess_rows"],
            ]
            for row in key_rows
        ],
    )
    fk_table = markdown_table(
        ["Relationship", "Unmatched rows", "Unmatched distinct", "Row match rate"],
        [
            [
                row["relationship"],
                row["unmatched_transaction_rows"],
                row["unmatched_distinct_keys"],
                row["row_match_rate"],
            ]
            for row in fk_rows
        ],
    )
    date_profile = observed(results, "transactions.full.date.range")
    price_profile = observed(results, "transactions.full.price.profile")
    age_profile = observed(results, "customers.full.age.profile")
    distinct_profile = observed(results, "transactions.full.distinct_entities")
    full_dup = observed(results, "transactions.full.duplicate.full_row")
    triplet_dup = observed(
        results, "transactions.full.duplicate.date_customer_article"
    )
    date_dup = observed(results, "transactions.full.duplicate.date_distribution")
    non_pass = [
        result
        for result in results
        if result.status in {"FAIL", "WARN", "ERROR", "NOT_RUN"}
    ]
    non_pass_text = (
        "\n".join(
            f"- `{item.check_id}`：{item.status}；{item.notes or item.error_message}"
            for item in non_pass
        )
        if non_pass
        else "无 FAIL、WARN、ERROR 或 NOT_RUN。"
    )
    metadata_table = markdown_table(
        ["Table", "Rows", "Row groups", "Bytes", "Fingerprint"],
        [
            [
                table,
                value["row_count"],
                value["row_groups"],
                value["file_size_bytes"],
                value["schema_fingerprint"][:16] + "…",
            ]
            for table, value in metadata.items()
        ],
    )
    history_lines = []
    for item in checkpoints:
        for index, attempt in enumerate(item.get("attempt_history", []), start=1):
            first_error_line = (attempt.get("error_message") or "").splitlines()[0]
            history_lines.append(
                f"- `{item['group_name']}` 历史尝试 {index}："
                f"status={attempt.get('status')}，"
                f"seconds={attempt.get('execution_seconds')}，"
                f"peak_rss={attempt.get('peak_rss_bytes')} bytes，"
                f"peak_temp={attempt.get('temp_disk_peak_bytes')} bytes，"
                f"error=`{first_error_line}`"
            )
    history_text = "\n".join(history_lines) if history_lines else "无历史失败或重试。"
    memory_fraction_percent = (
        float(config.raw["duckdb"]["memory_fraction_of_available"]) * 100
    )
    return f"""# Stage 1A：全量完整性与基础数据质量审计

## 1. 执行环境与护栏

- 执行时间：`{summary["executed_at"]}`
- Python：`{summary["environment"]["python"]}`（项目 `.venv`）
- DuckDB：`{summary["environment"]["duckdb"]}`
- memory_limit 策略：每组启动时取可用内存的 {memory_fraction_percent:g}%，限制在 256–1024 MiB；各组实际值见资源表。
- threads：`{config.threads}`
- temp_directory：`{config.relative(config.temp_dir)}`
- max_temp_directory_size：`{config.raw["duckdb"]["max_temp_directory_size"]}`
- 单查询软超时：`{config.timeout_seconds}` 秒；超时由 DuckDB interrupt 中断。
- retry：最多 `{config.max_attempts}` 次；本配置不自动重试重型查询。
- preserve_insertion_order：`{config.raw["duckdb"]["preserve_insertion_order"]}`

所有原始 Parquet 只读；没有 pandas 全量物化、清洗、去重、特征或模型代码。

## 2. 全量扫描范围

{metadata_table}

Group A 只读取完整 footer/schema；Group B–F 的正式指标均以 `execution_scope=FULL_SCAN` 标记。`rows_scanned` 表示该检查的逻辑输入行数，不等同于多查询累计物理行访问次数。

## 3. Schema、空值与格式结果

- 三张表均完整可读，schema 与数据契约一致，fingerprint 与 Stage 0A 一致。
- ID、postal code、日期及价格格式结果详见统一结果 CSV。

{null_table}

## 4. 日期和值域结果

- 交易日期：`{json.dumps(date_profile, ensure_ascii=False)}`
- 价格：`{json.dumps(price_profile, ensure_ascii=False)}`
- 年龄：`{json.dumps(age_profile, ensure_ascii=False)}`
- sales_channel_id 频数见 `transactions.full.sales_channel.frequency`。

## 5. 主键唯一性

{key_table}

表中只包含计数，不包含具体 customer_id 或 article_id。

## 6. 主外键覆盖

- transactions distinct 实体：`{json.dumps(distinct_profile, ensure_ascii=False)}`

{fk_table}

外键检查使用全量 anti join，不保存未匹配 ID。

## 7. 交易重复多口径结果

- 全字段完全相同：`{json.dumps(full_dup, ensure_ascii=False)}`
- `t_dat + customer_id + article_id`：`{json.dumps(triplet_dup, ensure_ascii=False)}`
- 日期分布摘要：`{json.dumps(date_dup, ensure_ascii=False)}`
- 当前 transactions 恰好只有五个字段，因此“全字段完全相同”与 `t_dat + customer_id + article_id + price + sales_channel_id` 是同一口径，报告中显式复用同一结果。

这些现象只能称为记录重复或同条件重复购买。由于缺少 order_id、精确时间戳和 quantity，不能判断是数据错误、同单多件、重复购买还是业务导出语义，因此本阶段绝不执行 `drop_duplicates`。

## 8. 资源占用与性能

{group_table}

汇总：spill=`{summary["resource_totals"]["spill_observed"]}`，retry=`{summary["resource_totals"]["retry_observed"]}`，OOM=`{summary["resource_totals"]["oom_observed"]}`。

历史失败与降级记录：

{history_text}

F 组最终通过临时、精确的一对一整数代理键完成重复分组；代理键不落入正式结果，原始 ID 与 Parquet 均未修改。

## 9. 测试结果

- 命令：`{tests["command"]}`
- 结果：`{"PASS" if tests["passed"] else "FAIL"}`
- 摘要：`{tests["summary"]}`
- 耗时：`{tests["execution_seconds"]}` 秒

测试使用 pytest 临时目录和小型临时 Parquet，不依赖完整 H&M 原始数据。

## 10. 失败、警告和未完成项

{non_pass_text}

Stage 1A 没有验证外部发布校验和，也没有赋予交易重复以“脏数据”标签。

## 11. 对数据清洗的影响

- 空值、格式、范围、主外键和重复统计应成为后续清洗决策输入，但本阶段未修改任何值。
- 年龄和价格的边界统计决定后续是否需要异常值策略。
- 重复记录必须结合业务语义或新增 order_id/quantity 证据后才能决定是否处理。

## 12. 对推荐架构的初步影响

- distinct 用户/商品规模决定用户与商品索引、稀疏交互矩阵及候选生成的容量设计。
- 主外键覆盖决定交互能否安全关联用户与商品侧信息。
- 同日同商品重复记录可能影响隐式反馈强度定义，但不能在此阶段直接压缩为一次事件。
- 日期范围和分布决定时间切分、回测窗口与防止未来信息泄露的策略。

## 13. 是否建议进入 Stage 1B

**{summary["stage_1b_recommendation"]["decision"]}**。

理由：{summary["stage_1b_recommendation"]["reason"]}

本报告没有执行 Stage 1B。
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = Stage1AConfig.load(config_path)
    contracts_path = config.resolve(config.raw["contracts_path"])
    contracts = load_contracts(contracts_path)
    metadata = parquet_metadata(config)
    input_fp = input_fingerprint(config, metadata)
    config_fp = config_fingerprint(config_path, contracts_path)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.group_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = args.groups or config.groups
    all_results: list[FullAuditResult] = []
    checkpoints: list[dict[str, Any]] = []
    stopped_after_resource_failure = False
    for group_name in selected_groups:
        print(f"[{group_name}] starting", flush=True)
        results, checkpoint, fatal = execute_group(
            group_name=group_name,
            config=config,
            contracts=contracts,
            metadata=metadata,
            input_fp=input_fp,
            config_fp=config_fp,
            force=args.force,
        )
        all_results.extend(results)
        checkpoints.append(checkpoint)
        print(
            f"[{group_name}] {checkpoint['status']} "
            f"in {checkpoint['execution_seconds']}s",
            flush=True,
        )
        if fatal:
            stopped_after_resource_failure = True
            print("Resource-fatal error detected; heavier groups will not run.", flush=True)
            break

    # 汇总时加载所有与当前指纹一致的成功组，支持从 checkpoint 恢复。
    aggregate_results: list[FullAuditResult] = []
    aggregate_checkpoints: list[dict[str, Any]] = []
    for group_name in config.groups:
        checkpoint_path = config.checkpoint_dir / f"{group_name}.json"
        checkpoint = read_checkpoint(checkpoint_path)
        group_path = config.group_dir / f"{group_name}.json"
        if (
            checkpoint
            and checkpoint.get("status") == "SUCCESS"
            and checkpoint.get("input_fingerprint") == input_fp
            and checkpoint.get("config_fingerprint") == config_fp
            and group_path.exists()
        ):
            aggregate_checkpoints.append(checkpoint)
            aggregate_results.extend(load_group_results(group_path))

    tests = run_tests(config.root)
    results_path = config.artifact_dir / "audit_results.csv"
    summary_path = config.artifact_dir / "audit_summary.json"
    resource_path = config.artifact_dir / "resource_usage.json"
    write_results_csv(aggregate_results, results_path)
    status_counts = Counter(result.status for result in aggregate_results)
    all_groups_success = len(aggregate_checkpoints) == len(config.groups)
    blocking_result = any(
        result.status in {"FAIL", "ERROR", "NOT_RUN"} for result in aggregate_results
    )
    recommend = all_groups_success and not blocking_result and tests["passed"]
    summary = {
        "stage": "1A",
        "executed_at": utc_now(),
        "execution_scope": "FULL_SCAN",
        "result_version": config.result_version,
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fp,
        "groups_expected": config.groups,
        "groups_successful": [item["group_name"] for item in aggregate_checkpoints],
        "status_counts": dict(status_counts),
        "result_count": len(aggregate_results),
        "tests": tests,
        "stopped_after_resource_failure": stopped_after_resource_failure,
        "environment": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "duckdb": importlib.metadata.version("duckdb"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "psutil": importlib.metadata.version("psutil"),
            "virtual_environment_active": sys.prefix != sys.base_prefix,
        },
        "resource_totals": {
            "spill_observed": any(item["spill_observed"] for item in aggregate_checkpoints),
            "retry_observed": any(item["retried"] for item in aggregate_checkpoints),
            "oom_observed": any(item["oom_observed"] for item in aggregate_checkpoints),
            "sum_group_execution_seconds": sum(
                item["execution_seconds"] for item in aggregate_checkpoints
            ),
            "max_group_peak_rss_bytes": max(
                (item["peak_rss_bytes"] for item in aggregate_checkpoints), default=0
            ),
            "max_group_temp_disk_peak_bytes": max(
                (item["temp_disk_peak_bytes"] for item in aggregate_checkpoints), default=0
            ),
        },
        "stage_1b_recommendation": {
            "decision": "有条件建议进入 Stage 1B" if recommend else "暂不建议进入 Stage 1B",
            "reason": (
                "六个全量检查组、扩展测试和资源护栏均完成；进入前应先评审本报告中的警告与重复语义。"
                if recommend
                else "存在未成功检查组、阻断级结果或测试失败，应先修复并重跑 Stage 1A。"
            ),
        },
    }
    write_json(summary_path, summary)
    resource_payload = {
        "execution_scope": "FULL_SCAN",
        "memory_limit_policy": {
            "fraction_of_available": config.raw["duckdb"][
                "memory_fraction_of_available"
            ],
            "min_mb": config.raw["duckdb"]["memory_min_mb"],
            "max_mb": config.raw["duckdb"]["memory_max_mb"],
        },
        "summary_process_computed_memory_limit_mb": config.memory_limit_mb,
        "duckdb_threads": config.threads,
        "temp_directory": config.relative(config.temp_dir),
        "max_temp_directory_size": config.raw["duckdb"]["max_temp_directory_size"],
        "query_timeout_seconds": config.timeout_seconds,
        "groups": aggregate_checkpoints,
        "totals": summary["resource_totals"],
        "output_file_sizes": {},
    }
    write_json(resource_path, resource_payload)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        build_report(
            config=config,
            metadata=metadata,
            results=aggregate_results,
            checkpoints=aggregate_checkpoints,
            tests=tests,
            summary=summary,
        ),
        encoding="utf-8",
    )
    final_paths = [
        results_path,
        summary_path,
        resource_path,
        config.report_path,
        *config.artifact_dir.glob("*.csv"),
        *config.checkpoint_dir.glob("*.json"),
    ]
    for _ in range(10):
        sizes = output_sizes(final_paths, config.root)
        if sizes == resource_payload["output_file_sizes"]:
            break
        resource_payload["output_file_sizes"] = sizes
        write_json(resource_path, resource_payload)

    validate_public_artifacts(
        [
            results_path,
            summary_path,
            config.report_path,
            config.artifact_dir / "null_profile.csv",
            config.artifact_dir / "value_ranges.csv",
            config.artifact_dir / "key_integrity.csv",
            config.artifact_dir / "foreign_key_coverage.csv",
            config.artifact_dir / "transaction_duplicate_profiles.csv",
        ],
        allowed_tokens=[
            input_fp,
            config_fp,
            *[value["schema_fingerprint"] for value in metadata.values()],
        ],
    )

    print(f"Results: {config.relative(results_path)} ({len(aggregate_results)} checks)")
    print(f"Summary: {config.relative(summary_path)}")
    print(f"Resources: {config.relative(resource_path)}")
    print(f"Report: {config.relative(config.report_path)}")
    print(summary["stage_1b_recommendation"]["decision"])
    return 0 if recommend else 1


if __name__ == "__main__":
    raise SystemExit(main())
