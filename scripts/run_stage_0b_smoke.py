from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

from data_audit.checks import SmokeChecks
from data_audit.config import AuditSettings, find_project_root
from data_audit.contracts import load_contracts
from data_audit.engine import DuckDBAuditEngine
from data_audit.reporting import AuditResult, write_json, write_results_csv


class ResourceMonitor:
    """后台采样当前进程 RSS 与 DuckDB 临时目录大小。"""

    def __init__(self, temp_dir: Path) -> None:
        self.process = psutil.Process()
        self.temp_dir = temp_dir
        self.peak_rss_bytes = self.process.memory_info().rss
        self.peak_temp_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _temp_bytes(self) -> int:
        return sum(
            path.stat().st_size for path in self.temp_dir.rglob("*") if path.is_file()
        )

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            self.peak_rss_bytes = max(
                self.peak_rss_bytes, self.process.memory_info().rss
            )
            self.peak_temp_bytes = max(self.peak_temp_bytes, self._temp_bytes())

    def __enter__(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak_rss_bytes = max(
            self.peak_rss_bytes, self.process.memory_info().rss
        )
        self.peak_temp_bytes = max(self.peak_temp_bytes, self._temp_bytes())


def public_error(value: str, root: Path) -> str:
    return value.replace(str(root.resolve()), ".")


def output_sizes(paths: list[Path], root: Path) -> dict[str, int]:
    return {
        path.resolve().relative_to(root.resolve()).as_posix(): path.stat().st_size
        for path in paths
        if path.exists()
    }


def report_markdown(
    results: list[AuditResult],
    summary: dict[str, Any],
    resources: dict[str, Any],
) -> str:
    rows = []
    for result in results:
        observed = result.as_record()["observed_value"]
        rows.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|").replace("\n", " ")
                for value in [
                    result.check_id,
                    result.status,
                    result.severity,
                    result.rows_scanned,
                    f"{result.execution_seconds:.6f}",
                    observed,
                ]
            )
            + " |"
        )
    result_table = "\n".join(
        [
            "| Check | Status | Severity | Rows scanned | Seconds | Observed |",
            "| --- | --- | --- | ---: | ---: | --- |",
            *rows,
        ]
    )
    statuses = summary["status_counts"]
    scope = summary["scan_scope"]
    recommendation = summary["stage_1_recommendation"]
    return f"""# Stage 0B：数据契约、审计框架与受限 Smoke Test

## 1. Stage 0B 目标

本阶段仅验证数据契约、DuckDB/PyArrow 审计框架、统一结果协议和资源边界；未执行正式全量 transaction 审计、清洗、特征工程、召回、排序或模型开发。

## 2. 数据契约摘要

- 契约版本：`{summary["contract_version"]}`
- 逻辑表：articles、customers、transactions
- 每条规则均包含 check ID、表、字段、规则类型、严重程度、预期和理由。
- sales channel 合法集合不硬编码，只记录 smoke 范围实际观察值。
- postal code、customer ID、article ID 只输出计数或格式，不输出真实值。

证据：`configs/data_contracts.yaml`。

## 3. Smoke 扫描范围

- articles：{scope["articles"]}
- customers：{scope["customers"]}
- transactions：{scope["transactions"]}
- DuckDB 另对三个 Parquet 各执行一次直接 `LIMIT 1` 可读性探测；该探测不被误作聚合范围。

transactions 的所有聚合均在首个 row group 前 `{scope["transaction_rows_materialized"]}` 行已物化临时表上执行，`LIMIT` 没有放在全表聚合之后。

## 4. 每条检查结果

{result_table}

状态汇总：PASS={statuses.get("PASS", 0)}，FAIL={statuses.get("FAIL", 0)}，WARN={statuses.get("WARN", 0)}，ERROR={statuses.get("ERROR", 0)}，NOT_RUN={statuses.get("NOT_RUN", 0)}。

## 5. 资源使用

- 总耗时：`{resources["total_execution_seconds"]}` 秒
- 进程峰值 RSS：`{resources["peak_rss_bytes"]}` bytes
- DuckDB 内存限制：`{resources["duckdb_memory_limit_mb"]}` MiB
- DuckDB threads：`{resources["duckdb_threads"]}`
- 临时目录：`{resources["temp_directory"]}`
- 临时磁盘峰值：`{resources["peak_temp_bytes"]}` bytes
- 观察到 spill：`{resources["spill_observed"]}`
- OOM：`{resources["oom_observed"]}`
- 输出文件大小：`{json.dumps(resources["output_file_sizes"], ensure_ascii=False)}`

## 6. 测试结果

- 命令：`{summary["tests"]["command"]}`
- 结果：`{"PASS" if summary["tests"]["passed"] else "FAIL"}`
- 返回码：`{summary["tests"]["return_code"]}`
- 耗时：`{summary["tests"]["execution_seconds"]}` 秒
- pytest 摘要：`{summary["tests"]["summary"]}`

测试不依赖完整 H&M 原始数据；临时 Parquet 由 pytest 临时目录创建并清理。

## 7. 失败与异常

{summary["failures_and_errors"]}

## 8. 当前框架局限

- transactions 只覆盖一个 row group 的固定前缀，无法代表完整时间跨度、全量唯一值、全量重复或全量外键覆盖。
- articles/customers 的值检查覆盖当前完整维表，但没有与外部权威校验和比较。
- DuckDB spill 结论只针对本次 smoke 查询和当时资源状态。
- schema fingerprint 稳定只表示相对 Stage 0A 未变化，不证明数据内容完整。

## 9. 局部结论边界

任何 smoke PASS 仅说明对应 SQL/契约在报告所列扫描范围内通过，**不得解释为全量 PASS 或数据百分之百完整**。交易检查尤其不能外推到其余 row groups。

## 10. 是否建议进入 Stage 1

结论：**{recommendation["decision"]}**。

理由：{recommendation["reason"]}

## 11. Stage 1 风险与建议顺序

1. 先冻结契约与结果协议，并为全量查询设置磁盘、内存和超时护栏。
2. 先做低基数空值、日期范围和值域，再做 customer/article 高基数 distinct。
3. 随后执行主键唯一性和 transaction 完全重复检查，监控 spill 与临时磁盘。
4. 最后执行全量外键覆盖率；该步骤可能同时构建大哈希表并扫描交易列。
5. 每一步单独落盘小型指标并可恢复，禁止物化或提交原始记录。
"""


def main() -> int:
    root = find_project_root()
    settings = AuditSettings.default(root)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    contracts = load_contracts(settings.contracts_path)
    started = time.perf_counter()
    test_started = time.perf_counter()
    test_process = subprocess.run(
        [sys.executable, "-m", "pytest"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    test_output_lines = [
        line.strip()
        for line in (test_process.stdout + "\n" + test_process.stderr).splitlines()
        if line.strip()
    ]
    test_result = {
        "command": "python -m pytest",
        "passed": test_process.returncode == 0,
        "return_code": test_process.returncode,
        "execution_seconds": round(time.perf_counter() - test_started, 6),
        "summary": public_error(
            test_output_lines[-1] if test_output_lines else "no pytest output",
            root,
        ),
    }
    results: list[AuditResult] = []
    engine_error = ""
    temp_bytes_after = 0

    with ResourceMonitor(settings.temp_dir) as monitor:
        try:
            with DuckDBAuditEngine(settings, contracts) as engine:
                engine.prepare()
                results = SmokeChecks(engine, contracts, root).run_all()
                temp_bytes_after = engine.temp_usage_bytes()
                transaction_scope = dict(engine.transaction_scope)
                table_rows = dict(engine.table_rows)
                row_groups = dict(engine.row_groups)
        except Exception as exc:
            engine_error = public_error(f"{type(exc).__name__}: {exc}", root)
            transaction_scope = {}
            table_rows = {}
            row_groups = {}

    total_seconds = round(time.perf_counter() - started, 6)
    csv_path = settings.artifact_dir / "audit_results.csv"
    summary_path = settings.artifact_dir / "audit_summary.json"
    resource_path = settings.artifact_dir / "resource_usage.json"
    report_path = root / "docs" / "stages" / "stage_0b_contract_and_smoke.md"

    write_results_csv(results, csv_path)
    status_counts = Counter(result.status for result in results)
    failures = [
        result
        for result in results
        if result.status in {"FAIL", "WARN", "ERROR", "NOT_RUN"}
    ]
    has_error = (
        not test_result["passed"]
        or bool(engine_error)
        or any(result.status in {"FAIL", "ERROR"} for result in results)
    )
    decision = "有条件建议进入 Stage 1" if not has_error else "暂不建议进入 Stage 1"
    reason = (
        "三个 Parquet 的 DuckDB 受限读取、契约检查、统一协议和测试均可运行；"
        "进入 Stage 1 前仍须保留分步执行与资源护栏。"
        if not has_error
        else "存在引擎错误或 error 级检查未通过，应先修复并重跑 Stage 0B。"
    )
    scan_scope = {
        "articles": (
            f"完整文件 {contracts.tables['articles'].file}；"
            f"{table_rows.get('articles', 0)} 行；"
            f"{row_groups.get('articles', 0)}/{row_groups.get('articles', 0)} row groups"
        ),
        "customers": (
            f"完整文件 {contracts.tables['customers'].file}；"
            f"{table_rows.get('customers', 0)} 行；"
            f"{row_groups.get('customers', 0)}/{row_groups.get('customers', 0)} row groups"
        ),
        "transactions": (
            f"文件 {contracts.tables['transactions'].file} 的 row group "
            f"{transaction_scope.get('row_group_index', 0)} 前 "
            f"{transaction_scope.get('rows_materialized', 0)} 行；"
            f"该 row group 共 {transaction_scope.get('row_group_rows', 0)} 行；"
            f"文件共 {transaction_scope.get('file_rows', 0)} 行/"
            f"{transaction_scope.get('file_row_groups', 0)} row groups"
        ),
        "transaction_rows_materialized": transaction_scope.get("rows_materialized", 0),
    }
    summary = {
        "stage": "0B",
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "contract_version": contracts.version,
        "status_counts": dict(status_counts),
        "result_count": len(results),
        "scan_scope": scan_scope,
        "engine_error": engine_error,
        "tests": test_result,
        "failures": [
            {
                "check_id": result.check_id,
                "status": result.status,
                "notes": result.notes,
                "error_message": result.error_message,
            }
            for result in failures
        ],
        "stage_1_recommendation": {"decision": decision, "reason": reason},
    }
    failures_text = (
        "\n".join(
            f"- `{result.check_id}`：{result.status}；"
            f"{result.notes or result.error_message or '详见结果 CSV。'}"
            for result in failures
        )
        if failures
        else "未观察到 FAIL、WARN、ERROR 或 NOT_RUN。"
    )
    if engine_error:
        failures_text += f"\n- 引擎异常：`{engine_error}`"
    if not test_result["passed"]:
        failures_text += f"\n- 单元测试失败：`{test_result['summary']}`"
    summary["failures_and_errors"] = failures_text
    write_json(summary, summary_path)

    resources = {
        "total_execution_seconds": total_seconds,
        "peak_rss_bytes": monitor.peak_rss_bytes,
        "duckdb_memory_limit_mb": settings.memory_limit_mb,
        "duckdb_threads": settings.threads,
        "temp_directory": settings.relative(settings.temp_dir),
        "temp_bytes_after_queries": temp_bytes_after,
        "peak_temp_bytes": monitor.peak_temp_bytes,
        "spill_observed": monitor.peak_temp_bytes > 0,
        "oom_observed": "out of memory" in engine_error.lower(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("duckdb", "pyarrow", "psutil", "pytest")
        },
        "python": {
            "version": ".".join(map(str, sys.version_info[:3])),
            "virtual_environment_active": sys.prefix != sys.base_prefix,
        },
        "output_file_sizes": {},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(resources, resource_path)
    report_path.write_text(report_markdown(results, summary, resources), encoding="utf-8")
    output_paths = [csv_path, summary_path, resource_path, report_path]
    # resource_usage 自身包含文件大小；迭代到序列化后的实际大小稳定。
    for _ in range(10):
        measured = output_sizes(output_paths, root)
        if measured == resources["output_file_sizes"]:
            break
        resources["output_file_sizes"] = measured
        write_json(resources, resource_path)
        report_path.write_text(
            report_markdown(results, summary, resources),
            encoding="utf-8",
        )

    print(f"Results: {settings.relative(csv_path)} ({len(results)} checks)")
    print(f"Summary: {settings.relative(summary_path)}")
    print(f"Resources: {settings.relative(resource_path)}")
    print(f"Report: {settings.relative(report_path)}")
    print(f"Stage 1 recommendation: {decision}")
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
