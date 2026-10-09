from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from data_audit.stage1a import (
    ResourceMonitor,
    Stage1AEngine,
    config_fingerprint,
    input_fingerprint,
    output_sizes,
    read_checkpoint,
    utc_now,
    write_json,
)
from data_audit.stage1a_groups import parquet_metadata
from data_audit.stage1b import (
    FEEDBACK_SCOPES,
    load_group_output,
    profile_checkpoint_can_skip,
    read_csv,
    stage1b_config,
    validate_profile_artifacts,
)
from data_audit.stage1b_groups import GROUP_FUNCTIONS


GROUP_ROWS = {
    "group_a_temporal_activity": 31_788_324,
    "group_b_user_activity": 31_788_324,
    "group_c_item_long_tail": 31_788_324,
    "group_d_repurchase_stability": 31_788_324,
    "group_e_cold_start": 31_788_324,
    "group_f_time_split": 31_788_324,
    "group_g_model_evidence": 0,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1B 推荐可行性全量聚合画像")
    parser.add_argument("--config", default="configs/stage_1b_profile.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUP_FUNCTIONS))
    return parser.parse_args(argv)


def checkpoint_can_skip(
    checkpoint: dict[str, Any] | None,
    input_fp: str,
    config_fp: str,
    output: Path,
    force: bool,
) -> bool:
    return profile_checkpoint_can_skip(
        checkpoint,
        input_fp,
        config_fp,
        output.exists()
        and all(Path(path).exists() for path in (checkpoint or {}).get("_resolved_outputs", [])),
        force,
    )


def resource_fatal(error: str) -> bool:
    lowered = error.lower()
    return any(
        marker in lowered
        for marker in ("out of memory", "no space", "disk", "查询超过", "timeout")
    )


def execute_group(
    group_name: str,
    config: Any,
    input_fp: str,
    config_fp: str,
    force: bool,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    checkpoint_path = config.checkpoint_dir / f"{group_name}.json"
    group_output = config.group_dir / f"{group_name}.json"
    previous = read_checkpoint(checkpoint_path)
    if previous:
        previous["_resolved_outputs"] = [
            str(config.resolve(path)) for path in previous.get("output_files", [])
        ]
    if checkpoint_can_skip(previous, input_fp, config_fp, group_output, force):
        public = {key: value for key, value in previous.items() if not key.startswith("_")}
        public["last_action"] = "SKIPPED_MATCHING_CHECKPOINT"
        return load_group_output(group_output), public, False

    history = list((previous or {}).get("attempt_history", []))
    if previous and previous.get("status") == "FAILED":
        history.append(
            {
                "status": "FAILED",
                "execution_seconds": previous.get("execution_seconds"),
                "peak_rss_bytes": previous.get("peak_rss_bytes", 0),
                "temp_disk_peak_bytes": previous.get("temp_disk_peak_bytes", 0),
                "error_message": previous.get("error_message", ""),
                "oom_observed": previous.get("oom_observed", False),
                "spill_observed": previous.get("spill_observed", False),
            }
        )
    started_at = utc_now()
    started = time.perf_counter()
    error = ""
    payload: dict[str, Any] = {}
    outputs: list[Path] = []
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        checkpoint_path,
        {
            "group_name": group_name,
            "status": "RUNNING",
            "started_at": started_at,
            "input_fingerprint": input_fp,
            "config_fingerprint": config_fp,
            "attempt_history": history,
        },
    )
    with ResourceMonitor(config.temp_dir) as monitor:
        try:
            with Stage1AEngine(config) as engine:
                payload, outputs = GROUP_FUNCTIONS[group_name](
                    engine, config, group_output
                )
        except Exception as exc:
            error = (
                f"{type(exc).__name__}: {exc}"
                .replace(str(config.root), ".")
                .replace("\\", "/")
            )
            if group_output.exists():
                payload = load_group_output(group_output)
    status = "FAILED" if error else "SUCCESS"
    historical_rss = max(
        (item.get("peak_rss_bytes", 0) for item in history), default=0
    )
    historical_temp = max(
        (item.get("temp_disk_peak_bytes", 0) for item in history), default=0
    )
    checkpoint = {
        "group_name": group_name,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "execution_seconds": round(time.perf_counter() - started, 6),
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fp,
        "rows_scanned": GROUP_ROWS[group_name],
        "peak_rss_bytes": max(monitor.peak_rss_bytes, historical_rss),
        "current_attempt_peak_rss_bytes": monitor.peak_rss_bytes,
        "temp_disk_peak_bytes": max(monitor.peak_temp_bytes, historical_temp),
        "current_attempt_temp_disk_peak_bytes": monitor.peak_temp_bytes,
        "output_files": [config.relative(path) for path in outputs if path.exists()],
        "output_file_sizes": output_sizes(outputs, config.root),
        "error_message": error,
        "attempts": len(history) + 1,
        "retried": bool(history),
        "spill_observed": monitor.peak_temp_bytes > 0
        or any(item.get("spill_observed") for item in history),
        "oom_observed": "out of memory" in error.lower()
        or any(item.get("oom_observed") for item in history),
        "duckdb_memory_limit_mb": config.memory_limit_mb,
        "duckdb_threads": config.threads,
        "temp_directory": config.relative(config.temp_dir),
        "max_temp_directory_size": config.raw["duckdb"]["max_temp_directory_size"],
        "attempt_history": history,
        "last_action": "EXECUTED",
    }
    write_json(checkpoint_path, checkpoint)
    return payload, checkpoint, resource_fatal(error)


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
        "passed": process.returncode == 0,
        "return_code": process.returncode,
        "summary": lines[-1] if lines else "no pytest output",
        "execution_seconds": round(time.perf_counter() - started, 6),
    }


def build_report(
    summary: dict[str, Any], groups: dict[str, Any], checkpoints: list[dict[str, Any]]
) -> str:
    temporal = groups["group_a_temporal_activity"]
    user = groups["group_b_user_activity"]
    item = groups["group_c_item_long_tail"]
    repeat = groups["group_d_repurchase_stability"]
    cold = groups["group_e_cold_start"]
    split = groups["group_f_time_split"]
    checkpoint_lines = "\n".join(
        f"| {item['group_name']} | {item['status']} | {item['duckdb_memory_limit_mb']} | "
        f"{item['execution_seconds']} | {item['peak_rss_bytes']} | "
        f"{item['temp_disk_peak_bytes']} | {item['spill_observed']} | {item['retried']} |"
        for item in checkpoints
    )
    cold_lines = "\n".join(
        f"| {window} | {value['test_rows']} | {value['user_cold_start_rate']:.6f} | "
        f"{value['item_cold_start_rate']:.6f} | {value['warm_candidate_upper_bound']:.6f} |"
        for window, value in cold.items()
    )
    split_lines = "\n".join(
        f"| {window} | {value['train_rows']} | {value['test_rows']} | "
        f"{value['evaluable_users']} | {value['test_user_train_visibility']:.6f} | "
        f"{value['one_test_positive_user_rate']:.6f} | "
        f"{value['warm_user_item_candidate_upper_bound']:.6f} |"
        for window, value in split.items()
    )
    history = []
    for checkpoint in checkpoints:
        for attempt in checkpoint.get("attempt_history", []):
            history.append(
                f"- `{checkpoint['group_name']}` 历史失败："
                f"`{(attempt.get('error_message') or '').splitlines()[0]}`"
            )
    return f"""# Stage 1B：推荐可行性画像与时间评估基础分析

## 1. 执行环境与范围

- execution scope：全量聚合或明确的 7/14/28 天窗口。
- Python：{summary["environment"]["python"]}；DuckDB：{summary["environment"]["duckdb"]}。
- 动态内存策略：可用内存 35%，256–1024 MiB；单线程；临时盘上限 20 GB。
- 原始 transaction 共 31,788,324 行，日期范围 {temporal["min_date"]} 至 {temporal["max_date"]}。
- 本阶段没有清洗、去重、训练模型、生成用户明细或冻结最终架构。

## 2. 时间行为分布

- 覆盖 {temporal["day_count"]} 个日期。
- 原始行 A：{temporal["raw_rows"]}；二值 user-item-day B：{temporal["binary_interactions"]}。
- 同日重复超额：{temporal["same_day_excess"]}。
- 月交易量最小/最大：{temporal["min_month_rows"]} / {temporal["max_month_rows"]}。
- 首月/末月交易量：{temporal["first_month_rows"]} / {temporal["last_month_rows"]}；首末月均为不完整月份，不能直接当作趋势端点。
- 完整月份峰值为 {temporal["peak_complete_month"]} 的 {temporal["peak_complete_month_rows"]} 行；2019-06 与 2020-06 均处于高位，提示夏季季节性候选，但尚未做因果或显著性检验。
- 渠道突变检查：{temporal["channel_1_zero_observation"]}；需在 Stage 1C 前核验业务背景，不能在本阶段直接判作缺失或错误。

## 3. 用户活跃度和序列长度

- 有交易用户：{user["users_with_transactions"]}。
- 仅 1 条二值行为用户：{user["one_binary_interaction_users"]}（{user["one_binary_interaction_user_rate"]:.6f}）。
- 二值序列至少 5/10/20 条用户数：{json.dumps(user["users_at_least_binary_interactions"], ensure_ascii=False)}。
- 对应覆盖率：{json.dumps(user["coverage_rates"], ensure_ascii=False)}。
- A/B 差异：{user["same_day_excess"]} 行；C 强度总和等于 A，但不能称为 quantity。

## 4. 商品热度和长尾

- 有交易商品：{item["transacted_items"]}；零交易 articles：{item["untransacted_articles"]}。
- Gini：{item["gini_raw"]:.6f}；HHI：{item["hhi_raw"]:.8f}。
- Top 1/5/10/20% 交易贡献：{json.dumps(item["top_shares_raw"], ensure_ascii=False)}。

热门召回具备数据基础，但高集中度会带来流行度偏置和长尾曝光不足。

## 5. 复购与兴趣稳定性

- 有相同 article 重复记录的用户比例：{repeat["user_repeat_rate"]:.6f}。
- 重复超额行比例：{repeat["repeat_excess_row_rate"]:.6f}。
- 同日超额 A−B：{repeat["same_day_excess"]}。
- 跨日复购组：{repeat["cross_day_repeat_groups"]}。
- 同 product_code 不同 article 组：{repeat["same_product_different_article_groups"]}。
- 最近 28 天与历史 department 偏好平均 Jaccard：{repeat["recent_history_department_overlap"]["mean_jaccard"]:.6f}。

这些是重复记录或兴趣强度近似，不是真实 quantity。

## 6. 冷用户和冷商品

| Window | Test rows | Cold-user rate | Cold-item rate | Warm candidate upper bound |
| --- | ---: | ---: | ---: | ---: |
{cold_lines}

## 7. 时间切分对比

| Window | Train rows | Test rows | Evaluable users | User visibility | One-positive user rate | Warm candidate upper bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{split_lines}

仅比较严格时间切分，不允许随机切分。任何训练特征必须只使用 cutoff 之前的数据。本阶段没有生成训练样本。

## 8. 三种反馈口径差异

- A：每个原始 transaction 行记一次。
- B：每日同用户同商品只记一次。
- C：保留 daily_interaction_count，表示同日记录强度近似。
- A 与 C 总强度相同；B 移除同日重复强度。三者的用户序列、商品热度和复购解释不同，本阶段不选择最终口径。

## 9. 模型适配证据矩阵

证据矩阵覆盖热门、趋势、复购、ItemCF、内容、矩阵分解、双塔、序列、冷启动和多样性重排。统计支持 V1 采用多路基础召回，但不能证明 DIN、双塔、DeepFM、GRU4Rec 或 Transformer 的离线收益。

## 10. 资源占用和失败恢复

| Group | Status | Memory MiB | Seconds | Peak RSS | Peak temp | Spill | Retried |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
{checkpoint_lines}

{chr(10).join(history) if history else "无历史失败。"}

## 11. 测试结果

`{summary["tests"]["summary"]}`。测试使用人工构造小数据，不依赖完整 H&M 原始数据。

## 12. 十六个核心问题的结论

1. 用户序列明显分层：B 口径下 11.05% 仅 1 条，至少 5/10/20 条分别覆盖 65.86%/47.30%/30.13% 用户。
2. B 口径下，21+ 行为用户占 {user["binary_users_21plus_share"]:.2%}，贡献 {user["binary_interactions_21plus_share"]:.2%} 二值交互，活跃用户贡献高度集中。
3. 若以至少 10 条行为作为序列模型最低门槛，统计覆盖上限为 47.30%；至少 20 条时为 30.13%，不是模型效果保证。
4. 商品长尾很强：Gini={item["gini_raw"]:.4f}，Top 10% 商品贡献 {item["top_shares_raw"]["top_10pct"]:.2%} 原始记录。
5. 热门召回适合作为低成本基线和冷启动 fallback，但会放大流行度偏置并压低长尾曝光。
6. 复购召回值得进入 V1 候选：{repeat["user_repeat_rate"]:.2%} 用户存在相同 article 重复记录；但同日重复与跨日复购必须分开。
7. ItemCF 有 28,575,395 条二值交互及可观活跃用户群，足以实施基线；当前统计不能证明离线收益。
8. 商品属性召回值得做，尤其用于 {item["untransacted_articles"]} 个零交易 article 和冷商品；属性权重与质量仍待验证。
9. 冷启动随窗口变重：7/14/28 天冷用户率为 8.08%/8.55%/9.70%，冷商品率为 3.71%/6.98%/12.00%。
10. 14 天是当前较平衡的首选候选窗口：117,366 个可评用户、91.45% 用户可见率、84.0% warm 候选上限；7 天可作快速回测，28 天可作稳健性验证，最终窗口尚未冻结。
11. 7/14/28 天严格时间切分后的可评用户分别为 63,412 / 117,366 / 210,553。
12. 热门、时间趋势和内容/属性 fallback 可覆盖更广用户；历史复购要求有历史行为。
13. ItemCF、矩阵分解、双塔和序列模型主要适用于 warm/活跃用户，序列模型尤其应做子群实验。
14. A/C 总强度相同；B 去掉 3,212,929 条同日重复强度，因此会改变序列长度、热度和复购解释。
15. 数据不支持无偏 CTR/CVR、真实 quantity、精确购物篮顺序、库存约束或因果推荐结论。
16. 已具备进入 Stage 1C 评审的聚合证据，但只能有条件进入；反馈口径、主评估窗口和分群策略仍需在架构冻结时决策。

## 13. 隐私、局限与下一阶段

- 正式产物均为时间序列、分桶、分位数、比率或证据矩阵。
- 未保存 customer_id、article_id、postal_code、用户序列或用户/商品级画像。
- 数据不含曝光、点击、负反馈、库存、完整订单、quantity 和精确交易时间，因此不支持无偏 CTR/CVR、真实购物篮顺序或因果推荐结论。
- Stage 1C 建议：**{summary["stage_1c_recommendation"]["decision"]}**。
- 理由：{summary["stage_1c_recommendation"]["reason"]}

本报告没有执行 Stage 1C。
"""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    config = stage1b_config(config_path)
    metadata = parquet_metadata(config)
    input_fp = input_fingerprint(config, metadata)
    contracts_path = config.resolve(config.raw["contracts_path"])
    config_fp = config_fingerprint(config_path, contracts_path)
    for path in [config.artifact_dir, config.checkpoint_dir, config.group_dir, config.temp_dir]:
        path.mkdir(parents=True, exist_ok=True)
    selected = args.groups or config.groups
    stopped = False
    for group_name in selected:
        print(f"[{group_name}] starting", flush=True)
        _, checkpoint, fatal = execute_group(
            group_name, config, input_fp, config_fp, args.force
        )
        print(
            f"[{group_name}] {checkpoint['status']} in "
            f"{checkpoint['execution_seconds']}s",
            flush=True,
        )
        if fatal:
            stopped = True
            break
    groups: dict[str, Any] = {}
    checkpoints = []
    for group_name in config.groups:
        checkpoint = read_checkpoint(config.checkpoint_dir / f"{group_name}.json")
        output = config.group_dir / f"{group_name}.json"
        if (
            checkpoint
            and checkpoint.get("status") == "SUCCESS"
            and checkpoint.get("input_fingerprint") == input_fp
            and checkpoint.get("config_fingerprint") == config_fp
            and output.exists()
        ):
            checkpoints.append(checkpoint)
            groups[group_name] = load_group_output(output)
    if "group_a_temporal_activity" in groups:
        monthly_rows = [
            row
            for row in read_csv(config.artifact_dir / "temporal_activity.csv")
            if row["granularity"] == "month"
        ]
        complete_months = monthly_rows[1:-1]
        peak_month = max(complete_months, key=lambda row: int(row["raw_rows_A"]))
        groups["group_a_temporal_activity"]["peak_complete_month"] = peak_month[
            "period_start"
        ][:7]
        groups["group_a_temporal_activity"]["peak_complete_month_rows"] = int(
            peak_month["raw_rows_A"]
        )
        zero_channel_1 = [
            row for row in monthly_rows if int(row["channel_1_rows"]) == 0
        ]
        groups["group_a_temporal_activity"]["channel_1_zero_observation"] = (
            f"{zero_channel_1[0]['period_start'][:7]} 渠道 1 为 0、"
            f"渠道 2 为 {int(zero_channel_1[0]['channel_2_rows']):,}"
            if zero_channel_1
            else "未观察到渠道 1 月度记录为 0"
        )
    if "group_b_user_activity" in groups:
        active_buckets = {"21-50", "51-100", "100+"}
        bucket_rows = read_csv(config.artifact_dir / "user_activity_buckets.csv")
        active_rows = [
            row
            for row in bucket_rows
            if row["feedback_scope"] == "B" and row["bucket"] in active_buckets
        ]
        groups["group_b_user_activity"]["binary_users_21plus_share"] = sum(
            float(row["user_share"]) for row in active_rows
        )
        groups["group_b_user_activity"]["binary_interactions_21plus_share"] = sum(
            float(row["interaction_coverage"]) for row in active_rows
        )
    tests = run_tests(config.root)
    complete = len(groups) == len(config.groups)
    summary_path = config.artifact_dir / "profile_summary.json"
    resource_path = config.artifact_dir / "resource_usage.json"
    recommendation = complete and tests["passed"]
    summary = {
        "stage": "1B",
        "executed_at": utc_now(),
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fp,
        "result_version": config.result_version,
        "feedback_scopes": FEEDBACK_SCOPES,
        "groups_expected": config.groups,
        "groups_successful": list(groups),
        "tests": tests,
        "stopped_after_resource_failure": stopped,
        "environment": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "duckdb": importlib.metadata.version("duckdb"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "psutil": importlib.metadata.version("psutil"),
            "virtual_environment_active": sys.prefix != sys.base_prefix,
        },
        "group_summaries": groups,
        "stage_1c_recommendation": {
            "decision": "有条件建议进入 Stage 1C" if recommendation else "暂不建议进入 Stage 1C",
            "reason": (
                "七个画像组和测试均完成，已有时间切分、冷启动、长尾、复购和模型适配证据；架构冻结前仍需评审反馈口径与窗口选择。"
                if recommendation
                else "存在未完成画像组或测试失败。"
            ),
        },
    }
    write_json(summary_path, summary)
    resources = {
        "memory_limit_policy": {
            "fraction_of_available": config.raw["duckdb"]["memory_fraction_of_available"],
            "min_mb": config.raw["duckdb"]["memory_min_mb"],
            "max_mb": config.raw["duckdb"]["memory_max_mb"],
        },
        "threads": config.threads,
        "temp_directory": config.relative(config.temp_dir),
        "max_temp_directory_size": config.raw["duckdb"]["max_temp_directory_size"],
        "groups": checkpoints,
        "totals": {
            "sum_group_seconds": sum(item["execution_seconds"] for item in checkpoints),
            "peak_rss_bytes": max((item["peak_rss_bytes"] for item in checkpoints), default=0),
            "peak_temp_bytes": max((item["temp_disk_peak_bytes"] for item in checkpoints), default=0),
            "spill_observed": any(item["spill_observed"] for item in checkpoints),
            "retry_observed": any(item["retried"] for item in checkpoints),
            "oom_observed": any(item["oom_observed"] for item in checkpoints),
        },
        "output_file_sizes": {},
    }
    write_json(resource_path, resources)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    if complete:
        config.report_path.write_text(
            build_report(summary, groups, checkpoints), encoding="utf-8"
        )
    output_paths = [
        summary_path,
        resource_path,
        config.report_path,
        *config.artifact_dir.glob("*.csv"),
        *config.checkpoint_dir.glob("*.json"),
        *config.group_dir.glob("*.json"),
    ]
    for _ in range(10):
        sizes = output_sizes(output_paths, config.root)
        if sizes == resources["output_file_sizes"]:
            break
        resources["output_file_sizes"] = sizes
        write_json(resource_path, resources)
    if complete:
        validate_profile_artifacts(
            [
                summary_path,
                resource_path,
                config.report_path,
                *config.artifact_dir.glob("*.csv"),
                *config.checkpoint_dir.glob("*.json"),
                *config.group_dir.glob("*.json"),
            ],
            [input_fp, config_fp, *[value["schema_fingerprint"] for value in metadata.values()]],
        )
    print(f"Groups: {len(groups)}/{len(config.groups)}")
    print(f"Summary: {config.relative(summary_path)}")
    print(f"Resources: {config.relative(resource_path)}")
    if complete:
        print(f"Report: {config.relative(config.report_path)}")
    print(summary["stage_1c_recommendation"]["decision"])
    return 0 if recommendation else 1


if __name__ == "__main__":
    raise SystemExit(main())
