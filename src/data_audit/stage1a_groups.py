from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .checks import schema_contract_diff
from .contracts import DataContracts
from .engine import arrow_schema_fingerprint
from .stage1a import GroupContext


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def full_scope(table: str, metadata: dict[str, dict[str, Any]]) -> str:
    value = metadata[table]
    return (
        f"FULL_SCAN: entire {value['file_path']}; {value['row_count']} rows; "
        f"{value['row_groups']}/{value['row_groups']} row groups"
    )


def _zero(value: int, label: str, notes: str = "") -> tuple[bool, Any, str]:
    return value == 0, {label: value}, notes


def group_a_metadata_schema(
    ctx: GroupContext,
    metadata: dict[str, dict[str, Any]],
    contracts: DataContracts,
) -> list[Path]:
    inventory_path = ctx.config.root / "artifacts" / "stage_0a" / "parquet_inventory.csv"
    with inventory_path.open(encoding="utf-8-sig", newline="") as handle:
        stage_0a = {
            row["logical_table"]: row["schema_fingerprint"]
            for row in csv.DictReader(handle)
        }
    metadata_rows: list[dict[str, Any]] = []
    for table, value in metadata.items():
        contract = contracts.tables[table]
        actual_fields = value["fields"]
        missing, mismatched = schema_contract_diff(actual_fields, contract.required_fields)
        ctx.add_check(
            check_id=f"{table}.full.metadata.readable",
            table=table,
            check_name="Parquet metadata readability",
            check_type="metadata",
            severity="error",
            expected="file exists and footer is readable",
            rows_scanned=0,
            scan_scope=f"FULL_SCAN metadata/footer: {value['file_path']}",
            operation=lambda value=value: (
                True,
                {
                    "file_size_bytes": value["file_size_bytes"],
                    "row_count": value["row_count"],
                    "row_groups": value["row_groups"],
                },
                "仅读取文件属性和 Parquet footer。",
            ),
            evidence_path="artifacts/stage_1a/metadata_schema.csv",
        )
        ctx.add_check(
            check_id=f"{table}.full.schema.contract",
            table=table,
            check_name="Full schema versus contract",
            check_type="schema",
            severity="error",
            expected="all required fields exist with exact Arrow types",
            rows_scanned=0,
            scan_scope=f"FULL_SCAN metadata/footer: {value['file_path']}",
            operation=lambda missing=missing, mismatched=mismatched, actual_fields=actual_fields: (
                not missing and not mismatched,
                {
                    "field_count": len(actual_fields),
                    "missing_fields": missing,
                    "type_mismatches": mismatched,
                },
                "Schema 检查不读取数据页。",
            ),
            evidence_path="configs/data_contracts.yaml",
        )
        current_fp = value["schema_fingerprint"]
        baseline_fp = stage_0a.get(table)
        ctx.add_check(
            check_id=f"{table}.full.schema.fingerprint",
            table=table,
            check_name="Schema fingerprint stability",
            check_type="schema_fingerprint",
            severity="error",
            expected="fingerprint equals Stage 0A inventory",
            rows_scanned=0,
            scan_scope=f"FULL_SCAN metadata/footer: {value['file_path']}",
            operation=lambda current_fp=current_fp, baseline_fp=baseline_fp: (
                bool(baseline_fp) and current_fp == baseline_fp,
                {
                    "matches_stage_0a": bool(baseline_fp) and current_fp == baseline_fp,
                    "fingerprint_prefix": current_fp[:16],
                },
                "只公开指纹前缀。",
            ),
            evidence_path="artifacts/stage_0a/parquet_inventory.csv",
        )
        metadata_rows.append(
            {
                "table_name": table,
                "file_path": value["file_path"],
                "file_size_bytes": value["file_size_bytes"],
                "row_count": value["row_count"],
                "row_groups": value["row_groups"],
                "schema_fingerprint": current_fp,
                "schema_matches_contract": not missing and not mismatched,
            }
        )
    output = ctx.config.artifact_dir / "metadata_schema.csv"
    write_csv(output, list(metadata_rows[0]), metadata_rows)
    return [output]


def _null_profile(
    ctx: GroupContext,
    table: str,
    columns: list[str],
    metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    expressions = ", ".join(
        f'count(*) FILTER (WHERE "{column}" IS NULL)' for column in columns
    )
    values = ctx.engine.row(f"SELECT {expressions} FROM {table}")
    total = metadata[table]["row_count"]
    profile = {}
    rows = []
    for column, null_count in zip(columns, values):
        null_count = int(null_count)
        null_rate = null_count / total if total else 0.0
        profile[column] = {"null_count": null_count, "null_rate": null_rate}
        rows.append(
            {
                "table_name": table,
                "column_name": column,
                "total_rows": total,
                "null_count": null_count,
                "null_rate": null_rate,
                "execution_scope": "FULL_SCAN",
            }
        )
    return profile, rows


def group_b_null_format(
    ctx: GroupContext,
    metadata: dict[str, dict[str, Any]],
    contracts: DataContracts,
) -> list[Path]:
    del contracts
    null_rows: list[dict[str, Any]] = []
    for table, value in metadata.items():
        profile, table_rows = _null_profile(ctx, table, list(value["fields"]), metadata)
        null_rows.extend(table_rows)
        ctx.add_check(
            check_id=f"{table}.full.null_profile",
            table=table,
            check_name="Full-column null profile",
            check_type="null_profile",
            severity="info",
            expected="report every column null count and rate without repair",
            rows_scanned=value["row_count"],
            scan_scope=full_scope(table, metadata),
            operation=lambda profile=profile: (
                True,
                profile,
                "全量聚合结果；未填补或删除空值。",
            ),
            evidence_path="artifacts/stage_1a/null_profile.csv",
        )

    checks = [
        (
            "articles.full.article_id.not_null",
            "articles",
            "article_id null count",
            "SELECT count(*) FROM articles WHERE article_id IS NULL",
            "null_count",
            "error",
        ),
        (
            "articles.full.article_id.format",
            "articles",
            "article_id format violations",
            "SELECT count(*) FROM articles WHERE article_id IS NOT NULL "
            "AND NOT regexp_full_match(article_id, '^[0-9]{10}$')",
            "violation_count",
            "error",
        ),
        (
            "customers.full.customer_id.not_null",
            "customers",
            "customer_id null count",
            "SELECT count(*) FROM customers WHERE customer_id IS NULL",
            "null_count",
            "error",
        ),
        (
            "customers.full.customer_id.format",
            "customers",
            "customer_id format violations",
            "SELECT count(*) FROM customers WHERE customer_id IS NOT NULL "
            "AND NOT regexp_full_match(customer_id, '^[0-9a-f]{64}$')",
            "violation_count",
            "error",
        ),
        (
            "customers.full.postal_code.format",
            "customers",
            "postal_code format violations",
            "SELECT count(*) FROM customers WHERE postal_code IS NOT NULL "
            "AND NOT regexp_full_match(postal_code, '^[0-9a-f]{64}$')",
            "violation_count",
            "error",
        ),
        (
            "transactions.full.customer_id.format",
            "transactions",
            "transaction customer_id format violations",
            "SELECT count(*) FROM transactions WHERE customer_id IS NOT NULL "
            "AND NOT regexp_full_match(customer_id, '^[0-9a-f]{64}$')",
            "violation_count",
            "error",
        ),
        (
            "transactions.full.article_id.format",
            "transactions",
            "transaction article_id format violations",
            "SELECT count(*) FROM transactions WHERE article_id IS NOT NULL "
            "AND NOT regexp_full_match(article_id, '^[0-9]{10}$')",
            "violation_count",
            "error",
        ),
        (
            "transactions.full.t_dat.parse",
            "transactions",
            "transaction date parse failures",
            "SELECT count(*) FROM transactions WHERE t_dat IS NOT NULL "
            "AND try_cast(t_dat AS DATE) IS NULL",
            "violation_count",
            "error",
        ),
        (
            "transactions.full.price.invalid",
            "transactions",
            "non-finite or negative prices",
            "SELECT count(*) FROM transactions WHERE price IS NOT NULL "
            "AND (NOT isfinite(price) OR price < 0)",
            "violation_count",
            "error",
        ),
    ]
    for check_id, table, name, sql, label, severity in checks:
        count = int(ctx.engine.row(sql)[0])
        ctx.add_check(
            check_id=check_id,
            table=table,
            check_name=name,
            check_type="format_or_validity",
            severity=severity,
            expected=f"{label} = 0",
            rows_scanned=metadata[table]["row_count"],
            scan_scope=full_scope(table, metadata),
            operation=lambda count=count, label=label: _zero(count, label),
            evidence_path="artifacts/stage_1a/audit_results.csv",
        )

    age_values = ctx.engine.row(
        "SELECT "
        "count(*) FILTER (WHERE age < 0), "
        "count(*) FILTER (WHERE age = 0), "
        "count(*) FILTER (WHERE age > 120), "
        "count(*) FILTER (WHERE age >= 100 AND age <= 120) "
        "FROM customers"
    )
    age_observed = dict(
        zip(["below_0", "equal_0", "above_120", "age_100_120"], map(int, age_values))
    )
    ctx.add_check(
        check_id="customers.full.age.boundaries",
        table="customers",
        check_name="Age invalid, boundary and extreme counts",
        check_type="range_profile",
        severity="warning",
        expected="below_0 = 0 and above_120 = 0; report boundaries",
        rows_scanned=metadata["customers"]["row_count"],
        scan_scope=full_scope("customers", metadata),
        operation=lambda: (
            age_observed["below_0"] == 0 and age_observed["above_120"] == 0,
            age_observed,
            "极端年龄仅统计，不自动修正。",
        ),
        evidence_path="artifacts/stage_1a/audit_results.csv",
    )

    channel_rows = [
        {"sales_channel_id": row[0], "row_count": int(row[1]), "execution_scope": "FULL_SCAN"}
        for row in ctx.engine.rows(
            "SELECT sales_channel_id, count(*) FROM transactions "
            "GROUP BY sales_channel_id ORDER BY sales_channel_id"
        )
    ]
    ctx.add_check(
        check_id="transactions.full.sales_channel.frequency",
        table="transactions",
        check_name="Observed sales-channel values and frequencies",
        check_type="observed_domain",
        severity="info",
        expected="discover full-scan values without a hard-coded domain",
        rows_scanned=metadata["transactions"]["row_count"],
        scan_scope=full_scope("transactions", metadata),
        operation=lambda: (
            True,
            {"frequencies": channel_rows},
            "这是本地完整 transaction 文件的观察集合。",
        ),
        evidence_path="artifacts/stage_1a/value_ranges.csv",
    )

    output = ctx.config.artifact_dir / "null_profile.csv"
    write_csv(
        output,
        ["table_name", "column_name", "total_rows", "null_count", "null_rate", "execution_scope"],
        null_rows,
    )
    return [output]


def group_c_date_value_ranges(
    ctx: GroupContext,
    metadata: dict[str, dict[str, Any]],
    contracts: DataContracts,
) -> list[Path]:
    del contracts
    range_rows: list[dict[str, Any]] = []
    date_row = ctx.engine.row(
        "SELECT min(try_cast(t_dat AS DATE)), max(try_cast(t_dat AS DATE)), "
        "count(*) FILTER (WHERE t_dat IS NOT NULL AND try_cast(t_dat AS DATE) IS NULL) "
        "FROM transactions"
    )
    date_observed = {
        "min_date": str(date_row[0]) if date_row[0] else None,
        "max_date": str(date_row[1]) if date_row[1] else None,
        "parse_failure_count": int(date_row[2]),
    }
    ctx.add_check(
        check_id="transactions.full.date.range",
        table="transactions",
        check_name="Transaction date range and parse quality",
        check_type="date_range",
        severity="error",
        expected="all non-null dates parse; report min and max",
        rows_scanned=metadata["transactions"]["row_count"],
        scan_scope=full_scope("transactions", metadata),
        operation=lambda: (
            date_observed["parse_failure_count"] == 0,
            date_observed,
            "日期范围来自完整 transaction 文件。",
        ),
        evidence_path="artifacts/stage_1a/value_ranges.csv",
    )
    range_rows.append(
        {
            "table_name": "transactions",
            "field_name": "t_dat",
            "metric": "date_range",
            "value": json.dumps(date_observed, ensure_ascii=False),
            "execution_scope": "FULL_SCAN",
        }
    )

    price_row = ctx.engine.row(
        "SELECT min(price), max(price), avg(price), "
        "quantile_cont(price, [0.01, 0.25, 0.5, 0.75, 0.99]), "
        "count(*) FILTER (WHERE price < 0), "
        "count(*) FILTER (WHERE price = 0) FROM transactions"
    )
    price_observed = {
        "min": price_row[0],
        "max": price_row[1],
        "mean": price_row[2],
        "quantiles": dict(zip(["p01", "p25", "p50", "p75", "p99"], price_row[3])),
        "negative_count": int(price_row[4]),
        "zero_count": int(price_row[5]),
    }
    ctx.add_check(
        check_id="transactions.full.price.profile",
        table="transactions",
        check_name="Full price range and quantiles",
        check_type="numeric_profile",
        severity="warning",
        expected="finite non-negative prices; report range, mean, quantiles and zero count",
        rows_scanned=metadata["transactions"]["row_count"],
        scan_scope=full_scope("transactions", metadata),
        operation=lambda: (
            price_observed["negative_count"] == 0,
            price_observed,
            "零价格只记录，不在本阶段解释或处理。",
        ),
        evidence_path="artifacts/stage_1a/value_ranges.csv",
    )
    range_rows.append(
        {
            "table_name": "transactions",
            "field_name": "price",
            "metric": "numeric_profile",
            "value": json.dumps(price_observed, ensure_ascii=False),
            "execution_scope": "FULL_SCAN",
        }
    )

    age_row = ctx.engine.row(
        "SELECT min(age), max(age), avg(age), "
        "quantile_cont(age, [0.01, 0.25, 0.5, 0.75, 0.99]), "
        "count(*) FILTER (WHERE age IS NULL), "
        "count(*) FILTER (WHERE age < 0 OR age > 120), "
        "count(*) FILTER (WHERE age >= 0 AND age < 18), "
        "count(*) FILTER (WHERE age >= 18 AND age <= 24), "
        "count(*) FILTER (WHERE age >= 25 AND age <= 64), "
        "count(*) FILTER (WHERE age >= 65 AND age <= 120) FROM customers"
    )
    age_observed = {
        "min": age_row[0],
        "max": age_row[1],
        "mean": age_row[2],
        "quantiles": dict(zip(["p01", "p25", "p50", "p75", "p99"], age_row[3])),
        "null_count": int(age_row[4]),
        "out_of_range": int(age_row[5]),
        "age_0_17": int(age_row[6]),
        "age_18_24": int(age_row[7]),
        "age_25_64": int(age_row[8]),
        "age_65_120": int(age_row[9]),
    }
    ctx.add_check(
        check_id="customers.full.age.profile",
        table="customers",
        check_name="Full age range, quantiles and buckets",
        check_type="numeric_profile",
        severity="warning",
        expected="non-null ages in [0, 120]; report quantiles and buckets",
        rows_scanned=metadata["customers"]["row_count"],
        scan_scope=full_scope("customers", metadata),
        operation=lambda: (
            age_observed["out_of_range"] == 0,
            age_observed,
            "年龄缺失和分桶仅用于数据质量评估。",
        ),
        evidence_path="artifacts/stage_1a/value_ranges.csv",
    )
    range_rows.append(
        {
            "table_name": "customers",
            "field_name": "age",
            "metric": "numeric_profile",
            "value": json.dumps(age_observed, ensure_ascii=False),
            "execution_scope": "FULL_SCAN",
        }
    )
    output = ctx.config.artifact_dir / "value_ranges.csv"
    write_csv(
        output,
        ["table_name", "field_name", "metric", "value", "execution_scope"],
        range_rows,
    )
    return [output]


def _key_integrity_row(ctx: GroupContext, table: str, field: str) -> dict[str, Any]:
    row = ctx.engine.row(
        f"WITH key_counts AS ("
        f"SELECT {field}, count(*) AS n FROM {table} "
        f"WHERE {field} IS NOT NULL GROUP BY {field}"
        f") SELECT "
        f"(SELECT count(*) FROM {table}), "
        f"COALESCE(sum(n), 0), count(*), "
        f"count(*) FILTER (WHERE n > 1), "
        f"COALESCE(sum(n - 1) FILTER (WHERE n > 1), 0) "
        f"FROM key_counts"
    )
    return {
        "table_name": table,
        "field_name": field,
        "total_rows": int(row[0]),
        "non_null_rows": int(row[1]),
        "distinct_count": int(row[2]),
        "duplicate_key_count": int(row[3]),
        "duplicate_excess_rows": int(row[4]),
        "execution_scope": "FULL_SCAN",
    }


def group_d_primary_keys(
    ctx: GroupContext,
    metadata: dict[str, dict[str, Any]],
    contracts: DataContracts,
) -> list[Path]:
    del contracts
    rows = [
        _key_integrity_row(ctx, "articles", "article_id"),
        _key_integrity_row(ctx, "customers", "customer_id"),
    ]
    for observed in rows:
        table = observed["table_name"]
        ctx.add_check(
            check_id=f"{table}.full.primary_key",
            table=table,
            check_name=f"{observed['field_name']} full candidate-key integrity",
            check_type="candidate_key",
            severity="error",
            expected="null_count = 0 and duplicate_key_count = 0",
            rows_scanned=metadata[table]["row_count"],
            scan_scope=full_scope(table, metadata),
            operation=lambda observed=observed: (
                observed["non_null_rows"] == observed["total_rows"]
                and observed["duplicate_key_count"] == 0,
                observed,
                "不输出任何具体重复键。",
            ),
            evidence_path="artifacts/stage_1a/key_integrity.csv",
        )
    output = ctx.config.artifact_dir / "key_integrity.csv"
    write_csv(output, list(rows[0]), rows)
    return [output]


def group_e_foreign_keys(
    ctx: GroupContext,
    metadata: dict[str, dict[str, Any]],
    contracts: DataContracts,
) -> list[Path]:
    del contracts
    transaction_rows = metadata["transactions"]["row_count"]
    distinct_row = ctx.engine.row(
        "SELECT count(DISTINCT customer_id), count(DISTINCT article_id), "
        "count(*) FILTER (WHERE customer_id IS NOT NULL), "
        "count(*) FILTER (WHERE article_id IS NOT NULL) FROM transactions"
    )
    distinct_customers, distinct_articles = int(distinct_row[0]), int(distinct_row[1])
    nonnull_customers, nonnull_articles = int(distinct_row[2]), int(distinct_row[3])
    ctx.add_check(
        check_id="transactions.full.distinct_entities",
        table="transactions",
        check_name="Full distinct customer and article counts",
        check_type="distinct_count",
        severity="info",
        expected="report full-scan distinct counts without emitting identifiers",
        rows_scanned=transaction_rows,
        scan_scope=full_scope("transactions", metadata),
        operation=lambda: (
            True,
            {
                "distinct_customer_count": distinct_customers,
                "distinct_article_count": distinct_articles,
            },
            "只输出聚合计数。",
        ),
        evidence_path="artifacts/stage_1a/foreign_key_coverage.csv",
    )
    rows: list[dict[str, Any]] = []
    for field, dimension, nonnull, distinct_total in [
        ("customer_id", "customers", nonnull_customers, distinct_customers),
        ("article_id", "articles", nonnull_articles, distinct_articles),
    ]:
        unmatched = ctx.engine.row(
            f"WITH missing AS ("
            f"SELECT t.{field} FROM transactions t ANTI JOIN {dimension} d "
            f"ON t.{field} = d.{field} WHERE t.{field} IS NOT NULL"
            f") SELECT count(*), count(DISTINCT {field}) FROM missing"
        )
        unmatched_rows, unmatched_distinct = int(unmatched[0]), int(unmatched[1])
        observed = {
            "relationship": f"transactions.{field}->{dimension}.{field}",
            "non_null_transaction_rows": nonnull,
            "distinct_transaction_keys": distinct_total,
            "unmatched_transaction_rows": unmatched_rows,
            "unmatched_distinct_keys": unmatched_distinct,
            "row_match_rate": (nonnull - unmatched_rows) / nonnull if nonnull else None,
            "distinct_key_match_rate": (
                (distinct_total - unmatched_distinct) / distinct_total
                if distinct_total
                else None
            ),
            "execution_scope": "FULL_SCAN",
        }
        rows.append(observed)
        ctx.add_check(
            check_id=f"transactions.full.{field}.foreign_key",
            table="transactions",
            check_name=f"Full foreign-key coverage for {field}",
            check_type="foreign_key_candidate",
            severity="warning",
            expected="unmatched rows and unmatched distinct keys = 0",
            rows_scanned=transaction_rows,
            scan_scope=full_scope("transactions", metadata),
            operation=lambda observed=observed: (
                observed["unmatched_transaction_rows"] == 0
                and observed["unmatched_distinct_keys"] == 0,
                observed,
                "使用 anti join；不输出任何未匹配 ID。",
            ),
            evidence_path="artifacts/stage_1a/foreign_key_coverage.csv",
        )
    output = ctx.config.artifact_dir / "foreign_key_coverage.csv"
    write_csv(output, list(rows[0]), rows)
    return [output]


def group_f_transaction_duplicates(
    ctx: GroupContext,
    metadata: dict[str, dict[str, Any]],
    contracts: DataContracts,
) -> list[Path]:
    del contracts
    total = metadata["transactions"]["row_count"]
    # 使用维表候选主键建立临时整数代码，降低全量高基数字符串分组的内存压力。
    # 映射与编码交易表仅存在 DuckDB 临时空间，不输出 ID，也不重写原始 Parquet。
    ctx.engine.execute(
        "CREATE TEMP TABLE _customer_codes AS "
        "SELECT customer_id, row_number() OVER ()::BIGINT AS customer_code "
        "FROM customers"
    )
    ctx.engine.execute(
        "CREATE TEMP TABLE _article_codes AS "
        "SELECT article_id, row_number() OVER ()::BIGINT AS article_code "
        "FROM articles"
    )
    ctx.engine.execute(
        "CREATE TEMP TABLE _transactions_coded AS "
        "SELECT t.t_dat, c.customer_code, a.article_code, t.price, t.sales_channel_id "
        "FROM transactions t "
        "LEFT JOIN _customer_codes c ON t.customer_id = c.customer_id "
        "LEFT JOIN _article_codes a ON t.article_id = a.article_id"
    )
    coded_quality = ctx.engine.row(
        "SELECT count(*), "
        "count(*) FILTER (WHERE customer_code IS NULL), "
        "count(*) FILTER (WHERE article_code IS NULL) "
        "FROM _transactions_coded"
    )
    if (
        int(coded_quality[0]) != total
        or int(coded_quality[1]) != 0
        or int(coded_quality[2]) != 0
    ):
        raise ValueError(
            "临时整数编码未覆盖全部交易，不能安全替代原始 ID 执行精确重复检查"
        )
    full_row = ctx.engine.row(
        "WITH counts AS ("
        "SELECT t_dat, customer_code, article_code, price, sales_channel_id, count(*) AS n "
        "FROM _transactions_coded "
        "GROUP BY t_dat, customer_code, article_code, price, sales_channel_id"
        ") SELECT "
        "count(*) FILTER (WHERE n > 1), "
        "COALESCE(sum(n - 1) FILTER (WHERE n > 1), 0), "
        "max(n), "
        "COALESCE(sum(n) FILTER (WHERE n > 1), 0), "
        "count(*) FILTER (WHERE n = 1), "
        "count(*) FILTER (WHERE n = 2), "
        "count(*) FILTER (WHERE n = 3), "
        "count(*) FILTER (WHERE n = 4), "
        "count(*) FILTER (WHERE n >= 5) "
        "FROM counts"
    )
    full_observed = {
        "full_identical_duplicate_groups": int(full_row[0]),
        "full_identical_duplicate_excess_rows": int(full_row[1]),
        "full_identical_max_repetition": int(full_row[2]),
        "rows_in_repeated_full_groups": int(full_row[3]),
        "duplicate_excess_ratio": int(full_row[1]) / total,
        "rows_in_repeated_groups_ratio": int(full_row[3]) / total,
        "group_count_buckets": {
            "1": int(full_row[4]),
            "2": int(full_row[5]),
            "3": int(full_row[6]),
            "4": int(full_row[7]),
            "5+": int(full_row[8]),
        },
    }
    ctx.add_check(
        check_id="transactions.full.duplicate.full_row",
        table="transactions",
        check_name="Full-field identical record repetitions",
        check_type="duplicate_profile",
        severity="warning",
        expected="report repetition metrics without classifying them as data errors",
        rows_scanned=total,
        scan_scope=full_scope("transactions", metadata),
        operation=lambda: (
            full_observed["full_identical_duplicate_excess_rows"] == 0,
            full_observed,
            "记录重复不等于脏数据；缺少 order_id、时间戳和 quantity，不能判断业务含义。",
        ),
        evidence_path="artifacts/stage_1a/transaction_duplicate_profiles.csv",
    )

    triplet = ctx.engine.row(
        "WITH counts AS ("
        "SELECT t_dat, customer_code, article_code, count(*) AS n "
        "FROM _transactions_coded GROUP BY t_dat, customer_code, article_code"
        ") SELECT count(*) FILTER (WHERE n > 1), "
        "COALESCE(sum(n - 1) FILTER (WHERE n > 1), 0), max(n) FROM counts"
    )
    triplet_observed = {
        "same_date_customer_article_multi_groups": int(triplet[0]),
        "same_date_customer_article_excess_rows": int(triplet[1]),
        "max_repetition": int(triplet[2]),
        "excess_ratio": int(triplet[1]) / total,
    }
    ctx.add_check(
        check_id="transactions.full.duplicate.date_customer_article",
        table="transactions",
        check_name="Same-date customer-article repeated purchase profile",
        check_type="repeat_purchase_profile",
        severity="info",
        expected="report repeated groups and excess rows without judging validity",
        rows_scanned=total,
        scan_scope=full_scope("transactions", metadata),
        operation=lambda: (
            True,
            triplet_observed,
            "该口径可能包含同日多件购买；缺少 quantity 和 order_id。",
        ),
        evidence_path="artifacts/stage_1a/transaction_duplicate_profiles.csv",
    )

    date_rows_raw = ctx.engine.rows(
        "WITH counts AS ("
        "SELECT t_dat, customer_code, article_code, price, sales_channel_id, count(*) AS n "
        "FROM _transactions_coded "
        "GROUP BY t_dat, customer_code, article_code, price, sales_channel_id"
        ") SELECT t_dat, "
        "count(*) FILTER (WHERE n > 1) AS duplicate_groups, "
        "COALESCE(sum(n - 1) FILTER (WHERE n > 1), 0) AS excess_rows, "
        "max(n) AS max_repetition "
        "FROM counts GROUP BY t_dat ORDER BY t_dat"
    )
    date_rows = [
        {
            "record_type": "date",
            "dimension": str(row[0]),
            "group_count": int(row[1]),
            "excess_rows": int(row[2]),
            "max_repetition": int(row[3]),
            "ratio": int(row[2]) / total,
            "execution_scope": "FULL_SCAN",
        }
        for row in date_rows_raw
    ]
    date_summary = {
        "date_count": len(date_rows),
        "dates_with_repeated_records": sum(row["excess_rows"] > 0 for row in date_rows),
        "max_daily_excess_rows": max((row["excess_rows"] for row in date_rows), default=0),
    }
    ctx.add_check(
        check_id="transactions.full.duplicate.date_distribution",
        table="transactions",
        check_name="Full-field repetition distribution by date",
        check_type="duplicate_date_profile",
        severity="info",
        expected="report aggregate daily distribution only",
        rows_scanned=total,
        scan_scope=full_scope("transactions", metadata),
        operation=lambda: (
            True,
            date_summary,
            "日期级汇总不包含用户、商品或交易明细。",
        ),
        evidence_path="artifacts/stage_1a/transaction_duplicate_profiles.csv",
    )

    profile_rows = [
        {
            "record_type": "metric",
            "dimension": "full_identical",
            "group_count": full_observed["full_identical_duplicate_groups"],
            "excess_rows": full_observed["full_identical_duplicate_excess_rows"],
            "max_repetition": full_observed["full_identical_max_repetition"],
            "ratio": full_observed["duplicate_excess_ratio"],
            "execution_scope": "FULL_SCAN",
        },
        {
            "record_type": "metric",
            "dimension": "t_dat+customer_id+article_id",
            "group_count": triplet_observed["same_date_customer_article_multi_groups"],
            "excess_rows": triplet_observed["same_date_customer_article_excess_rows"],
            "max_repetition": triplet_observed["max_repetition"],
            "ratio": triplet_observed["excess_ratio"],
            "execution_scope": "FULL_SCAN",
        },
        {
            "record_type": "metric",
            "dimension": "t_dat+customer_id+article_id+price+sales_channel_id",
            "group_count": full_observed["full_identical_duplicate_groups"],
            "excess_rows": full_observed["full_identical_duplicate_excess_rows"],
            "max_repetition": full_observed["full_identical_max_repetition"],
            "ratio": full_observed["duplicate_excess_ratio"],
            "execution_scope": "FULL_SCAN",
        },
    ]
    for bucket, count in full_observed["group_count_buckets"].items():
        profile_rows.append(
            {
                "record_type": "bucket",
                "dimension": bucket,
                "group_count": count,
                "excess_rows": "",
                "max_repetition": "",
                "ratio": "",
                "execution_scope": "FULL_SCAN",
            }
        )
    profile_rows.extend(date_rows)
    output = ctx.config.artifact_dir / "transaction_duplicate_profiles.csv"
    write_csv(
        output,
        [
            "record_type",
            "dimension",
            "group_count",
            "excess_rows",
            "max_repetition",
            "ratio",
            "execution_scope",
        ],
        profile_rows,
    )
    return [output]


GROUP_FUNCTIONS = {
    "group_a_metadata_schema": group_a_metadata_schema,
    "group_b_null_format": group_b_null_format,
    "group_c_date_value_ranges": group_c_date_value_ranges,
    "group_d_primary_keys": group_d_primary_keys,
    "group_e_foreign_keys": group_e_foreign_keys,
    "group_f_transaction_duplicates": group_f_transaction_duplicates,
}


def parquet_metadata(config: Any) -> dict[str, dict[str, Any]]:
    result = {}
    for table, path in config.data_paths.items():
        parquet_file = pq.ParquetFile(path)
        schema = parquet_file.schema_arrow
        result[table] = {
            "file_path": config.relative(path),
            "file_size_bytes": path.stat().st_size,
            "row_count": parquet_file.metadata.num_rows,
            "row_groups": parquet_file.metadata.num_row_groups,
            "schema_fingerprint": arrow_schema_fingerprint(schema),
            "fields": {field.name: str(field.type) for field in schema},
        }
    return result
