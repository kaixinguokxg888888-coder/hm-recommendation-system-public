from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

from .contracts import DataContracts
from .engine import DuckDBAuditEngine
from .reporting import AuditResult


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def schema_contract_diff(
    actual_fields: dict[str, str], required_fields: dict[str, str]
) -> tuple[list[str], dict[str, dict[str, str | None]]]:
    missing = sorted(set(required_fields) - set(actual_fields))
    mismatched = {
        field: {"expected": expected, "actual": actual_fields.get(field)}
        for field, expected in required_fields.items()
        if field in actual_fields and actual_fields[field] != expected
    }
    return missing, mismatched


def date_parse_failure_count(engine: DuckDBAuditEngine, relation: str, field: str) -> int:
    identifier = quote_identifier(field)
    return int(
        engine.scalar(
            f"SELECT count(*) FROM {relation} "
            f"WHERE {identifier} IS NOT NULL AND try_cast({identifier} AS DATE) IS NULL"
        )
    )


def candidate_key_duplicate_count(
    engine: DuckDBAuditEngine, relation: str, fields: list[str]
) -> int:
    identifiers = ", ".join(quote_identifier(field) for field in fields)
    return int(
        engine.scalar(
            "SELECT COALESCE(SUM(n - 1), 0) FROM "
            f"(SELECT count(*) AS n FROM {relation} GROUP BY {identifiers} HAVING count(*) > 1)"
        )
    )


class SmokeChecks:
    def __init__(
        self,
        engine: DuckDBAuditEngine,
        contracts: DataContracts,
        project_root: Path,
    ) -> None:
        self.engine = engine
        self.contracts = contracts
        self.project_root = project_root.resolve()
        self.results: list[AuditResult] = []

    def _public_error(self, exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}".replace(str(self.project_root), ".")

    def _add(
        self,
        *,
        check_id: str,
        table: str,
        name: str,
        check_type: str,
        severity: str,
        expected: str,
        rows_scanned: int,
        scope: str,
        operation: Callable[[], tuple[bool, Any, str]],
        evidence: str,
    ) -> None:
        started = time.perf_counter()
        try:
            passed, observed, notes = operation()
            status = "PASS" if passed else ("WARN" if severity == "warning" else "FAIL")
            error_message = ""
        except Exception as exc:
            observed = None
            notes = "该检查失败；其他检查继续执行。"
            status = "ERROR"
            error_message = self._public_error(exc)
        self.results.append(
            AuditResult(
                check_id=check_id,
                table_name=table,
                check_name=name,
                check_type=check_type,
                severity=severity,
                status=status,
                observed_value=observed,
                expected_condition=expected,
                rows_scanned=rows_scanned,
                scan_scope=scope,
                execution_seconds=round(time.perf_counter() - started, 6),
                evidence_path=evidence,
                notes=notes,
                error_message=error_message,
            )
        )

    def _full_scope(self, table: str) -> str:
        return (
            f"full table: {self.contracts.tables[table].file}; "
            f"{self.engine.table_rows[table]} rows; "
            f"{self.engine.row_groups[table]}/{self.engine.row_groups[table]} row groups"
        )

    def _transaction_scope(self) -> str:
        scope = self.engine.transaction_scope
        return (
            f"bounded smoke: {self.contracts.tables['transactions'].file}; "
            f"row_group={scope['row_group_index']}; first {scope['rows_materialized']} "
            f"of {scope['row_group_rows']} rows in that row group; "
            f"file has {scope['file_rows']} rows/{scope['file_row_groups']} row groups"
        )

    def run_direct_probes(self) -> None:
        for table in self.contracts.tables:
            self._add(
                check_id=f"{table}.duckdb.direct_parquet_probe",
                table=table,
                name="DuckDB direct Parquet probe",
                check_type="engine_stability",
                severity="error",
                expected="DuckDB returns one row using LIMIT 1",
                rows_scanned=1,
                scope=f"direct Parquet LIMIT 1 probe: {self.contracts.tables[table].file}",
                operation=lambda table=table: self._direct_probe_observation(table),
                evidence="artifacts/stage_0b/audit_results.csv",
            )

    def _direct_probe_observation(self, table: str) -> tuple[bool, Any, str]:
        rows_returned = self.engine.direct_parquet_probe(table)
        return (
            rows_returned == 1,
            {"rows_returned": rows_returned},
            "LIMIT 1 仅用于引擎可读性探测，不作为聚合扫描限制。",
        )

    def run_schema_checks(self) -> None:
        inventory = self._inventory_fingerprints()
        for table, contract in self.contracts.tables.items():
            schema = self.engine.arrow_schema(table)
            actual = {field.name: str(field.type) for field in schema}
            missing, mismatched = schema_contract_diff(actual, contract.required_fields)
            self._add(
                check_id=f"{table}.schema.required_fields",
                table=table,
                name="Required fields and Arrow types",
                check_type="schema",
                severity="error",
                expected="all declared fields exist with exact Arrow types",
                rows_scanned=0,
                scope=f"Parquet footer/schema only: {contract.file}",
                operation=lambda missing=missing, mismatched=mismatched, actual=actual: (
                    not missing and not mismatched,
                    {
                        "field_count": len(actual),
                        "missing_fields": missing,
                        "type_mismatches": mismatched,
                    },
                    "Schema 检查未扫描数据行。",
                ),
                evidence="configs/data_contracts.yaml",
            )
            current = self.engine.fingerprint(table)
            expected_fingerprint = inventory.get(table)
            self._add(
                check_id=f"{table}.schema.fingerprint_stable",
                table=table,
                name="Stage 0A schema fingerprint stability",
                check_type="schema_fingerprint",
                severity="error",
                expected="current fingerprint equals Stage 0A inventory",
                rows_scanned=0,
                scope=f"Parquet footer/schema only: {contract.file}",
                operation=lambda current=current, expected_fingerprint=expected_fingerprint: (
                    bool(expected_fingerprint) and current == expected_fingerprint,
                    {
                        "matches_stage_0a": bool(expected_fingerprint)
                        and current == expected_fingerprint,
                        "fingerprint_prefix": current[:16],
                    },
                    "仅公开 fingerprint 前缀。",
                ),
                evidence="artifacts/stage_0a/parquet_inventory.csv",
            )

    def _inventory_fingerprints(self) -> dict[str, str]:
        path = self.project_root / "artifacts" / "stage_0a" / "parquet_inventory.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return {
                row["logical_table"]: row["schema_fingerprint"]
                for row in csv.DictReader(handle)
            }

    def run_articles(self) -> None:
        rows = self.engine.table_rows["articles"]
        scope = self._full_scope("articles")
        self._count_zero_check(
            "articles.article_id.not_null",
            "articles",
            "Article ID null count",
            "not_null",
            "error",
            rows,
            scope,
            "SELECT count(*) FROM articles WHERE article_id IS NULL",
            "null_count = 0",
        )
        self._count_zero_check(
            "articles.article_id.format",
            "articles",
            "Article ID format violations",
            "regex",
            "error",
            rows,
            scope,
            "SELECT count(*) FROM articles WHERE article_id IS NOT NULL "
            "AND NOT regexp_full_match(article_id, '^[0-9]{10}$')",
            "exactly 10 ASCII digits",
        )
        self._add(
            check_id="articles.article_id.candidate_key",
            table="articles",
            name="Article candidate-key duplicates",
            check_type="candidate_key",
            severity="error",
            expected="duplicate_key_count = 0 within scan scope",
            rows_scanned=rows,
            scope=scope,
            operation=lambda: self._zero_observation(
                candidate_key_duplicate_count(self.engine, "articles", ["article_id"]),
                "duplicate_excess_rows",
                "articles 在 smoke 中使用完整维表；结论仍只属于当前本地文件。",
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )
        fields = self.contracts.tables["articles"].raw["nullable_fields"]
        expressions = ", ".join(
            f"count(*) FILTER (WHERE {quote_identifier(field)} IS NULL)" for field in fields
        )
        self._add(
            check_id="articles.attributes.null_profile",
            table="articles",
            name="Article attribute null profile",
            check_type="null_profile",
            severity="info",
            expected="report nullable attribute counts without repair",
            rows_scanned=rows,
            scope=scope,
            operation=lambda: (
                True,
                dict(zip(fields, self.engine.row(f"SELECT {expressions} FROM articles"))),
                "仅输出逐字段计数，不输出原始记录。",
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )

    def run_customers(self) -> None:
        rows = self.engine.table_rows["customers"]
        scope = self._full_scope("customers")
        self._count_zero_check(
            "customers.customer_id.not_null",
            "customers",
            "Customer ID null count",
            "not_null",
            "error",
            rows,
            scope,
            "SELECT count(*) FROM customers WHERE customer_id IS NULL",
            "null_count = 0",
        )
        self._count_zero_check(
            "customers.customer_id.format",
            "customers",
            "Customer ID format violations",
            "regex",
            "error",
            rows,
            scope,
            "SELECT count(*) FROM customers WHERE customer_id IS NOT NULL "
            "AND NOT regexp_full_match(customer_id, '^[0-9a-f]{64}$')",
            "exactly 64 lowercase hexadecimal characters",
        )
        self._add(
            check_id="customers.customer_id.candidate_key",
            table="customers",
            name="Customer candidate-key duplicates",
            check_type="candidate_key",
            severity="error",
            expected="duplicate_key_count = 0 within scan scope",
            rows_scanned=rows,
            scope=scope,
            operation=lambda: self._zero_observation(
                candidate_key_duplicate_count(self.engine, "customers", ["customer_id"]),
                "duplicate_excess_rows",
                "customers 在 smoke 中使用完整维表；结论仍只属于当前本地文件。",
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )
        self._add(
            check_id="customers.flags.nullability",
            table="customers",
            name="Customer nullable fields profile",
            check_type="null_profile",
            severity="info",
            expected="report FN, Active and age null counts without repair",
            rows_scanned=rows,
            scope=scope,
            operation=lambda: (
                True,
                dict(
                    zip(
                        ["FN", "Active", "age"],
                        self.engine.row(
                            'SELECT count(*) FILTER (WHERE "FN" IS NULL), '
                            'count(*) FILTER (WHERE "Active" IS NULL), '
                            "count(*) FILTER (WHERE age IS NULL) FROM customers"
                        ),
                    )
                ),
                "允许为空；不进行填补。",
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )
        self._add(
            check_id="customers.age.reasonable",
            table="customers",
            name="Age reasonableness profile",
            check_type="range_profile",
            severity="warning",
            expected="non-null ages in [0, 120]",
            rows_scanned=rows,
            scope=scope,
            operation=self._age_profile,
            evidence="artifacts/stage_0b/audit_results.csv",
        )
        self._count_zero_check(
            "customers.postal_code.privacy",
            "customers",
            "Postal-code format violations",
            "privacy_and_regex",
            "error",
            rows,
            scope,
            "SELECT count(*) FROM customers WHERE postal_code IS NOT NULL "
            "AND NOT regexp_full_match(postal_code, '^[0-9a-f]{64}$')",
            "non-null values are 64 lowercase hexadecimal characters; raw values never emitted",
            notes="只输出违规计数，不输出 postal_code 原值。",
        )

    def _age_profile(self) -> tuple[bool, Any, str]:
        values = self.engine.row(
            "SELECT "
            "count(*) FILTER (WHERE age IS NULL), "
            "count(*) FILTER (WHERE age < 0 OR age > 120), "
            "count(*) FILTER (WHERE age >= 0 AND age < 18), "
            "count(*) FILTER (WHERE age >= 18 AND age <= 24), "
            "count(*) FILTER (WHERE age >= 25 AND age <= 64), "
            "count(*) FILTER (WHERE age >= 65 AND age <= 120) "
            "FROM customers"
        )
        profile = dict(
            zip(
                ["null", "out_of_range", "age_0_17", "age_18_24", "age_25_64", "age_65_120"],
                values,
            )
        )
        return (
            profile["out_of_range"] == 0,
            profile,
            "分层计数仅用于审计，不执行年龄修正。",
        )

    def run_transactions(self) -> None:
        rows = self.engine.transaction_scope["rows_materialized"]
        scope = self._transaction_scope()
        self._count_zero_check(
            "transactions.t_dat.parseable",
            "transactions",
            "Transaction date parse failures",
            "date_parse",
            "error",
            rows,
            scope,
            "SELECT count(*) FROM transactions_smoke WHERE t_dat IS NOT NULL "
            "AND try_cast(t_dat AS DATE) IS NULL",
            "all non-null t_dat values parse as DATE",
        )
        for field in ("customer_id", "article_id", "price"):
            self._count_zero_check(
                f"transactions.{field}.not_null",
                "transactions",
                f"{field} null count",
                "not_null",
                "error",
                rows,
                scope,
                f"SELECT count(*) FROM transactions_smoke WHERE {quote_identifier(field)} IS NULL",
                "null_count = 0",
            )
        self._count_zero_check(
            "transactions.price.non_negative",
            "transactions",
            "Negative price count",
            "range",
            "error",
            rows,
            scope,
            "SELECT count(*) FROM transactions_smoke WHERE price < 0",
            "price >= 0",
        )
        self._add(
            check_id="transactions.sales_channel_id.observed_domain",
            table="transactions",
            name="Observed sales-channel domain",
            check_type="observed_domain",
            severity="info",
            expected="discover values from smoke scope; no predefined legal set",
            rows_scanned=rows,
            scope=scope,
            operation=lambda: (
                True,
                {
                    "observed_values": [
                        row[0]
                        for row in self.engine.connection.execute(
                            "SELECT DISTINCT sales_channel_id FROM transactions_smoke "
                            "WHERE sales_channel_id IS NOT NULL ORDER BY sales_channel_id"
                        ).fetchall()
                    ]
                },
                "该集合仅来自受限 smoke，不是全量合法值契约。",
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )
        self._add(
            check_id="transactions.full_row.duplicates",
            table="transactions",
            name="Full-row duplicate logic",
            check_type="duplicate_rows",
            severity="warning",
            expected="report duplicate excess rows within smoke scope",
            rows_scanned=rows,
            scope=scope,
            operation=lambda: self._zero_observation(
                int(
                    self.engine.scalar(
                        "SELECT COALESCE(SUM(n - 1), 0) FROM "
                        "(SELECT count(*) AS n FROM transactions_smoke "
                        "GROUP BY t_dat, customer_id, article_id, price, sales_channel_id "
                        "HAVING count(*) > 1)"
                    )
                ),
                "duplicate_excess_rows",
                "重复数只描述受限交易样本，不外推到全量。",
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )
        self._foreign_key_check(
            "transactions.customer_id.foreign_key",
            "customer_id",
            "customers",
            rows,
            scope,
        )
        self._foreign_key_check(
            "transactions.article_id.foreign_key",
            "article_id",
            "articles",
            rows,
            scope,
        )

    def _foreign_key_check(
        self,
        check_id: str,
        field: str,
        dimension: str,
        rows: int,
        scope: str,
    ) -> None:
        quoted = quote_identifier(field)
        self._add(
            check_id=check_id,
            table="transactions",
            name=f"Smoke foreign-key misses: {field}",
            check_type="foreign_key_candidate",
            severity="warning",
            expected=f"all non-null smoke {field} values match {dimension}.{field}",
            rows_scanned=rows,
            scope=scope,
            operation=lambda: self._zero_observation(
                int(
                    self.engine.scalar(
                        f"SELECT count(*) FROM transactions_smoke t "
                        f"LEFT JOIN {dimension} d ON t.{quoted} = d.{quoted} "
                        f"WHERE t.{quoted} IS NOT NULL AND d.{quoted} IS NULL"
                    )
                ),
                "unmatched_rows",
                "外键结果仅适用于受限 transaction 范围。",
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )

    def _count_zero_check(
        self,
        check_id: str,
        table: str,
        name: str,
        check_type: str,
        severity: str,
        rows: int,
        scope: str,
        sql: str,
        expected: str,
        notes: str = "",
    ) -> None:
        self._add(
            check_id=check_id,
            table=table,
            name=name,
            check_type=check_type,
            severity=severity,
            expected=expected,
            rows_scanned=rows,
            scope=scope,
            operation=lambda: self._zero_observation(
                int(self.engine.scalar(sql)),
                "violation_count",
                notes,
            ),
            evidence="artifacts/stage_0b/audit_results.csv",
        )

    @staticmethod
    def _zero_observation(value: int, label: str, notes: str) -> tuple[bool, Any, str]:
        return value == 0, {label: value}, notes

    def run_all(self) -> list[AuditResult]:
        self.run_direct_probes()
        self.run_schema_checks()
        self.run_articles()
        self.run_customers()
        self.run_transactions()
        return self.results
