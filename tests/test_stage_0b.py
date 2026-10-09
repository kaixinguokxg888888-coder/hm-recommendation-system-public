from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_audit.checks import (
    candidate_key_duplicate_count,
    date_parse_failure_count,
    schema_contract_diff,
)
from data_audit.config import AuditSettings, conservative_memory_limit_mb, find_project_root
from data_audit.contracts import load_contracts
from data_audit.reporting import (
    RESULT_FIELDS,
    AuditResult,
    redact_sensitive,
    validate_result_protocol,
)


class ScalarEngine:
    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self.connection = connection

    def scalar(self, sql: str, parameters: list[object] | None = None) -> object:
        row = self.connection.execute(sql, parameters or []).fetchone()
        return row[0]


def test_project_root_resolution_from_nested_path(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path


def test_default_settings_use_project_relative_locations() -> None:
    root = find_project_root(Path(__file__))
    settings = AuditSettings.default(root)
    assert settings.contracts_path == root / "configs" / "data_contracts.yaml"
    assert settings.relative(settings.temp_dir) == "artifacts/stage_0b/tmp"
    assert settings.transaction_row_group == 0
    assert settings.transaction_max_rows == 100_000


def test_memory_limit_is_conservative_and_bounded() -> None:
    assert conservative_memory_limit_mb(512 * 1024 * 1024) == 256
    assert conservative_memory_limit_mb(32 * 1024 * 1024 * 1024) == 1024


def test_contract_loading_and_rule_shape() -> None:
    root = find_project_root(Path(__file__))
    contracts = load_contracts(root / "configs" / "data_contracts.yaml")
    assert set(contracts.tables) == {"articles", "customers", "transactions"}
    assert contracts.tables["transactions"].required_fields["t_dat"] == "string"
    assert all(rule.field or rule.fields for table in contracts.tables.values() for rule in table.rules)


def test_contract_file_is_valid_json_compatible_yaml() -> None:
    root = find_project_root(Path(__file__))
    payload = json.loads((root / "configs" / "data_contracts.yaml").read_text(encoding="utf-8"))
    assert payload["contract_version"] == "0.1.0"


def test_schema_contract_diff_detects_missing_and_type_mismatch() -> None:
    missing, mismatched = schema_contract_diff(
        {"customer_id": "string", "age": "int64"},
        {"customer_id": "string", "age": "double", "postal_code": "string"},
    )
    assert missing == ["postal_code"]
    assert mismatched == {"age": {"expected": "double", "actual": "int64"}}


def test_date_parse_rule_counts_only_invalid_non_null_values() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE events(t_dat VARCHAR); "
        "INSERT INTO events VALUES ('2020-01-01'), ('not-a-date'), (NULL)"
    )
    try:
        assert date_parse_failure_count(ScalarEngine(connection), "events", "t_dat") == 1
    finally:
        connection.close()


def test_result_protocol_is_complete_and_stable() -> None:
    result = AuditResult(
        check_id="x",
        table_name="t",
        check_name="name",
        check_type="schema",
        severity="error",
        status="PASS",
        observed_value={"count": 0},
        expected_condition="count = 0",
        rows_scanned=0,
        scan_scope="test",
        execution_seconds=0.01,
        evidence_path="artifact.csv",
    )
    validate_result_protocol(result)
    assert tuple(result.as_record()) == RESULT_FIELDS
    with pytest.raises(ValueError):
        AuditResult(
            check_id="x",
            table_name="t",
            check_name="name",
            check_type="schema",
            severity="error",
            status="INVALID",
            observed_value={},
            expected_condition="x",
            rows_scanned=0,
            scan_scope="test",
            execution_seconds=0,
            evidence_path="x",
        )


def test_sensitive_identifier_redaction_is_recursive() -> None:
    payload = {
        "customer_id": "secret-customer",
        "nested": {"article_id": "secret-article", "count": 2},
        "postal_code": "secret-postal",
    }
    redacted = redact_sensitive(payload)
    assert redacted["customer_id"] == "<redacted>"
    assert redacted["nested"]["article_id"] == "<redacted>"
    assert redacted["postal_code"] == "<redacted>"
    assert "secret" not in json.dumps(redacted)


def test_unique_check_reads_small_temporary_parquet(tmp_path: Path) -> None:
    parquet_path = tmp_path / "tiny.parquet"
    pq.write_table(pa.table({"key": ["a", "a", "b"]}), parquet_path)
    connection = duckdb.connect(":memory:")
    safe_path = parquet_path.as_posix().replace("'", "''")
    connection.execute(
        f"CREATE VIEW tiny AS SELECT * FROM read_parquet('{safe_path}')"
    )
    try:
        assert candidate_key_duplicate_count(ScalarEngine(connection), "tiny", ["key"]) == 1
    finally:
        connection.close()
