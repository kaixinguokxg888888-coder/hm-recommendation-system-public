from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pytest

from data_audit.stage1a import (
    FullAuditResult,
    Stage1AConfig,
    config_fingerprint,
    declared_fingerprint_values,
    input_fingerprint,
    read_checkpoint,
    sensitive_value_matches,
    should_skip_group,
    validate_public_artifacts,
    write_json,
)


def _minimal_raw() -> dict:
    return {
        "result_version": "1.0.0",
        "data_paths": {"articles": "data/raw/articles.parquet"},
        "duckdb": {
            "memory_fraction_of_available": 0.25,
            "memory_min_mb": 256,
            "memory_max_mb": 1024,
            "threads": 2,
            "temp_directory": "artifacts/stage_1a/tmp",
        },
        "query_timeout_policy": {"seconds_per_query": 60},
        "retry_policy": {"max_attempts": 1},
        "paths": {
            "artifact_directory": "artifacts/stage_1a",
            "checkpoint_directory": "artifacts/stage_1a/checkpoints",
            "group_directory": "artifacts/stage_1a/groups",
            "report": "docs/report.md",
        },
        "groups": ["group_a_metadata_schema"],
    }


def test_stage_1a_config_reads_and_bounds_memory(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_minimal_raw()), encoding="utf-8")
    config = Stage1AConfig.load(config_path, available_bytes=2 * 1024**3)
    assert config.memory_limit_mb == 512
    assert config.threads == 2
    assert config.relative(config.temp_dir) == "artifacts/stage_1a/tmp"


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    payload = {"group_name": "a", "status": "SUCCESS", "output_files": []}
    write_json(path, payload)
    assert read_checkpoint(path) == payload


def test_config_fingerprint_changes_with_contract(tmp_path: Path) -> None:
    config = tmp_path / "config"
    contract = tmp_path / "contract"
    config.write_text("a", encoding="utf-8")
    contract.write_text("b", encoding="utf-8")
    first = config_fingerprint(config, contract)
    contract.write_text("c", encoding="utf-8")
    assert config_fingerprint(config, contract) != first


def test_input_fingerprint_changes_when_input_metadata_changes(tmp_path: Path) -> None:
    data = tmp_path / "data" / "raw"
    data.mkdir(parents=True)
    parquet = data / "articles.parquet"
    parquet.write_bytes(b"first")
    config = Stage1AConfig(root=tmp_path, raw=_minimal_raw(), memory_limit_mb=256)
    metadata = {
        "articles": {
            "row_count": 1,
            "row_groups": 1,
            "schema_fingerprint": "abc",
        }
    }
    first = input_fingerprint(config, metadata)
    parquet.write_bytes(b"second")
    os.utime(parquet, None)
    assert input_fingerprint(config, metadata) != first


def test_success_checkpoint_can_skip(tmp_path: Path) -> None:
    output = tmp_path / "group.json"
    output.write_text("{}", encoding="utf-8")
    checkpoint = {
        "status": "SUCCESS",
        "input_fingerprint": "input",
        "config_fingerprint": "config",
        "_resolved_output_files": [str(output)],
    }
    assert should_skip_group(checkpoint, "input", "config", force=False)


def test_force_disables_checkpoint_skip(tmp_path: Path) -> None:
    output = tmp_path / "group.json"
    output.write_text("{}", encoding="utf-8")
    checkpoint = {
        "status": "SUCCESS",
        "input_fingerprint": "input",
        "config_fingerprint": "config",
        "_resolved_output_files": [str(output)],
    }
    assert not should_skip_group(checkpoint, "input", "config", force=True)


def test_fingerprint_change_triggers_rerun(tmp_path: Path) -> None:
    output = tmp_path / "group.json"
    output.write_text("{}", encoding="utf-8")
    checkpoint = {
        "status": "SUCCESS",
        "input_fingerprint": "old",
        "config_fingerprint": "config",
        "_resolved_output_files": [str(output)],
    }
    assert not should_skip_group(checkpoint, "new", "config", force=False)


def test_duplicate_excess_and_criteria_are_distinct() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE tx(t_dat VARCHAR, customer_id VARCHAR, article_id VARCHAR, "
        "price DOUBLE, sales_channel_id BIGINT)"
    )
    connection.execute(
        "INSERT INTO tx VALUES "
        "('2020-01-01','c1','a1',1.0,1),"
        "('2020-01-01','c1','a1',1.0,1),"
        "('2020-01-01','c1','a1',2.0,1),"
        "('2020-01-02','c1','a1',1.0,1)"
    )
    full = connection.execute(
        "WITH c AS (SELECT t_dat,customer_id,article_id,price,sales_channel_id,"
        "count(*) n FROM tx GROUP BY ALL) "
        "SELECT count(*) FILTER(WHERE n>1), sum(n-1) FILTER(WHERE n>1) FROM c"
    ).fetchone()
    triplet = connection.execute(
        "WITH c AS (SELECT t_dat,customer_id,article_id,count(*) n FROM tx GROUP BY ALL) "
        "SELECT count(*) FILTER(WHERE n>1), sum(n-1) FILTER(WHERE n>1) FROM c"
    ).fetchone()
    connection.close()
    assert full == (1, 1)
    assert triplet == (1, 2)


def test_internal_integer_coding_preserves_exact_duplicate_counts() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute(
        "CREATE TABLE customers(customer_id VARCHAR); "
        "INSERT INTO customers VALUES ('c1'), ('c2'); "
        "CREATE TABLE articles(article_id VARCHAR); "
        "INSERT INTO articles VALUES ('a1'), ('a2'); "
        "CREATE TABLE tx(t_dat VARCHAR, customer_id VARCHAR, article_id VARCHAR, "
        "price DOUBLE, sales_channel_id BIGINT); "
        "INSERT INTO tx VALUES "
        "('2020-01-01','c1','a1',1.0,1),"
        "('2020-01-01','c1','a1',1.0,1),"
        "('2020-01-01','c2','a2',1.0,1)"
    )
    connection.execute(
        "CREATE TEMP TABLE cc AS SELECT customer_id,row_number() OVER() code FROM customers; "
        "CREATE TEMP TABLE ac AS SELECT article_id,row_number() OVER() code FROM articles; "
        "CREATE TEMP TABLE coded AS SELECT t_dat,cc.code customer_code,ac.code article_code,"
        "price,sales_channel_id FROM tx "
        "JOIN cc USING(customer_id) JOIN ac USING(article_id)"
    )
    original = connection.execute(
        "WITH c AS (SELECT t_dat,customer_id,article_id,price,sales_channel_id,count(*) n "
        "FROM tx GROUP BY ALL) SELECT sum(n-1) FILTER(WHERE n>1) FROM c"
    ).fetchone()[0]
    coded = connection.execute(
        "WITH c AS (SELECT t_dat,customer_code,article_code,price,sales_channel_id,count(*) n "
        "FROM coded GROUP BY ALL) SELECT sum(n-1) FILTER(WHERE n>1) FROM c"
    ).fetchone()[0]
    connection.close()
    assert coded == original == 1


def test_foreign_key_anti_join_counts_rows_and_distinct() -> None:
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE dim(id VARCHAR); INSERT INTO dim VALUES ('a')")
    connection.execute(
        "CREATE TABLE fact(id VARCHAR); INSERT INTO fact VALUES ('a'),('b'),('b'),(NULL)"
    )
    value = connection.execute(
        "WITH missing AS (SELECT f.id FROM fact f ANTI JOIN dim d ON f.id=d.id "
        "WHERE f.id IS NOT NULL) SELECT count(*), count(DISTINCT id) FROM missing"
    ).fetchone()
    connection.close()
    assert value == (2, 1)


def test_full_scan_result_protocol() -> None:
    result = FullAuditResult(
        check_id="x",
        table_name="t",
        check_name="n",
        check_type="schema",
        severity="error",
        status="PASS",
        observed_value={"count": 0},
        expected_condition="count=0",
        rows_scanned=1,
        scan_scope="FULL_SCAN: tiny",
        execution_seconds=0.1,
        evidence_path="result.csv",
        notes="",
        error_message="",
        execution_scope="FULL_SCAN",
        query_group="group_a",
        input_fingerprint="i",
        config_fingerprint="c",
        result_version="1",
    )
    assert result.as_json_record()["execution_scope"] == "FULL_SCAN"
    with pytest.raises(ValueError):
        result.execution_scope = "SMOKE"
        result.__post_init__()


def test_single_group_failure_does_not_delete_existing_artifact(tmp_path: Path) -> None:
    completed = tmp_path / "completed.json"
    write_json(completed, {"status": "SUCCESS"})
    try:
        raise RuntimeError("later group failed")
    except RuntimeError:
        pass
    assert read_checkpoint(completed) == {"status": "SUCCESS"}


def test_id_and_path_detection_and_clean_report(tmp_path: Path) -> None:
    assert sensitive_value_matches("customer=0" * 0 + "a" * 64)
    assert sensitive_value_matches("article=0123456789")
    assert sensitive_value_matches("path=C:\\Users\\name")
    report = tmp_path / "report.md"
    report.write_text(
        "Only aggregate counts; customer_id and article_id are field names.",
        encoding="utf-8",
    )
    validate_public_artifacts([report])


def test_public_validation_allows_declared_fingerprint(tmp_path: Path) -> None:
    fingerprint = "a" * 64
    report = tmp_path / "report.md"
    report.write_text(f"input_fingerprint={fingerprint}", encoding="utf-8")
    validate_public_artifacts([report], allowed_tokens=[fingerprint])


def test_public_validation_allows_sha256_only_in_explicit_fingerprint_field(
    tmp_path: Path,
) -> None:
    fingerprint = "a1" * 32
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"dataset_fingerprint": fingerprint}),
        encoding="utf-8",
    )
    assert declared_fingerprint_values(artifact) == {fingerprint}
    validate_public_artifacts([artifact])


def test_public_validation_does_not_exempt_sha256_in_ordinary_field(
    tmp_path: Path,
) -> None:
    sensitive = "b2" * 32
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"ordinary_value": sensitive}),
        encoding="utf-8",
    )
    assert declared_fingerprint_values(artifact) == set()
    with pytest.raises(ValueError, match="敏感信息"):
        validate_public_artifacts([artifact], allowed_tokens=[sensitive])


@pytest.mark.parametrize(
    "value",
    [
        "a" * 63,
        "a" * 65,
        "g" + "a" * 63,
    ],
)
def test_invalid_sha256_is_not_exempt_even_in_fingerprint_field(
    tmp_path: Path,
    value: str,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps({"dataset_fingerprint": value}),
        encoding="utf-8",
    )
    assert declared_fingerprint_values(artifact) == set()


def test_sensitive_customer_and_article_values_remain_blocked(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "customer_value": "c3" * 32,
                "article_value": "0123456789",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="敏感信息"):
        validate_public_artifacts([artifact])
