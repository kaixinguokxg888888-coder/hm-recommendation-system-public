from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import duckdb
import psutil

from .config import find_project_root
from .reporting import redact_sensitive


FULL_RESULT_FIELDS = (
    "check_id",
    "table_name",
    "check_name",
    "check_type",
    "severity",
    "status",
    "observed_value",
    "expected_condition",
    "rows_scanned",
    "scan_scope",
    "execution_seconds",
    "evidence_path",
    "notes",
    "error_message",
    "execution_scope",
    "query_group",
    "input_fingerprint",
    "config_fingerprint",
    "result_version",
)
VALID_STATUSES = {"PASS", "FAIL", "WARN", "ERROR", "NOT_RUN"}
PUBLIC_SENSITIVE_PATTERNS = {
    "windows_absolute_path": re.compile(r"[A-Za-z]:[\\/]"),
    "customer_or_postal_value": re.compile(r"\b[0-9a-f]{64}\b"),
    "article_value": re.compile(r"\b0[0-9]{9}\b"),
}
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
# 只有仓库正式协议中明确承载 fingerprint 的字段才允许豁免 64 位十六进制值。
# 不使用“字段名包含 fingerprint”这类宽松规则，避免未知字段静默绕过敏感扫描。
PUBLIC_FINGERPRINT_FIELDS = frozenset(
    {
        "artifact_fingerprint",
        "after_fingerprint",
        "before_fingerprint",
        "config_fingerprint",
        "dataset_fingerprint",
        "file_fingerprint",
        "fingerprint",
        "ground_truth_fingerprint",
        "input_fingerprint",
        "lock_fingerprint",
        "mapping_fingerprint",
        "output_fingerprint",
        "output_fingerprints",
        "private_feedback_fingerprint",
        "run_fingerprint",
        "schema_fingerprint",
        "source_fingerprint",
        "validation_metrics_fingerprint",
        "validation_selection_fingerprint",
    }
)


def utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json_yaml(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Stage1AConfig:
    root: Path
    raw: dict[str, Any]
    memory_limit_mb: int

    @classmethod
    def load(cls, path: Path, available_bytes: int | None = None) -> "Stage1AConfig":
        root = find_project_root(path)
        raw = load_json_yaml(path)
        duck = raw["duckdb"]
        available = available_bytes or psutil.virtual_memory().available
        calculated = int(
            available * float(duck["memory_fraction_of_available"]) / (1024 * 1024)
        )
        memory_limit = max(
            int(duck["memory_min_mb"]),
            min(int(duck["memory_max_mb"]), calculated),
        )
        return cls(root=root, raw=raw, memory_limit_mb=memory_limit)

    @property
    def result_version(self) -> str:
        return str(self.raw["result_version"])

    @property
    def groups(self) -> list[str]:
        return list(self.raw["groups"])

    @property
    def threads(self) -> int:
        return int(self.raw["duckdb"]["threads"])

    @property
    def timeout_seconds(self) -> int:
        return int(self.raw["query_timeout_policy"]["seconds_per_query"])

    @property
    def max_attempts(self) -> int:
        return int(self.raw["retry_policy"]["max_attempts"])

    def resolve(self, relative_path: str) -> Path:
        return (self.root / relative_path).resolve()

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    @property
    def data_paths(self) -> dict[str, Path]:
        return {
            table: self.resolve(path)
            for table, path in self.raw["data_paths"].items()
        }

    @property
    def artifact_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["artifact_directory"])

    @property
    def checkpoint_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["checkpoint_directory"])

    @property
    def group_dir(self) -> Path:
        return self.resolve(self.raw["paths"]["group_directory"])

    @property
    def temp_dir(self) -> Path:
        return self.resolve(self.raw["duckdb"]["temp_directory"])

    @property
    def report_path(self) -> Path:
        return self.resolve(self.raw["paths"]["report"])


def schema_fingerprint_text(schema_text: str) -> str:
    return sha256_bytes(schema_text.encode("utf-8"))


def input_fingerprint(config: Stage1AConfig, parquet_metadata: dict[str, Any]) -> str:
    """由相对路径、大小、mtime、行数、row group 和 schema 共同构成输入指纹。"""
    payload = {}
    for table, path in config.data_paths.items():
        stat = path.stat()
        metadata = parquet_metadata[table]
        payload[table] = {
            "path": config.relative(path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "row_count": metadata["row_count"],
            "row_groups": metadata["row_groups"],
            "schema_fingerprint": metadata["schema_fingerprint"],
        }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


def config_fingerprint(config_path: Path, contracts_path: Path) -> str:
    payload = config_path.read_bytes() + b"\0" + contracts_path.read_bytes()
    return sha256_bytes(payload)


@dataclass
class FullAuditResult:
    check_id: str
    table_name: str
    check_name: str
    check_type: str
    severity: str
    status: str
    observed_value: Any
    expected_condition: str
    rows_scanned: int
    scan_scope: str
    execution_seconds: float
    evidence_path: str
    notes: str
    error_message: str
    execution_scope: str
    query_group: str
    input_fingerprint: str
    config_fingerprint: str
    result_version: str

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"非法状态: {self.status}")
        if self.execution_scope != "FULL_SCAN":
            raise ValueError("Stage 1A 正式结果必须标记 FULL_SCAN")

    def as_json_record(self) -> dict[str, Any]:
        return redact_sensitive(asdict(self))

    def as_csv_record(self) -> dict[str, Any]:
        record = self.as_json_record()
        record["observed_value"] = json.dumps(
            record["observed_value"], ensure_ascii=False, sort_keys=True
        )
        return record

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> "FullAuditResult":
        return cls(**value)


def write_results_csv(results: Iterable[FullAuditResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FULL_RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            record = result.as_csv_record()
            if tuple(record.keys()) != FULL_RESULT_FIELDS:
                raise ValueError("Stage 1A 结果协议字段不稳定")
            writer.writerow(record)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(redact_sensitive(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def should_skip_group(
    checkpoint: dict[str, Any] | None,
    current_input_fingerprint: str,
    current_config_fingerprint: str,
    force: bool,
) -> bool:
    return bool(
        not force
        and checkpoint
        and checkpoint.get("status") == "SUCCESS"
        and checkpoint.get("input_fingerprint") == current_input_fingerprint
        and checkpoint.get("config_fingerprint") == current_config_fingerprint
        and all(Path(path).exists() for path in checkpoint.get("_resolved_output_files", []))
    )


class ResourceMonitor:
    def __init__(self, temp_dir: Path) -> None:
        self.temp_dir = temp_dir
        self.process = psutil.Process()
        self.peak_rss_bytes = self.process.memory_info().rss
        self.peak_temp_bytes = self._temp_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _temp_bytes(self) -> int:
        total = 0
        for path in self.temp_dir.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                # DuckDB 会并发创建/删除 spill 文件；遍历与 stat 之间消失是正常竞态。
                continue
        return total

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)
            self.peak_temp_bytes = max(self.peak_temp_bytes, self._temp_bytes())

    def __enter__(self) -> "ResourceMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)
        self.peak_temp_bytes = max(self.peak_temp_bytes, self._temp_bytes())


class QueryTimeoutError(TimeoutError):
    pass


class Stage1AEngine:
    def __init__(self, config: Stage1AConfig) -> None:
        self.config = config
        self.connection = duckdb.connect(database=":memory:")
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        self._configure()

    def _configure(self) -> None:
        duck = self.config.raw["duckdb"]
        temp_path = self.config.temp_dir.as_posix().replace("'", "''")
        self.connection.execute(f"SET temp_directory = '{temp_path}'")
        self.connection.execute(f"SET memory_limit = '{self.config.memory_limit_mb}MB'")
        self.connection.execute(f"SET threads = {self.config.threads}")
        preserve = "true" if duck["preserve_insertion_order"] else "false"
        self.connection.execute(f"SET preserve_insertion_order = {preserve}")
        try:
            max_temp = str(duck["max_temp_directory_size"]).replace("'", "''")
            self.connection.execute(f"SET max_temp_directory_size = '{max_temp}'")
        except duckdb.Error as exc:
            raise RuntimeError(
                "当前 DuckDB 不支持配置的 max_temp_directory_size"
            ) from exc
        for table, path in self.config.data_paths.items():
            sql_path = path.as_posix().replace("'", "''")
            self.connection.execute(
                f"CREATE VIEW {table} AS SELECT * FROM read_parquet('{sql_path}')"
            )

    def execute(
        self, sql: str, parameters: list[Any] | None = None
    ) -> duckdb.DuckDBPyConnection:
        timed_out = threading.Event()

        def interrupt() -> None:
            timed_out.set()
            self.connection.interrupt()

        timer = threading.Timer(self.config.timeout_seconds, interrupt)
        timer.daemon = True
        timer.start()
        try:
            return self.connection.execute(sql, parameters or [])
        except duckdb.Error as exc:
            if timed_out.is_set():
                raise QueryTimeoutError(
                    f"查询超过 {self.config.timeout_seconds} 秒并被中断"
                ) from exc
            raise
        finally:
            timer.cancel()

    def row(self, sql: str) -> tuple[Any, ...]:
        value = self.execute(sql).fetchone()
        if value is None:
            raise ValueError("查询未返回结果")
        return value

    def rows(self, sql: str) -> list[tuple[Any, ...]]:
        return self.execute(sql).fetchall()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Stage1AEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class GroupContext:
    def __init__(
        self,
        *,
        group_name: str,
        engine: Stage1AEngine,
        config: Stage1AConfig,
        input_fp: str,
        config_fp: str,
        group_output_path: Path,
    ) -> None:
        self.group_name = group_name
        self.engine = engine
        self.config = config
        self.input_fp = input_fp
        self.config_fp = config_fp
        self.group_output_path = group_output_path
        self.results: list[FullAuditResult] = []

    def _public_error(self, exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}".replace(str(self.config.root), ".")

    def add_check(
        self,
        *,
        check_id: str,
        table: str,
        check_name: str,
        check_type: str,
        severity: str,
        expected: str,
        rows_scanned: int,
        scan_scope: str,
        operation: Callable[[], tuple[bool, Any, str]],
        evidence_path: str,
    ) -> FullAuditResult:
        started = time.perf_counter()
        try:
            passed, observed, notes = operation()
            status = "PASS" if passed else ("WARN" if severity == "warning" else "FAIL")
            error_message = ""
        except Exception as exc:
            status = "ERROR"
            observed = None
            notes = "检查异常；已保留此前结果并继续同组其他检查。"
            error_message = self._public_error(exc)
        result = FullAuditResult(
            check_id=check_id,
            table_name=table,
            check_name=check_name,
            check_type=check_type,
            severity=severity,
            status=status,
            observed_value=observed,
            expected_condition=expected,
            rows_scanned=rows_scanned,
            scan_scope=scan_scope,
            execution_seconds=round(time.perf_counter() - started, 6),
            evidence_path=evidence_path,
            notes=notes,
            error_message=error_message,
            execution_scope="FULL_SCAN",
            query_group=self.group_name,
            input_fingerprint=self.input_fp,
            config_fingerprint=self.config_fp,
            result_version=self.config.result_version,
        )
        self.results.append(result)
        self.persist()
        return result

    def persist(self) -> None:
        write_json(
            self.group_output_path,
            {
                "group_name": self.group_name,
                "results": [result.as_json_record() for result in self.results],
            },
        )


def load_group_results(path: Path) -> list[FullAuditResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [FullAuditResult.from_record(record) for record in payload["results"]]


def output_sizes(paths: Iterable[Path], root: Path) -> dict[str, int]:
    return {
        path.resolve().relative_to(root.resolve()).as_posix(): path.stat().st_size
        for path in paths
        if path.exists()
    }


def sensitive_value_matches(text: str) -> dict[str, list[str]]:
    return {
        name: sorted(set(pattern.findall(text)))
        for name, pattern in PUBLIC_SENSITIVE_PATTERNS.items()
        if pattern.search(text)
    }


def _sha256_values(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {
            item
            for child in value.values()
            for item in _sha256_values(child)
        }
    if isinstance(value, list):
        return {
            item
            for child in value
            for item in _sha256_values(child)
        }
    text = str(value)
    return {text} if SHA256_PATTERN.fullmatch(text) else set()


def _structured_fingerprint_values(value: Any) -> set[str]:
    if isinstance(value, dict):
        allowed: set[str] = set()
        for field, child in value.items():
            if field in PUBLIC_FINGERPRINT_FIELDS:
                allowed.update(_sha256_values(child))
            else:
                allowed.update(_structured_fingerprint_values(child))
        return allowed
    if isinstance(value, list):
        return {
            item
            for child in value
            for item in _structured_fingerprint_values(child)
        }
    return set()


def declared_fingerprint_values(path: Path) -> set[str]:
    """提取合法字段中的完整 SHA-256；普通/未知字段不会获得豁免。"""
    text = path.read_text(encoding="utf-8-sig")
    allowed: set[str] = set()
    if path.suffix.lower() == ".json":
        try:
            allowed.update(_structured_fingerprint_values(json.loads(text)))
        except json.JSONDecodeError:
            pass
    elif path.suffix.lower() == ".csv":
        try:
            for row in csv.DictReader(text.splitlines()):
                for field, raw_value in row.items():
                    if field not in PUBLIC_FINGERPRINT_FIELDS or raw_value is None:
                        continue
                    try:
                        value = json.loads(raw_value)
                    except json.JSONDecodeError:
                        value = raw_value
                    allowed.update(_sha256_values(value))
        except csv.Error:
            pass

    # Markdown、日志式 key=value 与人类可读报告仍要求显式字段名。
    field_names = "|".join(
        re.escape(field).replace("_", r"[_ -]")
        for field in sorted(PUBLIC_FINGERPRINT_FIELDS, key=len, reverse=True)
    )
    pattern = re.compile(
        rf"(?i)\b(?:{field_names})\b\s*[:=：]\s*[`\"']?"
        r"([0-9a-f]{64})(?![0-9a-f])"
    )
    allowed.update(match.group(1) for match in pattern.finditer(text))
    return allowed


def validate_public_artifacts(
    paths: Iterable[Path], allowed_tokens: Iterable[str] = ()
) -> None:
    # 保留参数以兼容既有调用，但它不能单独授予豁免；值还必须实际位于
    # 显式白名单字段中并满足完整 SHA-256 格式。
    requested = set(allowed_tokens)
    findings = {}
    for path in paths:
        if not path.exists():
            continue
        declared = declared_fingerprint_values(path)
        allowed = declared | (requested & declared)
        matches = sensitive_value_matches(path.read_text(encoding="utf-8-sig"))
        matches = {
            name: [value for value in values if value not in allowed]
            for name, values in matches.items()
        }
        matches = {name: values for name, values in matches.items() if values}
        if matches:
            findings[path.name] = matches
    if findings:
        raise ValueError(f"公开产物敏感信息检查失败: {findings}")
