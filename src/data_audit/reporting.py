from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RESULT_FIELDS = (
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
)
VALID_STATUSES = {"PASS", "FAIL", "WARN", "ERROR", "NOT_RUN"}
SENSITIVE_FIELDS = {"customer_id", "article_id", "postal_code"}


@dataclass
class AuditResult:
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
    notes: str = ""
    error_message: str = ""

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUSES:
            raise ValueError(f"非法审计状态: {self.status}")

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["observed_value"] = json.dumps(
            redact_sensitive(record["observed_value"]),
            ensure_ascii=False,
            sort_keys=True,
        )
        return record


def redact_sensitive(value: Any, key: str | None = None) -> Any:
    """递归阻止敏感标识符原值进入小型公开产物。"""
    if key in SENSITIVE_FIELDS:
        return "<redacted>"
    if isinstance(value, dict):
        return {item_key: redact_sensitive(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [redact_sensitive(item) for item in value]
    return value


def validate_result_protocol(result: AuditResult) -> None:
    if tuple(result.as_record().keys()) != RESULT_FIELDS:
        raise ValueError("审计结果协议字段不完整或顺序不稳定")


def write_results_csv(results: Iterable[AuditResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            validate_result_protocol(result)
            writer.writerow(result.as_record())


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(redact_sensitive(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
