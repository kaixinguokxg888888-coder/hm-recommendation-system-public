from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_SEVERITIES = {"error", "warning", "info"}
REQUIRED_RULE_KEYS = {
    "check_id",
    "table",
    "rule_type",
    "severity",
    "expected",
    "rationale",
}


@dataclass(frozen=True)
class ContractRule:
    check_id: str
    table: str
    rule_type: str
    severity: str
    expected: str
    rationale: str
    field: str | None = None
    fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableContract:
    name: str
    file: str
    required_fields: dict[str, str]
    rules: tuple[ContractRule, ...]
    raw: dict[str, Any]


@dataclass(frozen=True)
class DataContracts:
    version: str
    tables: dict[str, TableContract]


def load_contracts(path: Path) -> DataContracts:
    """契约文件使用 JSON 兼容 YAML，避免为 Stage 0B 引入额外解析依赖。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    tables: dict[str, TableContract] = {}
    for name, raw_table in payload["tables"].items():
        rules = []
        for raw_rule in raw_table["rules"]:
            missing = REQUIRED_RULE_KEYS - raw_rule.keys()
            if missing:
                raise ValueError(f"{raw_rule.get('check_id', name)} 缺少规则字段: {sorted(missing)}")
            if not raw_rule.get("field") and not raw_rule.get("fields"):
                raise ValueError(f"{raw_rule['check_id']} 必须声明 field 或 fields")
            if raw_rule["severity"] not in VALID_SEVERITIES:
                raise ValueError(f"{raw_rule['check_id']} severity 非法")
            if raw_rule["table"] != name:
                raise ValueError(f"{raw_rule['check_id']} table 与所属契约不一致")
            rules.append(
                ContractRule(
                    check_id=raw_rule["check_id"],
                    table=raw_rule["table"],
                    field=raw_rule.get("field"),
                    fields=tuple(raw_rule.get("fields", [])),
                    rule_type=raw_rule["rule_type"],
                    severity=raw_rule["severity"],
                    expected=raw_rule["expected"],
                    rationale=raw_rule["rationale"],
                )
            )
        tables[name] = TableContract(
            name=name,
            file=raw_table["file"],
            required_fields=dict(raw_table["required_fields"]),
            rules=tuple(rules),
            raw=raw_table,
        )
    return DataContracts(version=payload["contract_version"], tables=tables)
