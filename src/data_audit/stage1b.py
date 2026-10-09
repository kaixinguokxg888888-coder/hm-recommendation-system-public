from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .stage1a import Stage1AConfig, sensitive_value_matches, write_json


FEEDBACK_SCOPES = {
    "A": "raw transaction rows",
    "B": "binary t_dat + customer_id + article_id interactions",
    "C": "daily interaction count; an intensity proxy, not true quantity",
}


def activity_bucket(value: int) -> str:
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 5:
        return "3-5"
    if value <= 10:
        return "6-10"
    if value <= 20:
        return "11-20"
    if value <= 50:
        return "21-50"
    if value <= 100:
        return "51-100"
    return "100+"


def gini_from_counts(values: list[int]) -> float:
    positive = sorted(value for value in values if value >= 0)
    if not positive or sum(positive) == 0:
        return 0.0
    n = len(positive)
    weighted = sum((index + 1) * value for index, value in enumerate(positive))
    return 2 * weighted / (n * sum(positive)) - (n + 1) / n


def top_share(values: list[int], fraction: float) -> float:
    if not values or sum(values) == 0:
        return 0.0
    count = max(1, int(len(values) * fraction + 0.999999))
    return sum(sorted(values, reverse=True)[:count]) / sum(values)


def feedback_counts(rows: list[tuple[str, str, str]]) -> dict[str, int]:
    raw = len(rows)
    binary = len(set(rows))
    return {
        "A_raw_rows": raw,
        "B_binary_user_item_day": binary,
        "C_daily_strength_sum": raw,
        "same_day_excess": raw - binary,
    }


def cold_start_metrics(
    train_rows: list[tuple[str, str]],
    test_rows: list[tuple[str, str]],
) -> dict[str, float | int]:
    train_users = {row[0] for row in train_rows}
    train_items = {row[1] for row in train_rows}
    test_users = {row[0] for row in test_rows}
    test_items = {row[1] for row in test_rows}
    cold_user_rows = sum(user not in train_users for user, _ in test_rows)
    cold_item_rows = sum(item not in train_items for _, item in test_rows)
    both_rows = sum(
        user not in train_users and item not in train_items for user, item in test_rows
    )
    return {
        "test_users": len(test_users),
        "new_users": len(test_users - train_users),
        "test_items": len(test_items),
        "new_items": len(test_items - train_items),
        "cold_user_row_rate": cold_user_rows / len(test_rows) if test_rows else 0.0,
        "cold_item_row_rate": cold_item_rows / len(test_rows) if test_rows else 0.0,
        "cold_both_row_rate": both_rows / len(test_rows) if test_rows else 0.0,
    }


def profile_checkpoint_can_skip(
    checkpoint: dict[str, Any] | None,
    input_fingerprint: str,
    config_fingerprint: str,
    outputs_exist: bool,
    force: bool,
) -> bool:
    return bool(
        not force
        and outputs_exist
        and checkpoint
        and checkpoint.get("status") == "SUCCESS"
        and checkpoint.get("input_fingerprint") == input_fingerprint
        and checkpoint.get("config_fingerprint") == config_fingerprint
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"不能写入空聚合表: {path.name}")
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_group_output(path: Path, group_name: str, payload: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "group_name": group_name,
            "feedback_scopes": FEEDBACK_SCOPES,
            "payload": payload,
        },
    )


def load_group_output(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["payload"]


def validate_profile_artifacts(
    paths: Iterable[Path],
    allowed_fingerprints: Iterable[str],
) -> None:
    def string_values(value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield str(key)
                yield from string_values(item)
        elif isinstance(value, list):
            for item in value:
                yield from string_values(item)

    allowed = set(allowed_fingerprints)
    findings = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        if path.suffix.lower() == ".json":
            # Scan decoded string values so JSON escapes such as ``:\n`` do
            # not masquerade as Windows drive paths.
            text = "\n".join(string_values(json.loads(text)))
        # ``https://`` contains ``s:/`` and would otherwise look like a
        # Windows drive path to the shared Stage 1A scanner.
        scan_text = re.sub(r"https?://[^\s\"']+", "<public-url>", text)
        matches = sensitive_value_matches(scan_text)
        cleaned = {
            name: [value for value in values if value not in allowed]
            for name, values in matches.items()
        }
        cleaned = {name: values for name, values in cleaned.items() if values}
        if cleaned:
            findings[path.name] = cleaned
    if findings:
        raise ValueError(f"Stage 1B 聚合产物敏感信息检查失败: {findings}")


def stage1b_config(path: Path) -> Stage1AConfig:
    return Stage1AConfig.load(path)
