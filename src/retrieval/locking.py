from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from data_foundation.manifest import file_fingerprint


class LockValidationError(ValueError):
    """Test 配置锁定门禁失败。"""


def select_trend_window(rows: Iterable[dict[str, Any]]) -> int:
    values = list(rows)
    if {int(row["window_days"]) for row in values} != {7, 14, 28}:
        raise ValueError("Trend selection 必须完整包含 7/14/28")
    winner = sorted(
        values,
        key=lambda row: (
            -float(row["recall_at_500"]),
            -float(row["map_at_12"]),
            -float(row["candidate_user_coverage"]),
            int(row["window_days"]),
        ),
    )[0]
    return int(winner["window_days"])


def lock_fingerprint(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "lock_fingerprint"}
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_test_lock(
    lock_path: Path,
    *,
    config_fingerprint: str,
    dataset_fingerprint: str,
    source_fingerprint: str,
    allowed_windows: list[int],
    expected_selection_path: Path | None = None,
) -> dict[str, Any]:
    if not lock_path.exists():
        raise LockValidationError("Test 禁止运行：Stage 3A lock 不存在")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "source_fingerprint": source_fingerprint,
    }
    for key, expected in checks.items():
        if lock.get(key) != expected:
            raise LockValidationError(f"Test 禁止运行：lock {key} 不匹配")
    if lock.get("lock_fingerprint") != lock_fingerprint(lock):
        raise LockValidationError("Test 禁止运行：lock 自身 fingerprint 不匹配")
    if int(lock.get("locked_trend_window", -1)) not in allowed_windows:
        raise LockValidationError("Test 禁止运行：锁定 Trend window 不在预注册集合")
    if expected_selection_path is not None:
        expected = lock.get("validation_selection_fingerprint")
        if not expected_selection_path.exists() or file_fingerprint(expected_selection_path) != expected:
            raise LockValidationError("Test 禁止运行：Validation selection artifact 已改变")
    return lock
