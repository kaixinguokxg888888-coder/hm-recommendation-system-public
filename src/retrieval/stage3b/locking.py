from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data_foundation.manifest import file_fingerprint


class Stage3BLockError(ValueError):
    """Stage 3B 一次性 Test 的锁定门禁失败。"""


def lock_fingerprint(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "lock_fingerprint"}
    return hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_test_lock(
    lock_path: Path,
    *,
    config_fingerprint: str,
    source_fingerprint: str,
    stage_3a_lock_fingerprint: str,
    selection_path: Path,
    expected_variant: str | None = None,
    expected_parameters: dict[str, Any] | None = None,
    expected_quotas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not lock_path.exists():
        raise Stage3BLockError("Test 禁止运行：Stage 3B lock 不存在")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    checks = {
        "config_fingerprint": config_fingerprint,
        "source_fingerprint": source_fingerprint,
        "stage_3a_lock_fingerprint": stage_3a_lock_fingerprint,
    }
    for field, expected in checks.items():
        if lock.get(field) != expected:
            raise Stage3BLockError(f"Test 禁止运行：lock {field} 不匹配")
    if lock.get("lock_fingerprint") != lock_fingerprint(lock):
        raise Stage3BLockError("Test 禁止运行：lock 自身 fingerprint 不匹配")
    if not selection_path.exists() or file_fingerprint(selection_path) != lock.get(
        "validation_selection_fingerprint"
    ):
        raise Stage3BLockError("Test 禁止运行：Validation selection 已改变")
    selected = str(lock.get("locked_itemcf_variant"))
    if selected not in {"cosine", "iuf"}:
        raise Stage3BLockError("Test 禁止运行：锁定 ItemCF variant 非法")
    if expected_variant is not None and selected != expected_variant:
        raise Stage3BLockError("Test 禁止运行：ItemCF variant 变化")
    if expected_parameters is not None and lock.get("itemcf_parameters") != expected_parameters:
        raise Stage3BLockError("Test 禁止运行：seed/neighbor/cooccurrence 参数变化")
    if expected_quotas is not None and lock.get("union_quotas") != expected_quotas:
        raise Stage3BLockError("Test 禁止运行：union quota 变化")
    return lock
