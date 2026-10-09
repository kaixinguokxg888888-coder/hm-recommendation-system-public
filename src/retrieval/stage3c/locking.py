from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data_foundation.manifest import file_fingerprint


class Stage3CLockError(ValueError):
    """Stage 3C 锁后 Test 门禁失败。"""


def lock_fingerprint(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "lock_fingerprint"}
    return hashlib.sha256(json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def validate_test_lock(
    path: Path,
    *,
    config_fingerprint: str,
    source_fingerprint: str,
    stage_3a_lock_fingerprint: str,
    stage_3b_lock_fingerprint: str,
) -> dict[str, Any]:
    if not path.exists():
        raise Stage3CLockError("Test 禁止运行：Stage 3C lock 不存在")
    lock = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "config_fingerprint": config_fingerprint,
        "source_fingerprint": source_fingerprint,
        "stage_3a_lock_fingerprint": stage_3a_lock_fingerprint,
        "stage_3b_lock_fingerprint": stage_3b_lock_fingerprint,
    }
    for field, expected in checks.items():
        if lock.get(field) != expected:
            raise Stage3CLockError(f"Test 禁止运行：lock {field} 不匹配")
    if lock.get("lock_fingerprint") != lock_fingerprint(lock):
        raise Stage3CLockError("Test 禁止运行：lock 自身 fingerprint 不匹配")
    for relative, expected in lock.get("validation_artifacts", {}).items():
        artifact = path.parents[2] / relative
        if not artifact.exists() or file_fingerprint(artifact) != expected:
            raise Stage3CLockError("Test 禁止运行：Validation 产物已改变")
    return lock
