from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from data_foundation.manifest import file_fingerprint


class Stage3DLockError(ValueError):
    """Stage 3D 锁后 Test 门禁失败。"""


def lock_fingerprint(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "lock_fingerprint"}
    return hashlib.sha256(json.dumps(clean, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def validate_test_lock(
    path: Path,
    *,
    config_fingerprint: str,
    source_fingerprint: str,
    upstream_lock_fingerprints: dict[str, str],
    requested_model: str | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise Stage3DLockError("Test 禁止运行：Stage 3D lock 不存在")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if not lock.get("test_authorized") or lock.get("stage_status") != "PASSED_VALUE_GATE":
        raise Stage3DLockError("Test 禁止运行：Validation value gate 未通过")
    checks = {"config_fingerprint": config_fingerprint, "source_fingerprint": source_fingerprint,
              **upstream_lock_fingerprints}
    for field, expected in checks.items():
        if lock.get(field) != expected:
            raise Stage3DLockError(f"Test 禁止运行：lock {field} 不匹配")
    if requested_model is not None and requested_model != lock.get("locked_model"):
        raise Stage3DLockError("Test 禁止运行：请求了非锁定模型")
    if lock.get("lock_fingerprint") != lock_fingerprint(lock):
        raise Stage3DLockError("Test 禁止运行：lock 自身 fingerprint 不匹配")
    final_fingerprint = lock.get("final_model_checkpoint_fingerprint")
    if (not isinstance(final_fingerprint, str)
            or lock.get("model_checkpoint_fingerprint") != final_fingerprint
            or lock.get("embedding_source_checkpoint_fingerprint") != final_fingerprint
            or lock.get("candidate_source_model_fingerprint") != final_fingerprint
            or lock.get("final_refit_reinitialized") is not True
            or int(lock.get("final_refit_internal_dev_examples", -1)) != 0
            or int(lock.get("final_refit_training_examples", 0)) <= 0
            or int(lock.get("locked_epoch", 0)) <= 0):
        raise Stage3DLockError("Test 禁止运行：Validation full-refit lineage 不完整或不一致")
    root = path.parents[2]
    for relative, expected in lock.get("validation_artifacts", {}).items():
        artifact = root / relative
        if not artifact.exists() or file_fingerprint(artifact) != expected:
            raise Stage3DLockError("Test 禁止运行：Validation 选择产物已改变")
    return lock
