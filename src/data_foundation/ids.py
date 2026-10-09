from __future__ import annotations

import hashlib
from typing import Iterable


KEY_STRATEGY = "full_dimension_lexicographic_dense_key_v1"


def stable_dense_mapping(values: Iterable[str]) -> dict[str, int]:
    unique = sorted(set(values))
    return {value: index for index, value in enumerate(unique)}


def validate_mapping(mapping: dict[str, int], expected_values: Iterable[str]) -> None:
    expected = set(expected_values)
    if set(mapping) != expected:
        raise ValueError("内部键映射覆盖不完整")
    keys = list(mapping.values())
    if len(keys) != len(set(keys)):
        raise ValueError("内部键不是一对一")
    if sorted(keys) != list(range(len(keys))):
        raise ValueError("内部键必须是从 0 开始的稠密整数")


def mapping_fingerprint(mapping: dict[str, int]) -> str:
    payload = "\n".join(f"{raw}\t{mapping[raw]}" for raw in sorted(mapping))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
