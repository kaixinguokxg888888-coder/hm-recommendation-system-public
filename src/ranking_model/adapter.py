from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class ModelInputAdapter:
    feature_set: str
    ordered_features: tuple[str, ...]
    dtypes: tuple[str, ...]
    categorical_features: tuple[str, ...]
    user_segment_mapping: Mapping[str, int]

    @classmethod
    def from_schema(
        cls, rows: Iterable[Mapping[str, Any]], feature_set: str,
        user_segment_mapping: Mapping[str, int],
    ) -> "ModelInputAdapter":
        selected = [
            row for row in rows
            if str(row["model_input_eligible"]).lower() == "true"
            and feature_set in str(row["feature_set"]).split(",")
        ]
        names = tuple(str(row["feature_name"]) for row in selected)
        dtypes = tuple(str(row["dtype"]) for row in selected)
        categorical = tuple(
            name for name in names
            if name == "user_segment" or (name.startswith("article_") and name.endswith("_code"))
        )
        expected = 30 if feature_set == "F0" else 52
        if len(names) != expected or len(names) != len(set(names)):
            raise ValueError(f"{feature_set} feature count/order contract invalid")
        return cls(feature_set, names, dtypes, categorical, dict(user_segment_mapping))

    @property
    def fingerprint(self) -> str:
        payload = self.manifest()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "feature_set": self.feature_set,
            "ordered_feature_names": list(self.ordered_features),
            "dtypes": dict(zip(self.ordered_features, self.dtypes)),
            "categorical_features": list(self.categorical_features),
            "categorical_mapping": {"user_segment": dict(self.user_segment_mapping),
                                    "article_category_codes": "frozen_nonnegative_contiguous_stage_4b3_codes"},
            "null_handling": "preserve_native_missing_no_imputation",
            "boolean_mapping": {"false": 0, "true": 1},
            "matrix_dtype": "float32",
        }

    def convert_table(self, table: pa.Table) -> np.ndarray:
        arrays: list[np.ndarray] = []
        for name, dtype in zip(self.ordered_features, self.dtypes):
            column = table[name].combine_chunks()
            if name == "user_segment":
                values = column.to_pylist()
                try:
                    array = np.asarray([self.user_segment_mapping[str(value)] for value in values], dtype=np.float32)
                except KeyError as exc:
                    raise ValueError(f"unknown user_segment: {exc}") from exc
            elif dtype == "bool":
                array = column.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            else:
                # Arrow null 转为 NaN；不把缺失 recency/source-local/affinity 偷换成 0。
                array = column.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            arrays.append(array)
        return np.column_stack(arrays)


def assert_adapter_deterministic(adapter: ModelInputAdapter, table: pa.Table) -> None:
    left = adapter.convert_table(table)
    right = adapter.convert_table(table)
    if not np.array_equal(left, right, equal_nan=True):
        raise AssertionError("model-input adapter is not deterministic")


def assert_query_group_order(
    query_rows: Sequence[tuple[str, int, int]], group_rows: Sequence[tuple[str, int, int, int]],
) -> None:
    expected = [(snapshot, customer, size) for snapshot, customer, _order, size in group_rows]
    if list(query_rows) != expected:
        raise ValueError("query/group canonical order mismatch")
