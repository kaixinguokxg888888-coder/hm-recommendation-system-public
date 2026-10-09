from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


MANIFEST_FIELDS = (
    "split_id", "split_version", "dataset_fingerprint", "contract_version",
    "feedback_version", "D_max", "history_start", "history_end", "label_start",
    "label_end", "history_rows_A", "history_rows_B", "label_rows_A",
    "label_rows_B", "history_users", "label_users", "history_items",
    "label_items", "evaluable_users", "cold_user_count", "cold_item_count",
    "C_cap", "C_cap_input_rows", "generated_at", "code_commit",
    "config_fingerprint", "output_fingerprints", "as_of_date",
)


def checkpoint_matches(
    checkpoint: dict[str, Any] | None,
    dataset_fingerprint: str,
    config_fingerprint: str,
    output_exists: bool,
    force: bool = False,
) -> bool:
    return bool(
        checkpoint
        and checkpoint.get("status") == "SUCCESS"
        and checkpoint.get("input_fingerprint") == dataset_fingerprint
        and checkpoint.get("config_fingerprint") == config_fingerprint
        and output_exists
        and not force
    )


def manifest_counts_match(
    manifest: dict[str, Any], actual: dict[str, Any]
) -> bool:
    fields = (
        "history_rows_A", "history_rows_B", "label_rows_A", "label_rows_B",
        "history_users", "label_users", "history_items", "label_items",
        "evaluable_users", "cold_user_count", "cold_item_count", "C_cap",
        "C_cap_input_rows",
    )
    return all(int(manifest[field]) == int(actual[field]) for field in fields)


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
