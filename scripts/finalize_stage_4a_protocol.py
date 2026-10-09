from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

from data_audit.stage1a import ResourceMonitor
from data_foundation.manifest import file_fingerprint, write_csv
from ranking_protocol.protocol import (
    Stage4AConfig,
    calendar_rows,
    leakage_audit_rows,
    no_ranking_implementation_audit,
    privacy_audit,
    protocol_gate_rows,
    replay_capability_rows,
    replay_contract_rows,
    source_fingerprint,
    validate_upstream,
)


CALENDAR_FIELDS = (
    "snapshot_id", "purpose", "candidate_snapshot_id", "as_of_date", "request_date",
    "history_cutoff", "label_start", "label_end", "label_window_days",
    "candidate_contract_version", "candidate_protocol_version", "is_model_selection",
    "is_test", "is_independent_request_snapshot",
)
REPLAY_FIELDS = (
    "component", "history_cutoff_rule", "label_access_allowed", "parameter_selection_allowed",
    "historical_rebuild_required", "locked_protocol_source", "semantic_drift_allowed",
    "replay_execution_stage", "stage_4a_full_replay_performed",
)
UPSTREAM_FIELDS = (
    "lineage_role", "artifact_path", "expected_fingerprint", "actual_fingerprint", "status",
)
LEAKAGE_FIELDS = ("leakage_gate_id", "status", "evidence")
CAPABILITY_FIELDS = ("component", "status", "evidence", "reason")
GATE_FIELDS = ("gate_id", "mandatory", "status", "reason", "stage_status")
PRIVACY_FIELDS = ("audit_id", "status", "evidence")
RESOURCE_FIELDS = (
    "task", "execution_seconds", "peak_rss_bytes", "peak_temp_bytes", "warning_count",
    "failure_count", "retry_count", "historical_full_candidate_replay",
    "ranking_training_executed", "stage_status",
)
MANIFEST_FIELDS = (
    "stage", "artifact_path", "artifact_fingerprint", "size_bytes", "config_fingerprint",
    "source_fingerprint", "upstream_fingerprint", "stage_status", "run_status",
)


def _write(path: Path, rows: Iterable[dict[str, Any]], fields: Iterable[str]) -> None:
    write_csv(path, rows, fields)


def _aggregate_upstream_fingerprint(rows: list[dict[str, Any]]) -> str:
    canonical = [{
        "lineage_role": row["lineage_role"],
        "artifact_path": row["artifact_path"],
        "actual_fingerprint": row["actual_fingerprint"],
    } for row in rows]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _manifest_rows(
    config: Stage4AConfig, *, source_fp: str, upstream_fp: str,
) -> list[dict[str, Any]]:
    paths = [
        config.root / "README.md",
        config.root / "AGENTS.md",
        config.path,
        config.decision_path,
        config.report_path,
        config.root / "src" / "ranking_protocol" / "__init__.py",
        config.root / "src" / "ranking_protocol" / "protocol.py",
        config.root / "scripts" / "finalize_stage_4a_protocol.py",
        config.root / "tests" / "test_stage_4a.py",
        *sorted(path for path in config.artifact_dir.glob("*.csv") if path.name != "stage_4a_manifest.csv"),
    ]
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Stage 4A formal artifact missing: {config.relative(path)}")
        rows.append({
            "stage": "4A", "artifact_path": config.relative(path),
            "artifact_fingerprint": file_fingerprint(path), "size_bytes": path.stat().st_size,
            "config_fingerprint": config.fingerprint, "source_fingerprint": source_fp,
            "upstream_fingerprint": upstream_fp, "stage_status": "PASSED_PROTOCOL_GATE",
            "run_status": "SUCCESS",
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize Stage 4A static ranking time protocol audits.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage_4a_ranking_time_protocol.yaml"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Stage4AConfig.load(args.config)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    temp_probe = config.artifact_dir / "tmp"
    with ResourceMonitor(temp_probe) as monitor:
        upstream = validate_upstream(config)
        calendar = calendar_rows(config)
        replay = replay_contract_rows(config)
        leakage = leakage_audit_rows(config)
        capabilities = replay_capability_rows(config)
        _write(config.artifact_dir / "upstream_fingerprint_manifest.csv", upstream, UPSTREAM_FIELDS)
        _write(config.artifact_dir / "ranking_calendar.csv", calendar, CALENDAR_FIELDS)
        _write(config.artifact_dir / "replay_contract.csv", replay, REPLAY_FIELDS)
        _write(config.artifact_dir / "leakage_audit.csv", leakage, LEAKAGE_FIELDS)
        _write(config.artifact_dir / "replay_capability_audit.csv", capabilities, CAPABILITY_FIELDS)

        public_base = [
            config.root / "README.md", config.root / "AGENTS.md",
            config.path, config.decision_path, config.report_path,
            config.root / "src" / "ranking_protocol" / "__init__.py",
            config.root / "src" / "ranking_protocol" / "protocol.py",
            config.root / "scripts" / "finalize_stage_4a_protocol.py",
            *sorted(config.artifact_dir.glob("*.csv")),
        ]
        no_ranking = no_ranking_implementation_audit(config)
        privacy = privacy_audit(config, public_base)
        gates = protocol_gate_rows(
            config, upstream, leakage, capabilities, no_ranking=no_ranking, privacy=privacy,
        )
        _write(config.artifact_dir / "protocol_gate.csv", gates, GATE_FIELDS)
        _write(config.artifact_dir / "privacy_audit.csv", [{
            "audit_id": "PUBLIC_STAGE4A_PRIVACY", "status": "PASS" if privacy[0] else "FAIL",
            "evidence": privacy[1],
        }], PRIVACY_FIELDS)

    runtime = round(time.perf_counter() - started, 6)
    resources = [{
        "task": "stage_4a_static_protocol_finalization", "execution_seconds": runtime,
        "peak_rss_bytes": monitor.peak_rss_bytes, "peak_temp_bytes": monitor.peak_temp_bytes,
        "warning_count": 0, "failure_count": 0, "retry_count": 0,
        "historical_full_candidate_replay": False, "ranking_training_executed": False,
        "stage_status": "PASSED_PROTOCOL_GATE",
    }]
    _write(config.artifact_dir / "resource_summary.csv", resources, RESOURCE_FIELDS)

    final_paths = [
        config.root / "README.md", config.root / "AGENTS.md",
        config.path, config.decision_path, config.report_path,
        config.root / "src" / "ranking_protocol" / "__init__.py",
        config.root / "src" / "ranking_protocol" / "protocol.py",
        config.root / "scripts" / "finalize_stage_4a_protocol.py",
        *sorted(config.artifact_dir.glob("*.csv")),
    ]
    final_privacy = privacy_audit(config, final_paths)
    if not final_privacy[0]:
        raise RuntimeError(f"G14_PRIVACY failed: {final_privacy[1]}")

    source_fp = source_fingerprint(config.root)
    upstream_fp = _aggregate_upstream_fingerprint(upstream)
    manifest = _manifest_rows(config, source_fp=source_fp, upstream_fp=upstream_fp)
    _write(config.artifact_dir / "stage_4a_manifest.csv", manifest, MANIFEST_FIELDS)

    if any(row["status"] != "PASS" for row in gates):
        raise RuntimeError("FAILED_PROTOCOL_GATE")
    print(f"CONFIG_FINGERPRINT={config.fingerprint}")
    print(f"SOURCE_FINGERPRINT={source_fp}")
    print(f"UPSTREAM_FINGERPRINT={upstream_fp}")
    print(f"EXECUTION_SECONDS={runtime}")
    print(f"PEAK_RSS_BYTES={monitor.peak_rss_bytes}")
    print("HISTORICAL_FULL_CANDIDATE_REPLAY=NOT_EXECUTED")
    print("PASSED_PROTOCOL_GATE")
    print("STOP. Do not enter Stage 4B.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
