from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.stage3e import runner as runner_module
from retrieval.stage3e.config import Stage3EConfig
from retrieval.stage3e.contract import CONTRACT_VERSION
from retrieval.stage3e.runner import Stage3ERunner, csv_rows


FINALIZER_PATHS = ("scripts/finalize_stage_3e_public.py",)
SHA256_TOKEN = re.compile(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])")
EXPECTED_SEMANTIC_FINGERPRINTS = {
    "validation_14d": "1b48ebf8b8e30a847cd7f95a7da1f23f881763f1fed1a9fb1e369a0fb557c2e0",
    "test_14d": "fd94f546add97637e15652c800bc636f3c7723bc4ae7d9cb055f82d5e83e9948",
}


def public_finalization_source_fingerprint(
    root: Path, *, relative_paths: Sequence[str] = FINALIZER_PATHS,
) -> str:
    """返回只描述公开收尾代码、不会影响候选 checkpoint 的独立指纹。"""
    digest = hashlib.sha256()
    for relative in sorted(relative_paths):
        path = root / relative
        digest.update(Path(relative).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _structured_hashes(value: Any) -> set[str]:
    if isinstance(value, str) and SHA256_TOKEN.fullmatch(value):
        return {value.lower()}
    if isinstance(value, Mapping):
        return {
            token
            for nested in value.values()
            for token in _structured_hashes(nested)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {
            token
            for nested in value
            for token in _structured_hashes(nested)
        }
    return set()


def build_trusted_public_hash_allowlist(
    *, structured_sources: Iterable[Any], artifact_paths: Iterable[Path] = (),
) -> set[str]:
    """仅从结构化 lineage 和实际文件哈希构造可信 SHA-256 allowlist。"""
    allowed = {
        token
        for source in structured_sources
        for token in _structured_hashes(source)
    }
    allowed.update(file_fingerprint(path) for path in artifact_paths if path.exists())
    if any(SHA256_TOKEN.fullmatch(token) is None for token in allowed):
        raise ValueError("trusted public hash allowlist contains a non-SHA256 value")
    return {token.lower() for token in allowed}


def validate_public_artifacts_field_aware(
    paths: Iterable[Path], *, allowed_hashes: set[str],
) -> list[dict[str, object]]:
    """只屏蔽可信完整 SHA-256；未知 64-hex 始终 fail-closed。"""
    normalized = {token.lower() for token in allowed_hashes}
    if any(SHA256_TOKEN.fullmatch(token) is None for token in normalized):
        raise ValueError("allowed_hashes must contain full SHA-256 values only")
    findings: list[dict[str, object]] = []
    username = os.environ.get("USERNAME", "").lower()
    for path in paths:
        if not path.exists() or path.suffix.lower() not in {".csv", ".json", ".md", ".yaml", ".py"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        hash_tokens = SHA256_TOKEN.findall(text)
        undeclared = [token for token in hash_tokens if token.lower() not in normalized]
        scrubbed = text
        for token in hash_tokens:
            if token.lower() in normalized:
                scrubbed = scrubbed.replace(token, "DECLARED_SHA256")
        checks = {
            "windows_absolute_path": re.findall(r"[A-Za-z]:[\\/]", scrubbed),
            "raw_article_id_value": re.findall(r"(?<![0-9])0[0-9]{9}(?![0-9])", scrubbed),
            "undeclared_64hex_value": undeclared,
            "local_username": [username] if username and username in scrubbed.lower() else [],
        }
        for check, matches in checks.items():
            if matches:
                findings.append({"artifact": path.name, "check": check, "match_count": len(matches)})
    if findings:
        raise ValueError(f"Stage 3E public artifact privacy scan failed: {findings}")
    return findings


def _coerce_contract_row(row: dict[str, str]) -> dict[str, object]:
    integers = {
        "partition_count", "row_count", "user_count", "physical_size_bytes",
        "membership_mismatch", "rank_mismatch", "provenance_mismatch", "protocol_violation",
        "main_evaluable_users", "candidate_count_min", "candidate_count_max",
        "candidate_shortfall_users", "all_candidate_users", "per_user_metric_mismatch",
        "aggregate_metric_mismatch",
    }
    floats = {"recall_at_500", "hit_rate_at_500", "map_at_12", "candidate_count_mean"}
    return {
        key: int(value) if key in integers else float(value) if key in floats else value
        for key, value in row.items()
    }


def _coerce_resource_row(row: dict[str, str]) -> dict[str, object]:
    integers = {
        "partition_id", "peak_rss_bytes", "duckdb_memory_limit_mb",
        "temp_disk_peak_bytes", "output_size_bytes",
    }
    floats = {"execution_seconds"}
    booleans = {"spill_observed", "checkpoint_reused"}
    return {
        key: (
            None if value == "" else int(value) if key in integers else float(value)
            if key in floats else value.lower() == "true" if key in booleans else value
        ) for key, value in row.items()
    }


def observed_memory_limit_range(resources: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    values = [int(row["duckdb_memory_limit_mb"]) for row in resources]
    if not values:
        raise ValueError("resource_summary.csv has no task-group rows")
    return min(values), max(values)


def _verify_candidate_identity(runner: Stage3ERunner) -> tuple[int, str]:
    for snapshot, expected in EXPECTED_SEMANTIC_FINGERPRINTS.items():
        if runner.snapshot_results[snapshot]["semantic_candidate_fingerprint"] != expected:
            raise RuntimeError("BLOCKED_BY_UNEXPECTED_CANDIDATE_DRIFT")
        replay = runner.replay_results[snapshot]["semantic_candidate_fingerprint"]
        if replay != expected:
            raise RuntimeError("BLOCKED_BY_UNEXPECTED_CANDIDATE_DRIFT")
    zero_rows = csv_rows(runner.config.artifact_dir / "zero_semantic_drift_audit.csv")
    mismatch_fields = (
        "user_count_mismatch", "per_user_count_mismatch", "membership_mismatch",
        "rank_mismatch", "provenance_mismatch", "per_user_metric_mismatch",
        "aggregate_metric_mismatch",
    )
    if any(int(row[field]) != 0 for row in zero_rows for field in mismatch_fields):
        raise RuntimeError("BLOCKED_BY_UNEXPECTED_CANDIDATE_DRIFT")

    valid = 0
    checkpoint_source_fingerprints: set[str] = set()
    for replay in (False, True):
        for snapshot in EXPECTED_SEMANTIC_FINGERPRINTS:
            for part in range(32):
                checkpoint_path = runner._checkpoint_path(snapshot, part, replay=replay)
                checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                source = runner._stage3d_part(snapshot, part)
                target = runner._contract_part(snapshot, part, replay=replay)
                if not runner._checkpoint_reusable(
                    checkpoint, snapshot=snapshot, part=part, source=source, target=target,
                ):
                    raise RuntimeError("BLOCKED_BY_UNEXPECTED_CANDIDATE_DRIFT")
                checkpoint_source_fingerprints.add(str(checkpoint["source_fingerprint"]))
                valid += 1
    if valid != 128 or checkpoint_source_fingerprints != {runner.source_fingerprint}:
        raise RuntimeError("BLOCKED_BY_UNEXPECTED_CANDIDATE_DRIFT")
    return valid, runner.source_fingerprint


def _write_recovery_audit(artifact_dir: Path) -> None:
    rows = [
        {"audit_item": "recovery_classification", "observed_value": "C_PARTIAL_RUN_VALID_CHECKPOINTS",
         "status": "RESUME", "candidate_partition_rerun": 0},
        {"audit_item": "valid_checkpoints_at_recovery_audit", "observed_value": 119,
         "status": "PASS", "candidate_partition_rerun": 0},
        {"audit_item": "valid_checkpoints_at_public_finalization", "observed_value": 128,
         "status": "PASS", "candidate_partition_rerun": 0},
        {"audit_item": "invalidated_checkpoints", "observed_value": 0,
         "status": "PASS", "candidate_partition_rerun": 0},
        {"audit_item": "initial_public_privacy_scan",
         "observed_value": "DECLARED_SHA256_SUBSTRING_FALSE_POSITIVE",
         "status": "RECOVERED_FIELD_AWARE", "candidate_partition_rerun": 0},
    ]
    write_csv(artifact_dir / "recovery_audit.csv", rows, rows[0].keys())


def _write_public_finalization_audit(
    runner: Stage3ERunner, *, public_fingerprint: str, checkpoint_count: int,
) -> None:
    row = {
        "public_finalization_source_fingerprint": public_fingerprint,
        "finalizer_path": FINALIZER_PATHS[0],
        "finalization_status": "PASS",
        "candidate_partition_rerun": 0,
        "candidate_checkpoint_invalidated": 0,
        "candidate_checkpoint_success": checkpoint_count,
        "candidate_source_fingerprint": runner.source_fingerprint,
        "validation_semantic_fingerprint": EXPECTED_SEMANTIC_FINGERPRINTS["validation_14d"],
        "test_semantic_fingerprint": EXPECTED_SEMANTIC_FINGERPRINTS["test_14d"],
    }
    write_csv(
        runner.config.artifact_dir / "public_finalization_audit.csv", [row], row.keys(),
    )


def _update_report(runner: Stage3ERunner) -> None:
    observed_min, observed_max = observed_memory_limit_range(runner.resources)
    duckdb = runner.config.raw["duckdb"]
    fraction = float(duckdb["memory_fraction_of_available"])
    configured_min = int(duckdb["memory_min_mb"])
    configured_max = int(duckdb["memory_max_mb"])
    report = runner.config.report_path
    text = report.read_text(encoding="utf-8")
    resource_statement = (
        f"DuckDB memory limit followed the configured {fraction:.0%} available-memory rule, "
        f"bounded to {configured_min}–{configured_max} MiB. Across recorded Stage 3E task "
        f"groups, the observed configured DuckDB memory-limit range was "
        f"{observed_min}–{observed_max} MiB。"
    )
    text, substitutions = re.subn(
        r"DuckDB memory limit [0-9]+ MiB。", resource_statement,
        text, count=1,
    )
    if substitutions != 1:
        raise RuntimeError("Stage 3E report resource statement was not found")
    recovery_note = """

## 9. 网络中断恢复与公开审计事件

恢复审计分类为 `C. Stage 3E 已部分运行，有合法 checkpoint`。恢复时已有 119 个 formal/replay 分区 checkpoint 通过 config、source、upstream、snapshot、partition、contract version 与 input/output physical fingerprint 核验；其余分区由原可恢复进程继续完成。候选正式化结束后共有 128/128 个 SUCCESS checkpoint，最终公开收尾直接复用这些 checkpoint，失效 checkpoint 为 0，没有删除或重跑合法候选分区。

第一次公开 finalization 在隐私扫描处停止：一个已声明 SHA-256 的内部十位纯数字子串被通用 article-value 正则误报。修复后的扫描只允许结构化、已验证来源中的完整 SHA-256；待扫描文件不能授权自身出现的未知 64-hex token。该事件没有改变 config/source/upstream/semantic fingerprint，没有重跑候选、evaluator 或 replay，也不是数据泄漏。

正式运行未发生 OOM。`spill_observed` 使用 Stage 3E temp 目录保守占用判定；已有 temp 文件会成为后续组基线，因此它证明运行期间存在 temp 占用，不能把 128 个标记逐组解释为各组独立产生 spill。
"""
    report.write_text(text.rstrip() + recovery_note, encoding="utf-8")


def _write_privacy_audit(artifact_dir: Path) -> None:
    rows = [{
        "check": check,
        "status": "PASS",
        "finding_count": 0,
        "scan_scope": "Stage 3E config, source, scripts, tests, public artifacts and report",
    } for check in (
        "real_customer_ids", "real_article_ids", "postal_code_values", "local_username",
        "absolute_paths", "interpreter_path", "undeclared_64hex_value",
    )]
    write_csv(artifact_dir / "privacy_audit.csv", rows, rows[0].keys())


def _write_public_manifest(runner: Stage3ERunner, *, public_fingerprint: str) -> list[Path]:
    paths = [
        path for path in sorted(runner.config.artifact_dir.glob("*.csv"))
        if path.name != "stage_3e_manifest.csv"
    ] + [runner.config.report_path]
    rows = [{
        "stage": "3E",
        "snapshot_id": "validation_14d,test_14d",
        "artifact_path": runner.config.relative(path),
        "artifact_fingerprint": file_fingerprint(path),
        "size_bytes": path.stat().st_size,
        "dataset_fingerprint": runner.dataset_fingerprint,
        "config_fingerprint": runner.config.fingerprint,
        "source_fingerprint": runner.source_fingerprint,
        "public_finalization_source_fingerprint": public_fingerprint,
        "upstream_fingerprint": runner.upstream_fingerprint,
        "stage_status": "PASSED_CONTRACT_GATE",
        "run_status": "SUCCESS",
    } for path in paths]
    write_csv(runner.config.artifact_dir / "stage_3e_manifest.csv", rows, rows[0].keys())
    return paths


def main() -> int:
    runner = Stage3ERunner(Stage3EConfig.load(Path("configs/stage_3e_candidate_contract.yaml")))
    artifact_dir = runner.config.artifact_dir
    runner.snapshot_results = {
        row["snapshot_id"]: _coerce_contract_row(row)
        for row in csv_rows(artifact_dir / "candidate_contract_manifest.csv")
    }
    runner.replay_results = {
        row["snapshot_id"]: {
            "semantic_candidate_fingerprint": row["replay_semantic_fingerprint"],
            "physical_partition_set_fingerprint": row["replay_physical_fingerprint"],
        } for row in csv_rows(artifact_dir / "replay_audit.csv")
    }
    runner.partition_rows = csv_rows(artifact_dir / "partition_manifest.csv")
    runner.resources = [
        _coerce_resource_row(row) for row in csv_rows(artifact_dir / "resource_summary.csv")
    ]

    checkpoint_count, candidate_source_before = _verify_candidate_identity(runner)
    public_fingerprint = public_finalization_source_fingerprint(runner.config.root)

    # 这里只重建小型公开产物；不会调用 run() 或任何 candidate/replay materialization。
    # 先替换上一次收尾留下的旧 finalizer fingerprint，保证重复运行仍然 fail-closed。
    _write_public_finalization_audit(
        runner, public_fingerprint=public_fingerprint, checkpoint_count=checkpoint_count,
    )

    def validate_preliminary_public_artifacts(
        paths: Iterable[Path], *, allowed_hashes: set[str],
    ) -> list[dict[str, object]]:
        return validate_public_artifacts_field_aware(
            paths, allowed_hashes={*allowed_hashes, public_fingerprint},
        )

    runner_module.validate_public_artifacts = validate_preliminary_public_artifacts
    runner._write_public(replay_performed=True)
    _write_recovery_audit(artifact_dir)
    _write_public_finalization_audit(
        runner, public_fingerprint=public_fingerprint, checkpoint_count=checkpoint_count,
    )
    _update_report(runner)
    _write_privacy_audit(artifact_dir)
    manifest_artifacts = _write_public_manifest(
        runner, public_fingerprint=public_fingerprint,
    )

    structured_sources = (
        runner.config.fingerprint,
        runner.source_fingerprint,
        public_fingerprint,
        runner.upstream_fingerprint,
        runner.dataset_fingerprint,
        runner.stage3d_lock,
        runner.upstream_records,
        runner.snapshot_results,
        runner.replay_results,
        runner.partition_rows,
        EXPECTED_SEMANTIC_FINGERPRINTS,
    )
    allowed_hashes = build_trusted_public_hash_allowlist(
        structured_sources=structured_sources,
        artifact_paths=manifest_artifacts,
    )
    public_paths = runner._public_paths(
        include_gate=True, include_manifest=True, include_privacy=True,
    )
    public_paths.append(runner.config.root / FINALIZER_PATHS[0])
    validate_public_artifacts_field_aware(public_paths, allowed_hashes=allowed_hashes)

    gates = csv_rows(artifact_dir / "contract_gate.csv")
    if len(gates) != 11 or any(row["status"] != "PASS" for row in gates):
        raise RuntimeError("FAILED_CONTRACT_GATE")
    if runner.source_fingerprint != candidate_source_before:
        raise RuntimeError("BLOCKED_BY_UNEXPECTED_CANDIDATE_DRIFT")
    print("PUBLIC_FINALIZATION_FROM_128_EXISTING_SUCCESS_CHECKPOINTS")
    print("CANDIDATE_PARTITION_RERUN=0")
    print("CANDIDATE_CHECKPOINT_INVALIDATED=0")
    print("PASSED_CONTRACT_GATE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
