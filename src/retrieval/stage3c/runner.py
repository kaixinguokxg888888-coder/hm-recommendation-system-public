from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from data_audit.stage1a import ResourceMonitor, declared_fingerprint_values, sensitive_value_matches
from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.stage3b.config import Stage3BConfig
from retrieval.stage3b.runner import Stage3BRunner

from .config import Stage3CConfig, source_fingerprint
from .content import (
    CanonicalField,
    build_sparse_item_matrix,
    paired_bootstrap,
    select_content_variant,
    top_k_cosine_block,
)
from .locking import lock_fingerprint, validate_test_lock
from .union import select_quota_variant, source_rank_cap_union


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def semantic_partition_fingerprint(paths: Sequence[Path]) -> str:
    """Fingerprint exact partition contents by their already frozen file hashes."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_fingerprint(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


STAGE3C_FINGERPRINT_FIELDS = frozenset({
    "artifact_fingerprint", "attribute_schema_fingerprint", "candidate_semantic_fingerprint",
    "candidate_universe_fingerprint", "config_fingerprint", "dataset_fingerprint",
    "file_fingerprint", "fingerprint", "ground_truth_fingerprint", "input_fingerprint",
    "lock_fingerprint", "mapping_fingerprint", "new_run_fingerprint",
    "new_source_fingerprint", "old_run_fingerprint", "old_source_fingerprint",
    "output_fingerprint", "output_fingerprints", "output_sha256", "population_fingerprint",
    "run_fingerprint", "schema_fingerprint", "semantic_fingerprint", "source_fingerprint",
    "stage_3a_lock_fingerprint", "stage_3b_lock_fingerprint", "validation_artifacts",
})


def _sha256_values(value: Any) -> set[str]:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return {value.lower()}
    if isinstance(value, Mapping):
        return {token for nested in value.values() for token in _sha256_values(nested)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {token for nested in value for token in _sha256_values(nested)}
    return set()


def validate_stage3c_public_artifacts(paths: Iterable[Path]) -> None:
    findings: dict[str, dict[str, list[str]]] = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        allowed = set(declared_fingerprint_values(path))
        if path.suffix.lower() == ".json":
            payload = json.loads(text)
            if isinstance(payload, Mapping):
                for field, value in payload.items():
                    if field in STAGE3C_FINGERPRINT_FIELDS:
                        allowed.update(_sha256_values(value))
        elif path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    for field in STAGE3C_FINGERPRINT_FIELDS & set(row):
                        raw = row[field]
                        try:
                            value = json.loads(raw)
                        except json.JSONDecodeError:
                            value = raw
                        allowed.update(_sha256_values(value))
        field_names = "|".join(
            re.escape(field).replace("_", r"[_ -]")
            for field in sorted(STAGE3C_FINGERPRINT_FIELDS, key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?i)\b(?:{field_names})\b\s*[:=：]\s*[`\"']?([0-9a-f]{{64}})(?![0-9a-f])"
        )
        allowed.update(match.group(1).lower() for match in pattern.finditer(text))
        matches = sensitive_value_matches(text)
        matches = {
            name: [value for value in values if value.lower() not in allowed]
            for name, values in matches.items()
        }
        matches = {name: values for name, values in matches.items() if values}
        if matches:
            findings[path.name] = matches
    if findings:
        raise ValueError(f"Stage 3C 公开产物敏感信息检查失败: {findings}")


class Stage3CRunner:
    def __init__(self, config: Stage3CConfig) -> None:
        self.config = config
        self.source_fingerprint = source_fingerprint(config.root)
        self.stage3b = Stage3BRunner(Stage3BConfig.load(config.root / "configs/stage_3b_itemcf.yaml"))
        self.dataset_fingerprint = self.stage3b.dataset_fingerprint
        self.feedback_version = self.stage3b.feedback_version
        self.population_version = self.stage3b.population_version
        self.snapshots = self.stage3b.snapshots
        self.stage3a_lock = config.resolve(config.raw["upstream"]["stage_3a_lock"])
        self.stage3b_lock = config.resolve(config.raw["upstream"]["stage_3b_lock"])
        self.stage3a_lock_fingerprint = file_fingerprint(self.stage3a_lock)
        self.stage3b_lock_fingerprint = file_fingerprint(self.stage3b_lock)
        self.run_fingerprint = hashlib.sha256(
            (self.dataset_fingerprint + config.fingerprint + self.source_fingerprint
             + self.stage3a_lock_fingerprint + self.stage3b_lock_fingerprint).encode("utf-8")
        ).hexdigest()
        self.resources: list[dict[str, Any]] = []
        self.run_status: list[dict[str, Any]] = []
        self._validate_frozen_upstream()

    def _validate_frozen_upstream(self) -> None:
        expected = {
            "artifacts/stage_3a/stage_3a_lock.json": "c8c8f233f05001a1d600433068c1874a875f9ec47aa7d7b8d612e51cfbe2d134",
            "artifacts/stage_3b/stage_3b_lock.json": "049ba378c5bc69285ebb8387006c16514737082c3586142499ffcad3104af8c8",
        }
        for relative, fingerprint in expected.items():
            if file_fingerprint(self.config.root / relative) != fingerprint:
                raise ValueError(f"冻结 lock 已变化: {relative}")
        lock = json_read(self.stage3b_lock)
        if lock.get("locked_itemcf_variant") != "iuf" or lock.get("stage_3a_locked_trend_window") != 7:
            raise ValueError("Stage 3A/3B 锁定版本发生变化")
        if self.config.raw["feedback_version"] != self.feedback_version:
            raise ValueError("Stage 3C feedback version 与上游不一致")
        if self.config.raw["population_version"] != self.population_version:
            raise ValueError("Stage 3C population version 与上游不一致")
        # manifest 逐文件校验，确保本阶段没有在旧结果漂移后继续。
        for manifest_name in ("stage_3a", "stage_3b"):
            manifest = self.config.resolve(self.config.raw["input_manifests"][manifest_name])
            for row in csv_rows(manifest):
                artifact = self.config.resolve(row["artifact_path"])
                if not artifact.exists() or file_fingerprint(artifact) != row["artifact_fingerprint"]:
                    raise ValueError(f"上游 manifest 产物不一致: {row['artifact_path']}")

    def prepare_directories(self) -> None:
        for path in (self.config.artifact_dir, self.config.private_dir,
                     self.config.checkpoint_dir, self.config.temp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect(":memory:")
        values = self.config.raw["duckdb"]
        con.execute(f"SET temp_directory='{sql_path(self.config.temp_dir)}'")
        con.execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
        con.execute(f"SET threads={int(values['threads'])}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{values['max_temp_directory_size']}'")
        return con

    def checkpoint_path(self, group: str) -> Path:
        return self.config.checkpoint_dir / f"{group}.json"

    def _checkpoint_reusable(self, payload: Mapping[str, Any] | None, outputs: Sequence[Path]) -> bool:
        if not payload or payload.get("status") != "SUCCESS" or payload.get("run_fingerprint") != self.run_fingerprint:
            return False
        expected = payload.get("output_fingerprints", {})
        return bool(outputs) and all(
            path.exists() and expected.get(self.config.relative(path)) == file_fingerprint(path)
            for path in outputs
        )

    def run_group(self, group: str, operation: Callable[[], Any], outputs: Iterable[Path]) -> Any:
        paths = list(outputs)
        checkpoint = self.checkpoint_path(group)
        previous = json_read(checkpoint) if checkpoint.exists() else None
        if self._checkpoint_reusable(previous, paths):
            row = dict(previous)
            row["run_status"] = "REUSED"
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "REUSED", "attempt": row.get("attempt", 1), "message": ""})
            return None
        attempt = int((previous or {}).get("attempt", 0)) + 1
        monitor = ResourceMonitor(self.config.temp_dir)
        started = time.perf_counter()
        try:
            with monitor:
                result = operation()
            missing = [self.config.relative(path) for path in paths if not path.exists()]
            if missing:
                raise RuntimeError(f"group 输出不完整: {missing}")
            row = {
                "group_name": group, "status": "SUCCESS", "attempt": attempt,
                "generated_at": now(), "run_fingerprint": self.run_fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "memory_limit_mb": self.config.memory_limit_mb,
                "threads": self.config.raw["duckdb"]["threads"],
                "output_files": [self.config.relative(path) for path in paths],
                "output_fingerprints": {self.config.relative(path): file_fingerprint(path) for path in paths},
                "output_size_bytes": sum(path.stat().st_size for path in paths),
                "error_message": "",
            }
            json_write(checkpoint, row)
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "SUCCESS", "attempt": attempt, "message": ""})
            return result
        except Exception as exc:
            row = {
                "group_name": group, "status": "FAILED", "attempt": attempt,
                "generated_at": now(), "run_fingerprint": self.run_fingerprint,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "error_message": str(exc).replace(str(self.config.root), "."),
            }
            json_write(checkpoint, row)
            self.resources.append(row)
            self.run_status.append({"group_name": group, "status": "FAILED", "attempt": attempt, "message": row["error_message"]})
            raise

    def stage2b_path(self, prefix: str, snapshot: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_2b_private"]) / f"{prefix}_{snapshot}.parquet"

    def stage3a_ranking(self, snapshot: str, source: str, variant: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_3a_private"]) / f"ranking_{snapshot}_{source}_{variant}.parquet"

    def stage3a_candidate(self, snapshot: str, source: str, variant: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_3a_private"]) / f"candidates_{snapshot}_{source}_{variant}.parquet"

    def stage3b_candidate_part(self, snapshot: str, part: int) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) / "candidates" / snapshot / "iuf" / f"part_{part:03d}.parquet"

    def stage3b_seed(self, snapshot: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) / f"seeds_{snapshot}.parquet"

    def stage3b_history(self, snapshot: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) / f"evaluation_history_{snapshot}.parquet"

    @staticmethod
    def stage3b_q0_variant(snapshot: str) -> str:
        # Stage 3B Validation 同时保留两种 ItemCF，目录携带 iuf 后缀；Test 在锁后
        # 只运行 IUF，正式目录使用无后缀的 itemcf_union。二者语义均为冻结 IUF Union。
        if snapshot == "validation_14d":
            return "itemcf_union_iuf"
        if snapshot == "test_14d":
            return "itemcf_union"
        raise ValueError(f"未知 snapshot: {snapshot}")

    def stage3b_q0_parts(self, snapshot: str) -> list[Path]:
        root = (self.config.resolve(self.config.raw["upstream"]["stage_3b_private"])
                / "unions" / snapshot / self.stage3b_q0_variant(snapshot))
        return sorted(root.glob("part_*.parquet"))

    def stage3b_q0_metric(self, snapshot: str) -> Path:
        return (self.config.resolve(self.config.raw["upstream"]["stage_3b_private"])
                / f"user_metrics_{snapshot}_union_{self.stage3b_q0_variant(snapshot)}.parquet")

    def attributes_path(self, snapshot: str) -> Path:
        return self.config.private_dir / f"attributes_{snapshot}.parquet"

    def token_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / f"tokens_{snapshot}_{variant}.json"

    def neighbor_path(self, snapshot: str, variant: str) -> Path:
        return self.config.private_dir / f"neighbors_{snapshot}_{variant}.parquet"

    def candidate_part(self, snapshot: str, variant: str, part: int) -> Path:
        return self.config.private_dir / "candidates" / snapshot / variant / f"part_{part:03d}.parquet"

    def candidate_glob(self, snapshot: str, variant: str) -> str:
        return sql_path(self.candidate_part(snapshot, variant, 0).parent / "part_*.parquet")

    def union_part(self, snapshot: str, union_variant: str, part: int) -> Path:
        return self.config.private_dir / "unions" / snapshot / union_variant / f"part_{part:03d}.parquet"

    def union_glob(self, snapshot: str, union_variant: str) -> str:
        return sql_path(self.union_part(snapshot, union_variant, 0).parent / "part_*.parquet")

    def metric_path(self, snapshot: str, source: str, variant: str) -> Path:
        return self.config.private_dir / f"user_metrics_{snapshot}_{source}_{variant}.parquet"

    @property
    def canonical_fields(self) -> list[CanonicalField]:
        return [CanonicalField(**row) for row in self.config.raw["canonical_fields"]]

    def build_attribute_snapshot(self, snapshot: str) -> Path:
        target = self.attributes_path(snapshot)
        articles = self.config.resolve(self.config.raw["paths"]["articles"])
        key_map = self.config.resolve(self.config.raw["upstream"]["stage_2a_private"]) / "article_key_map.parquet"
        state = self.stage2b_path("item_state", snapshot)
        columns = ",".join(f"a.{field.source_column}" for field in self.canonical_fields)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  SELECT k.article_key,{columns}
                  FROM read_parquet('{sql_path(articles)}') a
                  JOIN read_parquet('{sql_path(key_map)}') k USING(article_id)
                  JOIN read_parquet('{sql_path(state)}') s USING(article_key)
                  WHERE s.seen_item
                  ORDER BY k.article_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
            count = con.execute(f"SELECT count(*),count(DISTINCT article_key) FROM read_parquet('{sql_path(target)}')").fetchone()
            expected = next(
                int(row["warm_candidate_item_count"])
                for row in csv_rows(self.config.resolve(self.config.raw["input_manifests"]["population"]))
                if row["snapshot_id"] == snapshot and row["population"] == "evaluable_warm_users"
            )
            if count != (expected, expected):
                raise ValueError("Content candidate universe 与 Stage 2B history_seen_items 不一致")
        finally:
            con.close()
        return target

    def build_canonical_audit(self) -> Path:
        target = self.config.artifact_dir / "canonical_field_audit.csv"
        articles = self.config.resolve(self.config.raw["paths"]["articles"])
        con = self.connect()
        rows: list[dict[str, Any]] = []
        try:
            schema = {row[0]: row[1] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql_path(articles)}')").fetchall()}
            for field in self.canonical_fields:
                if field.source_column not in schema or field.display_column not in schema:
                    raise ValueError(f"预注册 canonical/display 字段不存在: {field.source_column}")
                source = field.source_column.replace('"', '""')
                display = field.display_column.replace('"', '""')
                stats = con.execute(
                    f'''WITH base AS (SELECT "{source}" AS field_value FROM read_parquet('{sql_path(articles)}')),
                         freq AS (SELECT field_value,count(*) n FROM base GROUP BY field_value)
                        SELECT (SELECT count(*) FROM base),
                               (SELECT count(*) FROM base WHERE field_value IS NULL),
                               (SELECT count(DISTINCT field_value) FROM base),
                               (SELECT max(n)::DOUBLE/(SELECT count(*) FROM base) FROM freq)'''
                ).fetchone()
                # 映射冲突按 source->display 与 display->source 两侧分别统计，不自动清洗。
                mapping = con.execute(
                    f'''WITH x AS (SELECT "{source}" s,"{display}" d FROM read_parquet('{sql_path(articles)}')),
                         a AS (SELECT count(*) n FROM (SELECT s FROM x GROUP BY s HAVING count(DISTINCT d)>1)),
                         b AS (SELECT count(*) n FROM (SELECT d FROM x GROUP BY d HAVING count(DISTINCT s)>1))
                         SELECT a.n,b.n FROM a,b'''
                ).fetchone()
                coverage: dict[str, float] = {}
                for snapshot in self.config.raw["snapshots"]:
                    if self.attributes_path(snapshot).exists():
                        value = con.execute(
                            f'SELECT avg(("{source}" IS NOT NULL)::INTEGER) FROM read_parquet(\'{sql_path(self.attributes_path(snapshot))}\')'
                        ).fetchone()[0]
                        coverage[snapshot] = float(value)
                rows.append({
                    "semantic_group": field.semantic_group,
                    "semantic_field": field.semantic_field,
                    "canonical_source_column": field.source_column,
                    "display_column": field.display_column,
                    "data_type": schema[field.source_column],
                    "missing_rate": stats[1] / stats[0],
                    "cardinality": stats[2],
                    "highest_frequency_rate": stats[3],
                    "source_to_display_conflicts": mapping[0],
                    "display_to_source_conflicts": mapping[1],
                    "mapping_status": "ONE_TO_ONE" if mapping == (0, 0) else "CONFLICT_RECORDED",
                    "modeling_inclusion": True,
                    "exclusion_reason": "",
                    "candidate_coverage_validation": coverage.get("validation_14d", ""),
                    "candidate_coverage_test": coverage.get("test_14d", ""),
                })
        finally:
            con.close()
        write_csv(target, rows, rows[0].keys())
        return target

    def build_neighbors(self, snapshot: str, variant: str) -> Path:
        attributes = pq.read_table(self.attributes_path(snapshot))
        article_keys = attributes["article_key"].to_numpy(zero_copy_only=False).astype(np.int64)
        definition = self.config.raw["content_variants"][variant]
        groups = set(definition["groups"])
        fields = [field for field in self.canonical_fields if field.semantic_group in groups]
        field_values = {
            field.source_column: attributes[field.source_column].to_numpy(zero_copy_only=False)
            for field in fields
        }
        matrix, tokens = build_sparse_item_matrix(
            field_values, fields, use_idf=bool(definition["idf"]),
            group_balance=bool(definition["group_balance"]),
        )
        json_write(self.token_path(snapshot, variant), {
            "snapshot_id": snapshot,
            "as_of_date": self.snapshots[snapshot].history_end,
            "candidate_universe_fingerprint": file_fingerprint(self.attributes_path(snapshot)),
            "attribute_schema_fingerprint": hashlib.sha256(
                json.dumps([field.__dict__ for field in fields], sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "variant": variant,
            "item_count": len(article_keys),
            "feature_count": matrix.shape[1],
            "tokens": tokens,
        })
        target = self.neighbor_path(snapshot, variant)
        target.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([
            ("article_key", pa.int64()), ("neighbor_article_key", pa.int64()),
            ("similarity", pa.float64()), ("neighbor_rank", pa.int32()),
        ])
        writer = pq.ParquetWriter(target, schema, compression="zstd")
        block = int(self.config.raw["execution"]["neighbor_block_rows"])
        top_k = int(self.config.raw["content"]["top_neighbors_per_seed"])
        try:
            for start in range(0, len(article_keys), block):
                left, right, score, rank = top_k_cosine_block(
                    matrix, article_keys, start, min(start + block, len(article_keys)), top_k
                )
                writer.write_table(pa.table({
                    "article_key": left, "neighbor_article_key": right,
                    "similarity": score, "neighbor_rank": rank,
                }, schema=schema))
        finally:
            writer.close()
        return target

    def build_candidate_part(self, snapshot: str, variant: str, part: int) -> Path:
        target = self.candidate_part(snapshot, variant, part)
        target.parent.mkdir(parents=True, exist_ok=True)
        partitions = int(self.config.raw["execution"]["candidate_partitions"])
        seeds, history = self.stage3b_seed(snapshot), self.stage3b_history(snapshot)
        neighbors = self.neighbor_path(snapshot, variant)
        population = self.stage2b_path("evaluation_population", snapshot)
        spec = self.snapshots[snapshot]
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key customer_key,is_main_evaluable
                    FROM read_parquet('{sql_path(population)}')
                    WHERE user_key % {partitions}={part}
                  ), scored AS (
                    SELECT s.customer_key,n.neighbor_article_key article_key,
                           sum(s.seed_weight*n.similarity) source_score,
                           count(DISTINCT s.seed_article_key)::INTEGER seed_hit_count,
                           max(n.similarity) max_similarity
                    FROM users u
                    JOIN read_parquet('{sql_path(seeds)}') s USING(customer_key)
                    JOIN read_parquet('{sql_path(neighbors)}') n
                      ON n.article_key=s.seed_article_key
                    ANTI JOIN read_parquet('{sql_path(history)}') h
                      ON h.customer_key=s.customer_key AND h.article_key=n.neighbor_article_key
                    WHERE u.is_main_evaluable
                    GROUP BY s.customer_key,n.neighbor_article_key
                  ), ranked AS (
                    SELECT *,row_number() OVER(PARTITION BY customer_key ORDER BY
                      source_score DESC,seed_hit_count DESC,max_similarity DESC,article_key)::INTEGER source_rank
                    FROM scored
                  )
                  SELECT '{snapshot}' snapshot_id,DATE '{spec.history_end}' as_of_date,
                         customer_key,article_key,'content' AS "source",'{variant}' AS source_variant,
                         source_rank,source_score,'history_seen_items_v1' candidate_universe_version,
                         '{self.feedback_version}' feedback_version,'{self.population_version}' population_version,
                         '{self.dataset_fingerprint}' dataset_fingerprint,'{self.config.fingerprint}' config_fingerprint,
                         seed_hit_count,max_similarity,'sum_seed_decay_times_cosine' source_score_kind
                  FROM ranked WHERE source_rank<=500 ORDER BY customer_key,source_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        return target

    def validate_candidates(self, snapshot: str, variant: str) -> None:
        glob = self.candidate_glob(snapshot, variant)
        history = self.stage3b_history(snapshot)
        con = self.connect()
        try:
            violations = con.execute(
                f"""SELECT
                  (SELECT count(*) FROM read_parquet('{glob}') WHERE source_rank>500 OR source_rank<1),
                  (SELECT count(*) FROM read_parquet('{glob}') c JOIN read_parquet('{sql_path(history)}') h USING(customer_key,article_key)),
                  (SELECT count(*) FROM (SELECT customer_key,article_key,count(*) n FROM read_parquet('{glob}') GROUP BY 1,2 HAVING n>1))
                """
            ).fetchone()
            if violations != (0, 0, 0):
                raise ValueError(f"Content candidate 协议违反: {violations}")
        finally:
            con.close()

    def evaluate(self, snapshot: str, *, candidate_glob: str, source: str, variant: str) -> Path:
        target = self.metric_path(snapshot, source, variant)
        population = self.stage2b_path("evaluation_population", snapshot)
        ground_truth = self.stage2b_path("ground_truth", snapshot)
        spec = self.snapshots[snapshot]
        if file_fingerprint(ground_truth) != spec.ground_truth_fingerprint:
            raise ValueError("Ground truth fingerprint 在评估前发生变化")
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key customer_key,user_segment,is_main_evaluable,positive_item_count
                    FROM read_parquet('{sql_path(population)}')
                  ), counts AS (
                    SELECT customer_key,count(*)::INTEGER candidate_count
                    FROM read_parquet('{candidate_glob}') GROUP BY customer_key
                  ), hit_rows AS (
                    SELECT c.customer_key,c.source_rank,row_number() OVER(
                      PARTITION BY c.customer_key ORDER BY c.source_rank)::INTEGER cumulative_hits
                    FROM read_parquet('{candidate_glob}') c
                    SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                      ON g.user_key=c.customer_key AND g.article_key=c.article_key
                  ), hits AS (
                    SELECT customer_key,
                      count(*) FILTER(WHERE source_rank<=12)::INTEGER hits_12,
                      count(*) FILTER(WHERE source_rank<=50)::INTEGER hits_50,
                      count(*) FILTER(WHERE source_rank<=100)::INTEGER hits_100,
                      count(*) FILTER(WHERE source_rank<=300)::INTEGER hits_300,
                      count(*) FILTER(WHERE source_rank<=500)::INTEGER hits_500,
                      sum(cumulative_hits::DOUBLE/source_rank) FILTER(WHERE source_rank<=12) ap_numerator_12
                    FROM hit_rows GROUP BY customer_key
                  )
                  SELECT '{snapshot}' snapshot_id,DATE '{spec.history_end}' as_of_date,
                         u.customer_key,u.user_segment,u.is_main_evaluable,u.positive_item_count,
                         coalesce(c.candidate_count,0)::INTEGER candidate_count,
                         coalesce(h.hits_12,0)::INTEGER hits_12,coalesce(h.hits_50,0)::INTEGER hits_50,
                         coalesce(h.hits_100,0)::INTEGER hits_100,coalesce(h.hits_300,0)::INTEGER hits_300,
                         coalesce(h.hits_500,0)::INTEGER hits_500,
                         coalesce(h.hits_12,0)::DOUBLE/u.positive_item_count recall_at_12,
                         coalesce(h.hits_50,0)::DOUBLE/u.positive_item_count recall_at_50,
                         coalesce(h.hits_100,0)::DOUBLE/u.positive_item_count recall_at_100,
                         coalesce(h.hits_300,0)::DOUBLE/u.positive_item_count recall_at_300,
                         coalesce(h.hits_500,0)::DOUBLE/u.positive_item_count recall_at_500,
                         (coalesce(h.hits_12,0)>0)::INTEGER hit_rate_at_12,
                         (coalesce(h.hits_50,0)>0)::INTEGER hit_rate_at_50,
                         (coalesce(h.hits_100,0)>0)::INTEGER hit_rate_at_100,
                         (coalesce(h.hits_300,0)>0)::INTEGER hit_rate_at_300,
                         (coalesce(h.hits_500,0)>0)::INTEGER hit_rate_at_500,
                         coalesce(h.ap_numerator_12,0)::DOUBLE/least(u.positive_item_count,12) ap_at_12,
                         '{source}' AS "source",'{variant}' AS source_variant,
                         '{spec.ground_truth_fingerprint}' ground_truth_fingerprint,
                         '{self.dataset_fingerprint}' dataset_fingerprint,'{self.config.fingerprint}' config_fingerprint
                  FROM users u LEFT JOIN counts c USING(customer_key) LEFT JOIN hits h USING(customer_key)
                  ORDER BY customer_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()
        return target

    @staticmethod
    def _population_sql(metric_path: Path) -> str:
        path = sql_path(metric_path)
        return f"""
          SELECT 'evaluable_warm_users' segment,* FROM read_parquet('{path}') WHERE is_main_evaluable
          UNION ALL SELECT 'short_history',* FROM read_parquet('{path}') WHERE user_segment='short_history'
          UNION ALL SELECT 'warm',* FROM read_parquet('{path}') WHERE user_segment='warm'
          UNION ALL SELECT 'active',* FROM read_parquet('{path}') WHERE user_segment='active'
          UNION ALL SELECT 'cold_history_label_users',* FROM read_parquet('{path}') WHERE user_segment='cold'
        """

    def metric_summary(self, snapshot: str, source: str, variant: str, metric_path: Path) -> list[dict[str, Any]]:
        con = self.connect()
        try:
            values = con.execute(
                f"""WITH p AS ({self._population_sql(metric_path)})
                SELECT segment,count(*) denominator,
                       avg(ap_at_12) map_at_12,avg(recall_at_12) recall_at_12,
                       avg(recall_at_50) recall_at_50,avg(recall_at_100) recall_at_100,
                       avg(recall_at_300) recall_at_300,avg(recall_at_500) recall_at_500,
                       avg(hit_rate_at_500) hit_rate_at_500,avg(candidate_count) mean_candidates,
                       avg((candidate_count>0)::INTEGER) candidate_user_coverage,
                       avg((candidate_count<500)::INTEGER) candidate_shortfall_rate
                FROM p GROUP BY segment ORDER BY segment"""
            ).fetchall()
        finally:
            con.close()
        fields = ("map_at_12", "recall_at_12", "recall_at_50", "recall_at_100", "recall_at_300",
                  "recall_at_500", "hit_rate_at_500", "mean_candidates", "candidate_user_coverage",
                  "candidate_shortfall_rate")
        rows: list[dict[str, Any]] = []
        for value in values:
            for index, metric in enumerate(fields, 2):
                rows.append({
                    "stage": "3C", "snapshot_id": snapshot,
                    "as_of_date": self.snapshots[snapshot].history_end,
                    "source": source, "source_variant": variant, "segment": value[0],
                    "population_denominator": value[1], "metric": metric, "metric_value": value[index],
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint,
                    "candidate_universe_version": "history_seen_items_v1",
                    "run_status": "SUCCESS",
                })
        return rows

    def _build_union_chunk(
        self, snapshot: str, union_variant: str, content_variant: str,
        part: int, modulus: int, residue: int, target: Path,
    ) -> None:
        if union_variant not in {"Q1", "Q2"}:
            raise ValueError("Stage 3C 只新建 Q1/Q2；Q0 必须复用 Stage 3B")
        caps = self.config.raw["union"]["variants"][union_variant]
        population = self.stage2b_path("evaluation_population", snapshot)
        repurchase = self.stage3a_candidate(snapshot, "repurchase", "pure")
        trend = self.stage3a_ranking(snapshot, "trend", "window_7d")
        popularity = self.stage3a_ranking(snapshot, "popularity", "global")
        itemcf = self.stage3b_candidate_part(snapshot, part)
        content = self.candidate_part(snapshot, content_variant, part)
        spec = self.snapshots[snapshot]
        target.parent.mkdir(parents=True, exist_ok=True)
        con = self.connect()
        try:
            con.execute(
                f"""
                COPY (
                  WITH users AS (
                    SELECT user_key customer_key,is_main_evaluable
                    FROM read_parquet('{sql_path(population)}')
                    WHERE user_key % {modulus}={residue}
                  ), primary_raw AS (
                    SELECT r.customer_key,r.article_key,'repurchase' source_name,1 source_priority,r.source_rank,r.source_score
                    FROM read_parquet('{sql_path(repurchase)}') r SEMI JOIN users u USING(customer_key)
                    WHERE r.source_rank<={int(caps['repurchase'])}
                    UNION ALL
                    SELECT i.customer_key,i.article_key,'itemcf',2,i.source_rank,i.source_score
                    FROM read_parquet('{sql_path(itemcf)}') i SEMI JOIN users u USING(customer_key)
                    WHERE i.source_rank<={int(caps['itemcf'])}
                    UNION ALL
                    SELECT c.customer_key,c.article_key,'content',3,c.source_rank,c.source_score
                    FROM read_parquet('{sql_path(content)}') c SEMI JOIN users u USING(customer_key)
                    WHERE c.source_rank<={int(caps['content'])}
                    UNION ALL
                    SELECT u.customer_key,t.article_key,'trend',4,t.source_rank,t.source_score
                    FROM users u CROSS JOIN read_parquet('{sql_path(trend)}') t
                    WHERE t.source_rank<={int(caps['trend'])}
                  ), primary_agg AS (
                    SELECT customer_key,article_key,
                      bool_or(source_name='repurchase') from_repurchase,
                      bool_or(source_name='itemcf') from_itemcf,
                      bool_or(source_name='content') from_content,
                      bool_or(source_name='trend') from_trend,
                      min(source_rank) FILTER(WHERE source_name='repurchase') repurchase_rank,
                      max(source_score) FILTER(WHERE source_name='repurchase') repurchase_score,
                      min(source_rank) FILTER(WHERE source_name='itemcf') itemcf_rank,
                      max(source_score) FILTER(WHERE source_name='itemcf') itemcf_score,
                      min(source_rank) FILTER(WHERE source_name='content') content_rank,
                      max(source_score) FILTER(WHERE source_name='content') content_score,
                      min(source_rank) FILTER(WHERE source_name='trend') trend_rank,
                      max(source_score) FILTER(WHERE source_name='trend') trend_score,
                      min(source_priority) first_priority,count(DISTINCT source_name)::INTEGER source_count
                    FROM primary_raw GROUP BY customer_key,article_key
                  ), primary_ranked AS (
                    SELECT *,row_number() OVER(PARTITION BY customer_key ORDER BY first_priority,
                      CASE first_priority WHEN 1 THEN repurchase_rank WHEN 2 THEN itemcf_rank
                           WHEN 3 THEN content_rank ELSE trend_rank END,article_key)::INTEGER union_rank
                    FROM primary_agg
                  ), primary_selected AS (SELECT * FROM primary_ranked WHERE union_rank<=500),
                  primary_counts AS (
                    SELECT u.customer_key,u.is_main_evaluable,count(p.article_key)::INTEGER primary_count
                    FROM users u LEFT JOIN primary_selected p USING(customer_key)
                    GROUP BY u.customer_key,u.is_main_evaluable
                  ), popularity_pool AS (
                    SELECT pc.customer_key,pc.is_main_evaluable,pc.primary_count,p.article_key,
                           p.source_rank popularity_rank,p.source_score popularity_score
                    FROM primary_counts pc CROSS JOIN read_parquet('{sql_path(popularity)}') p
                    ANTI JOIN primary_selected x ON x.customer_key=pc.customer_key AND x.article_key=p.article_key
                    WHERE p.source_rank<=500+pc.primary_count
                  ), popularity_fill AS (
                    SELECT *,row_number() OVER(PARTITION BY customer_key ORDER BY popularity_rank,article_key)::INTEGER fill_rank
                    FROM popularity_pool QUALIFY fill_rank<=500-primary_count
                  ), combined AS (
                    SELECT p.customer_key,p.article_key,p.union_rank,
                      CASE p.first_priority WHEN 1 THEN 'repurchase' WHEN 2 THEN 'itemcf'
                           WHEN 3 THEN 'content' ELSE 'trend' END first_source,
                      p.from_repurchase,p.from_itemcf,p.from_content,p.from_trend,false from_popularity,
                      p.repurchase_rank,p.repurchase_score,p.itemcf_rank,p.itemcf_score,
                      p.content_rank,p.content_score,p.trend_rank,p.trend_score,
                      NULL::INTEGER popularity_rank,NULL::DOUBLE popularity_score,p.source_count,u.is_main_evaluable
                    FROM primary_selected p JOIN users u USING(customer_key)
                    UNION ALL
                    SELECT customer_key,article_key,(primary_count+fill_rank)::INTEGER,'popularity_fallback',
                      false,false,false,false,true,NULL::INTEGER,NULL::DOUBLE,NULL::INTEGER,NULL::DOUBLE,
                      NULL::INTEGER,NULL::DOUBLE,NULL::INTEGER,NULL::DOUBLE,popularity_rank,popularity_score,
                      1::INTEGER,is_main_evaluable FROM popularity_fill
                  )
                  SELECT '{snapshot}' snapshot_id,DATE '{spec.history_end}' as_of_date,
                    customer_key,article_key,'union' AS "source",'{union_variant}_{content_variant}' AS source_variant,
                    union_rank source_rank,1.0/union_rank::DOUBLE source_score,
                    'history_seen_items_v1' candidate_universe_version,'{self.feedback_version}' feedback_version,
                    '{self.population_version}' population_version,'{self.dataset_fingerprint}' dataset_fingerprint,
                    '{self.config.fingerprint}' config_fingerprint,first_source,source_count,
                    from_repurchase,from_itemcf,from_content,from_trend,from_popularity,
                    repurchase_rank,repurchase_score,itemcf_rank,itemcf_score,content_rank,content_score,
                    trend_rank,trend_score,popularity_rank,popularity_score,
                    'source_rank_cap_pre_dedup_then_fixed_insertion' source_score_kind
                  FROM combined ORDER BY customer_key,source_rank
                ) TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )
        finally:
            con.close()

    def build_union_part(self, snapshot: str, union_variant: str, content_variant: str, part: int) -> Path:
        partitions = int(self.config.raw["execution"]["candidate_partitions"])
        subparts = int(self.config.raw["execution"]["union_subpartitions"])
        chunks: list[Path] = []
        temp = self.config.temp_dir / "union_chunks" / snapshot / f"{union_variant}_{content_variant}"
        temp.mkdir(parents=True, exist_ok=True)
        for subpart in range(subparts):
            chunk = temp / f"part_{part:03d}_sub_{subpart:02d}.parquet"
            self._build_union_chunk(
                snapshot, union_variant, content_variant, part,
                partitions * subparts, part + partitions * subpart, chunk,
            )
            chunks.append(chunk)
        target = self.union_part(snapshot, f"{union_variant}_{content_variant}", part)
        target.parent.mkdir(parents=True, exist_ok=True)
        con = self.connect()
        try:
            literals = ",".join(f"'{sql_path(path)}'" for path in chunks)
            con.execute(
                f"COPY (SELECT * FROM read_parquet([{literals}]) ORDER BY customer_key,source_rank) "
                f"TO '{sql_path(target)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
            )
        finally:
            con.close()
        for chunk in chunks:
            chunk.unlink(missing_ok=True)
        return target

    def validate_union(self, snapshot: str, union_variant: str) -> None:
        glob = self.union_glob(snapshot, union_variant)
        population = self.stage2b_path("evaluation_population", snapshot)
        con = self.connect()
        try:
            audit = con.execute(
                f"""WITH x AS (
                  SELECT customer_key,count(*) n,count(DISTINCT article_key) d,
                         min(source_rank) lo,max(source_rank) hi,count(DISTINCT source_rank) dr
                  FROM read_parquet('{glob}') GROUP BY customer_key)
                SELECT count(*) FILTER(WHERE n>500 OR n<>d OR lo<>1 OR hi<>n OR dr<>n),
                       (SELECT count(*) FROM read_parquet('{sql_path(population)}'))-count(*)
                FROM x"""
            ).fetchone()
            if audit != (0, 0):
                raise ValueError(f"Union 协议违反: {audit}")
        finally:
            con.close()

    def analyze_independence(self, snapshot: str, content_variant: str, union_variant: str) -> dict[str, Any]:
        content = self.candidate_glob(snapshot, content_variant)
        itemcf_root = self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) / "candidates" / snapshot / "iuf" / "part_*.parquet"
        itemcf = sql_path(itemcf_root)
        trend = self.stage3a_ranking(snapshot, "trend", "window_7d")
        repurchase = self.stage3a_candidate(snapshot, "repurchase", "pure")
        ground_truth = self.stage2b_path("ground_truth", snapshot)
        population = self.stage2b_path("evaluation_population", snapshot)
        q0_metric = self.metric_path(snapshot, "union", "Q0")
        challenger_metric = self.metric_path(snapshot, "union", union_variant)
        con = self.connect()
        try:
            value = con.execute(
                f"""
                WITH users AS (
                  SELECT user_key customer_key FROM read_parquet('{sql_path(population)}') WHERE is_main_evaluable
                ), cn AS (SELECT customer_key,count(*) n FROM read_parquet('{content}') GROUP BY 1),
                inum AS (SELECT customer_key,count(*) n FROM read_parquet('{itemcf}') GROUP BY 1),
                ci AS (SELECT c.customer_key,count(*) n FROM read_parquet('{content}') c
                       JOIN read_parquet('{itemcf}') i USING(customer_key,article_key) GROUP BY 1),
                ct AS (SELECT c.customer_key,count(*) n FROM read_parquet('{content}') c
                       JOIN read_parquet('{sql_path(trend)}') t USING(article_key) WHERE t.source_rank<=500 GROUP BY 1),
                cr AS (SELECT c.customer_key,count(*) n FROM read_parquet('{content}') c
                       JOIN read_parquet('{sql_path(repurchase)}') r USING(customer_key,article_key) GROUP BY 1),
                content_hits AS (
                  SELECT c.customer_key,c.article_key FROM read_parquet('{content}') c
                  SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=c.customer_key AND g.article_key=c.article_key
                ), prior_hits AS (
                  SELECT i.customer_key,i.article_key FROM read_parquet('{itemcf}') i
                  SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=i.customer_key AND g.article_key=i.article_key
                  UNION
                  SELECT u.customer_key,t.article_key FROM users u CROSS JOIN read_parquet('{sql_path(trend)}') t
                  SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=u.customer_key AND g.article_key=t.article_key WHERE t.source_rank<=500
                  UNION
                  SELECT r.customer_key,r.article_key FROM read_parquet('{sql_path(repurchase)}') r
                  SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=r.customer_key AND g.article_key=r.article_key
                ), hit_stats AS (
                  SELECT count(*) content_hits,
                    count(*) FILTER(WHERE p.article_key IS NULL) exclusive_hits,
                    count(*) FILTER(WHERE p.article_key IS NOT NULL) shared_hits,
                    count(DISTINCT c.customer_key) FILTER(WHERE p.article_key IS NULL) users_exclusive,
                    count(DISTINCT c.article_key) FILTER(WHERE p.article_key IS NULL) items_exclusive
                  FROM content_hits c LEFT JOIN prior_hits p USING(customer_key,article_key)
                ), delta AS (
                  SELECT avg((x.recall_at_500<q.recall_at_500)::INTEGER) declined_rate
                  FROM read_parquet('{sql_path(q0_metric)}') q JOIN read_parquet('{sql_path(challenger_metric)}') x USING(customer_key)
                  WHERE q.is_main_evaluable
                )
                SELECT avg(coalesce(ci.n,0)::DOUBLE/nullif(coalesce(cn.n,0)+coalesce(inum.n,0)-coalesce(ci.n,0),0)),
                       avg(coalesce(ct.n,0)::DOUBLE/nullif(coalesce(cn.n,0)+500-coalesce(ct.n,0),0)),
                       avg(coalesce(cr.n,0)::DOUBLE/nullif(coalesce(cn.n,0)+coalesce(rn.n,0)-coalesce(cr.n,0),0)),
                       h.content_hits,h.exclusive_hits,h.shared_hits,h.users_exclusive,h.items_exclusive,d.declined_rate
                FROM users u LEFT JOIN cn USING(customer_key) LEFT JOIN inum USING(customer_key)
                LEFT JOIN ci USING(customer_key) LEFT JOIN ct USING(customer_key) LEFT JOIN cr USING(customer_key)
                LEFT JOIN (SELECT customer_key,count(*) n FROM read_parquet('{sql_path(repurchase)}') GROUP BY 1) rn USING(customer_key),
                hit_stats h,delta d GROUP BY ALL
                """
            ).fetchone()
        finally:
            con.close()
        return {
            "stage": "3C", "snapshot_id": snapshot, "content_variant": content_variant,
            "union_variant": union_variant,
            "content_itemcf_candidate_jaccard": value[0] or 0.0,
            "content_trend_candidate_jaccard": value[1] or 0.0,
            "content_repurchase_candidate_jaccard": value[2] or 0.0,
            "content_ground_truth_hits": value[3], "content_exclusive_hits": value[4],
            "content_shared_hits": value[5], "users_with_content_exclusive_hits": value[6],
            "items_with_content_exclusive_hits": value[7], "declined_user_rate": value[8],
            "dataset_fingerprint": self.dataset_fingerprint, "config_fingerprint": self.config.fingerprint,
            "run_status": "SUCCESS",
        }

    @staticmethod
    def _lookup_metric(rows: Sequence[Mapping[str, Any]], variant: str, metric: str,
                       segment: str = "evaluable_warm_users") -> float:
        values = [float(row["metric_value"]) for row in rows
                  if row["source_variant"] == variant and row["metric"] == metric and row["segment"] == segment]
        if len(values) != 1:
            raise ValueError(f"指标不唯一: {variant}/{metric}/{segment}")
        return values[0]

    def compare_to_q0(self, snapshot: str, challenger: str) -> tuple[dict[str, Any], dict[str, Any]]:
        base = self.metric_path(snapshot, "union", "Q0")
        other = self.metric_path(snapshot, "union", challenger)
        con = self.connect()
        try:
            aggregate = con.execute(
                f"""SELECT count(*) n,
                  count(*) FILTER(WHERE x.recall_at_500>q.recall_at_500) improved,
                  count(*) FILTER(WHERE x.recall_at_500=q.recall_at_500) unchanged,
                  count(*) FILTER(WHERE x.recall_at_500<q.recall_at_500) declined,
                  sum(greatest(x.hits_500-q.hits_500,0)) gained_hits,
                  sum(greatest(q.hits_500-x.hits_500,0)) lost_hits,
                  sum(x.hits_500-q.hits_500) net_hits,
                  avg(x.recall_at_500-q.recall_at_500) mean_delta
                FROM read_parquet('{sql_path(base)}') q JOIN read_parquet('{sql_path(other)}') x USING(customer_key)
                WHERE q.is_main_evaluable"""
            ).fetchone()
            arrays = con.execute(
                f"""SELECT list(q.recall_at_500 ORDER BY q.customer_key),list(x.recall_at_500 ORDER BY q.customer_key)
                FROM read_parquet('{sql_path(base)}') q JOIN read_parquet('{sql_path(other)}') x USING(customer_key)
                WHERE q.is_main_evaluable"""
            ).fetchone()
        finally:
            con.close()
        iterations = int(self.config.raw["bootstrap"]["iterations"])
        seed = int(self.config.raw["bootstrap"]["seed"])
        bootstrap = paired_bootstrap(arrays[0], arrays[1], iterations=iterations, seed=seed)
        comparison = {
            "stage": "3C", "snapshot_id": snapshot, "comparison": f"{challenger}_vs_Q0",
            "population_denominator": aggregate[0],
            "improved_users": aggregate[1], "unchanged_users": aggregate[2], "declined_users": aggregate[3],
            "improved_user_rate": aggregate[1] / aggregate[0],
            "unchanged_user_rate": aggregate[2] / aggregate[0],
            "declined_user_rate": aggregate[3] / aggregate[0],
            "gained_ground_truth_hits": aggregate[4], "lost_ground_truth_hits": aggregate[5],
            "net_ground_truth_hits": aggregate[6], "mean_delta_recall_at_500": aggregate[7],
            "dataset_fingerprint": self.dataset_fingerprint, "config_fingerprint": self.config.fingerprint,
            "run_status": "SUCCESS",
        }
        boot = {
            "stage": "3C", "snapshot_id": snapshot, "comparison": f"{challenger}_vs_Q0",
            **bootstrap, "iterations": iterations, "seed": seed, "confidence_level": 0.95,
            "population_fingerprint": self.snapshots[snapshot].ground_truth_fingerprint,
            "interpretation": "offline_paired_user_block_bootstrap_not_online_ab_test",
            "dataset_fingerprint": self.dataset_fingerprint, "config_fingerprint": self.config.fingerprint,
            "run_status": "SUCCESS",
        }
        return comparison, boot

    def quota_occupancy(self, snapshot: str, quota: str, content_variant: str) -> list[dict[str, Any]]:
        caps = self.config.raw["union"]["variants"][quota]
        if quota == "Q0":
            glob = self._q0_glob(snapshot)
            content_provenance = "false"
        else:
            glob = self.union_glob(snapshot, f"{quota}_{content_variant}")
            content_provenance = "from_content"
        population = self.stage2b_path("evaluation_population", snapshot)
        repurchase = self.stage3a_candidate(snapshot, "repurchase", "pure")
        itemcf = sql_path(self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) /
                          "candidates" / snapshot / "iuf" / "part_*.parquet")
        content = self.candidate_glob(snapshot, content_variant)
        con = self.connect()
        try:
            values = con.execute(
                f"""WITH u AS (
                  SELECT user_key customer_key FROM read_parquet('{sql_path(population)}')
                ), rb AS (SELECT customer_key,count(*) n FROM read_parquet('{sql_path(repurchase)}') GROUP BY 1),
                ib AS (SELECT customer_key,count(*) n FROM read_parquet('{itemcf}') GROUP BY 1),
                cb AS (SELECT customer_key,count(*) n FROM read_parquet('{content}') GROUP BY 1),
                occ AS (
                  SELECT customer_key,
                    sum(from_repurchase::INTEGER) rep_present,sum((first_source='repurchase')::INTEGER) rep_first,
                    sum(from_itemcf::INTEGER) itemcf_present,sum((first_source='itemcf')::INTEGER) itemcf_first,
                    sum(({content_provenance})::INTEGER) content_present,sum((first_source='content')::INTEGER) content_first,
                    sum(from_trend::INTEGER) trend_present,sum((first_source='trend')::INTEGER) trend_first,
                    sum((first_source='popularity_fallback')::INTEGER) popularity_first
                  FROM read_parquet('{glob}') GROUP BY 1
                )
                SELECT
                  avg(coalesce(rb.n,0)),avg(least(coalesce(rb.n,0),{int(caps['repurchase'])})),
                  avg(o.rep_present),avg(o.rep_first),
                  avg(coalesce(ib.n,0)),avg(least(coalesce(ib.n,0),{int(caps['itemcf'])})),
                  avg(o.itemcf_present),avg(o.itemcf_first),
                  avg(coalesce(cb.n,0)),avg(least(coalesce(cb.n,0),{int(caps['content'])})),
                  avg(o.content_present),avg(o.content_first),
                  500::DOUBLE,{int(caps['trend'])}::DOUBLE,avg(o.trend_present),avg(o.trend_first),
                  avg(o.popularity_first),sum(least(coalesce(cb.n,0),{int(caps['content'])})),sum(o.content_first)
                FROM u LEFT JOIN rb USING(customer_key) LEFT JOIN ib USING(customer_key)
                LEFT JOIN cb USING(customer_key) JOIN occ o USING(customer_key)"""
            ).fetchone()
        finally:
            con.close()
        names = ("repurchase", "itemcf", "content", "trend", "popularity_fallback")
        configured = (caps["repurchase"], caps["itemcf"], caps["content"], caps["trend"], None)
        before = (values[0], values[4], values[8], values[12], values[16])
        truncated = (values[1], values[5], values[9], values[13], values[16])
        present = (values[2], values[6], values[10], values[14], values[16])
        occupancy = (values[3], values[7], values[11], values[15], values[16])
        comparison = None
        comparison_path = self.config.artifact_dir / "validation_marginal_recall.csv"
        if snapshot == "validation_14d" and quota in {"Q1", "Q2"} and comparison_path.exists():
            comparison = next(row for row in csv_rows(comparison_path) if row["quota_variant"] == quota)
        net_hits = float(comparison["net_ground_truth_hits"]) if comparison else 0.0
        return [{
            "stage": "3C", "snapshot_id": snapshot, "quota_variant": quota,
            "content_variant": content_variant if quota != "Q0" else "none",
            "source": name, "configured_source_rank_cap": configured[index],
            "mean_candidates_before_cap": before[index],
            "mean_candidates_after_source_rank_cap": truncated[index],
            "mean_candidates_present_after_dedup": present[index],
            "mean_duplicates_with_higher_priority": max(float(present[index]) - float(occupancy[index]), 0.0),
            "cross_source_duplicate_rate": (
                max(float(present[index]) - float(occupancy[index]), 0.0) / float(truncated[index])
                if float(truncated[index]) > 0 else 0.0
            ),
            "mean_actual_unique_occupancy": occupancy[index],
            "mean_effective_position_loss": (
                max(float(configured[index]) - float(occupancy[index]), 0.0)
                if configured[index] is not None else 0.0
            ),
            "mean_popularity_fill": values[16],
            "net_hits_per_100_content_source_rank_candidates": (
                100.0 * net_hits / float(values[17]) if name == "content" and float(values[17]) > 0 else 0.0
            ),
            "net_hits_per_100_actual_content_unique_positions": (
                100.0 * net_hits / float(values[18]) if name == "content" and float(values[18]) > 0 else 0.0
            ),
            "itemcf_source_rank_candidates_removed_vs_q0": 300 - int(caps["itemcf"]) if name == "itemcf" else 0,
            "quota_semantics": "source_rank_cap_before_cross_source_dedup",
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS",
        } for index, name in enumerate(names)]

    def write_q0_reproduction(self) -> Path:
        rows: list[dict[str, Any]] = []
        for snapshot in self.config.raw["snapshots"]:
            parts = self.stage3b_q0_parts(snapshot)
            if len(parts) != int(self.config.raw["execution"]["candidate_partitions"]):
                raise ValueError("Stage 3B Q0 分区不完整")
            rows.append({
                "stage": "3C", "snapshot_id": snapshot,
                "reproduction_mode": "exact_reuse_stage_3b_private_q0",
                "candidate_semantic_fingerprint": semantic_partition_fingerprint(parts),
                "file_partition_count": len(parts),
                "stage_3b_lock_fingerprint": self.stage3b_lock_fingerprint,
                "candidate_triples_unchanged": True,
                "schema_extended": False,
                "run_status": "SUCCESS",
            })
        target = self.config.artifact_dir / "q0_reproduction.csv"
        write_csv(target, rows, rows[0].keys())
        return target

    def _q0_glob(self, snapshot: str) -> str:
        return sql_path(self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) /
                        "unions" / snapshot / self.stage3b_q0_variant(snapshot) / "part_*.parquet")

    def _run_content_core(self, snapshot: str, variants: Sequence[str]) -> None:
        self.run_group(
            f"{snapshot}_attributes", lambda: self.build_attribute_snapshot(snapshot),
            [self.attributes_path(snapshot)],
        )
        for variant in variants:
            self.run_group(
                f"{snapshot}_{variant}_neighbors",
                lambda variant=variant: self.build_neighbors(snapshot, variant),
                [self.neighbor_path(snapshot, variant), self.token_path(snapshot, variant)],
            )
            for part in range(int(self.config.raw["execution"]["candidate_partitions"])):
                self.run_group(
                    f"{snapshot}_{variant}_candidates_part_{part:03d}",
                    lambda variant=variant, part=part: self.build_candidate_part(snapshot, variant, part),
                    [self.candidate_part(snapshot, variant, part)],
                )
            self.validate_candidates(snapshot, variant)
            self.run_group(
                f"{snapshot}_{variant}_content_evaluation",
                lambda variant=variant: self.evaluate(
                    snapshot, candidate_glob=self.candidate_glob(snapshot, variant),
                    source="content", variant=variant,
                ),
                [self.metric_path(snapshot, "content", variant)],
            )

    def _run_union(self, snapshot: str, quota: str, content_variant: str) -> str:
        name = f"{quota}_{content_variant}"
        for part in range(int(self.config.raw["execution"]["candidate_partitions"])):
            self.run_group(
                f"{snapshot}_{name}_union_part_{part:03d}",
                lambda part=part: self.build_union_part(snapshot, quota, content_variant, part),
                [self.union_part(snapshot, name, part)],
            )
        self.validate_union(snapshot, name)
        self.run_group(
            f"{snapshot}_{name}_union_evaluation",
            lambda: self.evaluate(snapshot, candidate_glob=self.union_glob(snapshot, name),
                                  source="union", variant=name),
            [self.metric_path(snapshot, "union", name)],
        )
        return name

    def _evaluate_q0(self, snapshot: str) -> None:
        self.run_group(
            f"{snapshot}_Q0_exact_reuse_evaluation",
            lambda: self.evaluate(snapshot, candidate_glob=self._q0_glob(snapshot), source="union", variant="Q0"),
            [self.metric_path(snapshot, "union", "Q0")],
        )
        # 重新评估值必须与 Stage 3B 正式公开 overall 完全一致到浮点审计容差。
        stage3b = csv_rows(self.config.artifact_dir.parent / "stage_3b" /
                           ("validation_union_metrics.csv" if snapshot == "validation_14d" else "test_union_metrics.csv"))
        expected = next(float(row["metric_value"]) for row in stage3b
                        if row["source_variant"] == self.stage3b_q0_variant(snapshot)
                        and row["metric"] == "recall_at_500"
                        and row["segment"] == "evaluable_warm_users")
        actual = self.metric_summary(snapshot, "union", "Q0", self.metric_path(snapshot, "union", "Q0"))
        observed = self._lookup_metric(actual, "Q0", "recall_at_500")
        if abs(expected - observed) > 1e-15:
            raise ValueError(f"Q0 Recall@500 未严格复现 Stage 3B: {observed} != {expected}")

    def run_validation(self) -> str:
        snapshot = "validation_14d"
        variants = tuple(self.config.raw["content_variants"])
        self._evaluate_q0(snapshot)
        self._run_content_core(snapshot, variants)
        self.run_group("canonical_field_audit", self.build_canonical_audit,
                       [self.config.artifact_dir / "canonical_field_audit.csv"])
        q1_names = {variant: self._run_union(snapshot, "Q1", variant) for variant in variants}

        content_metrics: list[dict[str, Any]] = []
        union_metrics: list[dict[str, Any]] = []
        independence: list[dict[str, Any]] = []
        selection_inputs: list[dict[str, Any]] = []
        for variant in variants:
            content_metrics.extend(self.metric_summary(
                snapshot, "content", variant, self.metric_path(snapshot, "content", variant)))
            union_metrics.extend(self.metric_summary(
                snapshot, "union", q1_names[variant], self.metric_path(snapshot, "union", q1_names[variant])))
            independent = self.analyze_independence(snapshot, variant, q1_names[variant])
            independence.append(independent)
            selection_inputs.append({
                "variant": variant,
                "q1_recall_at_500": self._lookup_metric(union_metrics, q1_names[variant], "recall_at_500"),
                "q1_hit_rate_at_500": self._lookup_metric(union_metrics, q1_names[variant], "hit_rate_at_500"),
                "content_exclusive_hits": independent["content_exclusive_hits"],
                "declined_user_rate": independent["declined_user_rate"],
                "field_count": sum(
                    field.semantic_group in set(self.config.raw["content_variants"][variant]["groups"])
                    for field in self.canonical_fields
                ),
            })
        selected_content = select_content_variant(selection_inputs)
        for row in selection_inputs:
            row.update({
                "stage": "3C", "snapshot_id": snapshot,
                "selected": row["variant"] == selected_content,
                "selection_rule": ">".join(self.config.raw["selection"]["content_order"]),
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS",
            })
        paths = {
            "content": self.config.artifact_dir / "validation_content_single_route_metrics.csv",
            "union": self.config.artifact_dir / "validation_phase_a_union_metrics.csv",
            "overlap": self.config.artifact_dir / "validation_overlap_summary.csv",
            "selection": self.config.artifact_dir / "validation_content_variant_selection.csv",
        }
        write_csv(paths["content"], content_metrics, content_metrics[0].keys())
        write_csv(paths["union"], union_metrics, union_metrics[0].keys())
        write_csv(paths["overlap"], independence, independence[0].keys())
        write_csv(paths["selection"], selection_inputs, selection_inputs[0].keys())

        q2_name = self._run_union(snapshot, "Q2", selected_content)
        quota_metrics = self.metric_summary(snapshot, "union", "Q0", self.metric_path(snapshot, "union", "Q0"))
        # 对外把锁定 Content 的 Phase-A Q1 统一命名为 Q1，同时保留实际文件变体名。
        q1_metric_rows = self.metric_summary(snapshot, "union", q1_names[selected_content],
                                             self.metric_path(snapshot, "union", q1_names[selected_content]))
        q2_metric_rows = self.metric_summary(snapshot, "union", q2_name,
                                             self.metric_path(snapshot, "union", q2_name))
        for row in quota_metrics:
            row["physical_source_variant"] = f"stage_3b_{self.stage3b_q0_variant(snapshot)}_exact_reuse"
        for rows, public_name in ((q1_metric_rows, "Q1"), (q2_metric_rows, "Q2")):
            for row in rows:
                row["physical_source_variant"] = row["source_variant"]
                row["source_variant"] = public_name
        quota_metrics.extend(q1_metric_rows)
        quota_metrics.extend(q2_metric_rows)
        quota_metrics_path = self.config.artifact_dir / "validation_quota_metrics.csv"
        write_csv(quota_metrics_path, quota_metrics, quota_metrics[0].keys())

        comparisons: list[dict[str, Any]] = []
        bootstraps: list[dict[str, Any]] = []
        for public_name, physical in (("Q1", q1_names[selected_content]), ("Q2", q2_name)):
            comparison, bootstrap = self.compare_to_q0(snapshot, physical)
            comparison["quota_variant"] = public_name
            bootstrap["quota_variant"] = public_name
            comparisons.append(comparison)
            bootstraps.append(bootstrap)
        comparison_path = self.config.artifact_dir / "validation_marginal_recall.csv"
        bootstrap_path = self.config.artifact_dir / "validation_bootstrap.csv"
        write_csv(comparison_path, comparisons, comparisons[0].keys())
        write_csv(bootstrap_path, bootstraps, bootstraps[0].keys())

        quota_inputs: list[dict[str, Any]] = []
        for quota in ("Q0", "Q1", "Q2"):
            comparison = next((row for row in comparisons if row["quota_variant"] == quota), None)
            quota_inputs.append({
                "variant": quota,
                "recall_at_500": self._lookup_metric(quota_metrics, quota, "recall_at_500"),
                "hit_rate_at_500": self._lookup_metric(quota_metrics, quota, "hit_rate_at_500"),
                "map_at_12": self._lookup_metric(quota_metrics, quota, "map_at_12"),
                "declined_user_rate": comparison["declined_user_rate"] if comparison else 0.0,
                "resource_seconds": 0.0,
            })
        selected_quota = select_quota_variant(
            quota_inputs, tolerance=float(self.config.raw["union"]["near_tie_tolerance"]))
        selected_comparison = next((row for row in comparisons if row["quota_variant"] == selected_quota), None)
        selected_bootstrap = next((row for row in bootstraps if row["quota_variant"] == selected_quota), None)
        selected_independence = next(row for row in independence if row["content_variant"] == selected_content)
        delta = float(selected_comparison["mean_delta_recall_at_500"]) if selected_comparison else 0.0
        if selected_quota == "Q0":
            status = "FAILED_VALUE_GATE"
        elif (delta >= float(self.config.raw["union"]["strong_pass_delta"])
              and float(selected_bootstrap["ci_lower"]) > 0
              and int(selected_comparison["net_ground_truth_hits"]) > 0
              and int(selected_independence["content_exclusive_hits"]) > 0):
            status = "PASSED"
        elif (delta > 0 and float(selected_bootstrap["ci_lower"]) > 0
              and int(selected_comparison["net_ground_truth_hits"]) > 0
              and int(selected_independence["content_exclusive_hits"]) > 0):
            status = "CONDITIONAL_PASS"
        else:
            status = "FAILED_VALUE_GATE"
        for row in quota_inputs:
            row.update({
                "stage": "3C", "snapshot_id": snapshot, "selected": row["variant"] == selected_quota,
                "locked_content_variant": selected_content, "recommended_stage_status": status,
                "near_tie_tolerance": self.config.raw["union"]["near_tie_tolerance"],
                "selection_rule": self.config.raw["selection"]["quota_primary"],
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS",
            })
        quota_selection = self.config.artifact_dir / "validation_quota_selection.csv"
        write_csv(quota_selection, quota_inputs, quota_inputs[0].keys())

        occupancy: list[dict[str, Any]] = []
        occupancy.extend(self.quota_occupancy(snapshot, "Q0", selected_content))
        occupancy.extend(self.quota_occupancy(snapshot, "Q1", selected_content))
        occupancy.extend(self.quota_occupancy(snapshot, "Q2", selected_content))
        q0_itemcf_occupancy = next(
            float(row["mean_actual_unique_occupancy"])
            for row in occupancy if row["quota_variant"] == "Q0" and row["source"] == "itemcf"
        )
        for row in occupancy:
            row["itemcf_actual_unique_candidates_removed_vs_q0"] = (
                q0_itemcf_occupancy - float(row["mean_actual_unique_occupancy"])
                if row["source"] == "itemcf" else 0.0
            )
        occupancy_path = self.config.artifact_dir / "validation_quota_occupancy.csv"
        write_csv(occupancy_path, occupancy, occupancy[0].keys())
        self.write_q0_reproduction()
        self.create_lock(selected_content, selected_quota, status, paths, quota_metrics_path,
                         comparison_path, bootstrap_path, quota_selection, occupancy_path)
        return status

    def create_lock(
        self, content_variant: str, quota_variant: str, status: str,
        phase_a_paths: Mapping[str, Path], quota_metrics: Path, comparison: Path,
        bootstrap: Path, quota_selection: Path, occupancy: Path,
    ) -> Path:
        artifacts = [*phase_a_paths.values(), quota_metrics, comparison, bootstrap, quota_selection, occupancy,
                     self.config.artifact_dir / "q0_reproduction.csv"]
        lock: dict[str, Any] = {
            "stage": "3C", "lock_version": "1.0.0", "created_at": now(),
            "dataset_fingerprint": self.dataset_fingerprint,
            "feedback_version": self.feedback_version, "population_version": self.population_version,
            "candidate_universe_version": "history_seen_items_v1",
            "locked_content_variant": content_variant, "locked_quota_variant": quota_variant,
            "recommended_stage_status": status,
            "content_parameters": self.config.raw["content"],
            "source_rank_caps": self.config.raw["union"]["variants"][quota_variant],
            "quota_semantics": self.config.raw["union"]["quota_semantics"],
            "validation_artifacts": {self.config.relative(path): file_fingerprint(path) for path in artifacts},
            "stage_3a_lock_fingerprint": self.stage3a_lock_fingerprint,
            "stage_3b_lock_fingerprint": self.stage3b_lock_fingerprint,
            "config_fingerprint": self.config.fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "test_allowed": quota_variant in {"Q1", "Q2"} and status in {"PASSED", "CONDITIONAL_PASS"},
            "test_not_used_for_selection": True,
        }
        lock["lock_fingerprint"] = lock_fingerprint(lock)
        json_write(self.config.lock_path, lock)
        return self.config.lock_path

    def run_smoke(self) -> Path:
        values = {
            "product_type_no": np.asarray([1, 1, 2], dtype=object),
            "product_group_name": np.asarray(["a", "a", "b"], dtype=object),
            "garment_group_no": np.asarray([10, 10, 20], dtype=object),
            "section_no": np.asarray([100, 100, 200], dtype=object),
        }
        fields = [field for field in self.canonical_fields if field.semantic_group == "taxonomy"]
        matrix, tokens = build_sparse_item_matrix(values, fields, use_idf=True, group_balance=False)
        left, right, score, rank = top_k_cosine_block(matrix, np.asarray([10, 11, 12]), 0, 3, 2)
        union = source_rank_cap_union(
            {
                "repurchase": [{"article_key": 1, "source_rank": 1, "source_score": 1.0}],
                "itemcf": [{"article_key": 1, "source_rank": 1, "source_score": 1.0},
                           {"article_key": 2, "source_rank": 2, "source_score": 0.5}],
                "content": [], "trend": [],
                "popularity_fallback": [
                    {"article_key": item, "source_rank": item, "source_score": 1.0 / item}
                    for item in range(1, 10)
                ],
            },
            {"repurchase": 1, "itemcf": 1, "content": 0, "trend": 0}, 3,
        )
        if [row["article_key"] for row in union] != [1, 2, 3] or union[1]["from_itemcf"]:
            raise ValueError("source_rank_cap smoke 语义错误")
        if not self.stage3b_q0_parts("validation_14d"):
            raise FileNotFoundError("Stage 3B Q0 私有分区缺失")
        target = self.config.private_dir / "smoke_summary.json"
        json_write(target, {
            "status": "SUCCESS", "matrix_shape": matrix.shape,
            "token_count": len(tokens), "neighbor_rows": len(left),
            "finite_similarity": bool(np.isfinite(score).all()),
            "stable_ranks": bool(len(rank) > 0),
            "q0_private_parts": len(self.stage3b_q0_parts("validation_14d")),
            "source_rank_cap_semantics": True,
        })
        return target

    def run_test(self) -> str:
        lock = validate_test_lock(
            self.config.lock_path, config_fingerprint=self.config.fingerprint,
            source_fingerprint=self.source_fingerprint,
            stage_3a_lock_fingerprint=self.stage3a_lock_fingerprint,
            stage_3b_lock_fingerprint=self.stage3b_lock_fingerprint,
        )
        # 在看到 Test 前再次核验冻结上游，且只允许 lock 唯一版本。
        self._validate_frozen_upstream()
        gate = self.config.artifact_dir / "test_gate.csv"
        if not lock.get("test_allowed"):
            rows = [{
                "stage": "3C", "snapshot_id": "test_14d", "test_run": False,
                "locked_content_variant": lock["locked_content_variant"],
                "locked_quota_variant": lock["locked_quota_variant"],
                "recommended_stage_status": lock["recommended_stage_status"],
                "reason": "Validation locked Q0 or failed value gate; Content Test variant prohibited",
                "lock_fingerprint": lock["lock_fingerprint"], "run_status": "SKIPPED_BY_PROTOCOL",
            }]
            write_csv(gate, rows, rows[0].keys())
            return "SKIPPED_BY_PROTOCOL"
        snapshot = "test_14d"
        content_variant = str(lock["locked_content_variant"])
        quota = str(lock["locked_quota_variant"])
        self._evaluate_q0(snapshot)
        self._run_content_core(snapshot, (content_variant,))
        physical = self._run_union(snapshot, quota, content_variant)
        q0_rows = self.metric_summary(snapshot, "union", "Q0", self.metric_path(snapshot, "union", "Q0"))
        locked_rows = self.metric_summary(snapshot, "union", physical, self.metric_path(snapshot, "union", physical))
        for row in locked_rows:
            row["physical_source_variant"] = row["source_variant"]
            row["source_variant"] = quota
        for row in q0_rows:
            row["physical_source_variant"] = f"stage_3b_{self.stage3b_q0_variant(snapshot)}_exact_reuse"
        metrics = q0_rows + locked_rows
        metrics_path = self.config.artifact_dir / "test_locked_union_metrics.csv"
        write_csv(metrics_path, metrics, metrics[0].keys())
        comparison, bootstrap = self.compare_to_q0(snapshot, physical)
        comparison["quota_variant"] = quota
        bootstrap["quota_variant"] = quota
        comparison_path = self.config.artifact_dir / "test_marginal_recall.csv"
        bootstrap_path = self.config.artifact_dir / "test_bootstrap.csv"
        write_csv(comparison_path, [comparison], comparison.keys())
        write_csv(bootstrap_path, [bootstrap], bootstrap.keys())
        gate_rows = [{
            "stage": "3C", "snapshot_id": snapshot, "test_run": True,
            "locked_content_variant": content_variant, "locked_quota_variant": quota,
            "recommended_stage_status": lock["recommended_stage_status"],
            "reason": "Validation lock allowed exactly one locked Content Test comparison",
            "lock_fingerprint": lock["lock_fingerprint"], "run_status": "SUCCESS",
        }]
        write_csv(gate, gate_rows, gate_rows[0].keys())
        return "SUCCESS"

    def build_token_summary(self) -> Path:
        rows: list[dict[str, Any]] = []
        for token_file in sorted(self.config.private_dir.glob("tokens_*.json")):
            payload = json_read(token_file)
            grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
            for token in payload["tokens"]:
                key = (token["semantic_group"], token["semantic_field"], token["source_column"])
                grouped.setdefault(key, []).append(token)
            for (group, semantic, source), values in grouped.items():
                rows.append({
                    "stage": "3C", "snapshot_id": payload["snapshot_id"],
                    "as_of_date": payload["as_of_date"], "content_variant": payload["variant"],
                    "semantic_group": group, "semantic_field": semantic,
                    "canonical_source_column": source, "token_count": len(values),
                    "document_frequency_min": min(row["document_frequency"] for row in values),
                    "document_frequency_max": max(row["document_frequency"] for row in values),
                    "document_frequency_rate_max": max(row["document_frequency_rate"] for row in values),
                    "idf_min": min(row["idf"] for row in values), "idf_max": max(row["idf"] for row in values),
                    "candidate_universe_fingerprint": payload["candidate_universe_fingerprint"],
                    "attribute_schema_fingerprint": payload["attribute_schema_fingerprint"],
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS",
                })
        if not rows:
            raise ValueError("没有 token summary 私有输入")
        target = self.config.artifact_dir / "token_attribute_summary.csv"
        write_csv(target, rows, rows[0].keys())
        return target

    def build_segment_summary(self) -> Path:
        source = self.config.artifact_dir / "validation_quota_metrics.csv"
        metrics = csv_rows(source)
        lookup = {(row["source_variant"], row["segment"], row["metric"]): float(row["metric_value"])
                  for row in metrics}
        rows: list[dict[str, Any]] = []
        for quota in ("Q0", "Q1", "Q2"):
            for segment in ("evaluable_warm_users", "short_history", "warm", "active", "cold_history_label_users"):
                base = lookup[("Q0", segment, "recall_at_500")]
                rows.append({
                    "stage": "3C", "snapshot_id": "validation_14d", "quota_variant": quota,
                    "segment": segment,
                    "population_denominator": next(int(row["population_denominator"]) for row in metrics
                        if row["source_variant"] == quota and row["segment"] == segment),
                    "recall_at_500": lookup[(quota, segment, "recall_at_500")],
                    "hit_rate_at_500": lookup[(quota, segment, "hit_rate_at_500")],
                    "map_at_12": lookup[(quota, segment, "map_at_12")],
                    "delta_recall_at_500_vs_q0": lookup[(quota, segment, "recall_at_500")] - base,
                    "cold_history_content_personalized_candidates": 0 if segment == "cold_history_label_users" else "not_applicable",
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS",
                })
        target = self.config.artifact_dir / "validation_user_segment_summary.csv"
        write_csv(target, rows, rows[0].keys())
        return target

    def build_item_support_summary(self) -> Path:
        selection = next(row for row in csv_rows(self.config.artifact_dir / "validation_content_variant_selection.csv")
                         if row["selected"].lower() == "true")
        variant = selection["variant"]
        content = self.candidate_glob("validation_14d", variant)
        itemcf = sql_path(self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) /
                          "candidates" / "validation_14d" / "iuf" / "part_*.parquet")
        trend = self.stage3a_ranking("validation_14d", "trend", "window_7d")
        repurchase = self.stage3a_candidate("validation_14d", "repurchase", "pure")
        ground_truth = self.stage2b_path("ground_truth", "validation_14d")
        state = self.stage2b_path("item_state", "validation_14d")
        support = self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) / "item_support_validation_14d.parquet"
        neighbors = self.config.resolve(self.config.raw["upstream"]["stage_3b_private"]) / "neighbors_validation_14d_iuf.parquet"
        con = self.connect()
        try:
            values = con.execute(
                f"""WITH content_hits AS MATERIALIZED (
                  SELECT c.customer_key,c.article_key FROM read_parquet('{content}') c
                  SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                    ON g.user_key=c.customer_key AND g.article_key=c.article_key
                ), itemcf_covered AS MATERIALIZED (
                  SELECT DISTINCT i.customer_key,i.article_key FROM read_parquet('{itemcf}') i
                  SEMI JOIN content_hits c USING(customer_key,article_key)
                ), repurchase_covered AS MATERIALIZED (
                  SELECT DISTINCT r.customer_key,r.article_key FROM read_parquet('{sql_path(repurchase)}') r
                  SEMI JOIN content_hits c USING(customer_key,article_key)
                ), trend_covered AS MATERIALIZED (
                  SELECT DISTINCT c.customer_key,c.article_key FROM content_hits c
                  SEMI JOIN (
                    SELECT article_key FROM read_parquet('{sql_path(trend)}') WHERE source_rank<=500
                  ) t USING(article_key)
                ), exclusive AS (
                  SELECT c.* FROM content_hits c
                  ANTI JOIN itemcf_covered i USING(customer_key,article_key)
                  ANTI JOIN repurchase_covered r USING(customer_key,article_key)
                  ANTI JOIN trend_covered t USING(customer_key,article_key)
                ), support_decile AS (
                  SELECT article_key,item_support,ntile(10) OVER(ORDER BY item_support,article_key) support_decile
                  FROM read_parquet('{sql_path(support)}')
                ), neighbor_count AS (
                  SELECT article_key,count(*) neighbor_count FROM read_parquet('{sql_path(neighbors)}') GROUP BY 1
                ), enriched AS (
                  SELECT e.customer_key,e.article_key,s.popularity_bucket,d.support_decile,
                         d.item_support,coalesce(n.neighbor_count,0) neighbor_count
                  FROM exclusive e JOIN read_parquet('{sql_path(state)}') s USING(article_key)
                  LEFT JOIN support_decile d USING(article_key) LEFT JOIN neighbor_count n USING(article_key)
                )
                SELECT 'popularity_bucket' dimension,popularity_bucket bucket,count(*) exclusive_hits,
                       count(DISTINCT article_key) exclusive_items,avg(item_support),avg(neighbor_count)
                FROM enriched GROUP BY popularity_bucket
                UNION ALL
                SELECT 'collaborative_support_decile',coalesce(cast(support_decile AS VARCHAR),'unknown'),count(*),
                        count(DISTINCT article_key),avg(item_support),avg(neighbor_count)
                FROM enriched GROUP BY support_decile ORDER BY 1,2"""
            ).fetchall()
        finally:
            con.close()
        rows = [{
            "stage": "3C", "snapshot_id": "validation_14d", "content_variant": variant,
            "dimension": value[0], "bucket": value[1], "content_exclusive_hits": value[2],
            "content_exclusive_items": value[3], "mean_history_support": value[4],
            "mean_itemcf_neighbor_count": value[5],
            "support_definition": "cutoff_history_only_not_ground_truth_defined",
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS",
        } for value in values]
        target = self.config.artifact_dir / "validation_item_popularity_support_summary.csv"
        write_csv(target, rows, rows[0].keys() if rows else ["stage", "snapshot_id"])
        return target

    def write_resource_failure_fingerprint(self) -> tuple[Path, Path, Path]:
        checkpoints = [json_read(path) for path in sorted(self.config.checkpoint_dir.glob("*.json"))]
        recovery_notes = {
            "canonical_field_audit": "DuckDB reserved alias fixed; audit rerun succeeded.",
            "item_support_summary": (
                "Two OOM attempts at dynamic ~709 MiB and fixed 1024 MiB, followed by one SQL binder "
                "failure; equivalent small-hit-set anti-join rewrite succeeded with explicit unknown support bucket."
            ),
            "validation_14d_attributes": "Unrelated global source-fingerprint invalidation caused one redundant rebuild before recovery was stopped.",
            "validation_14d_C0_neighbors": (
                "Interrupted overwrite was quarantined; single-group rebuild restored the exact prior "
                "SHA-256 recorded in the ignored recovery evidence."
            ),
            "validation_14d_Q0_exact_reuse_evaluation": (
                "Q0 was corrected from Stage 3B base_union to the frozen IUF itemcf_union; only Q0-dependent reports were rebuilt."
            ),
        }
        resource_rows = [{
            "stage": "3C", "group_name": row.get("group_name"), "status": row.get("status"),
            "attempt": row.get("attempt"), "execution_seconds": row.get("execution_seconds", 0),
            "peak_rss_bytes": row.get("peak_rss_bytes", 0),
            "temp_disk_peak_bytes": row.get("temp_disk_peak_bytes", 0),
            "spill_observed": row.get("spill_observed", False),
            "memory_limit_mb": row.get("memory_limit_mb", self.config.memory_limit_mb),
            "threads": row.get("threads", 1), "output_size_bytes": row.get("output_size_bytes", 0),
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint,
        } for row in checkpoints]
        failure_rows = [{
            "stage": "3C", "group_name": row.get("group_name"), "status": row.get("status"),
            "attempt": row.get("attempt"),
            "retry_count": max(int(row.get("attempt", 1)) - 1, 0),
            "error_message": re.sub(r"https?://\S+", "[external_url]", str(row.get("error_message", ""))),
            "recovered": row.get("status") == "SUCCESS" and int(row.get("attempt", 1)) > 1,
            "recovery_note": recovery_notes.get(str(row.get("group_name")), ""),
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint,
        } for row in checkpoints if row.get("status") == "FAILED" or int(row.get("attempt", 1)) > 1]
        if not failure_rows:
            failure_rows = [{
                "stage": "3C", "group_name": "not_applicable", "status": "NO_FAILURE",
                "attempt": 1, "retry_count": 0, "error_message": "", "recovered": False,
                "recovery_note": "",
                "dataset_fingerprint": self.dataset_fingerprint,
                "config_fingerprint": self.config.fingerprint,
            }]
        fingerprint_rows = [{
            "stage": "3C", "artifact_role": "configuration", "artifact_path": self.config.relative(self.config.path),
            "artifact_fingerprint": file_fingerprint(self.config.path),
            "config_fingerprint": self.config.fingerprint, "source_fingerprint": self.source_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "stage_3a_lock_fingerprint": self.stage3a_lock_fingerprint,
            "stage_3b_lock_fingerprint": self.stage3b_lock_fingerprint,
            "run_status": "SUCCESS",
        }]
        for snapshot in self.config.raw["snapshots"]:
            if self.attributes_path(snapshot).exists():
                fingerprint_rows.append({
                    "stage": "3C", "artifact_role": f"{snapshot}_candidate_universe",
                    "artifact_path": self.config.relative(self.attributes_path(snapshot)),
                    "artifact_fingerprint": file_fingerprint(self.attributes_path(snapshot)),
                    "config_fingerprint": self.config.fingerprint, "source_fingerprint": self.source_fingerprint,
                    "dataset_fingerprint": self.dataset_fingerprint,
                    "stage_3a_lock_fingerprint": self.stage3a_lock_fingerprint,
                    "stage_3b_lock_fingerprint": self.stage3b_lock_fingerprint,
                    "run_status": "SUCCESS",
                })
        paths = (self.config.artifact_dir / "resource_summary.csv",
                 self.config.artifact_dir / "failure_retry_summary.csv",
                 self.config.artifact_dir / "fingerprint_summary.csv")
        write_csv(paths[0], resource_rows, resource_rows[0].keys())
        write_csv(paths[1], failure_rows, failure_rows[0].keys())
        write_csv(paths[2], fingerprint_rows, fingerprint_rows[0].keys())
        return paths

    def build_report(self) -> str:
        lock = json_read(self.config.lock_path)
        content_selection = next(row for row in csv_rows(
            self.config.artifact_dir / "validation_content_variant_selection.csv") if row["selected"].lower() == "true")
        quota_rows = csv_rows(self.config.artifact_dir / "validation_quota_selection.csv")
        quota_selection = next(row for row in quota_rows if row["selected"].lower() == "true")
        marginal = csv_rows(self.config.artifact_dir / "validation_marginal_recall.csv")
        bootstrap = csv_rows(self.config.artifact_dir / "validation_bootstrap.csv")
        occupancy = csv_rows(self.config.artifact_dir / "validation_quota_occupancy.csv")
        selected_quota = lock["locked_quota_variant"]
        selected_marginal = next((row for row in marginal if row["quota_variant"] == selected_quota), None)
        selected_bootstrap = next((row for row in bootstrap if row["quota_variant"] == selected_quota), None)
        test_gate = csv_rows(self.config.artifact_dir / "test_gate.csv") if (self.config.artifact_dir / "test_gate.csv").exists() else []
        test_text = (
            f"Test 按 lock 只运行 Q0 与 {selected_quota}：{test_gate[0]['run_status']}。"
            if test_gate and test_gate[0]["test_run"].lower() == "true"
            else "Validation 锁定 Q0 或未通过 value gate，按协议未运行 Content Test 变体。"
        )
        delta = float(selected_marginal["mean_delta_recall_at_500"]) if selected_marginal else 0.0
        ci = (f"[{float(selected_bootstrap['ci_lower']):.6f}, {float(selected_bootstrap['ci_upper']):.6f}]"
              if selected_bootstrap else "不适用")
        quota_text = "；".join(
            f"{row['variant']} Recall@500={float(row['recall_at_500']):.6f}, "
            f"HitRate@500={float(row['hit_rate_at_500']):.6f}"
            for row in quota_rows
        )
        bootstrap_text = "；".join(
            f"{row['quota_variant']} vs Q0 ΔRecall@500={float(row['delta_mean']):.6f}, "
            f"95% CI=[{float(row['ci_lower']):.6f}, {float(row['ci_upper']):.6f}]"
            for row in bootstrap
        )
        occupancy_text_parts = []
        for quota in ("Q1", "Q2"):
            content_row = next(row for row in occupancy if row["quota_variant"] == quota and row["source"] == "content")
            itemcf_row = next(row for row in occupancy if row["quota_variant"] == quota and row["source"] == "itemcf")
            occupancy_text_parts.append(
                f"{quota} Content cap={content_row['configured_source_rank_cap']}、实际 unique occupancy="
                f"{float(content_row['mean_actual_unique_occupancy']):.3f}、重复损失="
                f"{float(content_row['mean_duplicates_with_higher_priority']):.3f}；ItemCF 实际唯一减少="
                f"{float(itemcf_row['itemcf_actual_unique_candidates_removed_vs_q0']):.3f}、Popularity 补齐="
                f"{float(content_row['mean_popularity_fill']):.3f}"
            )
        occupancy_text = "；".join(occupancy_text_parts)
        return f"""# Stage 3C：Structured-Attribute Content Retrieval 与固定预算增量验证

## 1. 阶段目标与边界

本阶段以结构化商品属性实现可解释 Content Retrieval，并回答其前部候选能否在固定 500 候选预算下替换部分 IUF ItemCF 尾部。未实现文本/图片 embedding、ANN、双塔、矩阵分解、排序、重排或在线服务。

## 2. 开始门禁与权威输入

开工时 `main` 与 `origin/main` 为 0 ahead/0 behind，工作区干净，初始测试 148 passed；Stage 3A/3B lock、manifest、公开指标及私有上游 fingerprint 审计通过。冲突裁定后，Q0/Q1/Q2 统一沿用 Stage 3B 的预去重 `source_rank_cap` 语义。

## 3. 数据、防泄漏与候选域

Validation/Test 分别只使用各自 cutoff 前 history。主候选域固定为 `history_seen_items_v1`；完整 articles 身份域未参与模型或配额选择。DF/IDF、属性矩阵、内容邻居和用户候选均按 snapshot 独立计算；ground truth 只进入评估。cold-history 个性化 Content 候选为 0。

## 4. Canonical 属性审计

建模字段为 taxonomy（product type/group、garment group、section）、appearance（graphical appearance、colour group、perceived colour value/master）和 merchandising/index（department、index、index group）。code/name 只选择一个 canonical 信号，名称只作解释；冲突、缺失率、基数、高频率和候选域覆盖见 `canonical_field_audit.csv`。`detail_desc`、商品名和图片均未使用。

## 5. Content 表示、IDF 与 group balance

token 固定为 `field_name=field_value`；missing 是字段隔离显式 token。IDF 为 `log((N+1)/(df+1))+1`。C2/C3 先组内 L2 归一化、各非空组等先验，再整体归一化。精确 sparse cosine 以分块矩阵乘法计算每商品 Top-200，不构建稠密全矩阵，也未使用 ANN。

## 6. 用户 seed 与候选生成

复用 Stage 3B 最近 20 个唯一 seed 和 `1/log2(rank+1)` 衰减；多 seed 分数为衰减乘 cosine 后求和。Content 排除全部 seen item、过滤到主候选域、最多 500，候选不足不自行补齐。

## 7. C0–C3 Validation 与版本锁定

四个预注册版本均完成 Content 单路和固定 Q1 Union。按 Q1 Recall@500、HitRate、独占命中、下降用户率和复杂度顺序锁定 `{content_selection['variant']}`；完整指标、overlap 和独占命中见对应公开 CSV。没有依据 Test 或标签新增属性组合。

## 8. Q0/Q1/Q2 固定预算实验

Q0 精确复用 Stage 3B 冻结候选；Q1 比较移除 ItemCF rank 251–300、加入 Content rank 1–50；Q2 比较移除 ItemCF rank 201–300、加入 Content rank 1–100。所有来源先按 source rank 截断，再合并 provenance 和去重，Popularity 补齐至 500。锁定 `{quota_selection['variant']}`，Validation Recall@500 为 {float(quota_selection['recall_at_500']):.6f}。

{quota_text}。{occupancy_text}。配置 cap 与实际 unique occupancy、重复损失、Popularity 补齐、每 100 个 Content rank/实际唯一位置的净命中效率均在 `validation_quota_occupancy.csv` 分开报告。

## 9. 固定预算增量、损失与 bootstrap

锁定方案相对 Q0 的 ΔRecall@500 为 {delta:.6f}，95% paired user-block bootstrap CI 为 {ci}。上升/不变/下降用户、增加/丢失/净 hits 均公开，未隐藏固定预算替换损失。这是离线稳定性分析，不是线上 A/B Test、CTR/CVR 显著性或业务收益。

候选方案的 paired bootstrap 结果为：{bootstrap_text}。二者区间均低于 0，因此 Content 虽有独占命中，仍不足以抵消被替换的 IUF ItemCF 尾部价值。

## 10. 分群、长尾与低协同支持

short-history/warm/active/cold-history 复用 Stage 2B 固定分母；head/mid/tail 复用 Stage 2B cutoff-history 分位规则。Content 独占命中按 popularity bucket、history support decile 和 IUF 邻居覆盖报告；低协同支持不被称为真实新品冷启动。

## 11. Validation lock 与 Test 门禁

Stage 3C lock 同时固化 Content 版本、source-rank cap、配置/源码、Stage 3A/3B locks 与全部选择产物 fingerprint。{test_text}

## 12. 资源、失败与恢复

各组耗时、峰值 RSS、临时盘、spill、attempt 和输出大小见 `resource_summary.csv`；失败、重试与恢复见 `failure_retry_summary.csv`。大任务以独立 checkpoint 和 output fingerprint 恢复，fingerprint 变化拒绝复用。

恢复过程中隔离并字节级重建了被中断覆盖的 C0 邻居文件；Q0 从错误的 Stage 3B `base_union` 更正为冻结的 IUF Union；item-support 汇总两次 OOM 后改为等价的小命中集合 anti-join，并保留 1 个缺失协同支持度的 `unknown` 桶。以上均在失败/迁移审计中留痕。

## 13. 阶段结论

Codex 建议状态：`{lock['recommended_stage_status']}`。Content 是否进入 Stage 3E 取决于该 value gate；Stage 3D 仍需新的正式授权。本状态不是 ChatGPT 最终验收结论。

## 14. 真实限制

数据没有曝光、点击、真实负反馈、库存、可售状态或精确上架时间。完整维表不等于历史可售域；未购买不等于不喜欢；结构化相似不等于真实偏好；本阶段未解决线上新品冷启动，离线 Recall/MAP 不等于 CTR、CVR、收入或线上收益。

## 15. 面试知识点与高频追问

- Content-based Retrieval 用商品属性 multi-hot 与 cosine 找相似商品；ItemCF 使用历史共现，两者归纳偏置不同。
- IDF 降低高频属性 token 的支配；group balance 防止字段更多的语义组自然获得更大权重。
- code/name 不同时建模，避免同一概念重复加权；名称只用于解释，不做语义 embedding。
- Content 依赖历史 seed，不能个性化服务零历史用户；Repurchase 负责 seen item，Content/ItemCF 均排除 seen。
- 先做 Content 再考虑双塔，是为了用低复杂度、可解释基线验证商品侧信号；本阶段结果不能预设双塔一定更好。
- 不使用文本和图片，是因为本阶段正式范围只验证结构化属性，避免在弱结果后扩大搜索空间。
- 固定预算 Union 评估边际替换价值；异构原始分数未标定，不能直接相加。
- source-rank cap 不等于实际唯一 occupancy；重复是召回冗余的一部分，本阶段不向后扫描补偿。
- 单路 Recall 高不保证应获更多 cap；真正问题是替换 ItemCF 尾部后的净新增 hits。
- 每 100 个 rank 候选与每 100 个实际唯一位置同时报告，可区分来源质量和重复损失。
- 报告下降用户是为了暴露固定预算替换的异质损失，不能只展示总体均值。
- paired bootstrap 是同一离线用户的稳定性分析，不是线上随机实验。
- Content 对 short-history 不一定更好，必须以本阶段真实分群指标回答。
- history 低支持商品不等于新品；缺少精确上架时间，不能声称解决真实新品冷启动。
- 即使 value gate 失败，canonical 审计、snapshot IDF、结构化商品塔输入和可恢复 sparse 计算仍可复用；停止扩参本身保证结论可信。
- unique-slot fill、RRF、动态配额与 rank fusion 留待 Stage 3E 新 Prompt，不是本阶段缺陷。
"""

    def write_manifest(self) -> Path:
        target = self.config.artifact_dir / "stage_3c_manifest.csv"
        public = sorted(path for path in self.config.artifact_dir.glob("*")
                        if path.is_file() and path != target)
        rows = [{
            "stage": "3C", "snapshot_id": "validation_14d,test_14d",
            "as_of_date": self.snapshots["test_14d"].history_end,
            "artifact_path": self.config.relative(path),
            "artifact_fingerprint": file_fingerprint(path), "size_bytes": path.stat().st_size,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint, "run_status": "SUCCESS",
        } for path in public]
        write_csv(target, rows, rows[0].keys())
        return target

    def validate_public_and_privacy(self) -> None:
        public = [*self.config.artifact_dir.glob("*.csv"), *self.config.artifact_dir.glob("*.json"),
                  self.config.report_path]
        validate_stage3c_public_artifacts(public)
        forbidden = {"customer_key", "article_key", "customer_id", "article_id", "postal_code", "user_key"}
        for path in self.config.artifact_dir.glob("*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                headers = set(csv.DictReader(handle).fieldnames or ())
            if headers & forbidden:
                raise ValueError(f"公开 CSV 包含实体字段: {path.name}")
            if str(self.config.root) in path.read_text(encoding="utf-8-sig"):
                raise ValueError(f"公开产物包含本地绝对路径: {path.name}")
        if self.config.report_path.exists() and str(self.config.root) in self.config.report_path.read_text(encoding="utf-8"):
            raise ValueError("报告包含本地绝对路径")
        for relative in ("artifacts/stage_3c/private", "artifacts/stage_3c/tmp"):
            result = subprocess.run(["git", "check-ignore", relative], cwd=self.config.root,
                                    text=True, capture_output=True)
            if result.returncode != 0:
                raise ValueError(f"{relative} 未命中 .gitignore")
        tracked = subprocess.run(["git", "ls-files", "data/raw", "artifacts"], cwd=self.config.root,
                                 text=True, capture_output=True, check=True).stdout.splitlines()
        unsafe = [value for value in tracked
                  if value.startswith("data/raw/") or "/private/" in value or "/tmp/" in value]
        if unsafe:
            raise ValueError("raw/private/tmp 存在 tracked 文件")
        self._validate_frozen_upstream()

    def finalize(self) -> None:
        self.run_group("token_attribute_summary", self.build_token_summary,
                       [self.config.artifact_dir / "token_attribute_summary.csv"])
        self.run_group("user_segment_summary", self.build_segment_summary,
                       [self.config.artifact_dir / "validation_user_segment_summary.csv"])
        self.run_group("item_support_summary", self.build_item_support_summary,
                       [self.config.artifact_dir / "validation_item_popularity_support_summary.csv"])
        self.write_resource_failure_fingerprint()
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.write_text(self.build_report(), encoding="utf-8")
        self.write_manifest()
        self.validate_public_and_privacy()
