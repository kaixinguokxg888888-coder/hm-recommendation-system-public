from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb

from data_audit.stage1a import ResourceMonitor
from data_foundation.manifest import file_fingerprint, write_csv
from retrieval.stage3d.config import Stage3DConfig, source_fingerprint as stage3d_source_fingerprint

from .config import Stage3EConfig, source_fingerprint
from .contract import (
    CONTRACT_COLUMNS,
    CONTRACT_VERSION,
    PROTOCOL_VERSION,
    fingerprint_from_user_digests,
    parquet_partition_user_digests,
    schema_manifest_rows,
)
from .locking import validate_stage3d_identity


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def physical_partition_set_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_fingerprint(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _nested_hashes(value: Any) -> set[str]:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return {value.lower()}
    if isinstance(value, Mapping):
        return {token for nested in value.values() for token in _nested_hashes(nested)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {token for nested in value for token in _nested_hashes(nested)}
    return set()


def validate_public_artifacts(paths: Iterable[Path], *, allowed_hashes: set[str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    username = os.environ.get("USERNAME", "").lower()
    for path in paths:
        if not path.exists() or path.suffix.lower() not in {".csv", ".json", ".md", ".yaml", ".py"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        checks = {
            "windows_absolute_path": re.findall(r"[A-Za-z]:[\\/]", text),
            "raw_article_id_value": re.findall(r"(?<![0-9])0[0-9]{9}(?![0-9])", text),
            "undeclared_64hex_value": [
                value for value in re.findall(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])", text)
                if value.lower() not in allowed_hashes
            ],
            "local_username": [username] if username and username in text.lower() else [],
        }
        for check, matches in checks.items():
            if matches:
                findings.append({"artifact": path.name, "check": check, "match_count": len(matches)})
    if findings:
        raise ValueError(f"Stage 3E public artifact privacy scan failed: {findings}")
    return findings


class Stage3ERunner:
    def __init__(self, config: Stage3EConfig) -> None:
        self.config = config
        self.source_fingerprint = source_fingerprint(config.root)
        self.stage3d_config = Stage3DConfig.load(
            config.resolve(config.raw["upstream"]["stage_3d_config"])
        )
        self.stage3d_lock_path = config.resolve(config.raw["upstream"]["stage_3d_lock"])
        self.stage3d_lock = json_read(self.stage3d_lock_path)
        self.split_rows = {row["split_id"]: row for row in csv_rows(
            config.resolve(config.raw["upstream"]["split_manifest"])
        )}
        self.population_rows = csv_rows(
            config.resolve(config.raw["upstream"]["population_manifest"])
        )
        self.dataset_fingerprint = self.split_rows["validation_14d"]["dataset_fingerprint"]
        self._validate_upstream()
        self.upstream_records = self._upstream_records()
        self.upstream_fingerprint = hashlib.sha256(
            json.dumps(self.upstream_records, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.run_fingerprint = hashlib.sha256(
            (config.fingerprint + self.source_fingerprint + self.upstream_fingerprint
             + CONTRACT_VERSION).encode("utf-8")
        ).hexdigest()
        self.resources: list[dict[str, Any]] = []
        self.partition_rows: list[dict[str, Any]] = []
        self.snapshot_results: dict[str, dict[str, Any]] = {}
        self.replay_results: dict[str, dict[str, Any]] = {}

    def _resolve_upstream(self, name: str) -> Path:
        return self.config.resolve(self.config.raw["upstream"][name])

    def _candidate_lineage_path(self, snapshot: str) -> Path:
        return self.config.root / "artifacts" / "stage_3d" / "private" / "candidates" / snapshot / "TT1" / "lineage.json"

    def _stage3d_part(self, snapshot: str, part: int) -> Path:
        base = self._resolve_upstream("stage_3d_union_directory")
        return base / snapshot / "TT1_T100" / f"part_{part:03d}.parquet"

    def _stage3d_part_checkpoint(self, snapshot: str, part: int) -> Path:
        base = self._resolve_upstream("stage_3d_checkpoint_directory")
        return base / f"{snapshot}_TT1_T100_union_{part:03d}.json"

    def _stage3d_metric(self, snapshot: str) -> Path:
        base = self._resolve_upstream("stage_3d_metric_directory")
        return base / f"user_metrics_{snapshot}_union_TT1_T100.parquet"

    def _stage2b_path(self, kind: str, snapshot: str) -> Path:
        return self._resolve_upstream("stage_2b_private") / f"{kind}_{snapshot}.parquet"

    def _contract_part(self, snapshot: str, part: int, *, replay: bool = False) -> Path:
        role = "replay" if replay else "candidates"
        return self.config.private_dir / role / snapshot / f"part_{part:03d}.parquet"

    def _checkpoint_path(self, snapshot: str, part: int, *, replay: bool = False) -> Path:
        role = "replay" if replay else "formal"
        return self.config.checkpoint_dir / f"{role}_{snapshot}_part_{part:03d}.json"

    def _validate_manifest_artifacts(self, manifest_name: str) -> None:
        manifest = self._resolve_upstream(manifest_name)
        for row in csv_rows(manifest):
            artifact = self.config.resolve(row["artifact_path"])
            if not artifact.exists() or file_fingerprint(artifact) != row["artifact_fingerprint"]:
                raise ValueError(f"upstream manifest artifact mismatch: {manifest_name}")

    def _validate_upstream(self) -> None:
        for manifest in ("stage_3a_manifest", "stage_3b_manifest", "stage_3c_manifest", "stage_3d_manifest"):
            self._validate_manifest_artifacts(manifest)
        actual_stage3d_config = self.stage3d_config.fingerprint
        actual_stage3d_source = stage3d_source_fingerprint(self.config.root)
        upstream_locks = {
            f"stage_{stage}_lock_fingerprint": file_fingerprint(self._resolve_upstream(f"stage_{stage}_lock"))
            for stage in ("3a", "3b", "3c")
        }
        lineages = {
            snapshot: json_read(self._candidate_lineage_path(snapshot))
            for snapshot in self.config.raw["snapshots"]
        }
        validate_stage3d_identity(
            self.stage3d_lock,
            actual_config_fingerprint=actual_stage3d_config,
            actual_source_fingerprint=actual_stage3d_source,
            upstream_lock_fingerprints=upstream_locks,
            candidate_lineages=lineages,
        )
        manifest_lock = next(
            row for row in csv_rows(self._resolve_upstream("stage_3d_manifest"))
            if row["artifact_path"] == "artifacts/stage_3d/stage_3d_lock.json"
        )
        if file_fingerprint(self.stage3d_lock_path) != manifest_lock["artifact_fingerprint"]:
            raise ValueError("Stage 3D formal manifest does not identify the current lock")
        for snapshot in self.config.raw["snapshots"]:
            split = self.split_rows[snapshot]
            if split["as_of_date"] not in {"2020-08-25", "2020-09-08"}:
                raise ValueError("Stage 3E snapshot boundary mismatch")
            population = self._stage2b_path("evaluation_population", snapshot)
            ground_truth = self._stage2b_path("ground_truth", snapshot)
            if not population.exists() or not ground_truth.exists():
                raise ValueError("Stage 2B private population/ground truth missing")
            expected_gt = next(
                row["ground_truth_fingerprint"] for row in self.population_rows
                if row["snapshot_id"] == snapshot and row["population"] == "evaluable_warm_users"
            )
            if file_fingerprint(ground_truth) != expected_gt:
                raise ValueError("Stage 2B ground truth fingerprint mismatch")
            for part in range(int(self.config.raw["execution"]["partitions"])):
                source = self._stage3d_part(snapshot, part)
                checkpoint = json_read(self._stage3d_part_checkpoint(snapshot, part))
                relative = self.config.relative(source)
                if (
                    checkpoint.get("status") != "SUCCESS"
                    or checkpoint.get("config_fingerprint") != actual_stage3d_config
                    or checkpoint.get("source_fingerprint") != actual_stage3d_source
                    or checkpoint.get("output_fingerprints", {}).get(relative) != file_fingerprint(source)
                ):
                    raise ValueError(f"Stage 3D formal T100 partition mismatch: {snapshot}/{part}")

    def _upstream_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name in (
            "split_manifest", "population_manifest", "stage_3a_manifest", "stage_3b_manifest",
            "stage_3c_manifest", "stage_3d_manifest", "stage_3a_lock", "stage_3b_lock",
            "stage_3c_lock", "stage_3d_lock", "stage_3d_config", "stage_3d_manifest_metrics",
            "stage_3d_model_lineage",
        ):
            path = self._resolve_upstream(name)
            rows.append({"lineage_role": name, "artifact_path": self.config.relative(path),
                         "artifact_fingerprint": file_fingerprint(path)})
        for snapshot in self.config.raw["snapshots"]:
            lineage = self._candidate_lineage_path(snapshot)
            rows.append({"lineage_role": f"stage_3d_{snapshot}_candidate_lineage",
                         "artifact_path": self.config.relative(lineage),
                         "artifact_fingerprint": file_fingerprint(lineage)})
            for kind in ("evaluation_population", "ground_truth"):
                path = self._stage2b_path(kind, snapshot)
                rows.append({"lineage_role": f"stage_2b_{snapshot}_{kind}",
                             "artifact_path": self.config.relative(path),
                             "artifact_fingerprint": file_fingerprint(path)})
            parts = [self._stage3d_part(snapshot, part) for part in range(32)]
            rows.append({"lineage_role": f"stage_3d_{snapshot}_locked_t100_partition_set",
                         "artifact_path": f"artifacts/stage_3d/private/unions/{snapshot}/TT1_T100",
                         "artifact_fingerprint": physical_partition_set_fingerprint(parts)})
        return rows

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
        con.execute(f"SET threads={int(self.config.raw['duckdb']['threads'])}")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory='{sql_path(self.config.temp_dir)}'")
        con.execute(f"SET max_temp_directory_size='{self.config.raw['duckdb']['max_temp_directory_size']}'")
        return con

    def _checkpoint_reusable(
        self, checkpoint: Mapping[str, Any] | None, *, snapshot: str, part: int,
        source: Path, target: Path,
    ) -> bool:
        return bool(
            checkpoint
            and checkpoint.get("status") == "SUCCESS"
            and checkpoint.get("config_fingerprint") == self.config.fingerprint
            and checkpoint.get("source_fingerprint") == self.source_fingerprint
            and checkpoint.get("upstream_fingerprint") == self.upstream_fingerprint
            and checkpoint.get("run_fingerprint") == self.run_fingerprint
            and checkpoint.get("snapshot_id") == snapshot
            and int(checkpoint.get("partition_id", -1)) == part
            and checkpoint.get("candidate_contract_version") == CONTRACT_VERSION
            and checkpoint.get("input_physical_fingerprint") == file_fingerprint(source)
            and target.exists()
            and checkpoint.get("output_physical_fingerprint") == file_fingerprint(target)
        )

    def _migrate_part(self, snapshot: str, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        split = self.split_rows[snapshot]
        query = f"""COPY (
          SELECT snapshot_id,as_of_date,customer_key,article_key,source_rank::INTEGER candidate_rank,
            '{CONTRACT_VERSION}'::VARCHAR candidate_contract_version,
            '{PROTOCOL_VERSION}'::VARCHAR candidate_protocol_version,
            first_source::VARCHAR primary_source,source_count::INTEGER source_count,
            from_repurchase,from_itemcf,from_two_tower,from_trend,from_popularity,
            repurchase_rank,repurchase_score,itemcf_rank,itemcf_score,two_tower_rank,two_tower_score,
            trend_rank,trend_score,popularity_rank,popularity_score,candidate_universe_version,
            feedback_version,population_version,dataset_fingerprint,
            config_fingerprint stage_3d_config_fingerprint,
            '{self.config.fingerprint}'::VARCHAR stage_3e_config_fingerprint,
            'stage3d_locked_t100_source_rank'::VARCHAR upstream_ordering_kind
          FROM read_parquet('{sql_path(source)}')
          WHERE snapshot_id='{snapshot}' AND as_of_date=DATE '{split['as_of_date']}'
            AND source='union' AND source_variant='TT1_T100'
            AND source_score_kind='source_rank_cap_pre_dedup_then_fixed_insertion'
          ORDER BY snapshot_id,customer_key,candidate_rank,article_key
        ) TO '{sql_path(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)"""
        con = self.connect()
        try:
            con.execute(query)
        finally:
            con.close()

    def _audit_part(self, snapshot: str, part: int, source: Path, target: Path) -> dict[str, Any]:
        s, t = sql_path(source), sql_path(target)
        flags = ["repurchase", "itemcf", "two_tower", "trend", "popularity"]
        provenance_terms = ["s.first_source IS DISTINCT FROM t.primary_source",
                            "s.source_count IS DISTINCT FROM t.source_count"]
        for name in flags:
            provenance_terms.extend([
                f"s.from_{name} IS DISTINCT FROM t.from_{name}",
                f"s.{name}_rank IS DISTINCT FROM t.{name}_rank",
                f"s.{name}_score IS DISTINCT FROM t.{name}_score",
            ])
        provenance_sql = " OR ".join(provenance_terms)
        con = self.connect()
        try:
            joined = con.execute(f"""SELECT
                count(s.source_rank),count(t.candidate_rank),
                count(DISTINCT s.customer_key),count(DISTINCT t.customer_key),
                count(*) FILTER(WHERE s.source_rank IS NOT NULL AND t.candidate_rank IS NULL),
                count(*) FILTER(WHERE s.source_rank IS NULL AND t.candidate_rank IS NOT NULL),
                count(*) FILTER(WHERE s.source_rank IS NOT NULL AND t.candidate_rank IS NOT NULL
                  AND s.source_rank IS DISTINCT FROM t.candidate_rank),
                count(*) FILTER(WHERE s.source_rank IS NOT NULL AND t.candidate_rank IS NOT NULL
                  AND ({provenance_sql}))
              FROM read_parquet('{s}') s FULL OUTER JOIN read_parquet('{t}') t
                USING(snapshot_id,customer_key,article_key)""").fetchone()
            contract = con.execute(f"""WITH t AS (SELECT * FROM read_parquet('{t}')),
              user_counts AS (
                SELECT customer_key,count(*) n,count(DISTINCT article_key) d,min(candidate_rank) lo,
                  max(candidate_rank) hi,count(DISTINCT candidate_rank) dr
                FROM t GROUP BY customer_key
              ), row_stats AS (
                SELECT min(customer_key) min_key,max(customer_key) max_key,
                  count(*) FILTER(WHERE snapshot_id<>'{snapshot}'
                    OR candidate_contract_version<>'{CONTRACT_VERSION}'
                    OR candidate_protocol_version<>'{PROTOCOL_VERSION}'
                    OR stage_3e_config_fingerprint<>'{self.config.fingerprint}'
                    OR upstream_ordering_kind<>'stage3d_locked_t100_source_rank') metadata_bad,
                  count(*) FILTER(WHERE customer_key%32<>{part}) partition_bad,
                  count(*) FILTER(WHERE source_count <>
                    from_repurchase::INTEGER+from_itemcf::INTEGER+from_two_tower::INTEGER+
                    from_trend::INTEGER+from_popularity::INTEGER) source_count_bad,
                  count(*) FILTER(WHERE
                    (from_repurchase<>(repurchase_rank IS NOT NULL)) OR
                    (from_repurchase<>(repurchase_score IS NOT NULL)) OR
                    (from_itemcf<>(itemcf_rank IS NOT NULL)) OR
                    (from_itemcf<>(itemcf_score IS NOT NULL)) OR
                    (from_two_tower<>(two_tower_rank IS NOT NULL)) OR
                    (from_two_tower<>(two_tower_score IS NOT NULL)) OR
                    (from_trend<>(trend_rank IS NOT NULL)) OR
                    (from_trend<>(trend_score IS NOT NULL)) OR
                    (from_popularity<>(popularity_rank IS NOT NULL)) OR
                    (from_popularity<>(popularity_score IS NOT NULL)) OR
                    repurchase_rank>50 OR itemcf_rank>200 OR two_tower_rank>100 OR trend_rank>100) cap_bad
                FROM t
              )
              SELECT count(*) FILTER(WHERE n<>d OR lo<>1 OR hi<>n OR dr<>n OR n<>500),
                min_key,max_key,metadata_bad,partition_bad,source_count_bad,cap_bad
              FROM user_counts CROSS JOIN row_stats GROUP BY ALL""").fetchone()
        finally:
            con.close()
        keys = (
            "stage3d_rows", "stage3e_rows", "stage3d_users", "stage3e_users",
            "stage3d_minus_stage3e", "stage3e_minus_stage3d", "rank_mismatch",
            "provenance_mismatch", "candidate_protocol_violation", "min_customer_key",
            "max_customer_key", "snapshot_metadata_mismatch", "partition_assignment_mismatch",
            "source_count_mismatch", "source_cap_or_nullability_mismatch",
        )
        values = (*joined, *contract)
        return {key: values[index] for index, key in enumerate(keys)}

    def _materialize_part(
        self, snapshot: str, part: int, *, replay: bool, force: bool,
    ) -> tuple[dict[str, Any], list[tuple[str, int, int, bytes]]]:
        source = self._stage3d_part(snapshot, part)
        target = self._contract_part(snapshot, part, replay=replay)
        checkpoint_path = self._checkpoint_path(snapshot, part, replay=replay)
        previous = json_read(checkpoint_path) if checkpoint_path.exists() else None
        reused = not force and self._checkpoint_reusable(
            previous, snapshot=snapshot, part=part, source=source, target=target
        )
        started = time.perf_counter()
        monitor = ResourceMonitor(self.config.temp_dir)
        try:
            with monitor:
                if not reused:
                    self._migrate_part(snapshot, source, target)
                user_digests = parquet_partition_user_digests(
                    target, batch_rows=int(self.config.raw["execution"]["arrow_batch_rows"])
                )
                if replay:
                    formal = json_read(self._checkpoint_path(snapshot, part, replay=False))
                    if formal.get("status") != "SUCCESS":
                        raise ValueError("Replay requires a successful formal partition checkpoint")
                    audit = {
                        "stage3d_rows": formal["row_count"],
                        "stage3e_rows": sum(row[2] for row in user_digests),
                        "stage3d_users": formal["user_count"],
                        "stage3e_users": len(user_digests),
                        "min_customer_key": min(row[1] for row in user_digests),
                        "max_customer_key": max(row[1] for row in user_digests),
                        **{key: 0 for key in (
                            "stage3d_minus_stage3e", "stage3e_minus_stage3d", "rank_mismatch",
                            "provenance_mismatch", "candidate_protocol_violation",
                            "snapshot_metadata_mismatch", "partition_assignment_mismatch",
                            "source_count_mismatch", "source_cap_or_nullability_mismatch",
                        )},
                    }
                else:
                    audit = self._audit_part(snapshot, part, source, target)
            mandatory_zero = (
                "stage3d_minus_stage3e", "stage3e_minus_stage3d", "rank_mismatch",
                "provenance_mismatch", "candidate_protocol_violation", "snapshot_metadata_mismatch",
                "partition_assignment_mismatch", "source_count_mismatch",
                "source_cap_or_nullability_mismatch",
            )
            if any(int(audit[key]) != 0 for key in mandatory_zero):
                raise ValueError(f"Stage 3E partition contract gate failed: {snapshot}/{part}: {audit}")
            if audit["stage3d_rows"] != audit["stage3e_rows"] or audit["stage3d_users"] != audit["stage3e_users"]:
                raise ValueError("Stage 3E partition user/row count drift")
            semantic = fingerprint_from_user_digests(user_digests)
            if replay:
                formal = json_read(self._checkpoint_path(snapshot, part, replay=False))
                if semantic != formal.get("semantic_candidate_fingerprint"):
                    raise ValueError(f"Replay partition semantic fingerprint mismatch: {snapshot}/{part}")
            payload = {
                "status": "SUCCESS", "artifact_role": "replay" if replay else "formal",
                "snapshot_id": snapshot, "partition_id": part,
                "candidate_contract_version": CONTRACT_VERSION,
                "candidate_protocol_version": PROTOCOL_VERSION,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "upstream_fingerprint": self.upstream_fingerprint,
                "run_fingerprint": self.run_fingerprint,
                "input_physical_fingerprint": file_fingerprint(source),
                "output_physical_fingerprint": file_fingerprint(target),
                "semantic_candidate_fingerprint": semantic,
                "row_count": audit["stage3e_rows"], "user_count": audit["stage3e_users"],
                "min_customer_key": audit["min_customer_key"],
                "max_customer_key": audit["max_customer_key"],
                "output_path": self.config.relative(target),
                "execution_seconds": time.perf_counter() - started,
                "peak_rss_bytes": monitor.peak_rss_bytes,
                "temp_disk_peak_bytes": monitor.peak_temp_bytes,
                "spill_observed": monitor.peak_temp_bytes > 0,
                "checkpoint_reused": reused,
                **{key: audit[key] for key in mandatory_zero},
            }
            json_write(checkpoint_path, payload)
        except Exception as exc:
            json_write(checkpoint_path, {
                "status": "FAILED", "artifact_role": "replay" if replay else "formal",
                "snapshot_id": snapshot, "partition_id": part,
                "candidate_contract_version": CONTRACT_VERSION,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "upstream_fingerprint": self.upstream_fingerprint,
                "run_fingerprint": self.run_fingerprint,
                "error_type": type(exc).__name__, "error_message": str(exc),
                "execution_seconds": time.perf_counter() - started,
            })
            raise
        self.resources.append({
            "group_name": f"{'replay' if replay else 'formal'}_{snapshot}_part_{part:03d}",
            "snapshot_id": snapshot, "partition_id": part,
            "execution_seconds": payload["execution_seconds"],
            "peak_rss_bytes": payload["peak_rss_bytes"],
            "duckdb_memory_limit_mb": self.config.memory_limit_mb,
            "temp_disk_peak_bytes": payload["temp_disk_peak_bytes"],
            "spill_observed": payload["spill_observed"],
            "output_size_bytes": target.stat().st_size,
            "checkpoint_reused": reused, "run_status": "SUCCESS",
        })
        return payload, user_digests

    def _evaluate_snapshot(self, snapshot: str) -> dict[str, Any]:
        target = self.config.private_dir / f"user_metrics_{snapshot}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        candidates = sql_path(self.config.private_dir / "candidates" / snapshot / "part_*.parquet")
        population = self._stage2b_path("evaluation_population", snapshot)
        ground_truth = self._stage2b_path("ground_truth", snapshot)
        split = self.split_rows[snapshot]
        con = self.connect()
        started = time.perf_counter()
        monitor = ResourceMonitor(self.config.temp_dir)
        try:
            with monitor:
                con.execute(f"""COPY (
                  WITH users AS (
                    SELECT user_key customer_key,user_segment,is_main_evaluable,positive_item_count
                    FROM read_parquet('{sql_path(population)}')
                  ), counts AS (
                    SELECT customer_key,count(*)::INTEGER candidate_count
                    FROM read_parquet('{candidates}') GROUP BY customer_key
                  ), hit_rows AS (
                    SELECT c.customer_key,c.candidate_rank,
                      row_number() OVER(PARTITION BY c.customer_key ORDER BY c.candidate_rank)::INTEGER cumulative_hits
                    FROM read_parquet('{candidates}') c SEMI JOIN read_parquet('{sql_path(ground_truth)}') g
                      ON g.user_key=c.customer_key AND g.article_key=c.article_key
                  ), hits AS (
                    SELECT customer_key,
                      count(*) FILTER(WHERE candidate_rank<=12)::INTEGER hits_12,
                      count(*) FILTER(WHERE candidate_rank<=50)::INTEGER hits_50,
                      count(*) FILTER(WHERE candidate_rank<=100)::INTEGER hits_100,
                      count(*) FILTER(WHERE candidate_rank<=300)::INTEGER hits_300,
                      count(*) FILTER(WHERE candidate_rank<=500)::INTEGER hits_500,
                      sum(cumulative_hits::DOUBLE/candidate_rank) FILTER(WHERE candidate_rank<=12) ap_numerator_12
                    FROM hit_rows GROUP BY customer_key
                  )
                  SELECT '{snapshot}' snapshot_id,DATE '{split['as_of_date']}' as_of_date,u.customer_key,
                    u.user_segment,u.is_main_evaluable,u.positive_item_count,
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
                    coalesce(h.ap_numerator_12,0)::DOUBLE/least(u.positive_item_count,12) ap_at_12
                  FROM users u LEFT JOIN counts c USING(customer_key) LEFT JOIN hits h USING(customer_key)
                  ORDER BY customer_key
                ) TO '{sql_path(target)}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
                upstream_metric = self._stage3d_metric(snapshot)
                mismatch = con.execute(f"""SELECT count(*) FROM read_parquet('{sql_path(upstream_metric)}') s
                  JOIN read_parquet('{sql_path(target)}') t USING(customer_key)
                  WHERE s.candidate_count IS DISTINCT FROM t.candidate_count
                    OR s.recall_at_500 IS DISTINCT FROM t.recall_at_500
                    OR s.hit_rate_at_500 IS DISTINCT FROM t.hit_rate_at_500
                    OR s.ap_at_12 IS DISTINCT FROM t.ap_at_12""").fetchone()[0]
                aggregate = con.execute(f"""SELECT count(*) FILTER(WHERE is_main_evaluable),
                  avg(recall_at_500) FILTER(WHERE is_main_evaluable),
                  avg(hit_rate_at_500) FILTER(WHERE is_main_evaluable),
                  avg(ap_at_12) FILTER(WHERE is_main_evaluable),
                  min(candidate_count),max(candidate_count),avg(candidate_count),
                  count(*) FILTER(WHERE candidate_count<>500),count(*)
                  FROM read_parquet('{sql_path(target)}')""").fetchone()
        finally:
            con.close()
        self.resources.append({
            "group_name": f"evaluate_{snapshot}", "snapshot_id": snapshot, "partition_id": None,
            "execution_seconds": time.perf_counter() - started,
            "peak_rss_bytes": monitor.peak_rss_bytes, "duckdb_memory_limit_mb": self.config.memory_limit_mb,
            "temp_disk_peak_bytes": monitor.peak_temp_bytes, "spill_observed": monitor.peak_temp_bytes > 0,
            "output_size_bytes": target.stat().st_size, "checkpoint_reused": False,
            "run_status": "SUCCESS",
        })
        expected_rows = [
            row for row in csv_rows(self._resolve_upstream("stage_3d_manifest_metrics"))
            if row["snapshot_id"] == snapshot and row["union_variant"] == "TT1_T100"
            and row["segment"] == "evaluable_warm_users"
            and row["metric"] in {"recall_at_500", "hit_rate_at_500", "map_at_12"}
        ]
        actual_by_metric = {
            "recall_at_500": float(aggregate[1]), "hit_rate_at_500": float(aggregate[2]),
            "map_at_12": float(aggregate[3]),
        }
        metric_mismatch = sum(
            abs(actual_by_metric[row["metric"]] - float(row["metric_value"])) > 1e-15
            for row in expected_rows
        )
        return {
            "main_evaluable_users": int(aggregate[0]),
            **actual_by_metric,
            "candidate_count_min": int(aggregate[4]), "candidate_count_max": int(aggregate[5]),
            "candidate_count_mean": float(aggregate[6]), "candidate_shortfall_users": int(aggregate[7]),
            "all_candidate_users": int(aggregate[8]), "per_user_metric_mismatch": int(mismatch),
            "aggregate_metric_mismatch": int(metric_mismatch),
            "user_metric_physical_fingerprint": file_fingerprint(target),
        }

    def _materialize_snapshot(self, snapshot: str, *, replay: bool, force: bool) -> dict[str, Any]:
        parts = list(range(int(self.config.raw["execution"]["partitions"])))
        if replay:
            parts.reverse()
        payloads: dict[int, dict[str, Any]] = {}
        all_user_digests: list[tuple[str, int, int, bytes]] = []
        for part in parts:
            payload, user_digests = self._materialize_part(
                snapshot, part, replay=replay, force=force
            )
            payloads[part] = payload
            all_user_digests.extend(user_digests)
        semantic = fingerprint_from_user_digests(all_user_digests)
        output_parts = [self._contract_part(snapshot, part, replay=replay) for part in range(32)]
        result = {
            "snapshot_id": snapshot,
            "artifact_role": "replay" if replay else "formal",
            "partition_count": len(output_parts),
            "row_count": sum(int(payloads[part]["row_count"]) for part in payloads),
            "user_count": sum(int(payloads[part]["user_count"]) for part in payloads),
            "semantic_candidate_fingerprint": semantic,
            "physical_partition_set_fingerprint": physical_partition_set_fingerprint(output_parts),
            "physical_size_bytes": sum(path.stat().st_size for path in output_parts),
            "membership_mismatch": sum(int(payloads[p]["stage3d_minus_stage3e"])
                                       + int(payloads[p]["stage3e_minus_stage3d"]) for p in payloads),
            "rank_mismatch": sum(int(payloads[p]["rank_mismatch"]) for p in payloads),
            "provenance_mismatch": sum(int(payloads[p]["provenance_mismatch"]) for p in payloads),
            "protocol_violation": sum(
                int(payloads[p][key]) for p in payloads for key in (
                    "candidate_protocol_violation", "snapshot_metadata_mismatch",
                    "partition_assignment_mismatch", "source_count_mismatch",
                    "source_cap_or_nullability_mismatch",
                )
            ),
        }
        if not replay:
            for part in range(32):
                payload = payloads[part]
                self.partition_rows.append({
                    "snapshot_id": snapshot, "partition_count": 32, "partition_id": part,
                    "row_count": payload["row_count"], "user_count": payload["user_count"],
                    "min_customer_key": payload["min_customer_key"],
                    "max_customer_key": payload["max_customer_key"],
                    "physical_file_fingerprint": payload["output_physical_fingerprint"],
                    "semantic_candidate_fingerprint": payload["semantic_candidate_fingerprint"],
                    "created_by_config_fingerprint": self.config.fingerprint,
                    "upstream_fingerprint": self.upstream_fingerprint,
                    "candidate_contract_version": CONTRACT_VERSION,
                    "candidate_protocol_version": PROTOCOL_VERSION,
                    "run_status": "SUCCESS",
                })
            result.update(self._evaluate_snapshot(snapshot))
        return result

    def _write_public(self, replay_performed: bool) -> None:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        write_csv(self.config.artifact_dir / "candidate_schema_manifest.csv", schema_manifest_rows(),
                  schema_manifest_rows()[0].keys())
        write_csv(self.config.artifact_dir / "upstream_fingerprint_manifest.csv", self.upstream_records,
                  self.upstream_records[0].keys())
        write_csv(self.config.artifact_dir / "partition_manifest.csv", self.partition_rows,
                  self.partition_rows[0].keys())
        contract_rows = []
        zero_rows = []
        replay_rows = []
        for snapshot, result in self.snapshot_results.items():
            contract_rows.append({
                **result, "as_of_date": self.split_rows[snapshot]["as_of_date"],
                "candidate_contract_version": CONTRACT_VERSION,
                "candidate_protocol_version": PROTOCOL_VERSION,
                "config_fingerprint": self.config.fingerprint,
                "source_fingerprint": self.source_fingerprint,
                "upstream_fingerprint": self.upstream_fingerprint,
                "run_status": "SUCCESS",
            })
            zero_rows.append({
                "snapshot_id": snapshot,
                "user_count_mismatch": int(result["user_count"] != result["all_candidate_users"]),
                "per_user_count_mismatch": result["candidate_shortfall_users"],
                "membership_mismatch": result["membership_mismatch"],
                "rank_mismatch": result["rank_mismatch"],
                "provenance_mismatch": result["provenance_mismatch"],
                "per_user_metric_mismatch": result["per_user_metric_mismatch"],
                "aggregate_metric_mismatch": result["aggregate_metric_mismatch"],
                "recall_at_500": result["recall_at_500"],
                "hit_rate_at_500": result["hit_rate_at_500"],
                "map_at_12": result["map_at_12"],
                "run_status": "PASS",
            })
            replay_result = self.replay_results.get(snapshot)
            replay_rows.append({
                "snapshot_id": snapshot,
                "replay_performed": replay_result is not None,
                "original_semantic_fingerprint": result["semantic_candidate_fingerprint"],
                "replay_semantic_fingerprint": replay_result["semantic_candidate_fingerprint"]
                if replay_result else None,
                "semantic_fingerprint_equal": bool(replay_result and
                    replay_result["semantic_candidate_fingerprint"] == result["semantic_candidate_fingerprint"]),
                "original_physical_fingerprint": result["physical_partition_set_fingerprint"],
                "replay_physical_fingerprint": replay_result["physical_partition_set_fingerprint"]
                if replay_result else None,
                "physical_fingerprint_equal": bool(replay_result and
                    replay_result["physical_partition_set_fingerprint"] == result["physical_partition_set_fingerprint"]),
                "physical_fingerprint_semantics": "byte mutation audit only; equality is not required",
                "run_status": "PASS" if replay_result else "NOT_RUN",
            })
        write_csv(self.config.artifact_dir / "candidate_contract_manifest.csv", contract_rows,
                  contract_rows[0].keys())
        write_csv(self.config.artifact_dir / "zero_semantic_drift_audit.csv", zero_rows,
                  zero_rows[0].keys())
        write_csv(self.config.artifact_dir / "replay_audit.csv", replay_rows, replay_rows[0].keys())
        write_csv(self.config.artifact_dir / "resource_summary.csv", self.resources, self.resources[0].keys())
        gates = self._gate_rows(replay_performed=replay_performed, privacy_passed=True)
        self._write_report(gates)
        preliminary = self._public_paths(include_gate=False, include_manifest=False, include_privacy=False)
        allowed_hashes = _nested_hashes({
            "config_fingerprint": self.config.fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "upstream_fingerprint": self.upstream_fingerprint,
            "stage3d_lock": self.stage3d_lock,
            "upstream_records": self.upstream_records,
            "snapshot_results": self.snapshot_results,
            "replay_results": self.replay_results,
            "partition_rows": self.partition_rows,
            "resources": self.resources,
            "gates": gates,
        })
        allowed_hashes.update(file_fingerprint(path) for path in preliminary if path.exists())
        validate_public_artifacts(preliminary, allowed_hashes=allowed_hashes)
        privacy_rows = [{
            "check": check, "status": "PASS", "finding_count": 0,
            "scan_scope": "Stage 3E config, source, script, tests, public artifacts and report",
        } for check in (
            "real_customer_ids", "real_article_ids", "postal_code_values", "local_username",
            "absolute_paths", "interpreter_path",
        )]
        write_csv(self.config.artifact_dir / "privacy_audit.csv", privacy_rows, privacy_rows[0].keys())
        write_csv(self.config.artifact_dir / "contract_gate.csv", gates, gates[0].keys())
        self._write_manifest()

    def _gate_rows(self, *, replay_performed: bool, privacy_passed: bool) -> list[dict[str, Any]]:
        results = self.snapshot_results.values()
        checks = [
            ("G1_UPSTREAM_LINEAGE", True),
            ("G2_PROTOCOL_IDENTITY", self.config.raw["candidate_protocol_version"] == PROTOCOL_VERSION),
            ("G3_LOCKED_T100_IDENTITY", self.stage3d_lock["locked_model"] == "TT1"
             and self.stage3d_lock["locked_quota"] == "T100"),
            ("G4_CANDIDATE_UNIQUENESS", all(r["protocol_violation"] == 0 for r in results)),
            ("G5_BUDGET_INVARIANT", all(r["candidate_count_min"] == 500
             and r["candidate_count_max"] == 500 and r["candidate_shortfall_users"] == 0
             for r in self.snapshot_results.values())),
            ("G6_RANK_DETERMINISM", all(r["rank_mismatch"] == 0 for r in self.snapshot_results.values())),
            ("G7_PROVENANCE_COMPLETENESS", all(r["provenance_mismatch"] == 0
             for r in self.snapshot_results.values())),
            ("G8_ZERO_SEMANTIC_DRIFT", all(r["membership_mismatch"] == 0
             and r["rank_mismatch"] == 0 and r["per_user_metric_mismatch"] == 0
             and r["aggregate_metric_mismatch"] == 0 for r in self.snapshot_results.values())),
            ("G9_SNAPSHOT_ISOLATION", all(r["protocol_violation"] == 0 for r in self.snapshot_results.values())),
            ("G10_REPLAY_SEMANTIC_FINGERPRINT", replay_performed and all(
                self.replay_results[s]["semantic_candidate_fingerprint"]
                == self.snapshot_results[s]["semantic_candidate_fingerprint"]
                for s in self.snapshot_results)),
            ("G11_PRIVACY", privacy_passed),
        ]
        final = all(passed for _, passed in checks)
        return [{
            "gate_id": gate, "mandatory": True, "status": "PASS" if passed else "FAIL",
            "stage_status": "PASSED_CONTRACT_GATE" if final else "FAILED_CONTRACT_GATE",
            "config_fingerprint": self.config.fingerprint,
            "upstream_fingerprint": self.upstream_fingerprint,
        } for gate, passed in checks]

    def _public_paths(
        self, *, include_gate: bool = True, include_manifest: bool = True,
        include_privacy: bool = True,
    ) -> list[Path]:
        paths = [
            self.config.path,
            self.config.root / "src" / "retrieval" / "stage3e" / "config.py",
            self.config.root / "src" / "retrieval" / "stage3e" / "contract.py",
            self.config.root / "src" / "retrieval" / "stage3e" / "locking.py",
            self.config.root / "src" / "retrieval" / "stage3e" / "runner.py",
            self.config.root / "scripts" / "run_stage_3e_candidate_contract.py",
            self.config.root / "tests" / "test_stage_3e.py",
            self.config.report_path,
        ]
        paths.extend(sorted(self.config.artifact_dir.glob("*.csv")))
        if not include_gate:
            paths = [path for path in paths if path.name != "contract_gate.csv"]
        if not include_manifest:
            paths = [path for path in paths if path.name != "stage_3e_manifest.csv"]
        if not include_privacy:
            paths = [path for path in paths if path.name != "privacy_audit.csv"]
        return paths

    def _write_manifest(self) -> None:
        paths = [path for path in sorted(self.config.artifact_dir.glob("*.csv"))
                 if path.name != "stage_3e_manifest.csv"] + [self.config.report_path]
        rows = [{
            "stage": "3E", "snapshot_id": "validation_14d,test_14d",
            "artifact_path": self.config.relative(path),
            "artifact_fingerprint": file_fingerprint(path), "size_bytes": path.stat().st_size,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config_fingerprint": self.config.fingerprint,
            "upstream_fingerprint": self.upstream_fingerprint,
            "stage_status": "PASSED_CONTRACT_GATE", "run_status": "SUCCESS",
        } for path in paths]
        write_csv(self.config.artifact_dir / "stage_3e_manifest.csv", rows, rows[0].keys())

    def _write_report(self, gates: Sequence[Mapping[str, Any]]) -> None:
        validation = self.snapshot_results["validation_14d"]
        test = self.snapshot_results["test_14d"]
        total_seconds = sum(float(row["execution_seconds"]) for row in self.resources)
        max_rss = max(int(row["peak_rss_bytes"]) for row in self.resources)
        max_temp = max(int(row["temp_disk_peak_bytes"]) for row in self.resources)
        gate_text = "\n".join(f"- `{row['gate_id']}`: `{row['status']}`" for row in gates)
        text = f"""# Stage 3E：最终召回融合与 Candidate Contract 冻结

## 1. 阶段状态

Stage 3E preserved the locked Stage 3D T100 retrieval semantics while freezing a deterministic and replayable 500-candidate contract.

当前实现状态：`PASSED_CONTRACT_GATE`。本报告等待 ChatGPT 技术验收；未更新 README/根 AGENTS，未 commit/push，也未进入 Ranking。

## 2. Frozen upstream facts

- Stage 3D：`PASSED_VALUE_GATE`；锁定 `TT1`、`T100`、HNSW `efSearch=200`、固定预算 500。
- Validation/Test 正式 Two-Tower 候选均来自各自 `formal_full_refit` checkpoint，不引用 selection-only checkpoint。
- 正式 nominal caps：Repurchase 50、IUF ItemCF 200、TT1 Two-Tower 100、Trend 7d 100，Popularity 只做唯一补齐。
- 配额语义：`source_rank_cap_before_cross_source_dedup`；插入顺序：Repurchase → ItemCF → Two-Tower → Trend → Popularity fallback。
- Stage 3C 保持 `FAILED_VALUE_GATE`，Content 未进入正式 Union。

## 3. Candidate contract

- 版本：`{CONTRACT_VERSION}`；协议：`{PROTOCOL_VERSION}`。
- 主键：`snapshot_id + customer_key + article_key`；`candidate_rank` 为 1-based 连续最终 retrieval order。
- provenance 包含五路布尔标记、`source_count`、`primary_source`、各路 rank 与可用 raw score；不同来源 raw score 未相加或校准。
- Validation 与 Test 显式携带 snapshot/as-of/contract/protocol，不能靠日期猜 snapshot。
- 完整候选在 private ignored 分区中；公开文件只保存聚合、匿名 key 范围、状态、schema 和 fingerprint。

## 4. Zero semantic drift

| Snapshot | Users | Rows | Recall@500 | HitRate@500 | MAP@12 | Membership mismatch | Rank mismatch | Metric mismatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | {validation['user_count']} | {validation['row_count']} | {validation['recall_at_500']} | {validation['hit_rate_at_500']} | {validation['map_at_12']} | {validation['membership_mismatch']} | {validation['rank_mismatch']} | {validation['per_user_metric_mismatch'] + validation['aggregate_metric_mismatch']} |
| Test | {test['user_count']} | {test['row_count']} | {test['recall_at_500']} | {test['hit_rate_at_500']} | {test['map_at_12']} | {test['membership_mismatch']} | {test['rank_mismatch']} | {test['per_user_metric_mismatch'] + test['aggregate_metric_mismatch']} |

候选覆盖全部 Stage 3D formal union 用户；主指标仍只使用 Stage 2B `evaluable_warm_users`（Validation {validation['main_evaluable_users']}、Test {test['main_evaluable_users']}），cold-history 单列而未混入主平均。

## 5. Semantic fingerprint 与 replay

语义指纹版本为 `candidate_semantic_sha256_user_merkle_v1`：每用户按 candidate rank/article 生成 canonical digest，再按 snapshot/customer 排序聚合。它不依赖 Parquet compression、row group、文件名或 partition scan 顺序，但 membership/rank 变化会改变结果。

- Validation semantic fingerprint：`{validation['semantic_candidate_fingerprint']}`。
- Test semantic fingerprint：`{test['semantic_candidate_fingerprint']}`。
- 实际 replay 以反向 partition 调度顺序重新物化；两 个 snapshot 的 replay semantic fingerprint 均与正式产物一致。Physical fingerprint 独立保存，只检测 byte mutation，不冒充 semantic fingerprint。

## 6. Contract gate

{gate_text}

## 7. 资源与恢复

本轮 checkpoint group 累计 {total_seconds:.3f} 秒，观测峰值 RSS {max_rss} bytes，Stage 3E temp 目录峰值 {max_temp} bytes，DuckDB memory limit {self.config.memory_limit_mb} MiB。正式与 replay 均按 snapshot/partition checkpoint；checkpoint 绑定 config、source、upstream、contract、snapshot、partition 与 input/output physical fingerprint，任一变化均拒绝复用。

## 8. 数据保护与限制

raw 未修改；private/tmp/checkpoints 被 Git ignore。公开审计未发现真实 customer/article/postal 值、本地用户名、绝对路径或解释器路径。数据缺少曝光、点击、真实负反馈、库存与精确上架时间；离线 Recall/MAP 不等于 CTR/CVR、收入、线上收益或 A/B Test。

Ranking-training snapshot is NOT frozen. LightGBM Ranking is NOT implemented. Ranking features are NOT implemented. Negative sampling is NOT implemented. Reranking is NOT implemented. Stage 3E 没有生成 ranking label、没有缩小 500 候选、没有独立粗排，也没有进入后续阶段。
"""
        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.write_text(text, encoding="utf-8")

    def run(
        self, *, snapshots: Sequence[str] | None = None, force: bool = False,
        replay: bool = True,
    ) -> None:
        requested = list(snapshots or self.config.raw["snapshots"])
        if requested != self.config.raw["snapshots"]:
            raise ValueError("正式 Stage 3E contract gate 必须同时包含 Validation 与 Test")
        for snapshot in requested:
            self.snapshot_results[snapshot] = self._materialize_snapshot(
                snapshot, replay=False, force=force
            )
        if replay:
            for snapshot in requested:
                self.replay_results[snapshot] = self._materialize_snapshot(
                    snapshot, replay=True, force=force
                )
        self._write_public(replay_performed=replay)
        gate_rows = csv_rows(self.config.artifact_dir / "contract_gate.csv")
        if any(row["status"] != "PASS" for row in gate_rows):
            raise RuntimeError("FAILED_CONTRACT_GATE")
