from __future__ import annotations

from pathlib import Path

import duckdb

from ranking_features.runner import Stage4B3Runner
from ranking_features.schema import feature_schema_rows, schema_fingerprint

from .config import Stage4B4Config


class ValidationFeatureBuilder(Stage4B3Runner):
    """复用 Stage 4B-3 的 SQL builder，仅参数化 snapshot 与隔离输出目录。"""

    def __init__(self, config: Stage4B4Config, stage4b3_raw: dict) -> None:
        self.config = config
        self.snapshot = config.raw["snapshots"]["validation"]
        self.fields = list(stage4b3_raw["canonical_fields"])
        self.schema_rows = feature_schema_rows(self.fields)
        self.schema_fp = schema_fingerprint(self.schema_rows)
        self._history_path = config.resolve(config.raw["upstream"]["validation_history"])
        self._candidate_dir = config.resolve(config.raw["upstream"]["validation_candidates"])
        self._base = config.private_dir / "validation_features"
        self._feature_upstream = dict(stage4b3_raw["upstream"])

    @property
    def upstream(self) -> dict[str, str]:
        return self._feature_upstream

    @property
    def history_path(self) -> Path:
        return self._history_path

    @property
    def candidate_dir(self) -> Path:
        return self._candidate_dir

    @property
    def user_aggregate_path(self) -> Path:
        return self._base / "aggregates" / "user_features.parquet"

    @property
    def item_aggregate_path(self) -> Path:
        return self._base / "aggregates" / "item_features.parquet"

    def ui_aggregate_part(self, part: int) -> Path:
        return self._base / "aggregates" / "user_item" / f"part_{part:03d}.parquet"

    @property
    def article_features_path(self) -> Path:
        return self._base / "aggregates" / "article_features.parquet"

    @property
    def history_partition_directory(self) -> Path:
        return self._base / "aggregates" / "history_parts"

    def history_part(self, part: int) -> Path:
        return self.history_partition_directory / f"part={part}" / "data_0.parquet"

    def category_aggregate_part(self, part: int) -> Path:
        return self._base / "aggregates" / "user_category" / f"part_{part:03d}.parquet"

    def feature_part(self, part: int, *, replay: bool = False) -> Path:
        role = "replay" if replay else "full500"
        return self._base / role / f"part_{part:03d}.parquet"

    def digest_part(self, part: int, *, replay: bool = False) -> Path:
        role = "replay_digests" if replay else "query_digests"
        return self._base / role / f"part_{part:03d}.parquet"

    @property
    def full_feature_glob(self) -> Path:
        return self._base / "full500" / "part_*.parquet"

    def connect(self) -> duckdb.DuckDBPyConnection:
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        con = duckdb.connect(":memory:")
        con.execute(f"SET temp_directory='{self.config.temp_dir.as_posix()}'")
        con.execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
        con.execute("SET threads=1")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET max_temp_directory_size='{self.config.raw['execution']['max_temp_directory_size']}'")
        return con
