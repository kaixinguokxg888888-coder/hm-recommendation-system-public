from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .config import AuditSettings
from .contracts import DataContracts


def arrow_schema_fingerprint(schema: pa.Schema) -> str:
    canonical = schema.remove_metadata().to_string(show_field_metadata=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DuckDBAuditEngine:
    """只读 Parquet 查询引擎；所有交易聚合均针对显式物化的受限批次。"""

    def __init__(self, settings: AuditSettings, contracts: DataContracts) -> None:
        self.settings = settings
        self.contracts = contracts
        self.connection = duckdb.connect(database=":memory:")
        self.table_rows: dict[str, int] = {}
        self.row_groups: dict[str, int] = {}
        self.transaction_scope: dict[str, int] = {}
        self._configure()

    def _configure(self) -> None:
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.settings.temp_dir.as_posix().replace("'", "''")
        self.connection.execute(f"SET temp_directory = '{temp_path}'")
        self.connection.execute(
            f"SET memory_limit = '{self.settings.memory_limit_mb}MB'"
        )
        self.connection.execute(f"SET threads = {self.settings.threads}")
        self.connection.execute("SET preserve_insertion_order = false")

    def parquet_path(self, table_name: str) -> Path:
        return (self.settings.project_root / self.contracts.tables[table_name].file).resolve()

    @staticmethod
    def _sql_path(path: Path) -> str:
        return path.as_posix().replace("'", "''")

    def prepare(self) -> None:
        for table_name in self.contracts.tables:
            path = self.parquet_path(table_name)
            parquet_file = pq.ParquetFile(path)
            self.table_rows[table_name] = parquet_file.metadata.num_rows
            self.row_groups[table_name] = parquet_file.metadata.num_row_groups

        for table_name in ("articles", "customers"):
            path = self._sql_path(self.parquet_path(table_name))
            self.connection.execute(
                f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{path}')"
            )

        transaction_path = self.parquet_path("transactions")
        parquet_file = pq.ParquetFile(transaction_path)
        row_group_index = self.settings.transaction_row_group
        if row_group_index >= parquet_file.metadata.num_row_groups:
            raise ValueError("transaction_row_group 超出文件 row group 范围")
        batch = next(
            parquet_file.iter_batches(
                batch_size=self.settings.transaction_max_rows,
                row_groups=[row_group_index],
            ),
            None,
        )
        if batch is None:
            raise ValueError("指定 transaction row group 为空")
        self.connection.register("_transactions_arrow_batch", pa.Table.from_batches([batch]))
        self.connection.execute(
            "CREATE TEMP TABLE transactions_smoke AS "
            "SELECT * FROM _transactions_arrow_batch"
        )
        self.connection.unregister("_transactions_arrow_batch")
        self.transaction_scope = {
            "row_group_index": row_group_index,
            "row_group_rows": parquet_file.metadata.row_group(row_group_index).num_rows,
            "rows_materialized": batch.num_rows,
            "file_row_groups": parquet_file.metadata.num_row_groups,
            "file_rows": parquet_file.metadata.num_rows,
        }

    def direct_parquet_probe(self, table_name: str) -> int:
        """由 DuckDB 直接对本地 Parquet 执行受限读取，返回结果行数。"""
        path = self._sql_path(self.parquet_path(table_name))
        return len(
            self.connection.execute(
                f"SELECT * FROM read_parquet('{path}') LIMIT 1"
            ).fetchall()
        )

    def arrow_schema(self, table_name: str) -> pa.Schema:
        return pq.ParquetFile(self.parquet_path(table_name)).schema_arrow

    def fingerprint(self, table_name: str) -> str:
        return arrow_schema_fingerprint(self.arrow_schema(table_name))

    def scalar(self, sql: str, parameters: list[Any] | None = None) -> Any:
        row = self.connection.execute(sql, parameters or []).fetchone()
        return None if row is None else row[0]

    def row(self, sql: str, parameters: list[Any] | None = None) -> tuple[Any, ...]:
        row = self.connection.execute(sql, parameters or []).fetchone()
        if row is None:
            raise ValueError("查询未返回结果")
        return row

    def temp_usage_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.settings.temp_dir.rglob("*")
            if path.is_file()
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "DuckDBAuditEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
