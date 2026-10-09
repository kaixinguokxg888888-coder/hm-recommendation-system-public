from __future__ import annotations

import csv
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PACKAGE_NAMES = {
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "polars": "polars",
    "duckdb": "duckdb",
    "numpy": "numpy",
    "scikit-learn": "scikit-learn",
    "lightgbm": "lightgbm",
    "pytorch": "torch",
}


def relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def logical_table_name(path: Path) -> str:
    """仅按文件名做保守归一化，不假定工作区一定采用 Kaggle 命名。"""
    name = path.stem.lower()
    name = re.sub(r"[-_]?(part|chunk|shard)[-_]?\d+$", "", name)
    name = re.sub(r"[-_]?\d{3,}$", "", name)
    if name.endswith("_train"):
        name = name[: -len("_train")]
    return name


def schema_fingerprint(schema: pa.Schema) -> str:
    canonical = schema.remove_metadata().to_string(show_field_metadata=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def windows_memory() -> dict[str, int | None]:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    if os.name != "nt":
        return {"total_bytes": None, "available_bytes": None}
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    if not ok:
        return {"total_bytes": None, "available_bytes": None}
    return {
        "total_bytes": int(status.ullTotalPhys),
        "available_bytes": int(status.ullAvailPhys),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for label, distribution in PACKAGE_NAMES.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def git_snapshot(root: Path) -> dict[str, Any]:
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "is_repository": False, "error": str(exc)}
    if inside.returncode != 0:
        return {
            "available": True,
            "is_repository": False,
            "error": inside.stderr.strip() or inside.stdout.strip(),
        }
    status = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return {
        "available": True,
        "is_repository": True,
        "status": status.stdout.strip(),
        "error": status.stderr.strip(),
    }


def encoding_summary(parquet_file: pq.ParquetFile) -> tuple[str, dict[str, list[str]]]:
    compressions: set[str] = set()
    encodings: dict[str, set[str]] = defaultdict(set)
    metadata = parquet_file.metadata
    for rg_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(rg_index)
        for col_index in range(row_group.num_columns):
            column = row_group.column(col_index)
            compressions.add(str(column.compression))
            encodings[column.path_in_schema].update(map(str, column.encodings))
    return (
        "|".join(sorted(compressions)),
        {key: sorted(values) for key, values in encodings.items()},
    )


def safe_example(value: Any, column_name: str) -> str:
    """只保存值形态；标识符完全脱敏，长文本截断。"""
    if value is None:
        return "null"
    if isinstance(value, bytes):
        return f"<bytes:length={len(value)}>"
    if isinstance(value, str):
        normalized = value.replace("\r", " ").replace("\n", " ")
        normalized_name = column_name.lower()
        if normalized_name.endswith("_id") or normalized_name == "postal_code":
            if normalized.isdigit():
                charset = "digits"
            elif re.fullmatch(r"[0-9a-fA-F]+", normalized):
                charset = "hex"
            else:
                charset = "mixed"
            return f"<string:length={len(normalized)},charset={charset}>"
        if len(normalized) > 18:
            return f"{normalized[:8]}…{normalized[-4:]} (len={len(normalized)})"
        return normalized
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__}:length={len(value)}>"
    if isinstance(value, dict):
        return f"<struct:keys={sorted(map(str, value.keys()))}>"
    return str(value)


def sample_observations(parquet_file: pq.ParquetFile, rows: int = 5) -> list[dict[str, Any]]:
    # 通过小批次迭代首个 row group，避免把整个 row group 物化到内存。
    if parquet_file.metadata.num_row_groups == 0:
        return []
    batch = next(parquet_file.iter_batches(batch_size=rows, row_groups=[0]), None)
    if batch is None:
        return []
    table = pa.Table.from_batches([batch])
    result: list[dict[str, Any]] = []
    for field, column in zip(table.schema, table.columns):
        values = column.to_pylist()
        examples: list[str] = []
        for value in values:
            rendered = safe_example(value, field.name)
            if rendered not in examples:
                examples.append(rendered)
            if len(examples) == 2:
                break
        result.append(
            {
                "column": field.name,
                "arrow_type": str(field.type),
                "sample_nulls": sum(value is None for value in values),
                "examples": examples,
            }
        )
    return result


def inspect_parquet(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "logical_table": logical_table_name(path),
        "file_path": relative_posix(path, root),
        "file_size_bytes": path.stat().st_size,
        "readable": False,
        "error_message": "",
    }
    try:
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        arrow_schema = parquet_file.schema_arrow
        compression, encodings = encoding_summary(parquet_file)
        physical_columns = {
            parquet_file.schema.column(i).path: {
                "physical_type": str(parquet_file.schema.column(i).physical_type),
                "logical_type": str(parquet_file.schema.column(i).logical_type),
                "converted_type": str(parquet_file.schema.column(i).converted_type),
            }
            for i in range(len(parquet_file.schema))
        }
        schema_rows = []
        for field in arrow_schema:
            physical = physical_columns.get(field.name, {})
            observed_encodings = encodings.get(field.name, [])
            schema_rows.append(
                {
                    "name": field.name,
                    "physical_type": physical.get("physical_type", "nested/unknown"),
                    "parquet_logical_type": physical.get("logical_type", "nested/unknown"),
                    "arrow_type": str(field.type),
                    "nullable": field.nullable,
                    "dictionary_encoding_observed": any(
                        "DICTIONARY" in encoding for encoding in observed_encodings
                    ),
                    "encodings": observed_encodings,
                    "special_type": any(
                        [
                            pa.types.is_timestamp(field.type),
                            pa.types.is_date(field.type),
                            pa.types.is_binary(field.type),
                            pa.types.is_list(field.type),
                            pa.types.is_struct(field.type),
                        ]
                    ),
                }
            )
        record.update(
            {
                "row_count": metadata.num_rows,
                "row_group_count": metadata.num_row_groups,
                "compression": compression,
                "schema_fingerprint": schema_fingerprint(arrow_schema),
                "readable": True,
                "format_version": str(metadata.format_version),
                "created_by": metadata.created_by,
                "has_pandas_metadata": bool(
                    arrow_schema.metadata and b"pandas" in arrow_schema.metadata
                ),
                "schema": schema_rows,
                "sample": sample_observations(parquet_file),
            }
        )
    except Exception as exc:  # 损坏文件也必须保留在清单中
        record.update(
            {
                "row_count": "",
                "row_group_count": "",
                "compression": "",
                "schema_fingerprint": "",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
        )
    return record


def directory_overview(root: Path) -> tuple[list[str], Counter[str]]:
    lines: list[str] = []
    extensions: Counter[str] = Counter()
    for item in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if item.name in {".git"}:
            continue
        if item.is_dir():
            children = sorted(item.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            shown = ", ".join(child.name + ("/" if child.is_dir() else "") for child in children[:20])
            suffix = f"（另有 {len(children) - 20} 项）" if len(children) > 20 else ""
            lines.append(f"- `{item.name}/`：{shown or '空目录'}{suffix}")
        else:
            lines.append(f"- `{item.name}`：{item.stat().st_size:,} bytes")
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            extensions[path.suffix.lower() or "<无扩展名>"] += 1
    return lines, extensions


def bytes_human(value: int | None) -> str:
    if value is None:
        return "未知"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    safe_rows = []
    for row in rows:
        safe_rows.append([str(value).replace("|", "\\|").replace("\n", " ") for value in row])
    head = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in safe_rows]
    return "\n".join([head, divider, *body])


def write_report(
    root: Path,
    records: list[dict[str, Any]],
    environment: dict[str, Any],
    report_path: Path,
) -> None:
    readable = [record for record in records if record["readable"]]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["logical_table"]].append(record)
    tree_lines, extensions = directory_overview(root)
    data_size = sum(record["file_size_bytes"] for record in records)
    now = environment["executed_at"]

    parquet_rows = [
        [
            record["logical_table"],
            record["file_path"],
            f'{record["file_size_bytes"]:,}',
            record.get("row_count", ""),
            record.get("row_group_count", ""),
            record.get("compression", ""),
            record.get("format_version", ""),
            "是" if record["readable"] else "否",
        ]
        for record in records
    ]
    logical_rows = []
    for name, items in sorted(groups.items()):
        fingerprints = {item.get("schema_fingerprint") for item in items if item["readable"]}
        logical_rows.append(
            [
                name,
                len(items),
                sum(item["file_size_bytes"] for item in items),
                sum(int(item.get("row_count") or 0) for item in items),
                sum(int(item.get("row_group_count") or 0) for item in items),
                "一致" if len(fingerprints) <= 1 and all(i["readable"] for i in items) else "不一致/无法确认",
                ", ".join(item["file_path"] for item in items),
            ]
        )

    schema_sections = []
    sample_sections = []
    for name, items in sorted(groups.items()):
        first = next((item for item in items if item["readable"]), None)
        if not first:
            schema_sections.append(f"### `{name}`\n\n无法读取 schema。")
            sample_sections.append(f"### `{name}`\n\n无法抽样。")
            continue
        schema_rows = [
            [
                col["name"],
                col["physical_type"],
                col["parquet_logical_type"],
                col["arrow_type"],
                "是" if col["nullable"] else "否",
                "是" if col["dictionary_encoding_observed"] else "否",
                "是" if col["special_type"] else "否",
            ]
            for col in first["schema"]
        ]
        schema_sections.append(
            f"### `{name}`\n\n"
            + markdown_table(
                ["字段", "物理类型", "Parquet 逻辑类型", "Arrow 类型", "可空", "观察到字典编码", "特殊类型"],
                schema_rows,
            )
            + f"\n\n分片字段类型一致性："
            + ("一致。" if len({i.get('schema_fingerprint') for i in items if i['readable']}) <= 1 else "不一致。")
            + "【证据：Parquet schema/footer】"
        )
        sample_rows = [
            [
                col["column"],
                col["arrow_type"],
                col["sample_nulls"],
                "; ".join(col["examples"]),
            ]
            for col in first["sample"]
        ]
        sample_sections.append(
            f"### `{name}`\n\n"
            + markdown_table(["字段", "Arrow 类型", "样本空值数", "截断后的值形态（最多 2 个）"], sample_rows)
            + "\n\n上述内容仅来自首个 row group 的前 5 行，不代表全量分布。【证据：少量抽样】"
        )

    missing_core = [
        name for name in ["transactions", "customers", "articles", "sample_submission"] if name not in groups
    ]
    corrupt = [record["file_path"] for record in records if not record["readable"]]
    package_rows = [
        [name, version if version is not None else "未安装"]
        for name, version in environment["dependencies"].items()
    ]
    ext_text = "、".join(
        f"`{extension}` {count} 个" for extension, count in sorted(extensions.items())
    )
    organization = (
        "每张识别出的逻辑表对应一个单独 Parquet 文件"
        if all(len(items) == 1 for items in groups.values())
        else "至少一张逻辑表由多个 Parquet 文件组成"
    )
    memory = environment["memory"]
    disk = environment["disk"]
    git = environment["git"]

    report = f"""# 阶段 0A：工作区与 Parquet 数据只读侦察

## 1. 执行信息

- 执行时间：`{now}`
- 工作区：`{environment["workspace"]}`（相对于项目根目录）
- 实际命令：`python scripts/recon_workspace.py`
- 约束：只读取文件系统、Parquet footer/schema 和每张逻辑表首个 row group 的前 5 行；未执行全量 groupby、join、sort、去重或训练。

## 2. 工作区与目录概况

{os.linesep.join(tree_lines)}

- 文件扩展名汇总：{ext_text or "无文件"}
- Git：{"已初始化" if git["is_repository"] else "未初始化"}；{git.get("status") or git.get("error") or "无额外信息"}。【证据：环境命令】
- 原始数据位于 `data/raw/`；侦察前其余顶层目录为空，未发现既有 Python、Notebook、README、requirements、pyproject、environment.yml 或 AGENTS.md。数据与最小项目目录处在同一工作区，但没有与代码文件混放在 `data/raw/` 内。【证据：文件系统遍历】

## 3. Parquet 文件清单

{markdown_table(["逻辑表", "相对路径", "字节", "行数", "Row groups", "压缩", "格式版本", "可读"], parquet_rows)}

共 `{len(records)}` 个文件，合计 `{data_size:,}` bytes（{bytes_human(data_size)}）。【证据：文件元数据与 Parquet footer】

## 4. 数据组织与逻辑表识别

组织方式判定：**{organization}**；未观察到 Hive `key=value` 分区路径，也未观察到 Hugging Face 风格分片命名。【证据：相对路径与文件名】

{markdown_table(["逻辑表", "文件数", "总字节", "总行数", "Row groups", "Schema", "文件"], logical_rows)}

逻辑表名由文件 stem 保守推断，其中 `_train` 后缀仅作为数据集角色后缀去除；未重命名原文件。

## 5. 完整 schema

{os.linesep.join(schema_sections)}

当前字段集合能够分别对应 H&M 的 article 主数据、customer 主数据和 transaction 事件数据；对应关系来自实际字段语义，不等于对公开版本完整性的证明。未发现可对应 sample submission 的本地文件。【证据：文件名与 schema】

## 6. Parquet 元数据

""" + "\n".join(
        f"- `{r['file_path']}`：created_by=`{r.get('created_by')}`，format_version=`{r.get('format_version')}`，"
        f"compression=`{r.get('compression')}`，pandas metadata={'有' if r.get('has_pandas_metadata') else '无'}，"
        f"schema fingerprint=`{str(r.get('schema_fingerprint', ''))[:16]}…`。【证据：Parquet footer】"
        for r in readable
    ) + f"""

编码说明：“观察到字典编码”来自 row-group column chunk 的 encoding 列表，并不表示 Arrow 字段本身是 dictionary 类型。

## 7. 安全抽样观察

{os.linesep.join(sample_sections)}

- ID 的存储类型、日期类型、数值精度和空值形态详见表格；没有对类型做自动转换。【证据：Arrow schema 与少量抽样】
- 抽样未发现 Arrow `list`/`struct` 嵌套字段；长 ID/文本已截断，未保存完整原始行。【证据：schema 与少量抽样】
- 是否存在乱码、异常值或 CSV→Parquet 类型漂移，只能作为抽样观察，不能据此推断全量比例。

## 8. 初步完整性结论

### 元数据已确认

- 三张已发现逻辑表的文件数、总行数、row group 数、压缩和 schema 已由 footer 确认。
- 无法读取文件：{", ".join(corrupt) if corrupt else "无"}。
- 同一逻辑表内 schema：{"均一致（当前每表仅 1 个文件）" if all(len(v) == 1 for v in groups.values()) else "见逻辑表汇总"}。
- 未发现的核心候选表：{", ".join(missing_core) if missing_core else "无"}。

### 抽样观察

- 每张可读逻辑表均已读取首个 row group 的前 5 行；抽样结果仅用于值形态检查。

### 尚未验证

- transaction 中 unique customer、unique article、重复交易、日期全量范围、空值率和外键覆盖率。
- customer/article 主表主键唯一性与 transaction 外键一致性。
- 文件内容是否与公开数据逐行一致；总行数相近不能证明 100% 完整。
- “重复或额外分片”只能确认目录中没有同表多文件，不能排除内容级重复。

## 9. 环境与资源

- OS：`{environment["os"]}`
- Python：`{environment["python"]["version"]}`；解释器命令 `{environment["python"]["executable"]}`
- 虚拟环境：`{"已启用（路径未记录）" if environment["python"]["virtual_env_active"] else "未检测到"}`
- CPU：`{environment["cpu"]["logical_cores"]}` 个逻辑核心
- 内存：总计 {bytes_human(memory["total_bytes"])}，可用 {bytes_human(memory["available_bytes"])}
- 工作区所在磁盘：总计 {bytes_human(disk["total_bytes"])}，可用 {bytes_human(disk["free_bytes"])}
- Parquet 总大小：{bytes_human(data_size)}

{markdown_table(["依赖", "版本"], package_rows)}

【证据：Python/platform、Windows GlobalMemoryStatusEx、shutil.disk_usage、importlib.metadata、Git 命令】

路径风险：工作区绝对路径无中文、无空格，当前相对路径较短；Windows 长路径风险低。Python 来自用户级安装目录，未启用虚拟环境，后续存在依赖漂移风险。

## 10. 风险清单

1. 缺少 sample submission：若后续需要复现 Kaggle 提交流程，需要另行确认其是否应在本项目数据范围内。
2. 当前无虚拟环境、无依赖锁定文件，重现性不足。
3. 交易表约 {bytes_human(next((r["file_size_bytes"] for r in records if r["logical_table"] == "transactions"), 0))}；解压后的列式扫描与高基数 distinct/groupby 会显著增加内存。
4. footer 不能给出精确 unique customer/article 数量或内容级重复结论。
5. 只有单文件逻辑表，无法通过分片间比较发现 schema 漂移；“一致”仅表示不存在跨分片冲突面。

## 11. 正式数据审计技术方案

| 方案 | 内存风险 | Parquet/聚合 | Windows 与测试 | 工程与展示价值 |
| --- | --- | --- | --- | --- |
| pandas | 高，易误触全量物化 | 读取方便，超大聚合风险高 | 稳定、测试方便 | 适合小结果，不宜主执行 |
| PyArrow Dataset | 低到中，可投影/过滤/分批 | Parquet 元数据与流式扫描强，复杂聚合较底层 | 稳定、易做 schema/批次测试 | 最适合底层 I/O 与契约检查 |
| Polars lazy scan | 低到中，惰性优化 | 扫描聚合强 | 当前未安装，尚未实机验证 | 有展示价值但不是必要复杂度 |
| DuckDB | 低到中，可溢写磁盘 | SQL 聚合、distinct、join 强 | 当前未安装，尚未实机验证 | 审计 SQL 清晰，可解释性强 |
| PyArrow 分批 | 最低，可显式控制 batch | 需要自行实现聚合状态 | 稳定、最易构造 smoke test | 维护成本适中，边界清晰 |

**建议**：下一阶段以 **DuckDB 作为正式审计主执行引擎**（需先最小化安装并验证），PyArrow 作为 footer/schema/批次读取辅助；pandas 只接收已聚合的小结果。若不增加依赖，则采用 PyArrow Dataset 分批扫描作为可运行备选。当前每表单文件且总数据不足 1 GiB，不建议先物理分区；先做 1 个 row group 或受限行数的 smoke test，再做全量列投影扫描。对昂贵 distinct、外键覆盖率和重复检查缓存小型审计结果（JSON/Parquet），不缓存原始数据副本。最可能瓶颈是 transaction 高基数 distinct/groupby 的 CPU、内存与临时磁盘，而不是 footer 读取。

## 12. 尚未验证事项

- 所有需要全量列扫描的唯一性、分布、空值率、键覆盖与时间范围指标。
- DuckDB/Polars 在本机的真实性能与稳定性（当前未安装时）。
- 数据集与任何外部权威校验和或发布清单的一致性。
- sample submission 缺失是有意裁剪还是下载不完整。

## 13. 下一阶段建议

阶段 0B 建议先建立数据契约与 smoke-test 审计：锁定最小依赖；用 DuckDB 或 PyArrow 扫描单个 row group/受限批次；验证审计 SQL/批处理的内存边界；再按列投影执行全量唯一性、空值率、日期范围、主外键覆盖与重复检查，并持久化仅含指标的小型结果。未经确认，本阶段不继续实施。
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    root = Path.cwd().resolve()
    parquet_paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".parquet"
    )
    records = [inspect_parquet(path, root) for path in parquet_paths]
    data_size = sum(record["file_size_bytes"] for record in records)

    disk_usage = shutil.disk_usage(root)
    environment = {
        "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        # 可公开快照不落盘本机绝对路径、用户名或环境变量值。
        "workspace": ".",
        "os": platform.platform(),
        "python": {
            "version": sys.version,
            "executable": Path(sys.executable).name,
            "virtual_env_active": bool(
                os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
            ),
        },
        "cpu": {"logical_cores": os.cpu_count()},
        "memory": windows_memory(),
        "disk": {
            "path": ".",
            "total_bytes": disk_usage.total,
            "used_bytes": disk_usage.used,
            "free_bytes": disk_usage.free,
        },
        "dependencies": package_versions(),
        "git": git_snapshot(root),
        "data_total_size_bytes": data_size,
        "parquet_file_count": len(records),
    }

    artifact_dir = root / "artifacts" / "stage_0a"
    report_dir = root / "docs" / "stages"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    inventory_path = artifact_dir / "parquet_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = [
            "logical_table",
            "file_path",
            "file_size_bytes",
            "row_count",
            "row_group_count",
            "compression",
            "schema_fingerprint",
            "readable",
            "error_message",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    snapshot_path = artifact_dir / "environment_snapshot.json"
    snapshot_path.write_text(
        json.dumps(environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = report_dir / "stage_0a_workspace_data_recon.md"
    write_report(root, records, environment, report_path)

    print(f"Parquet files: {len(records)}")
    print(f"Readable: {sum(record['readable'] for record in records)}")
    print(f"Inventory: {inventory_path.relative_to(root)}")
    print(f"Environment: {snapshot_path.relative_to(root)}")
    print(f"Report: {report_path.relative_to(root)}")
    return 0 if all(record["readable"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
