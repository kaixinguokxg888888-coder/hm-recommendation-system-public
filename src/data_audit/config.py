from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import psutil


def find_project_root(start: Path | None = None) -> Path:
    """向上查找 pyproject.toml，避免把本机绝对路径写入配置。"""
    current = (start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("无法从当前路径定位包含 pyproject.toml 的项目根目录")


def conservative_memory_limit_mb(available_bytes: int | None = None) -> int:
    """最多使用可用内存的四分之一，并限制在 256 MiB 到 1 GiB。"""
    available = available_bytes or psutil.virtual_memory().available
    return max(256, min(1024, int(available // (4 * 1024 * 1024))))


@dataclass(frozen=True)
class AuditSettings:
    project_root: Path
    contracts_path: Path
    artifact_dir: Path
    temp_dir: Path
    memory_limit_mb: int
    threads: int = 2
    transaction_row_group: int = 0
    transaction_max_rows: int = 100_000

    @classmethod
    def default(cls, project_root: Path | None = None) -> "AuditSettings":
        root = (project_root or find_project_root()).resolve()
        artifact_dir = root / "artifacts" / "stage_0b"
        return cls(
            project_root=root,
            contracts_path=root / "configs" / "data_contracts.yaml",
            artifact_dir=artifact_dir,
            temp_dir=artifact_dir / "tmp",
            memory_limit_mb=conservative_memory_limit_mb(),
        )

    def relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.project_root).as_posix()
