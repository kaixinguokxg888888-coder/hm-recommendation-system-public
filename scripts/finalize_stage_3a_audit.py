from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from data_audit.stage1a import validate_public_artifacts
from retrieval.config import RetrievalConfig
from retrieval.runner import Stage3ARunner


FORBIDDEN_FIELDS = {
    "customer_key",
    "article_key",
    "customer_id",
    "article_id",
    "postal_code",
}


def main() -> int:
    config = RetrievalConfig.load(Path("configs/stage_3a_recall.yaml"))
    runner = Stage3ARunner(config)
    public = [
        *config.artifact_dir.glob("*.csv"),
        *config.artifact_dir.glob("*.json"),
        config.report_path,
        config.root / "README.md",
    ]
    for path in public:
        if not path.exists():
            raise FileNotFoundError(config.relative(path))
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                if FORBIDDEN_FIELDS & set(row):
                    raise ValueError(
                        f"公开 CSV 包含实体字段: {config.relative(path)}"
                    )
        text = path.read_text(encoding="utf-8-sig")
        if str(config.root) in text:
            raise ValueError(
                f"公开产物包含本地绝对路径: {config.relative(path)}"
            )

    validate_public_artifacts(public)
    for relative in (
        "artifacts/stage_3a/private",
        "artifacts/stage_3a/tmp",
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", relative],
            cwd=config.root,
            text=True,
            capture_output=True,
        )
        if ignored.returncode != 0:
            raise ValueError(f"{relative} 未命中 .gitignore")
    tracked = subprocess.run(
        ["git", "ls-files", "data/raw", "artifacts"],
        cwd=config.root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    unsafe = [
        value
        for value in tracked
        if value.startswith("data/raw/")
        or "/private/" in value
        or "/tmp/" in value
    ]
    if unsafe:
        raise ValueError("raw/private/tmp 存在 tracked 文件")

    print(
        "Stage 3A final public/privacy audit SUCCESS: "
        f"{len(public)} public files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
