from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.runner import Stage3ARunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_3a_recall.yaml")
    parser.add_argument(
        "--phase",
        choices=("smoke", "validation", "test", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Stage3ARunner(RetrievalConfig.load(Path(args.config)))
    runner.prepare_directories()
    if args.phase == "smoke":
        path = runner.run_smoke()
        print(f"Stage 3A synthetic smoke SUCCESS: {runner.config.relative(path)}")
        return 0
    if args.phase in ("validation", "all"):
        runner.run_validation()
    if args.phase in ("test", "all"):
        runner.run_test()
    if args.phase == "all":
        public_outputs = [
            runner.config.artifact_dir / "stage_3a_manifest.csv",
            runner.config.artifact_dir / "resource_usage.csv",
            runner.config.artifact_dir / "run_status.csv",
        ]
        runner.run_group(
            "public_aggregation",
            runner.write_public_operations,
            public_outputs,
        )
        runner.run_group(
            "report_generation",
            runner.write_report_and_readme,
            [runner.config.report_path, runner.config.root / "README.md"],
        )
        runner.finalize_resource_files()
        runner.validate_public_and_privacy()
    print(f"Stage 3A {args.phase} SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
