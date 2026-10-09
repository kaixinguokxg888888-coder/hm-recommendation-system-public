from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.stage3b.config import Stage3BConfig
from retrieval.stage3b.runner import Stage3BRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_3b_itemcf.yaml")
    parser.add_argument(
        "--phase",
        choices=("smoke", "validation", "test", "finalize", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Stage3BRunner(Stage3BConfig.load(Path(args.config)))
    runner.prepare_directories()
    if args.phase == "smoke":
        runner.run_group("smoke", runner.run_smoke, [runner.config.private_dir / "smoke_summary.json"])
    elif args.phase == "validation":
        runner.run_validation()
    elif args.phase == "test":
        runner.run_test()
    elif args.phase == "finalize":
        runner.finalize()
    else:
        runner.run_group("smoke", runner.run_smoke, [runner.config.private_dir / "smoke_summary.json"])
        runner.run_validation()
        runner.run_test()
        runner.finalize()
    print(f"Stage 3B {args.phase} SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
