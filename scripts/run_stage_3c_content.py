from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.stage3c.config import Stage3CConfig
from retrieval.stage3c.runner import Stage3CRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_3c_content.yaml")
    parser.add_argument(
        "--phase", choices=("smoke", "validation", "test", "finalize", "all"), default="all"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Stage3CRunner(Stage3CConfig.load(Path(args.config)))
    runner.prepare_directories()
    phase_status = "SUCCESS"
    if args.phase == "smoke":
        runner.run_group("smoke", runner.run_smoke, [runner.config.private_dir / "smoke_summary.json"])
    elif args.phase == "validation":
        phase_status = runner.run_validation()
    elif args.phase == "test":
        phase_status = runner.run_test()
    elif args.phase == "finalize":
        runner.finalize()
    else:
        runner.run_group("smoke", runner.run_smoke, [runner.config.private_dir / "smoke_summary.json"])
        runner.run_validation()
        runner.run_test()
        runner.finalize()
    print(f"Stage 3C {args.phase} {phase_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
