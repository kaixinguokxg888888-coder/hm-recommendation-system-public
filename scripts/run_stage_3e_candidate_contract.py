from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.stage3e.config import Stage3EConfig
from retrieval.stage3e.runner import Stage3ERunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze and replay the Stage 3D TT1-T100 candidate contract")
    parser.add_argument("--snapshot", choices=["validation_14d", "test_14d", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()
    runner = Stage3ERunner(Stage3EConfig.load(Path("configs/stage_3e_candidate_contract.yaml")))
    snapshots = runner.config.raw["snapshots"] if args.snapshot == "all" else [args.snapshot]
    runner.run(snapshots=snapshots, force=args.force, replay=not args.skip_replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
