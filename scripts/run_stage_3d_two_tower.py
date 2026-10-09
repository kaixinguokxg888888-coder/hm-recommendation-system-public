from __future__ import annotations

import argparse
import json
from pathlib import Path

from retrieval.stage3d.config import Stage3DConfig
from retrieval.stage3d.runner import Stage3DRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fingerprinted Stage 3D Two-Tower workflow.")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/stage_3d_two_tower.yaml"),
        help="Repository-relative Stage 3D configuration path.",
    )
    parser.add_argument(
        "--phase", choices=("all", "validation", "test", "finalize", "smoke"), default="all",
        help="Resume at a protocol boundary; inner task groups reuse matching checkpoints.",
    )
    return parser.parse_args()


def smoke(runner: Stage3DRunner) -> dict[str, object]:
    runner.prepare_directories()
    runner.reproduce_q0("validation_14d")
    return {
        "stage": "3D", "status": "SUCCESS", "scope": "dependency_and_q0_smoke",
        "config_fingerprint": runner.config.fingerprint,
        "source_fingerprint": runner.source_fingerprint,
        "dataset_fingerprint": runner.dataset_fingerprint,
    }


def main() -> int:
    args = parse_args()
    runner = Stage3DRunner(Stage3DConfig.load(args.config))
    if args.phase == "all":
        status: object = runner.run_all()
    elif args.phase == "validation":
        status = runner.run_validation()
    elif args.phase == "test":
        status = runner.run_test()
    elif args.phase == "finalize":
        runner.finalize()
        status = "FINALIZED"
    else:
        status = smoke(runner)
    print(json.dumps({"stage": "3D", "phase": args.phase, "status": status},
                     ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
