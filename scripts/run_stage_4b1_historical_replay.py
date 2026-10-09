from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval.stage4b1.config import Stage4B1Config  # noqa: E402
from retrieval.stage4b1.runner import Stage4B1Runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-1 historical candidate replay")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/stage_4b1_historical_replay.yaml",
    )
    parser.add_argument("--smoke", action="store_true", help="run only the synthetic contract smoke")
    args = parser.parse_args()
    runner = Stage4B1Runner(Stage4B1Config.load(args.config))
    runner.prepare_directories()
    if args.smoke:
        result = {"smoke_path": runner.config.relative(runner.run_smoke()), "status": "PASS"}
    else:
        result = runner.run_all()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
