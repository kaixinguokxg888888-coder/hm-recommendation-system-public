from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ranking_features.config import Stage4B3Config  # noqa: E402
from ranking_features.runner import Stage4B3Runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-3 snapshot-safe ranking feature pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_4b3_ranking_features.yaml")
    parser.add_argument("--finalize-existing", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--reproduce-current-source-partition", type=int)
    args = parser.parse_args()
    runner = Stage4B3Runner(Stage4B3Config.load(args.config))
    if args.reproduce_current_source_partition is not None:
        result = runner.reproduce_current_source_partition(args.reproduce_current_source_partition)
    elif args.smoke_only:
        runner.prepare_directories()
        result = {"smoke_path": runner.config.relative(runner.run_smoke()), "status": "PASS"}
    else:
        result = runner.finalize_existing() if args.finalize_existing else runner.run_all()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
