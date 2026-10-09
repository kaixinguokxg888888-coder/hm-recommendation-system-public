from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model import Stage4B4Config, Stage4B4Runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 Phase 1 pre-label prediction freeze")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_lambdarank.yaml"))
    parser.add_argument("--resource-smoke", action="store_true")
    args = parser.parse_args()
    runner = Stage4B4Runner(Stage4B4Config.load(args.config))
    if args.resource_smoke:
        print(json.dumps(runner.resource_smoke(), indent=2))
        return
    result = runner.run()
    print(json.dumps({"status": "STAGE_4B4_PHASE1_PREDICTIONS_FROZEN",
                      "validation_queries": result["validation"]["query_count"],
                      "validation_rows": result["validation"]["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
