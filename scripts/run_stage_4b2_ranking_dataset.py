from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ranking_dataset.config import Stage4B2Config  # noqa: E402
from ranking_dataset.runner import Stage4B2Runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-2 ranking dataset and deterministic sampling")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/stage_4b2_ranking_dataset.yaml")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="verify complete producer checkpoints and run audit/public finalization only")
    args = parser.parse_args()
    runner = Stage4B2Runner(Stage4B2Config.load(args.config))
    result = runner.finalize_existing() if args.finalize_existing else runner.run_all()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
