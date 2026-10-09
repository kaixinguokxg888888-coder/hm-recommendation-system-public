from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.final_refit import FinalRefitConfig, preregister, preregister_and_materialize


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 Phase 4 preregistration and materialization")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_final_refit.yaml"))
    parser.add_argument("--step", choices=("preregister", "materialize"), default="materialize")
    args = parser.parse_args()
    if args.step == "preregister":
        result = preregister(FinalRefitConfig.load(args.config))
    else:
        result = preregister_and_materialize(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
