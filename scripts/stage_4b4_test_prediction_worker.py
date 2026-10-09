from __future__ import annotations

import argparse
from pathlib import Path

from ranking_model.test_prediction_freeze import materialize_features, run_predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("features", "predictions_formal", "predictions_replay"), required=True)
    args = parser.parse_args()
    if args.phase == "features":
        materialize_features(args.config)
    elif args.phase == "predictions_formal":
        run_predictions(args.config, replay=False)
    else:
        run_predictions(args.config, replay=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
