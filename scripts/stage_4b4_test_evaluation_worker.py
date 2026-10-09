from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.test_evaluation import (
    TestEvaluationConfig,
    aggregate_and_bootstrap,
    build_ground_truth,
    evaluate_partitions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--phase", choices=("ground_truth", "evaluation", "bootstrap"), required=True)
    args = parser.parse_args()
    config = TestEvaluationConfig.load(args.config)
    if args.phase == "ground_truth":
        result = build_ground_truth(config)
    elif args.phase == "evaluation":
        result = evaluate_partitions(config)
    else:
        result = aggregate_and_bootstrap(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
