from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.final_refit_post_training_recovery import (
    PostTrainingConfig,
    correct_post_training_metadata,
    finalize_post_training_recovery,
    preregister_post_training_recovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 post-training metadata/audit recovery")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_4b4_final_refit_post_training_recovery.yaml"),
    )
    parser.add_argument("--step", choices=("preregister", "correct", "finalize"), required=True)
    args = parser.parse_args()
    config = PostTrainingConfig.load(args.config)
    if args.step == "preregister":
        result = preregister_post_training_recovery(config)
    elif args.step == "correct":
        result = correct_post_training_metadata(config)
    else:
        result = finalize_post_training_recovery(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
