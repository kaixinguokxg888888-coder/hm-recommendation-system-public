from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.final_refit_resource_recovery import (
    RecoveryConfig,
    convert_binary_dataset,
    finalize_recovery,
    preregister_recovery,
    train_from_binary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4B-4 Phase 4 final-refit resource recovery")
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/stage_4b4_final_refit_resource_recovery.yaml"),
    )
    parser.add_argument("--step", choices=("preregister", "convert", "train", "finalize"), required=True)
    args = parser.parse_args()
    config = RecoveryConfig.load(args.config)
    if args.step == "preregister":
        result = preregister_recovery(config)
    elif args.step == "convert":
        result = convert_binary_dataset(config)
    elif args.step == "train":
        result = train_from_binary(config)
    else:
        result = finalize_recovery(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
