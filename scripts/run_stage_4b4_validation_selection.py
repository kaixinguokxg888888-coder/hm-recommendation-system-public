from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.validation_runner import FrozenArtifactDrift, ValidationSelectionRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_4b4_validation_selection.yaml")
    parser.add_argument("--preregister", action="store_true")
    parser.add_argument("--register-recovery", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = ValidationSelectionRunner(root, root / args.config)
    try:
        if args.preregister:
            result = runner.preregister()
        elif args.register_recovery:
            result = runner.register_recovery_correction()
        else:
            result = runner.evaluate()
    except FrozenArtifactDrift as exc:
        print(json.dumps({"status": "STOP", "reason": "FROZEN_ARTIFACT_DRIFT", "detail": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
