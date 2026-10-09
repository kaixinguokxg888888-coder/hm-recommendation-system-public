from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.first7d_runner import (
    First7dDiagnosticRunner,
    PostFirst7dImplementationFailure,
)
from ranking_model.validation_runner import FrozenArtifactDrift


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage_4b4_first7d_diagnostic.yaml")
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runner = First7dDiagnosticRunner(root, root / args.config)
    try:
        result = runner.preregister() if args.preregister else runner.evaluate()
    except FrozenArtifactDrift as exc:
        print(json.dumps({"status": "STOP", "reason": "FROZEN_ARTIFACT_DRIFT", "detail": str(exc)}))
        return 2
    except PostFirst7dImplementationFailure as exc:
        print(json.dumps({"status": "STOP", "reason": "POST_FIRST7D_IMPLEMENTATION_FAILURE", "detail": str(exc)}))
        return 3
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
