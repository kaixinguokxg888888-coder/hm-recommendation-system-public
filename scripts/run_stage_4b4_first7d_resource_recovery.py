from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.first7d_resource_recovery import (
    First7dResourceRecoveryRunner,
    RecoveryArtifactDrift,
    RecoveryScopeExceeded,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/stage_4b4_first7d_resource_recovery.yaml"
    )
    parser.add_argument("--targeted-result", required=True)
    parser.add_argument("--full-result", required=True)
    parser.add_argument("--compile-result", required=True)
    parser.add_argument("--diff-check-result", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        runner = First7dResourceRecoveryRunner(root, root / args.config)
        result = runner.finalize(
            targeted_result=args.targeted_result,
            full_result=args.full_result,
            compile_result=args.compile_result,
            diff_check_result=args.diff_check_result,
        )
    except RecoveryArtifactDrift as exc:
        print(json.dumps({"status": "STOP", "reason": "RECOVERY_ARTIFACT_DRIFT", "detail": str(exc)}))
        return 2
    except RecoveryScopeExceeded as exc:
        print(json.dumps({"status": "STOP", "reason": "RECOVERY_SCOPE_EXCEEDED", "detail": str(exc)}))
        return 3
    except Exception as exc:
        print(json.dumps({"status": "STOP", "reason": "RECOVERY_IMPLEMENTATION_FAILURE", "detail": str(exc)}))
        return 4
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
