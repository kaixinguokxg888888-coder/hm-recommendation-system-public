from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.pytest_skip_reconciliation import (
    ReconciliationConfig,
    finalize_reconciliation,
    preregister_reconciliation,
    prepare_attempt3,
    run_reconciliation_audit,
    supersede_failed_attempt1,
    supersede_failed_attempt2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile Stage 4B-4 Phase 4 pytest skip evidence")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage_4b4_pytest_skip_reconciliation.yaml"),
    )
    parser.add_argument(
        "--step",
        choices=(
            "supersede-attempt1",
            "supersede-attempt2",
            "prepare-attempt3",
            "preregister",
            "audit",
            "finalize",
        ),
        required=True,
    )
    args = parser.parse_args()
    config = ReconciliationConfig.load(args.config)
    if args.step == "supersede-attempt1":
        result = supersede_failed_attempt1(config)
    elif args.step == "supersede-attempt2":
        result = supersede_failed_attempt2(config)
    elif args.step == "prepare-attempt3":
        result = prepare_attempt3(config)
    elif args.step == "preregister":
        result = preregister_reconciliation(config)
    elif args.step == "audit":
        result = run_reconciliation_audit(config)
    else:
        result = finalize_reconciliation(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
