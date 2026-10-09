from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_model.final_refit import finalize_phase4


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize Stage 4B-4 Phase 4 lock, gate, and report")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_4b4_final_refit.yaml"))
    args = parser.parse_args()
    print(json.dumps(finalize_phase4(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
