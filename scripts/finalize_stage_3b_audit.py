from __future__ import annotations

from pathlib import Path

from retrieval.stage3b.config import Stage3BConfig
from retrieval.stage3b.runner import Stage3BRunner


def main() -> int:
    runner = Stage3BRunner(Stage3BConfig.load(Path("configs/stage_3b_itemcf.yaml")))
    runner.validate_public_and_privacy()
    print("Stage 3B final public/privacy/fingerprint audit SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
