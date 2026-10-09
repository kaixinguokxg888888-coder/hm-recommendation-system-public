from __future__ import annotations

from pathlib import Path

from retrieval.stage3c.config import Stage3CConfig
from retrieval.stage3c.runner import Stage3CRunner


def main() -> int:
    runner = Stage3CRunner(Stage3CConfig.load(Path("configs/stage_3c_content.yaml")))
    runner.validate_public_and_privacy()
    print("Stage 3C final public/privacy/fingerprint audit SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
