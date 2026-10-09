from __future__ import annotations

import csv
import json
from pathlib import Path

from ranking_model import Stage4B4Config, Stage4B4Runner


def main() -> None:
    config = Stage4B4Config.load(Path("configs/stage_4b4_lambdarank.yaml"))
    runner = Stage4B4Runner(config)
    with (config.artifact_dir / "validation_feature_fingerprint_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        feature = next(csv.DictReader(handle))
    validation = {
        "candidate_projection_fingerprint": feature["candidate_projection_fingerprint"],
        "feature_semantic_fingerprint": feature["feature_semantic_fingerprint"],
    }
    models = [
        json.loads((config.private_dir / "models" / f"{model_id}_metadata.json").read_text(encoding="utf-8"))
        for model_id in ("L0", "L1")
    ]
    if any(model["status"] != "ACCEPTED_AFTER_ISOLATED_RESOURCE_GATE" for model in models):
        raise RuntimeError("both isolated models must pass resource acceptance before prediction")
    predictions = runner.freeze_predictions(validation, models)
    print(json.dumps({"status": "PREDICTIONS_COMPLETE", "versions": len(predictions)}, indent=2))


if __name__ == "__main__":
    main()
