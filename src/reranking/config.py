from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ranking_model.validation_selection import canonical_fingerprint, file_fingerprint

VARIANTS = ("R0", "R1", "R2", "R3", "R4_N20", "R4_N30", "R4_N50")


@dataclass(frozen=True)
class Config:
    root: Path
    raw: dict

    @classmethod
    def load(cls, path: Path) -> "Config":
        root = Path(__file__).resolve().parents[2]
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (raw["snapshot_id"], raw["model_id"], raw["as_of_date"]) != (
            "validation_14d", "L1_600", "2020-08-25"
        ):
            raise ValueError("only frozen Validation L1_600 is authorized")
        if tuple(v["id"] for v in raw["variants"]) != VARIANTS:
            raise ValueError("unregistered experiment matrix")
        for value in [*raw["paths"].values(), *raw["upstream"].values()]:
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError("repository relative paths required")
        cfg = cls(root, raw)
        registration = json.loads((cfg.public / "preregistration.json").read_text(encoding="utf-8"))
        if registration["config_fingerprint"] != canonical_fingerprint(raw):
            raise ValueError("preregistered config mutation refused")
        body = {k: v for k, v in registration.items() if k != "preregistration_fingerprint"}
        if canonical_fingerprint(body) != registration["preregistration_fingerprint"]:
            raise ValueError("preregistration self-fingerprint mismatch")
        protocol = root / "docs/decisions/stage_5_reranking_protocol.md"
        if file_fingerprint(protocol) != registration["protocol_physical_fingerprint"]:
            raise ValueError("preregistered protocol mutation refused")
        return cfg

    def path(self, name: str) -> Path:
        return self.root / self.raw["paths"][name]

    def upstream(self, name: str) -> Path:
        return self.root / self.raw["upstream"][name]

    @property
    def public(self) -> Path:
        return self.path("public")

    @property
    def private(self) -> Path:
        return self.path("private")

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.raw)
