"""Stage 2A reproducible, leakage-safe data foundation."""

from .split import Snapshot, derive_snapshots

__all__ = ["Snapshot", "derive_snapshots"]
