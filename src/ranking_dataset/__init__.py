"""Stage 4B-2 ranking dataset contract implementation."""

from .config import Stage4B2Config
from .semantics import canonical_sampling_payload, deterministic_sample_rows

__all__ = ["Stage4B2Config", "canonical_sampling_payload", "deterministic_sample_rows"]
