"""Stage 3A leakage-safe baseline retrieval and offline evaluation."""

from .contracts import CandidateContext, CandidateProtocolError, validate_candidates

__all__ = ["CandidateContext", "CandidateProtocolError", "validate_candidates"]
