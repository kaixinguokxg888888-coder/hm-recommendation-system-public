"""Stage 4B-4 pre-label LambdaRank training and prediction freeze."""

from .config import Stage4B4Config

__all__ = ["Stage4B4Config", "Stage4B4Runner"]


def __getattr__(name: str):
    # Isolated training worker only needs config; lazy loading avoids importing DuckDB/PyArrow/Validation code.
    if name == "Stage4B4Runner":
        from .runner import Stage4B4Runner
        return Stage4B4Runner
    raise AttributeError(name)
