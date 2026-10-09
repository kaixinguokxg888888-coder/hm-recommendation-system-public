from __future__ import annotations

from collections import Counter
from datetime import date, timedelta
from typing import Iterable


def dates_inclusive(start: date, end: date) -> list[date]:
    if start > end:
        raise ValueError("start 不得晚于 end")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def weekday_composition(start: date, end: date) -> dict[int, int]:
    return dict(Counter(value.isoweekday() for value in dates_inclusive(start, end)))


def missing_dates(start: date, end: date, observed: Iterable[date]) -> list[date]:
    return sorted(set(dates_inclusive(start, end)) - set(observed))
