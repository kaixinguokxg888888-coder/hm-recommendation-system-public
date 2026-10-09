from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Snapshot:
    split_id: str
    history_start: date
    history_end: date
    label_start: date
    label_end: date
    window_days: int

    def validate(self) -> None:
        if self.history_end >= self.label_start:
            raise ValueError("history_end 必须严格早于 label_start")
        if self.label_start > self.label_end:
            raise ValueError("label_start 不得晚于 label_end")
        actual = (self.label_end - self.label_start).days + 1
        if actual != self.window_days:
            raise ValueError("标签窗口天数与 window_days 不一致")

    def contains_history(self, value: date) -> bool:
        return self.history_start <= value <= self.history_end

    def contains_label(self, value: date) -> bool:
        return self.label_start <= value <= self.label_end


def derive_snapshots(d_max: date, history_start: date) -> dict[str, Snapshot]:
    test_start = d_max - timedelta(days=13)
    validation_end = test_start - timedelta(days=1)
    validation_start = validation_end - timedelta(days=13)
    snapshots = {
        "validation_14d": Snapshot(
            "validation_14d",
            history_start,
            validation_start - timedelta(days=1),
            validation_start,
            validation_end,
            14,
        ),
        "test_14d": Snapshot(
            "test_14d",
            history_start,
            test_start - timedelta(days=1),
            test_start,
            d_max,
            14,
        ),
    }
    for snapshot in snapshots.values():
        snapshot.validate()
    if snapshots["validation_14d"].label_end >= snapshots["test_14d"].label_start:
        raise ValueError("validation 与 test 标签窗口重叠")
    return snapshots


def trailing_window(d_max: date, days: int) -> tuple[date, date]:
    if days <= 0:
        raise ValueError("窗口天数必须为正")
    return d_max - timedelta(days=days - 1), d_max
