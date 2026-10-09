from __future__ import annotations

from collections import Counter
from typing import Sequence

from .contracts import Candidate, validate_selection


def unique_rows(rows: Sequence[Candidate]) -> list[Candidate]:
    seen: set[int] = set()
    result = []
    for row in rows:
        if row.article not in seen:
            result.append(row)
            seen.add(row.article)
    return result


def counts(rows: Sequence[Candidate], field: str) -> Counter:
    return Counter(getattr(r, field) for r in rows if getattr(r, field) is not None)


def cap_ok(rows: Sequence[Candidate], product: int | None, garment: int | None) -> bool:
    return (product is None or max(counts(rows, "product").values(), default=0) <= product) and (
        garment is None or max(counts(rows, "garment").values(), default=0) <= garment)


def rerank(rows: Sequence[Candidate], *, product_cap: int | None = None,
           garment_cap: int | None = None, tail_window: int | None = None, k: int = 12,
           garment_relax: Sequence[int] = (5, 6), product_relax: Sequence[int] = (3, 4)) -> list[tuple[Candidate, str, int]]:
    if any(c is not None and c < 1 for c in (product_cap, garment_cap)) or k < 1:
        raise ValueError("invalid cap/K")
    candidates = unique_rows(rows)
    selected: list[tuple[Candidate, str, int]] = []
    seen: set[int] = set()
    products: Counter = Counter()
    garments: Counter = Counter()

    def fill(pc: int | None, gc: int | None, level: int, reason: str) -> None:
        for row in candidates:
            if len(selected) >= k:
                return
            if row.article in seen:
                continue
            if pc is not None and row.product is not None and products[row.product] >= pc:
                continue
            if gc is not None and row.garment is not None and garments[row.garment] >= gc:
                continue
            why = ("base" if row.rank <= k else "constraint_fill") if level == 0 else reason
            selected.append((row, why, level)); seen.add(row.article)
            if row.product is not None:
                products[row.product] += 1
            if row.garment is not None:
                garments[row.garment] += 1

    fill(product_cap, garment_cap, 0, "base")
    if garment_cap is not None:
        for gc in garment_relax:
            if gc > garment_cap:
                fill(product_cap, gc, 1, "relax_garment")
    relaxed_garment = max((garment_cap or 0, *garment_relax)) if garment_cap is not None else None
    if product_cap is not None:
        for pc in product_relax:
            if pc > product_cap:
                fill(pc, relaxed_garment, 2, "relax_product")
    fill(None, None, 3, "final_fill")
    selected.sort(key=lambda value: value[0].rank)
    # 探索不触发放宽，也不覆盖已有自然 tail；只在严格可行的列表上作一次末位替换。
    if tail_window is not None and len(selected) == k and all(level == 0 for _, _, level in selected) and not any(r.tail for r, _, _ in selected):
        retained = [r for r, _, _ in selected[:-1]]
        for row in candidates:
            if row.rank > tail_window:
                break
            if row.tail and row.article not in seen and cap_ok([*retained, row], product_cap, garment_cap):
                selected[-1] = (row, "tail_exploration", 0)
                selected.sort(key=lambda value: value[0].rank)
                break
    validate_selection(selected, rows, k)
    return selected
