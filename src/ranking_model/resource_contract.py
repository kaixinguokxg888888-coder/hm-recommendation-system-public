from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


FORMAL_RSS_FRACTION = 0.70


class ResourceContractError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenRssContract:
    available_memory_at_preregistration_bytes: int
    formal_fraction: float
    frozen_absolute_rss_cap_bytes: int
    runtime_available_memory_bytes: int
    source: str


def absolute_cap(available_memory_bytes: int, fraction: float) -> int:
    if available_memory_bytes <= 0:
        raise ResourceContractError("available memory must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ResourceContractError("RSS fraction must be in (0, 1]")
    return int(available_memory_bytes * fraction)


def validate_registered_fraction(registered: float, required: float = FORMAL_RSS_FRACTION) -> None:
    if registered != required:
        raise ResourceContractError(
            f"registered RSS fraction {registered!r} does not equal required {required!r}"
        )


def consume_frozen_rss_contract(
    preregistration: Mapping[str, Any],
    *,
    runtime_available_memory_bytes: int,
    recovery: Mapping[str, Any] | None = None,
) -> FrozenRssContract:
    """Resolve the formal cap without deriving it from runtime memory.

    Runtime available memory is observation-only. A normal run consumes the
    absolute cap already frozen in preregistration and verifies its 70%
    arithmetic. An authorized recovery consumes the corrected absolute cap
    bound to the original preregistration available-memory observation.
    """
    resource = preregistration.get("resource_contract")
    if not isinstance(resource, Mapping):
        raise ResourceContractError("missing preregistration resource contract")
    available = int(resource["available_memory_at_preregistration_bytes"])
    registered_cap = int(resource["rss_hard_limit_bytes"])
    expected_cap = absolute_cap(available, FORMAL_RSS_FRACTION)

    if recovery is None:
        if registered_cap != expected_cap:
            raise ResourceContractError(
                "preregistration absolute RSS cap does not equal frozen 70% arithmetic"
            )
        cap = registered_cap
        source = "preregistration_absolute_cap"
    else:
        if int(recovery["available_memory_at_original_preregistration_bytes"]) != available:
            raise ResourceContractError("recovery available-memory identity mismatch")
        if int(recovery["invalid_preregistered_cap_bytes"]) != registered_cap:
            raise ResourceContractError("recovery invalid-cap identity mismatch")
        if float(recovery["formal_required_fraction"]) != FORMAL_RSS_FRACTION:
            raise ResourceContractError("recovery formal fraction mismatch")
        if int(recovery["corrected_formal_cap_bytes"]) != expected_cap:
            raise ResourceContractError("recovery corrected-cap arithmetic mismatch")
        cap = int(recovery["corrected_formal_cap_bytes"])
        source = "authorized_recovery_absolute_cap"

    return FrozenRssContract(
        available_memory_at_preregistration_bytes=available,
        formal_fraction=FORMAL_RSS_FRACTION,
        frozen_absolute_rss_cap_bytes=cap,
        runtime_available_memory_bytes=int(runtime_available_memory_bytes),
        source=source,
    )


def resource_gate_passes(observed_peak_rss_bytes: int, frozen_absolute_rss_cap_bytes: int) -> bool:
    return int(observed_peak_rss_bytes) <= int(frozen_absolute_rss_cap_bytes)
