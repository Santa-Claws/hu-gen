"""Conservative GPU-admission checks for unattended detector inference."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reasons: list[str]


def assess_gpu(
    memory_used_mib: int,
    memory_total_mib: int,
    utilization_percent: int,
    *,
    maximum_memory_mib: int = 1024,
    maximum_utilization_percent: int = 10,
) -> GateDecision:
    """Allow only a demonstrably idle GPU under conservative default limits."""
    if min(memory_used_mib, memory_total_mib, utilization_percent) < 0:
        raise ValueError("GPU measurements cannot be negative")
    if memory_used_mib > memory_total_mib:
        raise ValueError("GPU memory use cannot exceed total memory")

    reasons: list[str] = []
    if utilization_percent > maximum_utilization_percent:
        reasons.append(
            f"GPU utilization {utilization_percent}% exceeds {maximum_utilization_percent}% admission limit"
        )
    if memory_used_mib > maximum_memory_mib:
        reasons.append(
            f"GPU memory use {memory_used_mib} MiB exceeds {maximum_memory_mib} MiB admission limit"
        )
    return GateDecision(allowed=not reasons, reasons=reasons)
