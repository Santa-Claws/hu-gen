"""Explicit, hardware-gated Fast-DetectGPT experiment configurations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Requirements are conservative operational minima, not merely FP16 parameter sizes.
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "neo_2p7b__neo_2p7b__analytic": {
        "id": "neo_2p7b__neo_2p7b__analytic",
        "sampling_model": "gpt-neo-2.7B",
        "scoring_model": "gpt-neo-2.7B",
        "criterion_mode": "analytic",
        "minimum_vram_gib": 9,
        "upstream_calibration": "gpt-neo-2.7B_gpt-neo-2.7B",
        "fingerprint": "fastdetectgpt:neo_2p7b__neo_2p7b__analytic:v1",
    },
    "neo_2p7b__neo_2p7b__monte_carlo": {
        "id": "neo_2p7b__neo_2p7b__monte_carlo",
        "sampling_model": "gpt-neo-2.7B",
        "scoring_model": "gpt-neo-2.7B",
        "criterion_mode": "monte_carlo",
        "minimum_vram_gib": 9,
        "upstream_calibration": None,
        "fingerprint": "fastdetectgpt:neo_2p7b__neo_2p7b__monte_carlo:v1",
    },
    "gptj_6b__neo_2p7b__analytic": {
        "id": "gptj_6b__neo_2p7b__analytic",
        "sampling_model": "gpt-j-6B",
        "scoring_model": "gpt-neo-2.7B",
        "criterion_mode": "analytic",
        "minimum_vram_gib": 20,
        "upstream_calibration": "gpt-j-6B_gpt-neo-2.7B",
        "fingerprint": "fastdetectgpt:gptj_6b__neo_2p7b__analytic:v1",
    },
}


def get_configuration(configuration_id: str) -> dict[str, Any]:
    """Return an isolated configuration dictionary by stable identifier."""
    try:
        return deepcopy(CONFIGURATIONS[configuration_id])
    except KeyError as exc:
        raise ValueError(f"unknown configuration: {configuration_id}") from exc


def validate_vram(configuration: dict[str, Any], available_gib: int | float) -> list[str]:
    """Return a refusal reason when a native configuration cannot fit safely."""
    required = configuration["minimum_vram_gib"]
    if available_gib < required:
        return [f"requires at least {required} GiB VRAM; only {available_gib:g} GiB available"]
    return []
