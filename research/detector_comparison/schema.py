"""Validation for provenance-labelled detector benchmark records."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

VALID_LABELS = {"human_authored", "ai_generated", "mixed"}


def validate_sample(sample: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for one frozen benchmark sample."""
    errors: list[str] = []
    for field in ("id", "label", "text", "word_count", "text_sha256", "source_type", "license_or_consent", "source_group"):
        if not sample.get(field):
            errors.append(f"{field} is required")

    if sample.get("label") not in VALID_LABELS:
        errors.append("label must be human_authored, ai_generated, or mixed")

    text = sample.get("text")
    if isinstance(text, str):
        if sample.get("word_count") != len(text.split()):
            errors.append("word_count does not match text")
        expected_hash = sha256(text.encode("utf-8")).hexdigest()
        if sample.get("text_sha256") != expected_hash:
            errors.append("text_sha256 does not match text")
    elif text is not None:
        errors.append("text must be a string")

    label = sample.get("label")
    if label == "human_authored" and not sample.get("source_url"):
        errors.append("source_url is required for human_authored samples")
    if label == "ai_generated":
        if not sample.get("generator_model"):
            errors.append("generator_model is required for ai_generated samples")
        if not sample.get("generator_prompt"):
            errors.append("generator_prompt is required for ai_generated samples")
    if label == "mixed" and len(sample.get("source_ids", [])) < 2:
        errors.append("mixed samples require at least two source_ids")

    return errors
