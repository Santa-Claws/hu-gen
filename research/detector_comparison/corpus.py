"""Utilities for fixed, labelled detector-comparison samples."""

from __future__ import annotations

from hashlib import sha256
from typing import Any


def limit_words(text: str, maximum: int) -> str:
    """Normalize whitespace and return no more than ``maximum`` words."""
    if maximum < 1:
        raise ValueError("maximum must be positive")
    return " ".join(text.split()[:maximum])


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""
    return sha256(text.encode("utf-8")).hexdigest()


def build_mixed_sample(
    *,
    sample_id: str,
    human_text: str,
    ai_text: str,
    human_source_id: str,
    ai_source_id: str,
) -> dict[str, Any]:
    """Create a transparently-labelled mixed sample from immutable segments."""
    text = f"{human_text.strip()}\n\n{ai_text.strip()}"
    return {
        "id": sample_id,
        "label": "mixed",
        "source_ids": [human_source_id, ai_source_id],
        "text": text,
        "text_sha256": sha256_text(text),
        "word_count": len(text.split()),
    }
