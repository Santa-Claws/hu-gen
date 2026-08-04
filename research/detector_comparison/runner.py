"""Append-only, resumable execution primitives for detector experiments."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

Scorer = Callable[[str], dict[str, Any]]


def _completed_keys(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "success":
            completed.add((record["sample_sha256"], record["configuration_fingerprint"]))
    return completed


def run_samples(
    samples: list[dict[str, Any]],
    scorer: Scorer,
    output_path: Path,
    run_id: str,
    configuration: dict[str, Any],
    host_metadata: dict[str, Any],
) -> dict[str, int]:
    """Score samples, append each outcome immediately, and resume prior successes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = configuration["fingerprint"]
    completed = _completed_keys(output_path)
    summary = {"completed": 0, "failed": 0, "skipped": 0}

    with output_path.open("a", encoding="utf-8") as stream:
        for sample in samples:
            key = (sample["text_sha256"], fingerprint)
            if key in completed:
                summary["skipped"] += 1
                continue

            started = time.perf_counter()
            record: dict[str, Any] = {
                "run_id": run_id,
                "run_at": datetime.now(timezone.utc).isoformat(),
                "configuration_id": configuration["id"],
                "configuration_fingerprint": fingerprint,
                "host_metadata": host_metadata,
                "sample_id": sample["id"],
                "sample_sha256": sample["text_sha256"],
                "known_label": sample["label"],
            }
            try:
                record.update(scorer(sample["text"]))
                record["status"] = "success"
                summary["completed"] += 1
            except Exception as exc:  # Records failure so later samples still run.
                record.update({
                    "status": "failure",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
                summary["failed"] += 1
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()

    return summary
