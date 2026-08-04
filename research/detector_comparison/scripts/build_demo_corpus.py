"""Create a small fixed corpus for detector comparison.

The two artificial entries were written by the assistant for this benchmark and
are labelled accordingly. Human entries are fetched from public-domain Project
Gutenberg sources, then frozen locally with hashes in the manifest.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from research.detector_comparison.corpus import build_mixed_sample, limit_words, sha256_text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEXTS = DATA / "texts"

HUMAN_SOURCES = [
    {
        "id": "human-sherlock-001",
        "title": "A Scandal in Bohemia",
        "author": "Arthur Conan Doyle",
        "url": "https://www.gutenberg.org/files/1661/1661-0.txt",
        "start": "To Sherlock Holmes she is always _the_ woman.",
        "license_or_consent": "Project Gutenberg public-domain text (United States)",
    },
    {
        "id": "human-walden-001",
        "title": "Walden",
        "author": "Henry David Thoreau",
        "url": "https://www.gutenberg.org/files/205/205-0.txt",
        "start": "When I wrote the following pages",
        "license_or_consent": "Project Gutenberg public-domain text (United States)",
    },
]

AI_SAMPLES = [
    {
        "id": "ai-story-001",
        "genre": "short_story",
        "text": """By the time Mara found the spare key, the rain had turned the porch into a shallow mirror. The brass tag on the key ring said GREENHOUSE, though the greenhouse had been empty since her grandfather stopped growing tomatoes. She let herself in anyway. Inside, the air smelled of wet soil and old pots. A radio sat on the workbench, playing a baseball game through a cloud of static. Her grandfather had been gone for eight years, but the announcer was describing a game from the summer Mara was twelve.\n\nShe looked behind the seed trays for the property papers. Instead she found a small notebook, the kind he used for plant dates and weather notes. Every page was filled with brief entries: frost, broken hinge, rabbit tracks, Mara home from school. On the last page, beneath a drawing of a crooked sunflower, he had written, “Leave the back window open when it rains. The plants like to hear it.”\n\nMara laughed once, quietly. Then she opened the window. Rain tapped against the glass roof, and the greenhouse sounded less abandoned than it had a minute earlier.""",
        "generator_model": "gpt-5.6-terra via Hermit",
        "generator_prompt": "Write a 250-word literary short story about finding a key in a rainstorm.",
    },
    {
        "id": "ai-explainer-001",
        "genre": "explainer",
        "text": """A neighborhood tool library works like a regular library, except the shelves hold drills, ladders, sewing machines, pressure washers, and other equipment that people may need only once or twice a year. Members pay a small fee, complete a basic safety orientation, and borrow items for a limited period. The model can save households money and reduce the number of rarely used tools sitting in garages.\n\nThe difficult part is maintenance. A borrowed circular saw needs more attention than a borrowed novel. Tool libraries usually inspect equipment when it comes back, replace worn blades or batteries, and set aside a repair budget. Some also run workshops so members can learn how to use unfamiliar equipment before taking it home.\n\nThe strongest programs rely on more than inventory software. They need volunteers who understand repairs, clear borrowing rules, and a location people can reach without a car. When those pieces are in place, the library becomes a practical shared resource rather than simply a room full of tools.""",
        "generator_model": "gpt-5.6-terra via Hermit",
        "generator_prompt": "Write a 250-word plain-language explainer of a neighborhood tool library.",
    },
]


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "hu-gen-detector-benchmark/0.1"})
    return urlopen(request, timeout=30).read().decode("utf-8", errors="replace")


def record(sample: dict) -> dict:
    text = sample["text"].strip()
    return {**sample, "text": text, "word_count": len(text.split()), "text_sha256": sha256_text(text)}


def main() -> None:
    TEXTS.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    for source in HUMAN_SOURCES:
        raw = fetch_text(source["url"])
        start = raw.index(source["start"])
        sample = record({
            **source,
            "label": "human_authored",
            "source_type": "public_domain_online",
            "text": limit_words(raw[start:], 360),
        })
        samples.append(sample)
    for item in AI_SAMPLES:
        samples.append(record({**item, "label": "ai_generated", "source_type": "assistant_generated", "license_or_consent": "synthetic-test-data"}))
    samples.extend([
        build_mixed_sample(sample_id="mixed-sherlock-story-001", human_text=samples[0]["text"], ai_text=samples[2]["text"], human_source_id=samples[0]["id"], ai_source_id=samples[2]["id"]),
        build_mixed_sample(sample_id="mixed-walden-explainer-001", human_text=samples[1]["text"], ai_text=samples[3]["text"], human_source_id=samples[1]["id"], ai_source_id=samples[3]["id"]),
    ])
    for sample in samples:
        (TEXTS / f'{sample["id"]}.txt').write_text(sample["text"] + "\n", encoding="utf-8")
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "samples": samples}
    (DATA / "samples.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(samples)} fixed samples to {TEXTS}")


if __name__ == "__main__":
    main()
