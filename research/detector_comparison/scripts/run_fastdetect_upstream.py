"""Run the upstream Fast-DetectGPT local implementation over a frozen manifest.

Usage (on the Fast-DetectGPT host):
    python run_fastdetect_upstream.py /path/to/fast-detect-gpt /path/to/samples.manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fastdetect_repo", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="gpt-neo-2.7B")
    args = parser.parse_args()

    scripts_dir = args.fastdetect_repo / "scripts"
    sys.path.insert(0, str(scripts_dir))
    from local_infer import FastDetectGPT  # imported from pinned upstream checkout

    detector_args = argparse.Namespace(
        sampling_model_name=args.model,
        scoring_model_name=args.model,
        device=args.device,
        cache_dir=str(args.fastdetect_repo / "cache"),
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    detector = FastDetectGPT(detector_args)
    for sample in manifest["samples"]:
        probability, criterion, tokens = detector.compute_prob(sample["text"])
        print(json.dumps({
            "provider": "fast-detect-gpt-upstream",
            "sampling_model": args.model,
            "scoring_model": args.model,
            "sample_id": sample["id"],
            "sample_sha256": sample["text_sha256"],
            "known_label": sample["label"],
            "criterion": criterion,
            "reported_ai_probability": probability,
            "tokens": tokens,
        }), flush=True)


if __name__ == "__main__":
    main()
