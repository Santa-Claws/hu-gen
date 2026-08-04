"""Throwaway feasibility spike: test quantized GPT-J + GPT-Neo GPU logits."""

from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CACHE_DIR = "/home/tmac/projects/hu-gen-fastdetect/spikes/001-quantized-gptj-neo/hf-cache"
TEXT = "The library opens at nine in the morning and closes at six."


def gpu_state(stage: str) -> None:
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({
        "stage": stage,
        "allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
        "reserved_mib": round(torch.cuda.memory_reserved() / 2**20, 1),
        "free_mib": round(free / 2**20, 1),
        "total_mib": round(total / 2**20, 1),
    }), flush=True)


def load_and_score(model_id: str, quantization: BitsAndBytesConfig):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        device_map={"": 0},
        quantization_config=quantization,
    )
    load_seconds = round(time.perf_counter() - started, 2)
    gpu_state(f"loaded:{model_id}")
    tokens = tokenizer(TEXT, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(**tokens).logits
    print(json.dumps({
        "model": model_id,
        "load_seconds": load_seconds,
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
    }), flush=True)
    del logits
    return model, tokenizer


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.empty_cache()
    gpu_state("start")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    gptj, _ = load_and_score("EleutherAI/gpt-j-6B", quantization)
    neo, _ = load_and_score("EleutherAI/gpt-neo-2.7B", quantization)
    gpu_state("pair_ready")
    # Keep both resident long enough to prove the simultaneous fit, then release.
    del neo, gptj
    torch.cuda.empty_cache()
    gpu_state("released")


if __name__ == "__main__":
    main()
