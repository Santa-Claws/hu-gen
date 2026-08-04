"""Throwaway runtime spike: upstream-equivalent analytical criterion on a NF4 pair."""

from __future__ import annotations

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CACHE_DIR = "/home/tmac/projects/hu-gen-fastdetect/spikes/001-quantized-gptj-neo/hf-cache"
TEXT = "The library opens at nine in the morning and closes at six."


def criterion(logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor) -> float:
    """Copied operation-for-operation from upstream get_sampling_discrepancy_analytic."""
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]
    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    return ((log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()).mean().item()


def main() -> None:
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    score_name = "EleutherAI/gpt-neo-2.7B"
    ref_name = "EleutherAI/gpt-j-6B"
    started = time.perf_counter()
    score_tokenizer = AutoTokenizer.from_pretrained(score_name, cache_dir=CACHE_DIR)
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_name, cache_dir=CACHE_DIR)
    score_model = AutoModelForCausalLM.from_pretrained(score_name, cache_dir=CACHE_DIR, device_map={"": 0}, quantization_config=quantization)
    ref_model = AutoModelForCausalLM.from_pretrained(ref_name, cache_dir=CACHE_DIR, device_map={"": 0}, quantization_config=quantization)
    score_model.eval()
    ref_model.eval()
    score_tokens = score_tokenizer(TEXT, return_tensors="pt").to("cuda")
    ref_tokens = ref_tokenizer(TEXT, return_tensors="pt").to("cuda")
    labels = score_tokens.input_ids[:, 1:]
    if not torch.equal(ref_tokens.input_ids[:, 1:], labels):
        raise RuntimeError("Tokenizer IDs differ; refusing to score")
    with torch.inference_mode():
        logits_score = score_model(**score_tokens).logits[:, :-1]
        logits_ref = ref_model(**ref_tokens).logits[:, :-1]
        value = criterion(logits_ref, logits_score, labels)
    free, total = torch.cuda.mem_get_info()
    print(json.dumps({
        "status": "success",
        "criterion": value,
        "sampling_model": ref_name,
        "scoring_model": score_name,
        "quantization": "nf4_4bit_double_quant",
        "tokenizer_ids_equal": True,
        "logits_ref_shape": list(logits_ref.shape),
        "logits_score_shape": list(logits_score.shape),
        "free_mib": round(free / 2**20, 1),
        "total_mib": round(total / 2**20, 1),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
