# Spike 001 — Quantized GPT-J + GPT-Neo Fast-DetectGPT feasibility

## Question

Can an RTX 3080 with 12 GB VRAM load **GPT-J 6B** and **GPT-Neo 2.7B** together in quantized form and calculate Fast-DetectGPT's analytical conditional-probability-curvature criterion?

## Approach

This is a disposable feasibility experiment, isolated from the verified upstream Fast-DetectGPT environment:

- Host: NESTLECRUNCH, NVIDIA RTX 3080 (12 GB)
- Environment: Python 3.11, PyTorch `2.13.0+cu130`, Transformers `4.57.1`, Accelerate `1.14.0`, BitsAndBytes `0.50.0`
- Quantization: BitsAndBytes **NF4 4-bit**, double quantization, FP16 compute
- Sampling/reference model: `EleutherAI/gpt-j-6B`
- Scoring model: `EleutherAI/gpt-neo-2.7B`
- Criterion: copied operation-for-operation from upstream `get_sampling_discrepancy_analytic` at Fast-DetectGPT commit `971b05202bac2bb504d60c0ac0812fea7a8f7c82`
- Input: one 13-token English sentence, used only to confirm model loading and finite forward/logit behavior.

## Runtime result

The actual two-model criterion execution completed successfully:

```json
{
  "status": "success",
  "criterion": -0.0776946097612381,
  "sampling_model": "EleutherAI/gpt-j-6B",
  "scoring_model": "EleutherAI/gpt-neo-2.7B",
  "quantization": "nf4_4bit_double_quant",
  "tokenizer_ids_equal": true,
  "logits_ref_shape": [1, 12, 50400],
  "logits_score_shape": [1, 12, 50257],
  "free_mib": 2445.1,
  "total_mib": 11876.2,
  "elapsed_seconds": 42.41
}
```

The paired loading probe previously measured 5,209.5 MiB PyTorch allocation and 5,782 MiB reservation after both models were resident. The separate criterion run retained 2,445.1 MiB free according to CUDA's memory report. GPU usage returned to ordinary desktop memory after the process exited.

### Tokenizer and vocabulary compatibility

The tested sentence produced identical token IDs in both tokenizers. GPT-J's logits have 50,400 columns while GPT-Neo's have 50,257. The upstream analytical Fast-DetectGPT implementation explicitly handles this by truncating both to the smaller vocabulary dimension before its softmax/log-probability calculation. This spike uses the same behavior.

## Verdict: PARTIAL

### What worked

- GPT-J 6B plus GPT-Neo 2.7B fit simultaneously in the RTX 3080 when both use NF4 4-bit quantization.
- Both models returned finite logits.
- Tokenizer IDs matched for the exercised input.
- The upstream-equivalent analytical criterion returned a numeric result.

### What this does **not** establish

- It does not show that the quantized pair improves detector accuracy.
- It does not validate the upstream native GPT-J/Neo calibration; quantization changes logits.
- It does not establish a production throughput estimate—first model loading is expensive and the criterion was tested on only one short input.
- It does not validate tokenizer identity over the full study corpus; that must be enforced per sample or as a corpus preflight check.

## Recommendation for the benchmark

Add this as a separately named experimental configuration:

`gptj_6b_nf4__neo_2p7b_nf4__analytic`

Use raw criteria. Select any threshold or score calibration on the development split only, then compare against the native `neo_2p7b__neo_2p7b__analytic` baseline on the same locked held-out corpus. Report accuracy, low-FPR behavior, latency, memory, failures, and tokenizer-compatibility failures separately. Never label it the upstream native GPT-J/Neo configuration.

## Evidence

- Raw runtime log: `artifacts/quantized-criterion.log`
- SHA-256: `d98fd2326bb01aef97f033ec21afc519884389dc8bca0d0f38d568fa92623bc6`
- Upstream paper: [Fast-DetectGPT, arXiv:2310.05130v3](https://arxiv.org/abs/2310.05130v3)
- Upstream code pinned at [commit `971b052`](https://github.com/baoguangsheng/fast-detect-gpt/tree/971b05202bac2bb504d60c0ac0812fea7a8f7c82)
