# Detector comparison — exploratory run (2026-08-04)

## Scope

This run used six fixed, labelled English samples:

- **Human-authored:** two public-domain Project Gutenberg passages (`A Scandal in Bohemia` and `Walden`), 360 words each.
- **AI-generated:** two passages written by Hermit for this benchmark, labelled `ai_generated` at creation.
- **Mixed:** each public-domain passage joined unchanged with one of the AI passages.

All samples and SHA-256 hashes are recorded in `../../data/samples.manifest.json`. No text was rewritten after a detector result.

## Fast-DetectGPT result

- **Implementation:** upstream `baoguangsheng/fast-detect-gpt`, shallow clone at test time.
- **Host:** NESTLECRUNCH (`tmac@192.168.254.84`)
- **GPU:** NVIDIA RTX 3080, 12 GB VRAM
- **Runtime configuration:** Python 3.11; PyTorch 2.13.0+cu130; Transformers 4.28.1; Datasets 2.12.0; NumPy 1.26.4; PyArrow 12.0.1.
- **Model configuration:** upstream-supported shared `gpt-neo-2.7B` sampling/scoring model on CUDA.
- **Raw log:** `raw.log`; normalized raw records: `results.json`.

| Sample | Known label | Criterion | Reported AI probability | Tokens |
|---|---|---:|---:|---:|
| `human-sherlock-001` | human-authored | 4.1953 | 99.98% | 434 |
| `human-walden-001` | human-authored | 1.4902 | 69.17% | 434 |
| `ai-story-001` | AI-generated | 1.3711 | 64.40% | 229 |
| `ai-explainer-001` | AI-generated | -1.2734 | 19.30% | 195 |
| `mixed-sherlock-story-001` | mixed | 3.6816 | 99.87% | 666 |
| `mixed-walden-explainer-001` | mixed | 0.3721 | 31.17% | 632 |

For an explicitly exploratory 50% threshold applied only to the four pure samples, the result is **TP=1, FN=1, FP=2, TN=0**. This is not an accuracy estimate; it is a very small reproducible counterexample set showing that this model/configuration can disagree substantially with known provenance.

## ZeroGPT status

- **Target:** `https://www.zerogpt.com/` (ZeroGPT, not GPTZero).
- **Action:** one supervised calibration submission using `ai-story-001` only.
- **Outcome:** the public page accepted the text but returned no detector result or visible error after `Detect Text` was clicked. The page loaded reCAPTCHA resources.
- **Decision:** stopped after the one calibration attempt. No CAPTCHA was clicked, no anti-automation behavior was used, and the remaining five samples were not submitted.

ZeroGPT is therefore recorded as **blocked/no-result** for this run. There is no valid Fast-DetectGPT-versus-ZeroGPT numeric comparison yet.

## Limitations

1. Fast-DetectGPT's reported value is a model/configuration-specific calibration, not proof of authorship.
2. The test is small, English-only, and includes historical public-domain prose that may differ from current everyday writing.
3. Detector behavior may change with provider/model updates and text length.
4. A proper multi-provider comparison requires normal authorized access or a documented API for each provider, while preserving the same frozen corpus.
