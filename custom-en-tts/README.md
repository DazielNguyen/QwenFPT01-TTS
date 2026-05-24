# QwenFPT01 — Custom English TTS

Custom English TTS system trained from scratch on LibriSpeech train-clean-100 (100h),
using the Qwen3-TTS architecture (0.6B backbone + Qwen3-TTS-Tokenizer-12Hz).

**Status:** 🚧 Under development — see [training plan](../plans/qwen3-tts-en-custom-voice/plan.md)

---

## Quick Start

```bash
# Install (Phase 7)
pip install qwenfpt-tts

# Synthesize speech
tts --voice ryan --text "Hello world" --output out.wav

# Run Gradio demo
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN qwenfptfpt-tts
# Open http://localhost:7860
```

---

## Voices

| Voice | Gender | Speaker ID |
|-------|--------|-----------|
| Ryan | Male, adult | LibriSpeech 1284 |
| Aiden | Male, young | LibriSpeech 3575 |
| Emma | Female, adult | LibriSpeech 2961 |
| Sophia | Female, clear | LibriSpeech 1221 |
| Oliver | Male, deep | LibriSpeech 4992 |
| Isabella | Female, bright | LibriSpeech 5142 |

---

## Architecture

- **Backbone**: 0.6B Transformer (GQA, RoPE/YaRN, SwiGLU, RMSNorm)
- **Audio tokenizer**: Qwen3-TTS-Tokenizer-12Hz (12.5 fps, 16 codebooks × 2048)
- **Training data**: LibriSpeech train-clean-100 (100h) — CC BY 4.0
- **Input format**: ChatML sequences of interleaved text and audio tokens

## Training Pipeline

```
S1 General (90h) → S2 HQ-CPT (40h Tier-1) → S3 Long-Context (8192 ctx)
    → DPO → GSPO → Speaker SFT × 6 voices
```

See [phase plans](../plans/qwen3-tts-en-custom-voice/) for full details.

---

## Reproduction

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Download and preprocess data (Phase 2)
python scripts/sprint_01/download_librispeech.py --output data/raw/
python scripts/sprint_01/tier_filter.py --input data/raw/ --output data/manifests/
python scripts/sprint_01/tokenize_dataset.py --manifest data/manifests/tier1.jsonl

# 3. Train (Phase 4–6) — requires GPU and approved budget (see D2)
python scripts/train_s1.py --config configs/training/s1_general.yaml --deepspeed
python scripts/train_s2.py --config configs/training/s2_hq_cpt.yaml --deepspeed
python scripts/train_s3.py --config configs/training/s3_long_context.yaml --deepspeed
python scripts/train_dpo.py --config configs/training/dpo.yaml
python scripts/train_gspo.py --config configs/training/gspo.yaml
```

---

## License

Training data: LibriSpeech CC BY 4.0. Model weights: see decision D5 in the training plan.
Tokenizer: Qwen3-TTS-Tokenizer-12Hz (license verified — D1 resolved).

**Acceptable use**: Research and personal use. Not intended for impersonation of real persons
or non-consenting voice synthesis.

---

## HuggingFace Hub

Model checkpoints and evaluation results: *link TBD — published in Phase 7*
