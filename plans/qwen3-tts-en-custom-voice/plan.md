# Plan: QwenFPT01 — Custom English TTS (Qwen3-TTS Architecture)
Status: 🟡 In Progress
Date: 2026-05-25
Mode: Hard

## Overview

Build a custom English TTS system by training a 0.6B language model backbone from scratch on
LibriSpeech train-clean-100 (100h), reusing the frozen Qwen3-TTS-Tokenizer-12Hz for audio
tokenization and waveform decoding. The 6-stage training pipeline (S1 General → S2 HQ-CPT →
S3 Long-Context → DPO → GSPO → Speaker SFT) produces 6 distinct English voices delivered via a
Gradio demo and pip-installable package.

Reference spec: `Qwen3-TTS-CustomVoice-English-Agileplan.md`

---

## ⚠️ Pre-Implementation Decisions (must resolve before Sprint 0)

These are unresolved blockers. Proceeding without them will cause mid-sprint rework.

| # | Decision | Blocks |
|---|----------|--------|
| D1 | ✅ **RESOLVED** — Qwen3-TTS-Tokenizer-12Hz weights are public, license permits derivative research. Pin exact checkpoint commit hash in `requirements.txt` before Phase 2. | Everything |
| D2 | ⏳ **PENDING** — Budget not yet approved (~$1,700–2,200). Must resolve before Phase 4 (pre-training). Phase 1–3 do not require large GPU; Phase 4 launch is blocked until D2 is approved. | Phase 4–5 |
| D3 | Define Tier-1 relaxation rule: if actual Tier-1 yield after filtering < 30h, lower SNR threshold to 30 dB and WER to 3%, document the change. Decide before Sprint 1 begins. | Phase 2, 5, 6 |
| D4 | ⏳ **PENDING** — RoPE extension method to be researched from Qwen3-TTS paper (arXiv:2601.15621 Section 3) before Phase 3 implementation. YaRN is the current recommendation but defer to paper's method. Phase 3 backbone implementation is blocked until D4 is resolved. | Phase 3, 4 |
| D5 | Decide outgoing license for model weights and Gradio demo (D1 confirms tokenizer license is OK; D5 is unblocked — can decide now or at Phase 7). | Phase 7 |

---

## Phases

- [x] **Phase 1** — Project Folder Setup: create entire `custom-en-tts/` directory tree, stubs, `.gitignore`, placeholder READMEs
- [ ] **Phase 2** — Bootstrap and Data Pipeline: Sprint 0–2 (env, tokenizer verify, download → tier → tokenize → DataLoader)
- [ ] **Phase 3** — Architecture and Training Infrastructure: Sprint 3–4 (0.6B backbone, DeepSpeed ZeRO-2, stage checkpoints)
- [ ] **Phase 4** — Pre-training Three Stages: Sprint 5–7 (S1 General, S2 HQ-CPT, S3 Long-Context)
- [ ] **Phase 5** — Post-training DPO and GSPO: Sprint 8–9 (preference pairs, DPO, GSPO reward RL)
- [ ] **Phase 6** — Speaker SFT and Evaluation: Sprint 10–11 (6 voices, WER/UTMOS/SIM benchmarks, ablation, MOS)
- [ ] **Phase 7** — Demo and Release: Sprint 12 (Gradio, pip, CLI, streaming, Docker, HF model card)

---

## Milestones

| Milestone | Description | End of Phase |
|-----------|-------------|--------------|
| M0 | D1–D5 decisions resolved, tokenizer round-trip verified | Pre-Phase 1 |
| M1 | Verified data pipeline: manifests built, tokenization locked, DataLoader green | Phase 2 |
| M2 | Backbone + training loop validated on tiny config; smoke test passes on 2×GPU | Phase 3 |
| M3 | S1 General checkpoint: model produces intelligible speech on held-out text | Phase 4 |
| M4 | S3 Base Model complete: WER < 7% short, stable up to 30s | Phase 4 |
| M5 | GSPO checkpoint: measurable improvement over S3 on WER + UTMOS | Phase 5 |
| M6 | 6 speaker voices: SIM ≥ 0.85, WER ≤ 5%, full benchmark report | Phase 6 |
| M7 | Demo released: Gradio live, pip install works, HF model card published | Phase 7 |

---

## Infrastructure Requirements

| Stage | Minimum | Recommended |
|-------|---------|-------------|
| Data preprocessing (Phase 2) | CPU 32 GB RAM | CPU 64 GB RAM |
| Smoke tests (Phase 3) | 1× A100 40 GB | 1× A100 40 GB |
| S1/S2/S3 Pre-training (Phase 4) | 1× A100 40 GB | 2× A100 80 GB |
| DPO + GSPO (Phase 5) | 1× A100 40 GB | 2× A100 80 GB |
| Speaker SFT ×6 (Phase 6) | 1× A100 40 GB | 1× A100 80 GB |
| Demo serving (Phase 7) | 1× T4 16 GB | 1× A10G 24 GB |

---

## Storage Estimate (revised)

| Item | Size |
|------|------|
| data/raw/ (LibriSpeech FLAC) | ~6 GB |
| data/processed/ (24kHz WAV) | ~6 GB |
| data/tokenized/ (.npy) | ~280 MB |
| data/long_form/ | ~300 MB |
| data/dpo_pairs/ + outputs/dpo_audio/ | ~4 GB |
| checkpoints/s1_general/ | ~50 GB |
| checkpoints/s2_hq_cpt/ | ~50 GB |
| checkpoints/s3_long_context/ | ~50 GB |
| checkpoints/dpo/ | ~50 GB |
| checkpoints/gspo/ | ~50 GB |
| checkpoints/speaker_sft/ (×6 voices) | ~36–60 GB |
| outputs/samples/ + outputs/evals/ | ~5 GB |
| **Total** | **~360–390 GB** |
| **Recommended disk budget (with 30% buffer)** | **500 GB** |

Checkpoint retention policy: keep only last 2 checkpoints per stage + final. Prune intermediate checkpoints after each stage gate is passed.

---

## Cost Estimate

| Phase | GPU Hours (est.) | Cost @ $3.50/hr A100 80GB |
|-------|-----------------|---------------------------|
| Phase 2 — data pipeline | 8h CPU | ~$0 GPU |
| Phase 3 — architecture smoke tests | 4h A100 | ~$14 |
| Phase 4 — S1 General | 120h A100 | ~$420 |
| Phase 4 — S2 HQ-CPT | 60h A100 | ~$210 |
| Phase 4 — S3 Long-Context | 40h A100 | ~$140 |
| Phase 5 — DPO + GSPO | 80h A100 | ~$280 |
| Phase 6 — Speaker SFT ×6 + eval | 60h A100 | ~$210 |
| Phase 7 — demo + eval | 8h A100 | ~$28 |
| **Total** | **~370h GPU** | **~$1,302** |

---

## Stage Transition Gates (rollback policy)

Each stage checkpoint must pass a quality gate before the next stage begins.
For each gate: one retry from the previous checkpoint is allowed; if gate still not met after retry, proceed with best checkpoint and document the delta in the eval JSON.

| Gate | From | To | Accept Criteria | Retry Budget |
|------|------|----|-----------------|--------------|
| G1 | S1 | S2 | Intelligible speech on 80% of 50 held-out sentences | 1 retry (reduce LR, add warmup) |
| G2 | S2 | S3 | WER ≥ 15% relative improvement vs S1; UTMOS ≥ +0.1 | 1 retry (mix S1 data 20%) |
| G3 | S3 | DPO | WER < 12% short; stable generation to 30s | 1 retry (reduce seq_len, extend training steps) |
| G4 | DPO | GSPO | DPO preference accuracy > 60% on val split | 1 retry (increase human spot-check coverage) |
| G5 | GSPO | SFT | WER ≥ 10% relative improvement vs S3; UTMOS ≥ +0.15 | 1 retry (adjust reward weights) |
| G6 | SFT | Eval | Per-voice SIM ≥ 0.82 on 3 of 6 voices | 1 retry per failing voice (more data or unfreeze +2 layers) |

---

## Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Tokenizer weights gated or license-restricted | Medium | **CRITICAL** | Verify before Phase 2; fallback: EnCodec 24kHz |
| W&B / HF tokens leaked into Git or Docker image | Medium | High | All secrets via env vars only; CI secret scan in Phase 1 |
| Upsample 16kHz→24kHz causes codebook mismatch | Medium | High | Round-trip PESQ/STOI validation on 20 clips in Phase 2 |
| Tier-1 yield < 30h after strict SNR/WER filter | Medium | High | Relaxation rule defined in D3; gate checked at end of Sprint 1 |
| GSPO VRAM overflow (policy + Whisper + UTMOS + SIM) | High | High | Profile combined VRAM before Sprint 9; async CPU offload for reward models |
| RoPE naive extension breaks attention at 4096–8192 | High | High | Use YaRN interpolation (decision D4); add perplexity regression test |
| Long-form clip concatenation produces prosody discontinuities | Medium | Medium | Chapter-aware concatenation (same SPEAKER-CHAPTER); add silence token at boundaries |
| DPO scoring systematic bias (model scores its own outputs) | Medium | Medium | Add MFA forced-aligner score as 3rd criterion; increase human spot-check to 25% |
| Speaker SFT storage exceeds 500 GB budget | Medium | Medium | Run storage_report.sh before Phase 6; prune intermediate checkpoints aggressively |
| S2 CPT causes catastrophic forgetting | Medium | High | Mix 20% S1 data into S2; gate G2 catches regression |
| GSPO reward collapse (output distribution narrows) | Medium | High | Entropy bonus in reward; monitor output diversity every 500 steps |
| DPO + GSPO interact negatively (DPO shifts distribution GSPO must undo) | Medium | Medium | Ablation: compare S3→GSPO vs S3→DPO→GSPO in Sprint 11; drop DPO if combined pipeline does not outperform direct GSPO |
| MOS evaluation has too few raters for publishable claim | Low | Medium | Label result as "pilot MOS (n=5)"; target 20 raters for model card; use stratified sampling |
| nemo_text_processing heavyweight dep leaks into pip package | Low | Medium | Use NeMo offline in Phase 2 only; inference package uses lightweight num2words + regex |
| LibriSpeech 100h insufficient for WER < 5% | High | Medium | Accept WER < 8% as success; do not compare directly with Qwen3-TTS trained on 5M hours |
