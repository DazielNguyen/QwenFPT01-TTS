# Phase 4: Pre-training — Three Stages

## Overview

Train the 0.6B backbone through three sequential pre-training stages producing a checkpoint after
each one. Each stage must pass a quality gate (see `plan.md` Stage Transition Gates G1–G3) before
the next stage begins. Corresponds to Sprint 5, 6, and 7.

---

## Sprint 5 — Pre-training S1: General Stage (2 weeks)

### Tasks

- [ ] Verify storage budget before launch: run `scripts/storage_report.sh`; confirm ≥ 200 GB free (S1 + S2 + S3 checkpoints need ~150 GB total)
- [ ] Launch S1 training: `python scripts/train_s1.py --config configs/training/s1_general.yaml --deepspeed` on Tier-1 + Tier-2 data (~90h), targeting 50K–80K steps
- [ ] Monitor every 500 steps: training loss curve, gradient norm (abort and reduce LR if grad norm > 5.0 for 1000 consecutive steps)
- [ ] Generate 20 test sentences after every 5K steps; save to `outputs/samples/s1_step{N}/`; human spot-check for basic intelligibility
- [ ] At training end: compute WER on held-out test set using Whisper-large; save to `outputs/evals/s1_eval.json`
- [ ] **Gate G1:** intelligible speech on ≥ 80% of 50 held-out sentences (human spot-check). If gate fails: retry once from step 30K with LR × 0.5. If still fails: document and proceed with best checkpoint.
- [ ] Promote best S1 checkpoint to `checkpoints/s1_general/final.pt`; log W&B artifact

### Targets (S1)

| Metric | Target |
|--------|--------|
| WER | < 20% |
| Audio intelligible | ≥ 80% of held-out sentences |
| Training loss | Converged, not diverging |

### DoD

- `checkpoints/s1_general/final.pt` exists with `trainer_state.json` confirming completed steps
- `outputs/evals/s1_eval.json` with WER, intelligibility rate
- Gate G1 verdict documented (pass / pass-after-retry / proceed-with-delta)

---

## Sprint 6 — Pre-training S2: High-Quality CPT (2 weeks)

### Tasks

- [ ] Resume from `checkpoints/s1_general/final.pt`; switch to Tier-1 only data (~40h) with config `s2_hq_cpt.yaml` (LR=2e-5, warmup=500)
- [ ] Train ~20K–30K steps; **mix 20% S1 (Tier-1+Tier-2) data into S2 batches** to prevent catastrophic forgetting
- [ ] Generate 50 test sentences every 5K steps; count repetitions and word skips vs S1 baseline
- [ ] At training end: compute WER + UTMOS on 100 test sentences; save to `outputs/evals/s2_eval.json`; compare against S1 baseline
- [ ] **Gate G2:** WER ≥ 15% relative improvement vs S1; UTMOS ≥ +0.1 vs S1. If gate fails: retry once from S1 checkpoint with 20% S1 data mix increased to 40%. If still fails: proceed with best checkpoint, document delta.
- [ ] Stress test: generate 10 paragraphs (200+ words each), count hallucination events (repetition, babbling, word skipping)
- [ ] Promote best S2 checkpoint to `checkpoints/s2_hq_cpt/final.pt`

### Targets (S2)

| Metric | S1 Baseline | S2 Target |
|--------|-------------|-----------|
| WER | < 20% | < 10% |
| UTMOS | — | > 3.3 |
| Repetition rate | ~20% of outputs | < 5% |

### DoD

- `checkpoints/s2_hq_cpt/final.pt` exists
- `outputs/evals/s2_eval.json` with WER, UTMOS, hallucination count (S1 vs S2 comparison)
- Gate G2 verdict documented

---

## Sprint 7 — Pre-training S3: Long-Context Stage (2 weeks)

### Tasks

- [ ] Verify YaRN RoPE scaling factor in `configs/training/s3_long_context.yaml` — this is the same implementation from Phase 3 Sprint 3; set `max_seq_len=8192` in config; confirm no code changes needed (only config change)
- [ ] Build long-form data mix: `data/long_form/` sequences (2K–4K frames, same-chapter concatenation from Phase 2) + Tier-1+2 standard sequences in a 60/40 ratio
- [ ] Enable gradient checkpointing for S3 (seq_len 8192 doubles activation memory); if OOM on 2×A100 80GB: reduce per-device batch size and increase gradient accumulation steps proportionally
- [ ] Resume from `checkpoints/s2_hq_cpt/final.pt`; train with `s3_long_context.yaml` (LR=1e-5, ~20K steps)
- [ ] **Perplexity regression test (new — red-team finding):** at end of S3, verify that perplexity on short sequences (< 4096 tokens) does not increase > 10% relative vs S2 checkpoint; if regression detected, reduce YaRN scaling factor and continue training for 5K more steps
- [ ] Long-form eval: generate 30–60s utterances on 20 test prompts; measure WER + subjective prosody quality; save to `outputs/evals/s3_eval.json`
- [ ] **Gate G3:** WER < 12% on short utterances; stable generation to 30s without truncation/babbling. If gate fails: retry once with 80% standard + 20% long-form data ratio. If still fails: document and proceed.
- [ ] Promote best S3 checkpoint to `checkpoints/s3_long_context/final.pt` — this is the Base Model for post-training

### Targets (S3 Base Model)

| Metric | S2 | S3 Target |
|--------|-----|-----------|
| WER (short < 10s) | < 10% | < 7% |
| WER (long > 20s) | — | < 12% |
| Max stable length | ~10s | 30–60s |
| UTMOS | > 3.3 | > 3.5 |

### DoD

- `checkpoints/s3_long_context/final.pt` exists — this is the Base Model
- `outputs/evals/s3_eval.json` with short-form + long-form WER and UTMOS
- Perplexity regression test result logged (≤ 10% regression from S2 on short sequences)
- Gate G3 verdict documented

---

## Risks

| Risk | Mitigation |
|------|------------|
| S1 training diverges (grad norm spike) | Abort if grad norm > 5 for 1000 steps; halve LR, resume from last saved checkpoint |
| S2 catastrophic forgetting despite 20% S1 mix | Gate G2 catches this; retry with 40% S1 mix |
| S3 OOM at seq_len 8192 on 2×A100 80GB | Gradient checkpointing is default for S3; reduce batch size further if needed |
| YaRN scaling degrades short-sequence quality | Perplexity regression test in DoD catches this; adjust scaling factor |
| Cloud GPU unavailability for 5–7 day runs | Reserve GPU instance before Sprint 5 begins; test DeepSpeed launch on instance before committing training budget |

## Estimated Duration

Sprint 5: 2 weeks · Sprint 6: 2 weeks · Sprint 7: 2 weeks → **~6 weeks total** (includes GPU wait time)

## Dependencies

- Phase 3 complete (backbone, training loop, YaRN RoPE, 2×GPU smoke test all verified)
- Phase 2 complete (`data/tokenized/.lock`, `data/long_form/` built, manifests final)
- Decision D2 resolved (cloud GPU provider and budget authorized)
- UTMOS scoring model downloaded and verified before Sprint 6 evaluation
- S1/S2: 2× A100 80GB recommended · S3: 2× A100 80GB required for seq_len 8192
