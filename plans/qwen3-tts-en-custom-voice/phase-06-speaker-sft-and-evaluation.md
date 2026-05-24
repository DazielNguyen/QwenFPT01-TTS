# Phase 6: Speaker SFT and Evaluation

## Overview

Fine-tune the GSPO checkpoint to produce 6 distinct, recognizable English voices by training with
frozen backbone (only speaker embedding + top 4 Transformer layers updated), then run a
comprehensive benchmark suite and ablation study. Corresponds to Sprint 10 and Sprint 11.

---

## Sprint 10 — Speaker SFT: 6 Custom Voices (3 weeks)

### Tasks

- [ ] **Storage check before starting (critical):** run `scripts/storage_report.sh`; confirm ≥ 80 GB free disk space after all Phase 4–5 checkpoints; prune intermediate checkpoints from Phase 4 (keep only `final.pt` per stage); document available space in `outputs/evals/pre_sft_storage.json`
- [ ] Curate per-speaker reference audio: for each of the 6 personas (Ryan/Speaker 1284, Aiden/3575, Emma/2961, Sophia/1221, Oliver/4992, Isabella/5142), select Tier-1 clips; verify ≥ 2h per speaker from `outputs/evals/tier1_speaker_hours.json` (computed in Phase 2); place in `data/speaker_sft/{name}/`
- [ ] Configure `configs/training/speaker_sft.yaml`: freeze backbone layers 0–23, keep trainable: speaker embedding table + Transformer layers 24–27 (top 4); LR=5e-6, 3–5 epochs; per-speaker early stopping on val SIM
- [ ] Fine-tune each voice sequentially (or in parallel if ≥ 2 GPUs): resume from `checkpoints/gspo/final.pt` for each voice; save checkpoints to `checkpoints/speaker_sft/{name}/`
- [ ] After each voice: evaluate WER + UTMOS + SIM (using ECAPA-TDNN speaker verification, not raw cosine similarity on speaker embeddings); log to `outputs/evals/speaker_sft_{name}.json`
- [ ] If a voice fails SIM < 0.80 after first fine-tune: retry with 5h reference audio (if available) or unfreeze 6 layers instead of 4; document retry in eval JSON
- [ ] **Gate G6:** ≥ 3 of 6 voices achieve SIM ≥ 0.82 on first attempt. If gate fails: revise speaker embedding initialization or increase reference audio; one retry per failing voice.
- [ ] Cross-voice test: synthesize the same 10 sentences with all 6 voices; confirm distinct perceptual difference between voices (human spot-check)

### Voice Personas

| Voice ID | Gender | LibriSpeech Speaker | Expected Tier-1 Hours |
|----------|--------|--------------------|-----------------------|
| Ryan | Male, adult | Speaker 1284 | ~5h (verify in Phase 2 output) |
| Aiden | Male, young | Speaker 3575 | ~4h |
| Emma | Female, adult | Speaker 2961 | ~5h |
| Sophia | Female, clear | Speaker 1221 | ~4h |
| Oliver | Male, deep | Speaker 4992 | ~3h |
| Isabella | Female, bright | Speaker 5142 | ~3h |

### DoD

- `checkpoints/speaker_sft/{name}/final.pt` exists for all 6 voices
- All 6 voices evaluated: `outputs/evals/speaker_sft_{name}.json` for each
- Cross-voice test confirms perceptual distinctiveness (human spot-check documented)
- Gate G6 verdict documented

---

## Sprint 11 — Evaluation and Benchmarks (2 weeks)

### Tasks

- [ ] **Full benchmark suite** (`src/evaluation/benchmark.py`) on 100 test sentences × 6 voices:
  - WER: Whisper-large transcription on generated audio
  - UTMOS: MOS proxy score
  - SIM: ECAPA-TDNN speaker similarity against held-out reference audio
  - Save to `outputs/evals/full_benchmark.json`
- [ ] Long-form stability test: generate 30–60s utterances on 20 prompts per voice; measure WER + check for repetition/truncation
- [ ] **Ablation study** on 50 shared test sentences across checkpoints: S1 → S2 → S3 → DPO → GSPO → Speaker SFT (per voice). **Critically: include row "S3 Base → GSPO (no DPO)"** to test DPO necessity. Save to `outputs/evals/ablation_table.json`
- [ ] **Human MOS evaluation:**
  - Target: 20+ raters for publishable claim; minimum 5 raters for pilot result
  - 20 test sentences × 6 voices = 120 samples per rater
  - Rater criteria: native English speakers, headphone listening, ITU-T P.808 protocol
  - Label result clearly: "pilot MOS (n=5)" if < 20 raters; "MOS (n=20)" if ≥ 20 raters
  - Save to `outputs/evals/mos_results.json` with per-rater, per-sample, per-voice breakdown
- [ ] Inference speed benchmark: measure RTF (real-time factor) and first-packet latency on A10G (or T4 fallback); save to `outputs/evals/inference_speed.json`
- [ ] Compile final benchmark report: summary table linking all eval JSON files; identify which stages contributed the most improvement from the ablation table

### Final Quality Targets

| Metric | Target | Note |
|--------|--------|------|
| WER (test-clean, short) | < 5% | LibriSpeech Whisper-large |
| UTMOS | > 3.8 | Qwen3-TTS trained on 5M hours achieves ~4.16 — we have 100h |
| SIM | > 0.82 per speaker | ECAPA-TDNN verification |
| First-packet latency | < 300ms | Single GPU |

### DoD

- `outputs/evals/full_benchmark.json` with per-voice per-metric scores for all 6 voices
- `outputs/evals/ablation_table.json` with all 6 stage rows + "S3→GSPO no DPO" row
- `outputs/evals/mos_results.json` with rater count documented
- Inference speed logged in `outputs/evals/inference_speed.json`
- Final benchmark report compiled (can be a Markdown summary in `outputs/evals/README.md`)

---

## Risks

| Risk | Mitigation |
|------|------------|
| Storage exhausted before SFT starts | `storage_report.sh` check is first task; prune Phase 4 intermediate checkpoints |
| A voice speaker has < 2h Tier-1 (discovered late) | Phase 2 gate already verified this; if still occurs, select replacement speaker from manifest |
| SIM metric dominated by pitch not timbre | Use ECAPA-TDNN speaker verification model, not raw cosine on embeddings |
| MOS blocked by rater availability | Schedule MOS session at Sprint 10 kickoff; prepare 120 audio samples in advance |
| Ablation shows DPO regresses GSPO | Document finding, recommend dropping DPO for future runs; does not block release |

## Estimated Duration

Sprint 10: 3 weeks · Sprint 11: 2 weeks → **~5 weeks total**

## Dependencies

- Phase 5 complete (`checkpoints/gspo/final.pt` exists, Gate G5 passed)
- `outputs/evals/tier1_speaker_hours.json` from Phase 2 confirming per-speaker Tier-1 hours
- ECAPA-TDNN speaker verification model downloaded and verified
- UTMOS model downloaded and verified
- MFA forced-aligner installed (from Phase 5 DPO pair building)
- ≥ 80 GB free disk confirmed before Sprint 10 starts
- 1× A100 40 GB minimum; 2× A100 80 GB for parallel voice SFT
