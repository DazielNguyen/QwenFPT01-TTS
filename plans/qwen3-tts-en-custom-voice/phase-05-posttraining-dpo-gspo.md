# Phase 5: Post-training — DPO and GSPO

## Overview

Improve speech quality and naturalness beyond the S3 Base Model by building a preference dataset and
running DPO, then applying GSPO with a composite reward signal. Each stage passes a gate before the
next begins. Corresponds to Sprint 8 and Sprint 9.

---

## Sprint 8 — Post-training: DPO (2 weeks)

### Tasks

- [ ] **VRAM profiling before DPO (new — red-team finding):** load S3 Base Model + TRL DPO setup (policy + reference model) on target GPU; measure peak VRAM. If > 70 GB on 2×A100 80GB: apply LoRA to the reference model; if > 80 GB: reduce batch size and increase gradient accumulation. Document configuration in `outputs/evals/dpo_vram_profile.json`
- [ ] Generate ~2000 audio samples from S3 Base Model on diverse prompt set (~500 unique prompts × 4 samples via different seeds/temperatures); save to `outputs/dpo_audio/`
- [ ] Auto-score each sample: Whisper-small WER + UTMOS + **MFA forced-aligner phoneme alignment score (new — red-team finding)** as 3rd scoring criterion to catch model-distribution bias; select (chosen, rejected) pairs where score gap ≥ minimum threshold; write to `data/dpo_pairs/pairs.jsonl`
- [ ] **Human spot-check ≥ 25% of pairs (new — red-team finding, increased from 10%):** sample stratified across UTMOS score ranges (low/mid/high) to detect systematic scoring artifacts; reject any pair where audio difference is imperceptible to the rater
- [ ] Confirm ≥ 1500 valid pairs after filtering; if fewer: lower score gap threshold and re-filter, or generate 500 more samples
- [ ] Launch DPO training from S3 checkpoint using TRL; config `dpo.yaml` (LR=1e-6, beta=0.1, 1–2 epochs); monitor KL divergence from reference — abort if KL > 5.0 (collapse signal)
- [ ] Evaluate DPO checkpoint: WER + UTMOS on 100 test sentences vs S3 baseline; save to `outputs/evals/dpo_eval.json`
- [ ] **Gate G4:** DPO preference accuracy > 60% on held-out val split. If gate fails: increase human spot-check to 50% of pairs, re-filter, retry DPO training once.
- [ ] Promote best DPO checkpoint to `checkpoints/dpo/final.pt`

### Targets (after DPO)

| Metric | S3 Base | After DPO |
|--------|---------|-----------|
| UTMOS | > 3.5 | > 3.65 |
| WER | < 7% | < 6% |
| Prefer DPO over S3 (human) | — | > 60% |

### DoD

- `data/dpo_pairs/pairs.jsonl` with ≥ 1500 pairs, MFA + UTMOS + WER scores per pair
- ≥ 25% of pairs human-verified, stratified by UTMOS range
- `checkpoints/dpo/final.pt` exists
- `outputs/evals/dpo_eval.json` with WER, UTMOS, preference accuracy vs S3 baseline
- Gate G4 verdict documented

---

## Sprint 9 — Post-training: GSPO (2 weeks)

### Tasks

- [ ] **VRAM and throughput profiling for GSPO reward stack (new — red-team finding, CRITICAL):**
  Profile the combined VRAM footprint of policy model (0.6B) + Whisper-tiny + UTMOS + WeSpeaker SIM on target GPU config (2×A100 80GB) with G=8 samples per step. If VRAM > 140 GB total (across 2 GPUs): offload Whisper + UTMOS to CPU async process; implement reward server pattern where reward models run on a separate GPU or CPU thread decoupled from the policy update loop. Document the chosen architecture in `outputs/evals/gspo_reward_profile.json`.
- [ ] Implement `src/training/gspo/reward_fn.py`: composite reward `r = 0.4×(1-WER) + 0.3×UTMOS + 0.2×SIM + 0.1×length_OK`; use Whisper-tiny for WER (speed vs accuracy tradeoff documented); batch all G=8 reward calls in a single Whisper forward pass per step; decouple reward computation from policy gradient update
- [ ] Implement `src/training/gspo/gspo_trainer.py`: group relative policy gradient with G=8 samples, normalize rewards within group, entropy bonus in reward to prevent collapse
- [ ] Launch GSPO training from `checkpoints/dpo/final.pt` using `gspo.yaml`; monitor reward curve and output diversity (unique n-gram rate) every 500 steps; if diversity drops below 0.3: double entropy bonus weight
- [ ] Edge-case stress test at 5K and 10K steps: generate speech for 20 challenging prompts (numbers, names, abbreviations, long sentences); count failure rate
- [ ] Evaluate GSPO checkpoint: full benchmark on 100 test sentences; save to `outputs/evals/gspo_eval.json`
- [ ] **Ablation note:** during Sprint 11, the ablation table must include a row for "S3 Base → GSPO without DPO" to verify that DPO+GSPO outperforms GSPO alone. If it does not, document and recommend dropping DPO from the final pipeline for future training runs.
- [ ] **Gate G5:** WER ≥ 10% relative improvement vs S3; UTMOS ≥ +0.15 vs S3. If gate fails: adjust reward weights (increase UTMOS weight to 0.4) and retry for 5K more steps once.
- [ ] Promote best GSPO checkpoint to `checkpoints/gspo/final.pt` — this is the Post-trained Base for Speaker SFT

### Targets (after GSPO)

| Metric | After DPO | After GSPO |
|--------|-----------|------------|
| WER | < 6% | < 4% |
| UTMOS | > 3.65 | > 3.8 |
| Edge case robustness | Moderate | High |

### DoD

- `outputs/evals/gspo_reward_profile.json` exists with VRAM/throughput measurements
- Reward curves converge (not plateau at max within first 500 steps)
- Output diversity (unique 5-gram rate) ≥ 0.3 at end of training
- `checkpoints/gspo/final.pt` exists
- `outputs/evals/gspo_eval.json` with WER, UTMOS vs S3 and DPO baselines
- Gate G5 verdict documented

---

## Risks

| Risk | Mitigation |
|------|------------|
| DPO KL divergence explodes | Monitor KL per step; set KL budget constraint in config; start beta=0.1, increase gradually only |
| GSPO VRAM overflow from 3 concurrent reward models | VRAM profile in pre-task; async CPU offload for Whisper/UTMOS if needed |
| GSPO reward hacking (Whisper-gameable outputs) | UTMOS + SIM together ≥ 50% of reward weight; inspect samples every 200 steps |
| DPO + GSPO interact negatively | Ablation in Sprint 11 tests this explicitly; decision rule: drop DPO if combined pipeline loses to direct GSPO |
| 2000 sample generation is slow | Batch inference with beam=1; parallelize across GPUs if available |

## Estimated Duration

Sprint 8: 2 weeks · Sprint 9: 2 weeks → **~4 weeks total**

## Dependencies

- Phase 4 complete (`checkpoints/s3_long_context/final.pt` exists, Gate G3 passed)
- TRL ≥ 0.9 installed (pinned in `requirements.lock`)
- UTMOS, WeSpeaker SIM, and MFA forced-aligner models downloaded and verified
- `src/inference/generator.py` functional for batched sample generation (pull forward from Phase 7 if needed)
- 1× A100 40 GB minimum; 2× A100 80 GB recommended for GSPO reward batching
