# Phase 2: Bootstrap and Data Pipeline

## Overview

Deliver a fully verified, reproducible data pipeline from raw LibriSpeech audio through quality-tier
assignment, offline tokenization, long-form sequence construction, manifest building, and a working
DataLoader. Corresponds to Sprint 0, Sprint 1, and Sprint 2.

---

## Sprint 0 — Bootstrap (1 week)

### Tasks

- [ ] Pin all dependencies in `requirements.txt` with exact versions; create conda/venv environment
- [ ] Verify Qwen3-TTS-Tokenizer-12Hz loads and the tokenizer's vocab size matches the expected 2048 tokens per codebook × 16 codebooks (decision D1 must be resolved before this step)
- [ ] **Tokenizer round-trip quality check (new — red-team finding):** encode 20 LibriSpeech FLAC clips (upsampled 16kHz→24kHz) through the tokenizer, decode back to waveform, compute PESQ and STOI against the upsampled original; abort and document if PESQ < 2.5 or STOI < 0.85 — this validates that 16k→24k upsample does not cause codebook mismatch
- [ ] Set up Git repo with `.gitignore` verified; add CI stub that runs `grep -r "WANDB_API_KEY\|HF_TOKEN" scripts/ src/` and fails if any hardcoded secrets are found
- [ ] Configure W&B: project name, API key loaded from `WANDB_API_KEY` env var only — never hardcoded; test live logging with a dummy run
- [ ] Verify all secrets (W&B, HF token, cloud provider keys) are sourced exclusively from env vars or `.env` file listed in `.gitignore`

### DoD

- `tokenizer.encode(clip).shape == (T_frames, 16)` for at least 20 test clips
- PESQ ≥ 2.5 and STOI ≥ 0.85 on all 20 round-trip clips (logged to `outputs/evals/tokenizer_roundtrip.json`)
- W&B live dashboard shows test run
- CI secret-scan check passes (no hardcoded API keys found)
- `pip install -e .` succeeds from the project root

---

## Sprint 1 — Data Pipeline I: Download → Tier Assignment (2 weeks)

### Tasks

- [ ] Download LibriSpeech train-clean-100 (~6.3 GB FLAC), verify MD5 checksum, place in `data/raw/`; write `data/raw/.lock` sentinel after checksum passes — all subsequent scripts abort if this lock is absent
- [ ] Resample pipeline: FLAC → 24kHz WAV using ffmpeg batch script (`scripts/sprint_01/02_resample.py`); write output to `data/processed/`
- [ ] Duration filter: keep clips 3–30 seconds; log rejected count
- [ ] Amplitude normalization: peak normalize to −3 dBFS
- [ ] Text normalization: run `nemo_text_processing` **offline only** on all transcripts; store normalized text in manifest — this tool is NOT a runtime dependency of the pip package
- [ ] ASR alignment check: run Whisper-small (GPU preferred; Whisper-tiny fallback with documented accuracy tradeoff) to compute per-clip WER; assign clips to Tier 1/2/3
- [ ] Quality scoring: compute SNR per clip; combine SNR + WER + duration → Tier 1 (SNR > 35dB, WER < 2%), Tier 2 (SNR 25–35dB, WER 2–8%), Tier 3 (discard)
- [ ] **Per-speaker Tier-1 hour verification (new — red-team finding):** after tier assignment, compute Tier-1 hours per speaker ID and confirm that each of the 6 chosen speaker personas (Speaker 1284, 3575, 2961, 1221, 4992, 5142) has ≥ 2h of Tier-1 audio. If any speaker falls below 2h, select a replacement speaker with ≥ 2h Tier-1 from the manifest. Log results to `outputs/evals/tier1_speaker_hours.json`
- [ ] **Tier-1 yield gate:** if total Tier-1 yield < 30h, apply decision D3 relaxation rule (lower SNR threshold to 30 dB, WER to 3%) and re-run quality scorer; document threshold change in manifest header
- [ ] Build preliminary manifest JSONL with: id, speaker_id, text (normalized), duration_s, tier, snr_db, asr_wer, wav_path

### DoD

- Total Tier-1 yield ≥ 30h (with or without D3 relaxation, logged)
- All 6 speaker personas have ≥ 2h Tier-1 audio confirmed in `outputs/evals/tier1_speaker_hours.json`
- Manifest JSONL exists with tier/wer/snr/duration for every retained clip
- Tier-3 clips are excluded and their count logged

---

## Sprint 2 — Data Pipeline II: Tokenize → DataLoader (2 weeks)

### Tasks

- [ ] **Pre-tokenize ~90h audio (Tier 1 + Tier 2):** run `Qwen3-TTS-Tokenizer-12Hz.encode()` on every retained clip; save `.npy` files to `data/tokenized/` with shape `(T_frames, 16)`
- [ ] Verify tokenization: encode → decode 50 clips, listen-check and compute PESQ; write `data/tokenized/.lock` after verification passes — all training scripts assert this lock exists before reading token files
- [ ] **Chapter-aware long-form sequence construction (new — red-team finding):** concatenate 3–5 clips from the **same speaker AND same chapter** (LibriSpeech filenames encode `SPEAKER-CHAPTER-SEGMENT`); insert a short silence token (3–5 frames of silence codebook tokens) at each clip boundary to prevent prosody discontinuities; save to `data/long_form/`
- [ ] Pre-compute token lengths for all clips; store in manifest for bucketed batching (target: padding < 5% per batch)
- [ ] Build final manifest JSONL with all fields including `tokens_path`, `token_len`, `split`; split 95/4/1 train/val/test with constraint that same speaker does not appear in both train and test
- [ ] Implement `src/data/dataset.py` (reads `.npy` + text from manifest, asserts `.lock` exists) and `src/data/collator.py` (bucketed batching by token length)
- [ ] DataLoader smoke test: iterate 100 batches from `train.jsonl`; verify batch shape `(B, T, 16)`, no OOM on target GPU, padding ratio < 5%
- [ ] Dataset statistics report: duration distribution, speaker distribution, tier breakdown — log to `outputs/evals/dataset_stats.json`

### DoD

- `data/tokenized/.lock` exists
- Token count in `data/tokenized/` matches retained clip count in `data/manifests/train.jsonl` exactly
- DataLoader iterates 100 batches without error, correct shapes, padding < 5%
- `outputs/evals/dataset_stats.json` exists
- No `.wav` files are read at any point during the DataLoader smoke test — only `.npy` integers

---

## Risks

| Risk | Mitigation |
|------|------------|
| PESQ/STOI round-trip below threshold → tokenizer rejects 16k→24k audio | Abort and select fallback codec (decision D1); document tradeoff |
| Tier-1 yield < 30h even after D3 relaxation | Extend Tier-2 to SNR > 20dB as last resort; this data goes into S1 only, never S2 or DPO |
| Whisper-small too slow on CPU for 100h | Switch to Whisper-tiny; document WER accuracy delta vs Whisper-small in eval JSON |
| nemo_text_processing install fails (Cython, CUDA deps) | Use `num2words` + regex fallback for number/abbreviation expansion; document coverage gaps |
| `.lock` sentinel logic missing in a write script → silent overwrite | Implement shared `lock_guard` utility in `src/data/` imported by all data-writing scripts |

## Estimated Duration

Sprint 0: 1 week · Sprint 1: 2 weeks · Sprint 2: 2 weeks → **~5 weeks total**

## Dependencies

- Phase 1 complete (directory tree exists, `.gitignore` verified)
- Decision D1 resolved (tokenizer license confirmed, checkpoint pinned)
- Decision D3 resolved (Tier-1 relaxation rule defined)
- GPU with ≥ 8 GB VRAM for Whisper ASR check (optional but strongly recommended)
- LibriSpeech train-clean-100 accessible via download or local archive
