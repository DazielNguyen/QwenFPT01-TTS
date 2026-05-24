# scripts/

Training launchers and preprocessing scripts.

## Top-level launchers

| Script | Stage | Phase |
|--------|-------|-------|
| `train_s1.py` | S1 General pre-training | Phase 4 Sprint 5 |
| `train_s2.py` | S2 High-Quality CPT | Phase 4 Sprint 6 |
| `train_s3.py` | S3 Long-Context | Phase 4 Sprint 7 |
| `train_dpo.py` | DPO post-training | Phase 5 Sprint 8 |
| `train_gspo.py` | GSPO post-training | Phase 5 Sprint 9 |
| `storage_report.sh` | Storage check | Run before any Phase 4+ |

## Sprint 1 — Data Pipeline

| Script | Description |
|--------|-------------|
| `sprint_01/download_librispeech.py` | Download LibriSpeech train-clean-100 |
| `sprint_01/tier_filter.py` | Filter clips into Tier-1/2 by SNR and WER |
| `sprint_01/verify_roundtrip.py` | Tokenizer round-trip PESQ/STOI validation |
| `sprint_01/tokenize_dataset.py` | Pre-tokenize all audio to .npy files |
| `sprint_01/verify_speaker_hours.py` | Verify ≥2h Tier-1 audio per speaker |

## Sprint 2 — Long-form and DataLoader

| Script | Description |
|--------|-------------|
| `sprint_02/build_longform.py` | Chapter-aware long-form sequence builder |
| `sprint_02/build_dataloader.py` | Validate DataLoader pipeline end-to-end |
