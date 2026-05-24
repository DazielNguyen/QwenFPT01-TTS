# data/

Runtime data directory. All subdirectories are `.gitignore`d — too large for git.

| Directory | Contents | Phase |
|-----------|----------|-------|
| `raw/` | LibriSpeech train-clean-100 FLAC (~6 GB) | Phase 2 |
| `processed/` | Resampled 24kHz WAV (~6 GB) | Phase 2 |
| `tokenized/` | Pre-tokenized .npy arrays (~280 MB) + `.lock` sentinel | Phase 2 |
| `manifests/` | JSONL manifests per tier (tier1.jsonl, tier2.jsonl) | Phase 2 |
| `quality_tiers/tier1/` | Tier-1 clips: SNR>35dB, WER<2% (~40h) | Phase 2 |
| `quality_tiers/tier2/` | Tier-2 clips: SNR>25dB, WER<5% (~50h) | Phase 2 |
| `long_form/` | Concatenated long-form sequences (~300 MB) | Phase 2 Sprint 2 |
| `dpo_pairs/` | DPO preference pairs — pairs.jsonl (~4 GB) | Phase 5 Sprint 8 |
| `speaker_sft/{name}/` | Per-voice reference audio (~2–5h each) | Phase 6 Sprint 10 |

## Download

```bash
python scripts/sprint_01/download_librispeech.py --output data/raw/
```

## Tokenization lock

`data/tokenized/.lock` is a write-once sentinel created after tokenization completes.
If absent, tokenization is incomplete — re-run `tokenize_dataset.py`.
