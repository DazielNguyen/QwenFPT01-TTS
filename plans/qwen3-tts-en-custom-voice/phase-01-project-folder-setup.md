# Phase 1: Project Folder Setup

## Overview

Create the entire `custom-en-tts/` directory tree, all Python `__init__.py` stubs, placeholder
READMEs, root project files, and `.gitignore` — before any functional code is written. Every
subsequent phase assumes this structure exists.

## Tasks

- [ ] Create top-level directory `custom-en-tts/` and all first-level subdirectories: `configs/`, `data/`, `checkpoints/`, `outputs/`, `src/`, `scripts/`, `demo/`, `tests/`
- [ ] Create nested subdirectories under `configs/` — `model/` (backbone_0.6b, backbone_tiny, backbone_small) and `training/` (one YAML stub per stage)
- [ ] Create nested subdirectories under `data/` — `raw/`, `processed/`, `quality_tiers/tier1/`, `quality_tiers/tier2/`, `tokenized/`, `long_form/`, `dpo_pairs/`, `speaker_sft/{ryan,aiden,emma,sophia,oliver,isabella}/`, `manifests/`
- [ ] Create nested subdirectories under `checkpoints/` — `s1_general/`, `s2_hq_cpt/`, `s3_long_context/`, `dpo/`, `gspo/`, `speaker_sft/{ryan,aiden,emma,sophia,oliver,isabella}/`
- [ ] Create nested subdirectories under `outputs/` — `samples/`, `evals/`, `dpo_audio/`
- [ ] Create all `src/` subdirectories and add `__init__.py` to every Python package: `src/model/`, `src/data/`, `src/data/preprocessing/`, `src/training/`, `src/training/pretrain/`, `src/training/dpo/`, `src/training/gspo/`, `src/training/sft/`, `src/inference/`, `src/evaluation/`
- [ ] Create stub `.py` files under `src/` (module docstring + `pass` only — no functional code yet): `config.py`, `backbone.py`, `tts_model.py`, `dataset.py`, `collator.py`, all preprocessing files, all training files, all inference and evaluation files
- [ ] Create per-sprint script subdirectories: `scripts/sprint_01/` and `scripts/sprint_02/` with empty stub scripts
- [ ] Create top-level training/eval script stubs: `train_s1.py` through `train_gspo.py`, `train_speaker_sft.py`, `evaluate.py`, `synthesize.py`, `storage_report.sh`
- [ ] Write `demo/gradio_app.py` stub and all `tests/test_*.py` stubs
- [ ] Write `.gitignore` excluding: `data/raw/`, `data/processed/`, `data/tokenized/`, `checkpoints/`, `outputs/`, `*.pyc`, `__pycache__/`, `.env`, `*.env`, `.DS_Store`, `wandb/`, `*.log`
- [ ] Write `configs/base.yaml` with shared architecture + DeepSpeed ZeRO-2 base settings using YAML `_base_` inheritance pattern
- [ ] Write `requirements.txt` with pinned dependency placeholders including: `torch>=2.3.0`, `deepspeed>=0.14.0`, `trl>=0.9.0`, `transformers>=4.40.0`, `wandb>=0.17.0` — all exact versions to be pinned after Phase 2 env verification
- [ ] Write `setup.py` with package metadata skeleton, `tts` CLI entry point placeholder, and version `0.1.0.dev0`
- [ ] Write `Dockerfile` base skeleton (FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime)
- [ ] Write `README.md` project overview with placeholder sections for quick-start, architecture, and training pipeline
- [ ] Write short `README.md` in each major directory explaining: purpose, write-once or mutable status, and `.lock` sentinel convention (for `data/raw/` and `data/tokenized/`)
- [ ] Add a secret-scan check comment in `setup.py` and CI config stub — all secrets sourced from env vars only, never hardcoded

## Full Directory Tree

```
custom-en-tts/
├── configs/
│   ├── base.yaml                   # shared arch + DeepSpeed base (YAML inheritance)
│   ├── model/
│   │   ├── backbone_0.6b.yaml
│   │   ├── backbone_tiny.yaml      # for smoke tests
│   │   └── backbone_small.yaml
│   └── training/
│       ├── s1_general.yaml         # extends base.yaml
│       ├── s2_hq_cpt.yaml
│       ├── s3_long_context.yaml
│       ├── dpo.yaml
│       ├── gspo.yaml
│       └── speaker_sft.yaml
├── data/
│   ├── raw/                        # LibriSpeech FLAC — write-once, .lock after verify
│   ├── processed/                  # 24kHz WAV — write-once after Sprint 1
│   ├── quality_tiers/
│   │   ├── tier1/
│   │   └── tier2/
│   ├── tokenized/                  # .npy token files — write-once, .lock after S2-2
│   ├── long_form/
│   ├── dpo_pairs/
│   ├── speaker_sft/
│   │   ├── ryan/
│   │   ├── aiden/
│   │   ├── emma/
│   │   ├── sophia/
│   │   ├── oliver/
│   │   └── isabella/
│   └── manifests/
│       ├── train.jsonl             # placeholder
│       ├── val.jsonl
│       └── test.jsonl
├── checkpoints/
│   ├── s1_general/
│   ├── s2_hq_cpt/
│   ├── s3_long_context/
│   ├── dpo/
│   ├── gspo/
│   └── speaker_sft/
│       ├── ryan/
│       ├── aiden/
│       ├── emma/
│       ├── sophia/
│       ├── oliver/
│       └── isabella/
├── outputs/
│   ├── samples/
│   ├── evals/
│   └── dpo_audio/
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── backbone.py
│   │   └── tts_model.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── collator.py
│   │   └── preprocessing/
│   │       ├── __init__.py
│   │       ├── audio_filter.py
│   │       ├── text_normalize.py
│   │       ├── asr_check.py
│   │       ├── quality_scorer.py
│   │       ├── tokenize_audio.py
│   │       └── build_longform.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain/
│   │   │   ├── __init__.py
│   │   │   └── trainer.py
│   │   ├── dpo/
│   │   │   ├── __init__.py
│   │   │   ├── dpo_trainer.py
│   │   │   └── pair_builder.py
│   │   ├── gspo/
│   │   │   ├── __init__.py
│   │   │   ├── gspo_trainer.py
│   │   │   └── reward_fn.py
│   │   └── sft/
│   │       ├── __init__.py
│   │       └── speaker_sft.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── streaming.py
│   └── evaluation/
│       ├── __init__.py
│       ├── wer.py
│       ├── utmos.py
│       ├── speaker_sim.py
│       └── benchmark.py
├── scripts/
│   ├── sprint_01/
│   │   ├── 01_download_data.sh
│   │   ├── 02_resample.py
│   │   ├── 03_filter_duration.py
│   │   ├── 04_normalize_amplitude.py
│   │   ├── 05_text_normalize.py     # offline only — not a package dep
│   │   ├── 06_asr_check.py
│   │   └── 07_quality_score_tier.py
│   ├── sprint_02/
│   │   ├── 01_tokenize_offline.py
│   │   ├── 02_verify_tokenization.py
│   │   ├── 03_build_longform.py
│   │   ├── 04_compute_lengths.py
│   │   └── 05_build_manifest.py
│   ├── train_s1.py
│   ├── train_s2.py
│   ├── train_s3.py
│   ├── train_dpo.py
│   ├── train_gspo.py
│   ├── train_speaker_sft.py
│   ├── evaluate.py
│   ├── synthesize.py
│   └── storage_report.sh           # du aggregation across checkpoints/
├── demo/
│   └── gradio_app.py
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_preprocessing.py
│   ├── test_dpo.py
│   └── test_gspo_reward.py
├── .gitignore
├── requirements.txt
├── setup.py
├── Dockerfile
└── README.md
```

## Definition of Done

- [ ] `find custom-en-tts -type d | wc -l` matches expected directory count
- [ ] Every Python package directory has `__init__.py`
- [ ] `git check-ignore -v data/raw/ checkpoints/ outputs/ .env` — all are ignored
- [ ] `python -c "import src.model; import src.training.dpo"` succeeds (stubs importable)
- [ ] All major directories have a `README.md` with at least a one-line purpose + write-once status note
- [ ] No functional Python code in `src/` — only module docstrings and `pass`

## Risks

- Stub files left empty (no docstring) → import errors in later phases; mitigation: enforce docstring in every stub at creation
- `.gitignore` misconfiguration accidentally tracks large binaries; mitigation: verify with `git check-ignore` before first commit
- Directory naming inconsistency (hyphens vs underscores) breaks Python imports; mitigation: all Python package dirs use underscores only

## Estimated Duration

2–4 hours (scripted creation recommended)

## Dependencies

None — first phase, no predecessors.
