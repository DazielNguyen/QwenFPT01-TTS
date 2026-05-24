# Phase 7: Demo and Release

## Overview

Deliver a publicly usable TTS system: Gradio demo with 6 voices, pip-installable package with CLI,
streaming inference, Docker image, and HuggingFace model card. Corresponds to Sprint 12.

---

## Tasks

- [ ] Implement `src/inference/generator.py`: batched synthesis with KV cache; pre-load and cache all 6 speaker embeddings at server startup to minimize first-token latency (pre-computation at init time, not per-request)
- [ ] Implement `src/inference/streaming.py`: token-streaming output with overlap-add buffering at chunk boundaries (prevents audio artifacts between chunks)
- [ ] Build `demo/gradio_app.py`: voice selector dropdown (Ryan, Aiden, Emma, Sophia, Oliver, Isabella), text input, audio playback, streaming mode toggle; verify latency for a 20-word sentence < 3s on A10G (< 5s on T4 fallback)
- [ ] **Gradio demo security (new — red-team finding):** add a terms-of-use accept gate on first load; add server-side rate limiting (max 10 requests/minute per IP); add a model card link stating allowed and disallowed uses (no voice cloning for impersonation of real persons)
- [ ] Package as pip library: complete `setup.py` with `tts` CLI entry point accepting `--voice`, `--text`, `--output`; verify the inference package does NOT import `nemo_text_processing` (offline preprocessing tool only); use `num2words` + regex for runtime text normalization
- [ ] Check PyPI for naming conflicts before publishing; use `qwenfpt-tts` as package name if `custom-en-tts` is taken; publish to TestPyPI first, verify `pip install qwenfpt-tts && tts --help` from a clean Python 3.10 venv
- [ ] Write `Dockerfile`: multi-stage build (build stage installs deps, runtime stage is minimal); checkpoints downloaded from HuggingFace Hub at container startup (not baked into image to keep image < 5 GB); expose port 7860; test `docker build` + `docker run` end-to-end
- [ ] **HuggingFace model card (new — red-team finding must address):** the HF README must include:
  - Architecture summary (0.6B backbone, 6-stage training, 100h LibriSpeech)
  - Full evaluation results table from `outputs/evals/full_benchmark.json`
  - **Misuse disclosures**: model can clone speaking style from speaker IDs; not intended for impersonation of real persons or non-consenting voice synthesis; CC BY 4.0 derivative if tokenizer license allows (verify decision D5)
  - Training data statement: LibriSpeech CC BY 4.0 only
  - License: state the outgoing license (depends on decision D5 and tokenizer license from D1)
- [ ] Push all 6 voice checkpoints to HuggingFace Hub using Git-LFS; shard files to ≤ 4.9 GB using `save_pretrained(max_shard_size="4.9GB")`; verify download works from a fresh machine
- [ ] Write project `README.md`: quick-start (pip install, docker run, Gradio URL), architecture diagram, training pipeline overview, HF Hub link, reproduction guide (from data download to final checkpoint)
- [ ] CLI test from clean venv: `pip install qwenfpt-tts && tts --voice ryan --text "Hello world" --output /tmp/out.wav` produces valid WAV

---

## Definition of Done

- [ ] `pip install qwenfpt-tts` succeeds from a clean Python 3.10 venv
- [ ] `tts --voice ryan --text "Hello world" --output out.wav` produces a valid WAV file
- [ ] Gradio demo: all 6 voices selectable, synthesis completes, rate limiting active, terms-of-use gate visible
- [ ] Streaming inference: first audio chunk within 1s on A10G
- [ ] `docker build && docker run` completes; demo accessible at localhost:7860
- [ ] All 6 voice checkpoints live on HuggingFace Hub with model card including misuse disclosures
- [ ] Project README contains working quick-start commands verified from a clean environment
- [ ] Inference package does NOT import `nemo_text_processing` (verified by `pip show nemo_text_processing` in clean install — must not appear)

---

## Risks

| Risk | Mitigation |
|------|------------|
| Demo latency > 3s due to checkpoint load per request | Pre-load model at server startup; keep resident in GPU memory between requests |
| Checkpoint shard size exceeds HF Hub limits | Use `max_shard_size="4.9GB"` in `save_pretrained`; test upload before demo day |
| Docker image > 5 GB | Multi-stage build; checkpoints downloaded at runtime not baked in |
| Package name collision on PyPI | Check before publishing; fallback name: `qwenfpt-tts` |
| Streaming chunk boundary artifacts | Overlap-add buffering in `streaming.py`; test on 100+ word sentences |
| Model license restricted by tokenizer license (D1/D5) | Resolve D5 before pushing to HF Hub; add explicit license file to HF repo |
| Gradio demo abused for voice cloning at scale | Rate limiting + terms-of-use gate; model card states acceptable use |

---

## Estimated Duration

Sprint 12: 2 weeks

## Dependencies

- Phase 6 complete (all 6 speaker SFT checkpoints verified, `outputs/evals/full_benchmark.json` exists)
- Decision D5 resolved (outgoing license confirmed)
- HuggingFace account with write token (sourced from `HF_TOKEN` env var, never hardcoded)
- PyPI / TestPyPI account for package publishing
- Docker with buildx installed
- Git-LFS installed for checkpoint push
- A10G or equivalent GPU for latency testing (T4 acceptable fallback)
