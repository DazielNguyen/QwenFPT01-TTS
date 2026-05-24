"""Sprint 1: Verify tokenizer round-trip quality on 20 sample clips.

Checks:
    - Upsample 16kHz → 24kHz
    - Tokenize with Qwen3-TTS-Tokenizer-12Hz
    - Decode back to waveform
    - Measure PESQ (target ≥ 2.5) and STOI (target ≥ 0.85)

Secrets: HF_TOKEN from env var for tokenizer download.

Usage:
    python scripts/sprint_01/verify_roundtrip.py --clips data/raw/samples/
"""

raise NotImplementedError("Implement in Phase 2 Sprint 0")
