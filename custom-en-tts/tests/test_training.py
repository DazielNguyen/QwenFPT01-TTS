"""Tests for src/training/ — pretrain, DPO, GSPO, speaker SFT trainers.

Tests to implement in Phase 3–5:
    - Smoke test: 10-step training loop on tiny config (8-layer, 256-dim)
    - DeepSpeed ZeRO-2 launch without error on 2×GPU
    - DPO preference accuracy > random on synthetic pairs
    - GSPO reward function outputs expected range [0, 1]
    - Speaker SFT freezes backbone layers 0–23
"""

import pytest


def test_placeholder():
    """Placeholder — replace with real tests in Phase 3."""
    pass
