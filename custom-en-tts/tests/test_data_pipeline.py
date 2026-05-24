"""Tests for src/data/ — dataset, collator, and preprocessing.

Tests to implement in Phase 2:
    - Tokenizer round-trip: PESQ ≥ 2.5, STOI ≥ 0.85 on 5 sample clips
    - DataLoader iterates without error
    - Batch shapes match expected token dimensions
    - .lock sentinel written after tokenization completes
    - Long-form sequences are from same SPEAKER-CHAPTER (no cross-chapter concat)
    - Tier-1 yield ≥ 30h after filtering
"""

import pytest


def test_placeholder():
    """Placeholder — replace with real tests in Phase 2."""
    pass
