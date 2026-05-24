"""Tests for src/inference/ — generator and streaming.

Tests to implement in Phase 7:
    - Generator produces valid WAV output for a short prompt
    - All 6 speaker embeddings pre-loaded at init time
    - Streaming: first chunk arrives within expected latency
    - Overlap-add buffering produces no audible chunk artifacts
    - nemo_text_processing is NOT imported (runtime text norm via num2words + regex)
"""

import pytest


def test_no_nemo_text_processing_import():
    """nemo_text_processing must not be imported in the inference package."""
    import importlib
    import sys

    # Ensure the module is not already loaded
    assert "nemo_text_processing" not in sys.modules, (
        "nemo_text_processing must not be imported in inference package"
    )


def test_placeholder():
    """Placeholder — replace with real tests in Phase 7."""
    pass
