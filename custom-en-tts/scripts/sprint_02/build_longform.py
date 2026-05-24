"""Sprint 2: Build long-form training sequences by concatenating clips.

Chapter-aware concatenation: only combine clips from the same SPEAKER-CHAPTER
to prevent prosody discontinuities. Insert silence token at clip boundaries.
Target sequence length: 2K–4K audio frames.

Outputs: data/long_form/

Usage:
    python scripts/sprint_02/build_longform.py \
        --manifest data/manifests/tier1.jsonl \
        --output data/long_form/
"""

raise NotImplementedError("Implement in Phase 2 Sprint 2")
