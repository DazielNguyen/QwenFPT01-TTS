"""Sprint 1: Filter audio clips into quality tiers (Tier-1 / Tier-2 / discard).

Criteria:
    Tier-1: SNR > 35 dB, WER < 2%   (~40h target)
    Tier-2: SNR > 25 dB, WER < 5%   (~50h target)
    Discard: below Tier-2 thresholds

D3 relaxation: if Tier-1 yield < 30h, lower SNR to 30 dB and WER to 3%.

Outputs manifests to data/manifests/.

Usage:
    python scripts/sprint_01/tier_filter.py --input data/raw/ --output data/manifests/
"""

raise NotImplementedError("Implement in Phase 2 Sprint 1")
