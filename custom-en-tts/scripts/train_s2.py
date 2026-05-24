"""Top-level launcher: S2 High-Quality CPT stage.

Resumes from checkpoints/s1_general/final.pt. Uses Tier-1 only data
with 20% S1 data mix to prevent catastrophic forgetting.

Usage:
    python scripts/train_s2.py --config configs/training/s2_hq_cpt.yaml --deepspeed

Secrets required (env vars):
    WANDB_API_KEY
"""

raise NotImplementedError("Implement in Phase 4 Sprint 6")
