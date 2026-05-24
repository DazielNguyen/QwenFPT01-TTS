"""Top-level launcher: GSPO post-training stage.

Resumes from checkpoints/dpo/final.pt. Uses composite reward:
    r = 0.4*(1-WER) + 0.3*UTMOS + 0.2*SIM + 0.1*length_OK

WARNING: Profile combined VRAM before launching (see phase-05 plan).

Usage:
    python scripts/train_gspo.py --config configs/training/gspo.yaml

Secrets required (env vars):
    WANDB_API_KEY
"""

raise NotImplementedError("Implement in Phase 5 Sprint 9")
