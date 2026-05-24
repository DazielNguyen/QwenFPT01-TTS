"""Top-level launcher: DPO post-training stage.

Resumes from checkpoints/s3_long_context/final.pt using TRL DPO trainer.
Reads preference pairs from data/dpo_pairs/pairs.jsonl.

Usage:
    python scripts/train_dpo.py --config configs/training/dpo.yaml

Secrets required (env vars):
    WANDB_API_KEY
"""

raise NotImplementedError("Implement in Phase 5 Sprint 8")
