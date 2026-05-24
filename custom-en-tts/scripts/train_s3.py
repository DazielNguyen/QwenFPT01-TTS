"""Top-level launcher: S3 Long-Context pre-training stage.

Resumes from checkpoints/s2_hq_cpt/final.pt. Enables gradient checkpointing
and YaRN RoPE extension to max_seq_len=8192.

Usage:
    python scripts/train_s3.py --config configs/training/s3_long_context.yaml --deepspeed

Secrets required (env vars):
    WANDB_API_KEY
"""

raise NotImplementedError("Implement in Phase 4 Sprint 7")
