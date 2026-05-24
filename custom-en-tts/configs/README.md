# configs/

Hydra-style YAML configuration files. Stage-specific configs inherit from `base.yaml`.

| File | Description |
|------|-------------|
| `base.yaml` | Base config — all stages inherit from this |
| `model/backbone_0_6b.yaml` | 0.6B backbone architecture |
| `training/s1_general.yaml` | S1 General pre-training (50K–80K steps, Tier-1+2) |
| `training/s2_hq_cpt.yaml` | S2 High-Quality CPT (20K–30K steps, Tier-1 only) |
| `training/s3_long_context.yaml` | S3 Long-Context (seq_len=8192, gradient checkpointing) |
| `training/dpo.yaml` | DPO post-training (TRL, beta=0.1) |
| `training/gspo.yaml` | GSPO post-training (G=8, composite reward) |
| `training/speaker_sft.yaml` | Speaker SFT (freeze layers 0–23, per-voice override) |

## Inheritance pattern

```yaml
# In any stage config:
_base_: ../base.yaml

# Override only what changes:
training:
  lr: 2e-5
```

## Secrets

All API keys and tokens are sourced **exclusively from environment variables**:
- `WANDB_API_KEY` — Weights & Biases
- `HF_TOKEN` — HuggingFace Hub

Never hardcode secrets in YAML files.
