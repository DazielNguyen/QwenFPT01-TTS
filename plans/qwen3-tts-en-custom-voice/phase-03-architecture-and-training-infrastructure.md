# Phase 3: Architecture and Training Infrastructure

## Overview

Implement and smoke-test the 0.6B Transformer backbone with speaker conditioning and 16 parallel LM
heads, plus a stage-aware training loop supporting DeepSpeed ZeRO-2, checkpoint save/resume, and
cosine LR scheduling — validated end-to-end on the tiny config. Corresponds to Sprint 3 and Sprint 4.

---

## Sprint 3 — Model Architecture (2 weeks)

### Tasks

- [ ] Implement `src/model/config.py`: YAML-driven dataclasses for all architecture hyperparameters; both `backbone_tiny.yaml` and `backbone_0.6b.yaml` must load through the same code path; add assertion on expected param count (target: ~0.5B for 0.6B config)
- [ ] Implement `src/model/backbone.py`: RMSNorm, Grouped-Query Attention (16 query heads / 8 KV heads), **YaRN-aware RoPE** with `rope_theta=1_000_000.0` and `max_seq_len` parameterized in config (decision D4); SwiGLU FFN; full stacked TransformerBackbone class
- [ ] **YaRN RoPE implementation (new — red-team finding):** implement NTK-aware (YaRN) position interpolation so that extending `max_seq_len` from 4096 to 8192 in Phase 4 (S3) does not produce unseen RoPE frequencies; parameterize the scaling factor in the config so it can be tuned in Sprint 7
- [ ] Implement `src/model/tts_model.py`: speaker embedding table (one embedding per speaker ID) + 16 parallel LM heads (one per codebook, each vocab_size=2048) + full TTS model wrapping the backbone; verify forward pass produces logits shape `(B, T, 16, 2048)`
- [ ] Implement ChatML sequence builder: format system + user + assistant turns with `[SPEAKER: X]` prefix and `[TTS_START] ... [TTS_END]` audio token delimiters
- [ ] Write `tests/test_model.py`: shape checks on all components, GQA attention correctness against reference MHA on identical inputs, RoPE frequency sanity check at positions 0, 2048, 4095, loss > 0 on random batch
- [ ] All unit tests pass on both tiny and 0.6B config

### DoD

- Forward pass on 0.6B config, batch size 1, seq_len 4096: completes without OOM on A100 40 GB
- Logits shape `(B, T, 16, 2048)` confirmed by test
- GQA attention output matches reference MHA output within 1e-4 tolerance on identical inputs
- `tests/test_model.py` passes (all assertions green)
- YaRN RoPE: perplexity at seq positions 4096–8192 does not spike more than 10% vs positions 0–4095 (measured on random token sequence — not speech data yet)

---

## Sprint 4 — Training Infrastructure (2 weeks)

### Tasks

- [ ] **Pin all exact versions (new — red-team finding):** in `requirements.txt`, pin specific patch versions of: `torch==2.3.x`, `deepspeed==0.14.x`, `nccl` (via CUDA toolkit pin), `triton`, `flash-attn` (if used). Run `pip freeze > requirements.lock` after env is built. Add a CI step that reinstalls from `requirements.lock` and re-runs smoke test to catch silent breakage.
- [ ] Implement `src/training/pretrain/trainer.py`: AdamW optimizer, cosine LR schedule with linear warmup, gradient clipping (max_norm=1.0), step/epoch/checkpoint logging
- [ ] Integrate DeepSpeed ZeRO-2 via `deepspeed_config` in `configs/base.yaml`; implement multi-GPU launch script (`scripts/train_s1.py` accepts `--deepspeed` flag); verify ZeRO-2 initializes without error on **2-GPU setup specifically** — not single GPU only
- [ ] Add bfloat16 mixed precision; verify no NaN gradients on 100 steps with tiny config on 2×GPU
- [ ] Stage-aware checkpoint save/resume: save `{stage}_{step:06d}.pt` with model state + optimizer state + step index; implement resume that restores exact loss at step N+1; checkpoint naming goes to `checkpoints/{stage}/`
- [ ] Stage-specific training config YAMLs: `s1_general.yaml` (LR=1e-4, tiers=[1,2], seq_len=4096, warmup=2000), `s2_hq_cpt.yaml` (LR=2e-5, tiers=[1], warmup=500), `s3_long_context.yaml` (LR=1e-5, seq_len=8192)
- [ ] Logging: W&B log loss, grad norm, LR, tokens/sec every 100 steps; log checkpoint path as W&B artifact after each save
- [ ] Validation loop: eval on `val.jsonl` every epoch, log val loss
- [ ] Smoke test: 100 steps with `backbone_tiny.yaml` on 2×GPU with ZeRO-2 + bfloat16; verify loss decreases monotonically over first 50 steps, no crash, no NaN gradients, checkpoint save+resume exact match

### DoD

- Smoke test (100 steps, tiny config, 2×GPU, ZeRO-2 + bfloat16) completes without crash or NaN
- Loss at step 51 after resume from step 50 checkpoint matches continuous run at step 51 (within floating point tolerance)
- All 6 stage YAML configs load without error
- W&B run live with training metrics visible
- `requirements.lock` exists and CI reinstall + smoke test passes

---

## Risks

| Risk | Mitigation |
|------|------------|
| GQA produces incorrect attention (silent correctness bug) | Test against reference MHA before any training run |
| YaRN RoPE implementation wrong → attention degrades at extended positions | Perplexity regression test in Sprint 3 DoD; compare with published YaRN paper coefficients |
| DeepSpeed + bfloat16 silent NaN only visible at scale (not in smoke test) | Run smoke test specifically on 2×GPU with bfloat16; monitor grad norm histogram first 500 steps of S1 |
| Version mismatch between DeepSpeed and PyTorch 2.x → NCCL hang | Use `requirements.lock`; test on exact target cloud GPU instance before Phase 4 |
| 16 LM heads exceed VRAM budget with ZeRO-2 | If OOM: apply weight tying across codebook heads or switch to ZeRO-3; log VRAM peak in smoke test |

## Estimated Duration

Sprint 3: 2 weeks · Sprint 4: 2 weeks → **~4 weeks total**

## Dependencies

- Phase 2 complete (DataLoader and manifests verified, `.lock` files exist)
- Decision D4 resolved (YaRN RoPE extension method confirmed)
- At least 1× A100 40 GB for smoke-test; 2× GPU required for ZeRO-2 smoke test
- `requirements.lock` created in this phase (carried into Phase 4)
