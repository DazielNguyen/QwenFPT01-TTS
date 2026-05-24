# outputs/

Evaluation results and generated audio samples.

| Directory | Contents | Committed to git? |
|-----------|----------|-------------------|
| `evals/` | JSON benchmark results, eval reports | ✅ YES — commit eval files |
| `samples/` | Generated audio samples per stage | ❌ No — gitignored |
| `dpo_audio/` | DPO scoring audio (~2000 samples, ~4 GB) | ❌ No — gitignored |

## Key eval files (written per phase)

| File | Phase | Contents |
|------|-------|----------|
| `tier1_speaker_hours.json` | Phase 2 | Per-speaker Tier-1 hours for 6 personas |
| `s1_eval.json` | Phase 4 | WER, intelligibility rate |
| `s2_eval.json` | Phase 4 | WER, UTMOS, hallucination count |
| `s3_eval.json` | Phase 4 | Short+long WER, perplexity regression |
| `dpo_vram_profile.json` | Phase 5 | DPO VRAM measurements |
| `dpo_eval.json` | Phase 5 | WER, UTMOS, preference accuracy |
| `gspo_reward_profile.json` | Phase 5 | GSPO reward stack VRAM/throughput |
| `gspo_eval.json` | Phase 5 | WER, UTMOS vs DPO baseline |
| `pre_sft_storage.json` | Phase 6 | Available disk space before SFT |
| `speaker_sft_{name}.json` | Phase 6 | Per-voice WER, UTMOS, SIM |
| `full_benchmark.json` | Phase 6 | All voices × all metrics (100 sentences) |
| `ablation_table.json` | Phase 6 | S1→S2→S3→DPO→GSPO→SFT + S3→GSPO direct |
| `mos_results.json` | Phase 6 | MOS scores with rater count |
| `inference_speed.json` | Phase 6 | RTF and first-packet latency |
