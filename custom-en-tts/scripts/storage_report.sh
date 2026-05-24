#!/usr/bin/env bash
# Storage report: check available disk space and summarize checkpoint sizes.
# Run before starting any training phase.
#
# Usage: bash scripts/storage_report.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== Storage Report ==="
echo "Root: ${ROOT_DIR}"
echo ""

echo "--- Disk available ---"
df -h "${ROOT_DIR}" | tail -1

echo ""
echo "--- Directory sizes ---"
for dir in data checkpoints outputs; do
    if [ -d "${ROOT_DIR}/${dir}" ]; then
        du -sh "${ROOT_DIR}/${dir}" 2>/dev/null || echo "${dir}: empty"
    fi
done

echo ""
echo "--- Checkpoint sizes per stage ---"
for stage in s1_general s2_hq_cpt s3_long_context dpo gspo speaker_sft; do
    ckpt_dir="${ROOT_DIR}/checkpoints/${stage}"
    if [ -d "${ckpt_dir}" ] && [ "$(ls -A "${ckpt_dir}" 2>/dev/null)" ]; then
        du -sh "${ckpt_dir}" 2>/dev/null
    else
        echo "checkpoints/${stage}: (empty)"
    fi
done
