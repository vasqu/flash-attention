#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-benchmark_results/glm_moe_dsa_45_batch_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

WARMUP="${WARMUP:-7}"
ROUNDS="${ROUNDS:-9}"
TARGET_ROUND_MS="${TARGET_ROUND_MS:-250}"
MIN_ITERS="${MIN_ITERS:-1}"
MAX_ITERS="${MAX_ITERS:-300}"
CASE_REGEX="${CASE_REGEX:-}"

# Large-batch Q/K/V and candidate outputs benefit from expandable segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Bound persistent indexed workspaces even when the sweep visits B=1..16.
export FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB="${FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB:-1024}"

args=(
  --suite glm-moe-dsa-45-batch
  --dtype bf16
  --native all
  --sdpa none
  --compare-backends none
  --prefill-lab
  --block-sparse-lab
  --cute-block-metadata
  --warmup "$WARMUP"
  --rounds "$ROUNDS"
  --target-round-ms "$TARGET_ROUND_MS"
  --min-iters "$MIN_ITERS"
  --max-iters "$MAX_ITERS"
  --output-dir "$OUT"
  --run-name glm_moe_dsa_45_batch_bf16
)
if [[ -n "$CASE_REGEX" ]]; then
  args+=(--case-regex "$CASE_REGEX")
fi

set -o pipefail
PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py "${args[@]}" \
  2>&1 | tee "$OUT/benchmark.log"

python benchmarks/summarize_glm_moe_dsa_45_batch.py \
  "$OUT/glm_moe_dsa_45_batch_bf16.json" \
  --output "$OUT/GLM_MOE_DSA_45_BATCH_REPORT.md"

echo "Results: $OUT"
echo "Report:  $OUT/GLM_MOE_DSA_45_BATCH_REPORT.md"
