#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-benchmark_results/glm_moe_dsa_45_overnight_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

WARMUP="${WARMUP:-7}"
ROUNDS="${ROUNDS:-9}"
TARGET_ROUND_MS="${TARGET_ROUND_MS:-250}"
MIN_ITERS="${MIN_ITERS:-1}"
MAX_ITERS="${MAX_ITERS:-300}"
SDPA_MAX_MASK_MIB="${SDPA_MAX_MASK_MIB:-768}"
CASE_REGEX="${CASE_REGEX:-}"
BLOCK_SPARSE_LAB="${BLOCK_SPARSE_LAB:-1}"
CUTE_BLOCK_METADATA="${CUTE_BLOCK_METADATA:-1}"

# Avoid allocator fragmentation between long prefill and high-batch decode.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

args=(
  --suite glm-moe-dsa-45-overnight
  --dtype bf16
  --native all
  --sdpa all
  --sdpa-max-mask-mib "$SDPA_MAX_MASK_MIB"
  --compare-backends row-dense
  --prefill-lab
  --warmup "$WARMUP"
  --rounds "$ROUNDS"
  --target-round-ms "$TARGET_ROUND_MS"
  --min-iters "$MIN_ITERS"
  --max-iters "$MAX_ITERS"
  --output-dir "$OUT"
  --run-name glm_moe_dsa_45_overnight_bf16
)
if [[ "$BLOCK_SPARSE_LAB" == "1" ]]; then
  args+=(--block-sparse-lab)
  if [[ "$CUTE_BLOCK_METADATA" == "1" ]]; then
    args+=(--cute-block-metadata)
  fi
fi
if [[ -n "$CASE_REGEX" ]]; then
  args+=(--case-regex "$CASE_REGEX")
fi

set -o pipefail
PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py "${args[@]}" \
  2>&1 | tee "$OUT/benchmark.log"

python benchmarks/summarize_glm_moe_dsa_45_overnight.py \
  "$OUT/glm_moe_dsa_45_overnight_bf16.json" \
  --output "$OUT/GLM_MOE_DSA_45_OVERNIGHT_REPORT.md"

echo "Results: $OUT"
echo "Report:  $OUT/GLM_MOE_DSA_45_OVERNIGHT_REPORT.md"
