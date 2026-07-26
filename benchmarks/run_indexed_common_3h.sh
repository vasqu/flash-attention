#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-benchmark_results/indexed_common_3h_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

# These defaults deliberately target a multi-hour H100 qualification run.
WARMUP="${WARMUP:-7}"
ROUNDS="${ROUNDS:-12}"
TARGET_ROUND_MS="${TARGET_ROUND_MS:-1000}"
MIN_ITERS="${MIN_ITERS:-1}"
MAX_ITERS="${MAX_ITERS:-600}"
SDPA_MAX_MASK_MIB="${SDPA_MAX_MASK_MIB:-512}"
CASE_REGEX="${CASE_REGEX:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB="${FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB:-2048}"

args=(
  --suite common-indexed-3h
  --dtype bf16
  --native all
  --sdpa subset
  --sdpa-max-mask-mib "$SDPA_MAX_MASK_MIB"
  --compare-backends all
  --prefill-lab
  --block-sparse-lab
  --cute-block-metadata
  --precompile-all
  --shuffle
  --warmup "$WARMUP"
  --rounds "$ROUNDS"
  --target-round-ms "$TARGET_ROUND_MS"
  --min-iters "$MIN_ITERS"
  --max-iters "$MAX_ITERS"
  --output-dir "$OUT"
  --run-name indexed_common_3h_bf16
)
if [[ -n "$CASE_REGEX" ]]; then
  args+=(--case-regex "$CASE_REGEX")
fi

set -o pipefail
PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py "${args[@]}" \
  2>&1 | tee "$OUT/benchmark.log"

python benchmarks/summarize_indexed_common_3h.py \
  "$OUT/indexed_common_3h_bf16.json" \
  --output "$OUT/INDEXED_COMMON_3H_REPORT.md"

echo "Results: $OUT"
echo "Report:  $OUT/INDEXED_COMMON_3H_REPORT.md"
