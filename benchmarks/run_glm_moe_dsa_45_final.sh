#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${1:-benchmark_results/glm_moe_dsa_45_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"

WARMUP="${WARMUP:-5}"
ROUNDS="${ROUNDS:-7}"
TARGET_ROUND_MS="${TARGET_ROUND_MS:-150}"
MIN_ITERS="${MIN_ITERS:-1}"
MAX_ITERS="${MAX_ITERS:-200}"
SDPA_MAX_MASK_MIB="${SDPA_MAX_MASK_MIB:-512}"
CASE_REGEX="${CASE_REGEX:-}"

args=(
  --suite glm-moe-dsa-45
  --dtype bf16
  --native all
  --sdpa all
  --sdpa-max-mask-mib "$SDPA_MAX_MASK_MIB"
  --compare-backends row-dense
  --warmup "$WARMUP"
  --rounds "$ROUNDS"
  --target-round-ms "$TARGET_ROUND_MS"
  --min-iters "$MIN_ITERS"
  --max-iters "$MAX_ITERS"
  --output-dir "$OUT"
  --run-name glm_moe_dsa_45_bf16
)
if [[ -n "$CASE_REGEX" ]]; then
  args+=(--case-regex "$CASE_REGEX")
fi

set -o pipefail
PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py "${args[@]}" \
  2>&1 | tee "$OUT/benchmark.log"

python benchmarks/summarize_glm_moe_dsa_45.py \
  "$OUT/glm_moe_dsa_45_bf16.json" \
  --output "$OUT/GLM_MOE_DSA_45_REPORT.md"

echo "Results: $OUT"
echo "Report:  $OUT/GLM_MOE_DSA_45_REPORT.md"
