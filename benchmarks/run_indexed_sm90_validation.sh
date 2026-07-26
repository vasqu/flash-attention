#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/benchmark_results/indexed_sm90_validation}"
LEGACY_RANDOM_CASES="${FA4_RANDOM_SHAPE_CASES:-32}"
CORRECTNESS_RANDOM_CASES="${FA4_CORRECTNESS_RANDOM_CASES:-$LEGACY_RANDOM_CASES}"
BENCHMARK_RANDOM_CASES="${FA4_BENCHMARK_RANDOM_CASES:-$LEGACY_RANDOM_CASES}"
RANDOM_BACKEND_STRESS_CASES="${FA4_RANDOM_BACKEND_STRESS_CASES:-0}"
CORRECTNESS_LEVEL="${FA4_INDEXED_CORRECTNESS_LEVEL:-standard}"
STRICT="${FA4_VALIDATION_STRICT:-1}"
RUN_REGRESSION="${FA4_RUN_REGRESSION:-1}"
RUN_SERVING_MEDIUM="${FA4_RUN_SERVING_MEDIUM:-1}"
RUN_DSA_MODELS="${FA4_RUN_DSA_MODELS:-1}"
RUN_ITERATION_MEDIUM="${FA4_RUN_ITERATION_MEDIUM:-0}"
RUN_BROAD_PERF="${FA4_RUN_BROAD_PERF:-0}"
RUN_RANDOM_BENCHMARK="${FA4_RUN_RANDOM_BENCHMARK:-0}"
RUN_UNION_CANDIDATES="${FA4_RUN_UNION_CANDIDATES:-0}"
SDPA_MODE="${FA4_SDPA_BASELINE:-subset}"
SDPA_MAX_MASK_MIB="${FA4_SDPA_MAX_MASK_MIB:-256}"
KEEP_OLD_RESULTS="${FA4_KEEP_OLD_RESULTS:-0}"
COMMON_COMPARE_BACKENDS="${FA4_COMMON_COMPARE_BACKENDS:-subset}"
BENCH_WARMUP="${FA4_BENCH_WARMUP:-5}"
BENCH_ROUNDS="${FA4_BENCH_ROUNDS:-3}"
BENCH_TARGET_ROUND_MS="${FA4_BENCH_TARGET_ROUND_MS:-100}"
BENCH_MIN_ITERS="${FA4_BENCH_MIN_ITERS:-10}"
BENCH_MAX_ITERS="${FA4_BENCH_MAX_ITERS:-500}"
AUTO_REGRET_THRESHOLD="${FA4_AUTO_REGRET_THRESHOLD:-1.03}"
FAIL_AUTO_REGRET="${FA4_FAIL_AUTO_REGRET:-}"
FA4_PY="$ROOT/benchmarks/fa4_only_exec.py"
JUNIT="$OUT_DIR/correctness.xml"
STATUS=0

read -r -a BENCH_DTYPE_LIST <<< "${FA4_BENCH_DTYPES:-bf16}"
if [[ ${#BENCH_DTYPE_LIST[@]} -eq 0 ]]; then
  echo "FA4_BENCH_DTYPES must contain bf16 and/or fp16" >&2
  exit 2
fi
for dtype in "${BENCH_DTYPE_LIST[@]}"; do
  case "$dtype" in
    bf16|fp16) ;;
    *)
      echo "Unsupported benchmark dtype: $dtype (expected bf16 or fp16)" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT_DIR"
if [[ "$KEEP_OLD_RESULTS" != "1" ]]; then
  rm -f -- "$OUT_DIR"/*.json "$OUT_DIR"/*.csv "$OUT_DIR"/correctness.xml "$OUT_DIR"/SUMMARY.md
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FLASH_ATTN_FA4_ROOT="${FLASH_ATTN_FA4_ROOT:-$ROOT}"
export FLASH_ATTENTION_ARCH="${FLASH_ATTENTION_ARCH:-sm_90a}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_90a}"
export FA4_RANDOM_SHAPE_CASES="$CORRECTNESS_RANDOM_CASES"
export FA4_INDEXED_CORRECTNESS_LEVEL="$CORRECTNESS_LEVEL"

cd "$ROOT"

run_step() {
  local title="$1"
  shift
  echo
  echo "== $title =="
  "$@"
  local code=$?
  if [[ $code -ne 0 ]]; then
    echo "STEP FAILED ($code): $title" >&2
    STATUS=1
  fi
  return 0
}

run_benchmark() {
  local title="$1"
  local suite="$2"
  local compare="$3"
  local run_name="$4"
  local dtype="$5"
  shift 5
  run_step "$title ($dtype)" \
    python "$FA4_PY" benchmarks/benchmark_indexed_sm90.py \
      --suite "$suite" \
      --dtype "$dtype" \
      --warmup "$BENCH_WARMUP" \
      --rounds "$BENCH_ROUNDS" \
      --target-round-ms "$BENCH_TARGET_ROUND_MS" \
      --min-iters "$BENCH_MIN_ITERS" \
      --max-iters "$BENCH_MAX_ITERS" \
      --native all \
      --sdpa "$SDPA_MODE" \
      --sdpa-max-mask-mib "$SDPA_MAX_MASK_MIB" \
      --compare-backends "$compare" \
      --output-dir "$OUT_DIR" \
      --run-name "${run_name}_${dtype}" \
      "$@"
}

run_step "FA4-only import preflight" \
  python "$FA4_PY" benchmarks/fa4_only_preflight.py

run_step "Correctness: upstream FA4 matrix, top-k sweep, eager + SDPA, odd/random shapes" \
  python "$FA4_PY" -m pytest -q \
    tests/cute/test_indexed_policy.py \
    tests/cute/test_indexed_metrics.py \
    tests/cute/test_indexed_score_mod_utils.py \
    tests/cute/test_indexed_benchmark_utils.py \
    tests/cute/test_sm90_tile_safety.py \
    tests/cute/test_indexed_sm90_sdpa.py \
    tests/cute/test_indexed_sm90_numerics.py \
    tests/cute/test_indexed_sm90_score_mod_numerics.py \
    --junitxml="$JUNIT"

run_step "CuTe fake compilation: dense and indexed backends" \
  python "$FA4_PY" tests/cute/smoke_compile_indexed_sm90.py

run_step "CuTe fake compilation: indexed score_mod" \
  python "$FA4_PY" tests/cute/smoke_compile_indexed_score_mod_sm90.py

for dtype in "${BENCH_DTYPE_LIST[@]}"; do
  if [[ "$RUN_REGRESSION" == "1" ]]; then
    run_benchmark "Indexed regression benchmark" "regression" "all" "regression" "$dtype"
  fi
  if [[ "$RUN_SERVING_MEDIUM" == "1" ]]; then
    run_benchmark "Serving decode + same-length prefill benchmark" \
      "serving-medium" "all" "serving_medium" "$dtype" --native subset
  fi
  if [[ "$RUN_DSA_MODELS" == "1" ]]; then
    run_benchmark "DeepSeek-V3.2 + GLM MoE DSA benchmark" \
      "dsa-models" "all" "dsa_models" "$dtype" --native subset
  fi
  if [[ "$RUN_ITERATION_MEDIUM" == "1" ]]; then
    run_benchmark "Legacy 2K medium policy benchmark" "iteration-medium" "all" "iteration_medium" "$dtype"
  fi
  if [[ "$RUN_BROAD_PERF" == "1" ]]; then
    run_benchmark "Odd/non-aligned shape benchmark" "odd" "all" "odd_shapes" "$dtype"
    run_benchmark "Top-k crossover benchmark" "topk-sweep" "all" "topk_sweep" "$dtype"
    run_benchmark "Focused chunked-prefill crossover benchmark" "prefill-focus" "all" "prefill_focus" "$dtype"
    run_benchmark "Common + large-batch benchmark" "common" "$COMMON_COMPARE_BACKENDS" "common" "$dtype"

    if [[ "$RUN_UNION_CANDIDATES" == "1" ]]; then
      run_benchmark "Union/MQA candidate benchmark" "union-candidates" "all" "union_candidates" "$dtype"
    fi
  fi

  if [[ "$RUN_RANDOM_BENCHMARK" == "1" ]]; then
    # Random shapes are compatibility/perf fuzzing, not the primary serving
    # objective. Keep them in a separate process and opt in explicitly.
    run_benchmark "Deterministic random-shape benchmark" \
      "random" "subset" "random_shapes" "$dtype" \
      --random-cases "$BENCHMARK_RANDOM_CASES" \
      --random-max-k 8192 \
      --no-precompile-all \
      --no-shuffle
  fi

  if [[ "$RANDOM_BACKEND_STRESS_CASES" -gt 0 ]]; then
    run_benchmark "Random-shape all-backend stress benchmark" \
      "random" "all" "random_all_backends" "$dtype" \
      --random-cases "$RANDOM_BACKEND_STRESS_CASES" \
      --random-max-k 8192 \
      --no-precompile-all \
      --no-shuffle
  fi
done

SUMMARY_ARGS=(
  "$OUT_DIR"
  --junit "$JUNIT"
  --output "$OUT_DIR/SUMMARY.md"
  --auto-regret-threshold "$AUTO_REGRET_THRESHOLD"
)
if [[ -n "$FAIL_AUTO_REGRET" ]]; then
  SUMMARY_ARGS+=(--fail-auto-regret "$FAIL_AUTO_REGRET")
fi
if [[ "$STRICT" == "1" ]]; then
  SUMMARY_ARGS+=(--strict)
fi

run_step "Comprehensive good / parity / bad summary" \
  python "$FA4_PY" benchmarks/summarize_indexed_sm90_validation.py "${SUMMARY_ARGS[@]}"

if [[ $STATUS -ne 0 ]]; then
  echo
  echo "Validation completed with failures. See $OUT_DIR/SUMMARY.md" >&2
  exit 2
fi

echo
echo "Validation completed successfully. Summary: $OUT_DIR/SUMMARY.md"
