#!/usr/bin/env bash
# BF16-first serving-manifold loop. Broad compatibility suites remain opt-in.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DTYPE="${DTYPE:-bf16}"
SUITE="${SUITE:-serving-medium}"
case "${SUITE}" in
  iteration|iteration-medium|serving-medium|serving-decode|serving-prefill) ;;
  *)
    echo "SUITE must be iteration, iteration-medium, serving-medium, serving-decode, or serving-prefill (got: ${SUITE})" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/benchmark_results/${SUITE}}"
RUN_NAME="${RUN_NAME:-${SUITE//-/_}_${DTYPE}_$(date -u +%Y%m%d_%H%M%S)}"
CSV_OUT="${OUTPUT_DIR}/${RUN_NAME}.csv"
JSON_OUT="${OUTPUT_DIR}/${RUN_NAME}.json"

mkdir -p "${OUTPUT_DIR}"

echo "Indexed SM90 policy benchmark: suite=${SUITE} dtype=${DTYPE}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
FLASH_ATTENTION_ARCH="${FLASH_ATTENTION_ARCH:-sm_90a}" \
CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_90a}" \
"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_indexed_sm90.py" \
  --suite "${SUITE}" \
  --dtype "${DTYPE}" \
  --warmup "${WARMUP:-1}" \
  --rounds "${ROUNDS:-3}" \
  --target-round-ms "${TARGET_ROUND_MS:-40}" \
  --min-iters "${MIN_ITERS:-1}" \
  --max-iters "${MAX_ITERS:-50}" \
  --native "${NATIVE_MODE:-subset}" \
  --sdpa "${SDPA_MODE:-none}" \
  --compare-backends "${COMPARE_BACKENDS:-all}" \
  --skip-metrics \
  --no-precompile-all \
  --no-shuffle \
  --fail-fast \
  --repo-root "${REPO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --csv-out "${CSV_OUT}" \
  --json-out "${JSON_OUT}" \
  "$@"

summary_args=(
  "${CSV_OUT}"
  --max-regret "${MAX_AUTO_REGRET:-1.05}"
  --top "${SUMMARY_TOP:-60}"
)
if [[ -n "${BASELINE_CSV:-}" ]]; then
  summary_args+=(--baseline "${BASELINE_CSV}")
fi
if [[ -n "${FAIL_ABOVE_REGRET:-}" ]]; then
  summary_args+=(--fail-above-regret "${FAIL_ABOVE_REGRET}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_indexed_sm90_iteration.py" "${summary_args[@]}"

echo
printf 'JSON: %s\nCSV:  %s\n' "${JSON_OUT}" "${CSV_OUT}"
