from pathlib import Path

from flash_attn.cute.indexed_policy import (
    IndexedPath,
    choose_indexed_plan,
    clear_indexed_plan_cache,
    indexed_plan_cache_info,
)
from benchmarks.summarize_indexed_common_3h import _prefill_certificates


def _plan(**kwargs):
    defaults = dict(
        batch_size=1,
        query_length=1024,
        kv_length=1024,
        query_heads=32,
        kv_heads=32,
        qk_head_dim=128,
        value_head_dim=128,
        topk=512,
        backend="auto",
    )
    defaults.update(kwargs)
    return choose_indexed_plan(**defaults)


def test_scalar_plan_cache_hits_repeated_serving_shape():
    clear_indexed_plan_cache()
    first = _plan()
    before = indexed_plan_cache_info()
    second = _plan()
    after = indexed_plan_cache_info()
    assert first is second
    assert after.hits == before.hits + 1


def test_measured_common_prefill_profiles_are_auto_promoted():
    assert _plan().path is IndexedPath.FA4_BITMASK_INDEXED
    assert _plan(query_length=2048, kv_length=2048).path is IndexedPath.BLOCK_SPARSE_INDEXED
    assert _plan(query_heads=64, kv_heads=8).path is IndexedPath.FA4_BITMASK_INDEXED
    # Ratio-2 GQA is in the expanded qualification suite, not production.
    assert _plan(query_heads=32, kv_heads=16).path is IndexedPath.DENSE_INDEXED
    assert _plan(backend="fa4_bitmask_general").path is IndexedPath.FA4_BITMASK_INDEXED
    assert _plan(backend="block_sparse_general").path is IndexedPath.BLOCK_SPARSE_INDEXED


def test_general_sparse_supports_gqa_and_asymmetric_dims_explicitly():
    plan = _plan(
        query_length=2048,
        kv_length=2048,
        query_heads=32,
        kv_heads=8,
        qk_head_dim=192,
        value_head_dim=128,
        topk=512,
        backend="block_sparse_general",
    )
    assert plan.path is IndexedPath.BLOCK_SPARSE_INDEXED


def test_hot_workspace_cache_sources_use_weak_references():
    root = Path(__file__).parents[2]
    for rel in (
        "flash_attn/cute/indexed_bitmask.py",
        "flash_attn/cute/indexed_block_sparse.py",
    ):
        source = (root / rel).read_text()
        assert "_workspace_hot = local()" in source
        assert "weakref.ref(workspace)" in source
        assert '"hot_hits"' in source


def test_common_3h_suite_and_runner_are_long_form():
    root = Path(__file__).parents[2]
    benchmark = (root / "benchmarks/benchmark_indexed_sm90.py").read_text()
    runner = (root / "benchmarks/run_indexed_common_3h.sh").read_text()
    expanded_runner = (root / "benchmarks/run_indexed_common_5h.sh").read_text()
    summary = (root / "benchmarks/summarize_indexed_common_3h.py").read_text()
    assert "def _common_indexed_3h_cases" in benchmark
    assert "def _common_indexed_5h_cases" in benchmark
    assert '"common-indexed-3h"' in benchmark
    assert '"common-indexed-5h"' in benchmark
    assert 'ROUNDS="${ROUNDS:-12}"' in runner
    assert 'TARGET_ROUND_MS="${TARGET_ROUND_MS:-1000}"' in runner
    assert "--compare-backends all" in runner
    assert 'ROUNDS="${ROUNDS:-12}"' in expanded_runner
    assert 'TARGET_ROUND_MS="${TARGET_ROUND_MS:-1000}"' in expanded_runner
    assert "--suite common-indexed-5h" in expanded_runner
    assert "Expanded 420-case promotion" in benchmark
    assert "Largest remaining opportunities" in summary
    assert "Prefill candidate certificates" in summary
    assert "Ratio-8 GQA now joins" in summary


def test_prefill_certificate_enforces_gain_floor_and_correctness():
    def row(name: str, candidate_ms: float, max_abs: float = 0.01):
        return {
            "case": {
                "name": name,
                "category": "common5h-gqa2-prefill",
                "query_length": 1024,
                "kv_length": 1024,
            },
            "plan": {"path": "dense_indexed"},
            "timings": {
                "indexed_auto_end_to_end": {"median_ms": 2.0},
                "fa4_topk_cute_128x64_end_to_end": {
                    "median_ms": candidate_ms
                },
            },
            "correctness": {
                "indexed_auto_vs_fa4_topk_cute_128x64": {
                    "max_abs": max_abs,
                    "mean_abs": 0.001,
                }
            },
        }

    qualifies = _prefill_certificates([row("a", 1.0), row("b", 1.1)])
    bitmask = next(item for item in qualifies if item[3] == "native bitmask")
    assert bitmask[-1] == "QUALIFIES"

    regression = _prefill_certificates([row("a", 1.0), row("b", 2.1)])
    bitmask = next(item for item in regression if item[3] == "native bitmask")
    assert bitmask[-1] == "REGRESSES"

    bad_correctness = _prefill_certificates([row("a", 1.0, max_abs=0.1)])
    bitmask = next(item for item in bad_correctness if item[3] == "native bitmask")
    assert bitmask[-1] == "CORRECTNESS_FAIL"
