from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_BENCHMARK = Path(__file__).parents[2] / "benchmarks" / "benchmark_indexed_sm90.py"
_SPEC = spec_from_file_location("benchmark_indexed_sm90", _BENCHMARK)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_rough_speedup_text_handles_faster_and_slower():
    assert _MODULE._rough_speedup_text(2.0, 1.0) == "~2.00x faster"
    assert _MODULE._rough_speedup_text(1.0, 2.0) == "~2.00x slower"


def test_regression_suite_contains_all_q_stride_failures():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("regression", single)
    shapes = {(case.batch, case.query_length) for case in cases}
    assert {(1, 4), (4, 4), (16, 4), (1, 8), (4, 8), (1, 16), (4, 16), (1, 64)} <= shapes


def test_speedup_summary_groups_by_selected_path():
    results = [
        {
            "status": "ok",
            "case": {"name": "a", "category": "decode"},
            "plan": {"path": "warp_decode"},
            "derived": {"indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e": 4.0},
        },
        {
            "status": "ok",
            "case": {"name": "b", "category": "prefill"},
            "plan": {"path": "dense_indexed"},
            "derived": {"indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e": 1.0},
        },
    ]
    summary = _MODULE.summarize_exact_speedups(results)
    assert summary["cases"] == 2
    assert summary["faster_cases"] == 1
    assert summary["parity_cases"] == 1
    assert summary["slower_cases"] == 0
    assert summary["by_path"]["warp_decode"]["geomean_speedup"] == 4.0


def test_fa4_namespace_loader_bypasses_parent_init(tmp_path, monkeypatch):
    import importlib

    package = tmp_path / "flash_attn"
    cute = package / "cute"
    cute.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('parent init executed')\n")
    (cute / "__init__.py").write_text("\n")
    (cute / "interface.py").write_text("PROBE = 123\n")

    old_parent = sys.modules.pop("flash_attn", None)
    old_cute = sys.modules.pop("flash_attn.cute", None)
    old_interface = sys.modules.pop("flash_attn.cute.interface", None)
    try:
        info = _MODULE.ensure_fa4_namespace(tmp_path)
        imported = importlib.import_module("flash_attn.cute.interface")
        assert imported.PROBE == 123
        assert info["bypassed_parent_init"] is True
        assert info["flash_attn_2_cuda_required"] is False
    finally:
        for name in ("flash_attn.cute.interface", "flash_attn.cute", "flash_attn"):
            sys.modules.pop(name, None)
        if old_parent is not None:
            sys.modules["flash_attn"] = old_parent
        if old_cute is not None:
            sys.modules["flash_attn.cute"] = old_cute
        if old_interface is not None:
            sys.modules["flash_attn.cute.interface"] = old_interface


def test_fa4_discovery_accepts_package_directory(tmp_path):
    package = tmp_path / "flash_attn"
    (package / "cute").mkdir(parents=True)
    (package / "cute" / "interface.py").write_text("\n")
    found = _MODULE._discover_fa4_package_dirs(package)
    assert package.resolve() in found


def test_odd_suite_contains_non_aligned_lengths():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("odd", single)
    assert any(case.kv_length == 2500 for case in cases)
    assert any(case.query_length == 65 for case in cases)
    assert all(case.topk <= case.kv_length for case in cases)
    assert any(case.topk % 2 == 1 for case in cases)


def test_random_suite_is_seeded_and_awkward():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    first = _MODULE.build_suite(
        "random", single, seed=7, random_cases=12, random_max_k=2501
    )
    second = _MODULE.build_suite(
        "random", single, seed=7, random_cases=12, random_max_k=2501
    )
    assert [case.to_dict() for case in first] == [case.to_dict() for case in second]
    assert len(first) == 12
    assert all(case.topk <= case.kv_length <= 2501 for case in first)
    assert any(case.query_length not in (1, 4, 8, 16, 64, 128) for case in first)


def test_union_candidates_do_not_change_auto_policy():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("union-candidates", single)
    assert cases
    assert all(case.category == "union-candidate" for case in cases)


def test_speedup_summary_uses_two_percent_parity_band():
    results = [
        {
            "status": "ok",
            "case": {"name": "near", "category": "prefill"},
            "plan": {"path": "dense_indexed"},
            "derived": {"indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e": 0.99},
        },
        {
            "status": "ok",
            "case": {"name": "slow", "category": "prefill"},
            "plan": {"path": "dense_indexed"},
            "derived": {"indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e": 0.95},
        },
    ]
    summary = _MODULE.summarize_exact_speedups(results)
    assert summary["parity_cases"] == 1
    assert summary["slower_cases"] == 1


def test_topk_sweep_contains_boundary_and_near_dense_widths():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("topk-sweep", single)
    assert len(cases) >= 30
    widths = {case.topk for case in cases}
    assert {1, 17, 73, 257, 513, 1025} <= widths
    assert any(case.topk == case.kv_length - 1 for case in cases)
    assert any(case.query_heads == 64 and case.kv_heads == 1 for case in cases)


def test_random_suite_covers_extended_head_dimensions():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite(
        "random", single, seed=123, random_cases=128, random_max_k=5003
    )
    dims = {case.head_dim for case in cases} | {case.value_head_dim for case in cases}
    assert {64, 96, 128, 192, 256} <= dims
    assert any(case.kv_length == 5003 for case in cases)


def test_sdpa_mask_is_broadcastable_and_exact():
    import torch

    indices = torch.tensor(
        [[[3, 1, 7, -1], [0, -1, 2, 0]]], dtype=torch.int32
    )
    mask = _MODULE.build_sdpa_topk_mask(indices, 8)
    assert mask.shape == (1, 1, 2, 8)
    assert mask.dtype is torch.bool
    assert mask[0, 0, 0].nonzero().flatten().tolist() == [1, 3, 7]
    assert mask[0, 0, 1].nonzero().flatten().tolist() == [0, 2]


def test_sdpa_subset_and_memory_cap():
    case = _MODULE.BenchmarkCase(
        "odd", "odd-shape", 1, 65, 2501, 257, query_heads=32, kv_heads=8
    )
    required = _MODULE._sdpa_mask_bytes(case)
    assert _MODULE.should_run_sdpa(case, "subset", "odd", required)
    assert not _MODULE.should_run_sdpa(case, "all", "odd", required - 1)


def test_random_suite_excludes_known_unsafe_dense_fa4_corner():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite(
        "random", single, seed=0, random_cases=256, random_max_k=8192
    )
    assert not any(
        case.head_dim == 64
        and case.value_head_dim == 256
        for case in cases
    )
    assert _MODULE._known_unsafe_dense_fa4_random_shape(
        query_length=1, head_dim=64, value_head_dim=256
    )


def test_forced_union_benchmark_stays_in_supported_cross_query_domain():
    case = _MODULE.BenchmarkCase(
        "tiny_q", "random-shape", 1, 7, 509, 127,
        query_heads=64, kv_heads=1, head_dim=64, value_head_dim=64,
    )
    aligned = _MODULE.BenchmarkCase(
        "aligned", "random-shape", 1, 32, 509, 127,
        query_heads=64, kv_heads=1, head_dim=64, value_head_dim=64,
    )
    non_aligned = _MODULE.BenchmarkCase(
        "non_aligned", "random-shape", 1, 64, 509, 127,
        query_heads=64, kv_heads=1, head_dim=96, value_head_dim=96,
    )
    assert not _MODULE._union_benchmark_eligible(case)
    assert _MODULE._union_benchmark_eligible(aligned)
    assert not _MODULE._union_benchmark_eligible(non_aligned)


def test_sdpa_speedup_summary():
    results = [
        {
            "status": "ok",
            "case": {"name": "a", "category": "decode"},
            "plan": {"path": "warp_decode"},
            "derived": {"indexed_e2e_speedup_vs_torch_sdpa_topk_exact_e2e": 3.0},
        }
    ]
    summary = _MODULE.summarize_sdpa_speedups(results)
    assert summary["cases"] == 1
    assert summary["faster_cases"] == 1
    assert abs(summary["geomean_speedup"] - 3.0) < 1e-12


def test_masked_sdpa_matches_eager_gqa_reference_on_cpu():
    import math
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    batch, q_len, k_len, hq, hkv, dim, value_dim = 1, 3, 7, 4, 2, 8, 6
    q = torch.randn(batch, hq, q_len, dim)
    k = torch.randn(batch, hkv, k_len, dim)
    v = torch.randn(batch, hkv, k_len, value_dim)
    indices = torch.tensor([[[0, 3, 6], [1, 2, 5], [0, 4, 6]]], dtype=torch.int32)
    mask = _MODULE.build_sdpa_topk_mask(indices, k_len)

    actual = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, enable_gqa=True
    )
    repeat = hq // hkv
    k_expanded = k.repeat_interleave(repeat, dim=1)
    v_expanded = v.repeat_interleave(repeat, dim=1)
    scores = torch.matmul(q, k_expanded.transpose(-1, -2)) / math.sqrt(dim)
    scores = scores.masked_fill(~mask, float("-inf"))
    expected = torch.matmul(torch.softmax(scores, dim=-1), v_expanded)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_prefill_focus_suite_covers_gqa_sdpa_and_mqa_crossovers():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("prefill-focus", single)
    assert len(cases) == 36
    assert any(case.query_length == 64 and case.kv_heads == 8 for case in cases)
    assert any(case.query_length == 512 and case.kv_heads == 8 for case in cases)
    assert any(case.query_heads == 64 and case.kv_heads == 1 for case in cases)
    assert any(case.query_heads == 64 and case.kv_heads == 8 for case in cases)
    assert any(case.query_heads == case.kv_heads == 32 for case in cases)
    assert {2048, 4096, 8192} <= {
        case.topk for case in cases if case.query_heads == 64 and case.kv_heads == 1
    }


def test_summary_reports_grouped_prefill_experiments():
    summary_path = Path(__file__).parents[2] / "benchmarks" / "summarize_indexed_sm90_validation.py"
    spec = spec_from_file_location("summarize_indexed_sm90_validation", summary_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    payloads = [
        {
            "_path": "prefill.json",
            "suite": "prefill-focus",
            "results": [
                {
                    "status": "ok",
                    "case": {"name": "gqa"},
                    "timings": {
                        "indexed_warp_end_to_end": {"median_ms": 1.0},
                        "indexed_warp_scalar_end_to_end": {"median_ms": 3.0},
                        "indexed_union_end_to_end": {"median_ms": 1.2},
                    },
                }
            ],
        }
    ]
    result = module.summarize_kernel_experiments(payloads)
    assert abs(result["grouped_geomean"] - 3.0) < 1e-12


def test_iteration_suite_covers_policy_crossovers_and_regressions():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("iteration", single)
    names = {case.name for case in cases}
    assert len(cases) == 13
    assert {
        "iter_decode_row",
        "iter_spec_dense_b16_q8",
        "iter_spec_row_b32_q8",
        "iter_union_identical",
        "iter_union_random_control",
        "iter_regress_wide_short_near_dense",
    } <= names
    identical = next(case for case in cases if case.name == "iter_union_identical")
    random_control = next(case for case in cases if case.name == "iter_union_random_control")
    assert identical.union_compute_inflation_hint == 1.0
    assert random_control.union_compute_inflation_hint == 16.0


def test_all_suite_actually_contains_extended_suites():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    common = _MODULE.build_suite("common", single, random_cases=4)
    all_cases = _MODULE.build_suite("all", single, random_cases=4)
    categories = {case.category for case in all_cases}
    assert len(all_cases) > len(common)
    assert {"odd-shape", "random-shape", "topk-sweep", "union-candidate"} <= categories
    assert any(case.batch == 16 and case.query_length == 4 for case in all_cases)


def test_pattern_suite_supplies_content_hints_without_changing_indices():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("patterns", single)
    by_name = {case.name: case for case in cases}
    assert by_name["pattern_union_identical_b1_q128_k32768_t2048"].union_compute_inflation_hint == 1.0
    assert by_name["pattern_union_random_b1_q128_k32768_t2048"].union_compute_inflation_hint == 16.0


def test_iteration_summary_finds_auto_regret(tmp_path):
    summary_path = Path(__file__).parents[2] / "benchmarks" / "summarize_indexed_sm90_iteration.py"
    spec = spec_from_file_location("summarize_indexed_sm90_iteration", summary_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = [
        {
            "name": "union-miss",
            "dtype": "bf16",
            "plan_path": "dense_indexed",
            "indexed_auto_end_to_end_median_ms": "2.0",
            "indexed_dense_end_to_end_median_ms": "2.0",
            "indexed_union_end_to_end_median_ms": "1.0",
            "indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e": "3.0",
        }
    ]
    result = module.summarize_rows(rows)
    assert result[0]["best_backend"] == "union"
    assert result[0]["regret"] == 2.0


def test_iteration_medium_suite_is_bounded_and_2k_centered():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("iteration-medium", single)
    names = {case.name for case in cases}
    categories = {case.category for case in cases}
    assert len(cases) == 58
    assert sum(case.topk == 2048 for case in cases) == 46
    assert {
        "iteration-2k-decode-context",
        "iteration-2k-decode-batch",
        "iteration-2k-speculative",
        "iteration-2k-prefill",
        "iteration-2k-prefill-context",
        "iteration-2k-head-layout",
        "iteration-2k-model-proxy",
        "iteration-2k-density-control",
    } <= categories
    assert {
        "iter2k_decode_context_k131072",
        "iter2k_decode_batch_b128",
        "iter2k_spec_b64_q4",
        "iter2k_prefill_b4_q512",
        "iter2k_prefill_context_q128_k131072",
        "iter2k_prefill_context_b2_q128_k65536",
        "iter2k_prefill_context_b4_q64_k65536",
        "iter2k_prefill_context_b1_q512_k65536",
        "iter2k_prefill_context_heads_h64_8",
        "iter2k_prefill_context_heads_h32_32",
        "iter2k_deepseek_v32_decode_long",
        "iter2k_glm_moe_dsa_decode_long",
    } <= names
    deepseek = next(case for case in cases if case.name == "iter2k_deepseek_v32_decode_long")
    glm = next(case for case in cases if case.name == "iter2k_glm_moe_dsa_decode_long")
    assert (deepseek.topk, deepseek.head_dim, deepseek.value_head_dim) == (2048, 192, 128)
    assert (glm.topk, glm.head_dim, glm.value_head_dim) == (2048, 256, 256)


def test_iteration_summary_clamps_noise_below_unit_regret():
    summary_path = Path(__file__).parents[2] / "benchmarks" / "summarize_indexed_sm90_iteration.py"
    spec = spec_from_file_location("summarize_indexed_sm90_iteration_noise", summary_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = [
        {
            "name": "same-kernel-noise",
            "dtype": "bf16",
            "plan_path": "row_sparse",
            "indexed_auto_end_to_end_median_ms": "0.99",
            "indexed_warp_end_to_end_median_ms": "1.0",
        }
    ]
    result = module.summarize_rows(rows)
    assert result[0]["best_backend"] == "row"
    assert result[0]["regret"] == 1.0


def test_iteration_summary_retains_category_and_topk():
    summary_path = Path(__file__).parents[2] / "benchmarks" / "summarize_indexed_sm90_iteration.py"
    spec = spec_from_file_location("summarize_indexed_sm90_iteration_grouped", summary_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    rows = [
        {
            "name": "decode-2k",
            "category": "iteration-2k-decode-context",
            "topk": "2048",
            "dtype": "bf16",
            "plan_path": "row_sparse",
            "indexed_auto_end_to_end_median_ms": "1.01",
            "indexed_warp_end_to_end_median_ms": "1.0",
            "indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e": "2.0",
        }
    ]
    result = module.summarize_rows(rows)
    assert result[0]["category"] == "iteration-2k-decode-context"
    assert result[0]["topk"] == 2048.0


def test_validation_summary_distinguishes_dtypes():
    summary_path = Path(__file__).parents[2] / "benchmarks" / "summarize_indexed_sm90_validation.py"
    spec = spec_from_file_location("summarize_indexed_sm90_validation_dtype", summary_path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.payload_label(
        {
            "suite": "common",
            "configuration": {"dtype": "bf16"},
            "_path": "common_bf16.json",
        }
    ) == "common/bf16"
    assert module.payload_label(
        {
            "suite": "random",
            "configuration": {"dtype": "bf16"},
            "_path": "random_shapes_bf16.json",
        }
    ) == "random-shapes/bf16"
    assert module.payload_label(
        {
            "suite": "random",
            "configuration": {"dtype": "fp16"},
            "_path": "random_all_backends_fp16.json",
        }
    ) == "random-all-backends/fp16"
    label = module.case_label(
        {
            "name": "case",
            "dtype": "fp16",
            "path": "row_sparse",
            "batch": 1,
            "q": 1,
            "k": 32768,
            "topk": 2048,
            "hq": 32,
            "hkv": 8,
            "d": 128,
            "dv": 128,
        }
    )
    assert "[fp16, row_sparse]" in label
    dispatch = module.summarize_auto_dispatch(
        [
            {
                "suite": "common",
                "configuration": {"dtype": "bf16"},
                "_path": "common_bf16.json",
                "results": [
                    {
                        "status": "ok",
                        "dtype": "bf16",
                        "case": {
                            "name": "policy-miss",
                            "category": "test",
                            "batch": 1,
                            "query_length": 128,
                            "kv_length": 65536,
                            "topk": 2048,
                            "query_heads": 32,
                            "kv_heads": 8,
                            "head_dim": 128,
                            "value_head_dim": 128,
                        },
                        "plan": {"path": "dense_indexed"},
                        "timings": {
                            "indexed_auto_end_to_end": {"median_ms": 2.0},
                            "indexed_dense_end_to_end": {"median_ms": 2.0},
                            "indexed_warp_end_to_end": {"median_ms": 1.0},
                        },
                    }
                ],
            }
        ],
        1.03,
    )
    assert dispatch["max"] == 2.0
    assert dispatch["misses"][0]["best_backend"] == "row"


def test_causal_prefill_patterns_are_prefix_valid_unique_and_padded():
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    for pattern in ("causal-window", "causal-mixed"):
        indices = _MODULE.make_indices(
            pattern=pattern,
            batch=2,
            query_length=8,
            kv_length=8,
            topk=4,
            tail=2,
            device=torch.device("cpu"),
            generator=generator,
        )
        assert indices.shape == (2, 8, 4)
        for batch in range(2):
            for query in range(8):
                row = indices[batch, query]
                valid = row[row >= 0]
                assert valid.numel() == min(4, query + 1)
                assert valid.unique().numel() == valid.numel()
                assert int(valid.max()) <= query


def test_serving_medium_suite_prioritizes_decode_and_same_length_prefill():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("serving-medium", single)
    categories = {case.category for case in cases}
    assert len(cases) == 49
    assert sum(case.topk == 2048 for case in cases) == 36
    assert sum(case.query_length == 1 for case in cases) == 21
    assert sum(case.query_length == case.kv_length for case in cases) == 19
    assert sum(
        case.query_length == 1 or case.query_length == case.kv_length
        for case in cases
    ) == 40
    assert categories == {"serving-decode", "serving-prefill", "serving-guardrail"}
    assert all(
        case.pattern in ("causal-window", "causal-mixed", "random")
        for case in cases
        if case.category == "serving-prefill"
    )


def test_dsa_model_suite_is_decode_and_true_prefill_only():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("dsa-models", single)
    assert len(cases) == 24
    assert {case.category for case in cases} == {
        "dsa-model-decode",
        "dsa-model-prefill",
    }
    assert sum(case.query_length == 1 for case in cases) == 12
    assert sum(case.query_length == case.kv_length for case in cases) == 12
    assert all(
        case.query_length == 1 or case.query_length == case.kv_length
        for case in cases
    )
    assert {
        (case.query_heads, case.kv_heads, case.head_dim, case.value_head_dim)
        for case in cases
    } == {
        (128, 128, 192, 128),
        (64, 64, 256, 256),
    }
    assert sum(case.topk == 2048 for case in cases) == 16


def test_dsa_model_prefill_patterns_are_prefix_valid():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("dsa-models", single)
    prefill = [case for case in cases if case.category == "dsa-model-prefill"]
    assert prefill
    assert all(case.query_length == case.kv_length for case in prefill)
    assert all(case.pattern in ("causal-mixed", "causal-window") for case in prefill)


def test_glm_moe_dsa_45_profile_matches_supplied_config():
    profile = _MODULE.model_profile_for_suite("glm-moe-dsa-45")
    assert profile is not None
    assert profile.num_hidden_layers == 45
    assert profile.num_attention_heads == profile.num_key_value_heads == 64
    assert profile.qk_nope_head_dim == 256
    assert profile.qk_rope_head_dim == 0
    assert profile.qk_head_dim == profile.v_head_dim == 256
    assert profile.q_lora_rank == 1536
    assert profile.kv_lora_rank == 512
    assert profile.n_routed_experts == 288
    assert profile.num_experts_per_tok == 8
    assert profile.index_topk == 2048


def test_glm_moe_dsa_45_suite_is_decode_and_true_prefill_only():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("glm-moe-dsa-45", single)
    assert len(cases) == 23
    assert sum(case.query_length == 1 for case in cases) == 11
    assert sum(case.query_length == case.kv_length for case in cases) == 12
    assert all(
        case.query_length == 1 or case.query_length == case.kv_length
        for case in cases
    )
    assert {
        (case.query_heads, case.kv_heads, case.head_dim, case.value_head_dim)
        for case in cases
    } == {(64, 64, 256, 256)}
    assert all(not case.benchmark_union for case in cases)
    assert sum(case.topk == 2048 for case in cases) == 13


def test_glm_moe_dsa_45_prefill_patterns_are_prefix_valid():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("glm-moe-dsa-45", single)
    prefill = [case for case in cases if case.category == "glm45-prefill"]
    assert prefill
    assert all(case.pattern in ("causal-mixed", "causal-window") for case in prefill)


def test_row_dense_compare_mode_excludes_union():
    case = _MODULE.BenchmarkCase(
        "wide",
        "test",
        1,
        4096,
        4096,
        2048,
        query_heads=64,
        kv_heads=64,
        head_dim=256,
        value_head_dim=256,
    )
    assert _MODULE.should_compare_backends(case, "row-dense")
    assert not _MODULE.should_compare_union(case, "row-dense")


def test_glm_overnight_suite_is_model_specific_and_bounded():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    cases = _MODULE.build_suite("glm-moe-dsa-45-overnight", single)
    assert len(cases) == 61
    assert sum("decode" in case.category for case in cases) == 15
    assert sum("prefill" in case.category for case in cases) == 46
    assert all(case.query_heads == case.kv_heads == 64 for case in cases)
    assert all(case.head_dim == case.value_head_dim == 256 for case in cases)
    assert max(
        case.batch * case.kv_length
        for case in cases
        if "decode" in case.category
    ) <= 524288


def test_common_5h_suite_expands_profiles_and_paired_crossovers():
    single = _MODULE.BenchmarkCase("single", "single", 1, 1, 32768, 2048)
    stable = _MODULE.build_suite("common-indexed-3h", single)
    expanded = _MODULE.build_suite("common-indexed-5h", single)
    assert len(stable) == 252
    assert len(expanded) == 420

    profiles = {
        (
            case.query_heads,
            case.kv_heads,
            case.head_dim,
            case.value_head_dim,
        )
        for case in expanded
    }
    assert (32, 16, 128, 128) in profiles
    assert (28, 4, 128, 128) in profiles
    assert (32, 32, 96, 96) in profiles

    gqa8_prefill = [
        case
        for case in expanded
        if case.category == "common5h-gqa8-prefill"
    ]
    paired = {
        (case.batch, case.query_length, case.topk, case.pattern)
        for case in gqa8_prefill
    }
    assert (8, 512, 256, "causal-mixed") in paired
    assert (16, 512, 256, "causal-window") in paired
    assert (4, 1024, 512, "causal-mixed") in paired
    assert (16, 1024, 512, "causal-window") in paired
    assert (16, 2048, 1024, "causal-mixed") in paired
