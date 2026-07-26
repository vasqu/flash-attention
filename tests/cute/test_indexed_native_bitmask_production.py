from pathlib import Path


def test_native_bitmask_kernel_is_single_launch_cute_atomic_or():
    root = Path(__file__).parents[2]
    source = (root / "flash_attn" / "cute" / "indexed_bitmask_sm90.py").read_text()

    assert "@cute.kernel" in source
    assert "cute.arch.atomic_or" in source
    assert ".scatter_add_(" not in source
    assert "torch.zeros" not in source
    assert "mBitmask.iterator" in source
    assert "get_jit_cache" in source


def test_short_and_sparse_production_paths_use_native_bitmask():
    root = Path(__file__).parents[2]
    short = (root / "flash_attn" / "cute" / "indexed_bitmask.py").read_text()
    sparse = (root / "flash_attn" / "cute" / "indexed_block_sparse.py").read_text()

    assert "build_topk_bitmask_cute" in short
    assert "build_topk_bitmask(" not in short
    assert "build_topk_bitmask_cute" in sparse
    assert "build_topk_bitmask(" not in sparse


def test_overnight_reports_native_bitmask_cost_and_candidate():
    root = Path(__file__).parents[2]
    benchmark = (root / "benchmarks" / "benchmark_indexed_sm90.py").read_text()
    summary = (
        root / "benchmarks" / "summarize_glm_moe_dsa_45_overnight.py"
    ).read_text()

    assert "bitmask_build_cute_only" in benchmark
    assert "fa4_topk_cute_128x64_end_to_end" in benchmark
    assert "fa4_cute_bitmask_128x64" in summary
    assert "Native CuTe build ms" in summary
