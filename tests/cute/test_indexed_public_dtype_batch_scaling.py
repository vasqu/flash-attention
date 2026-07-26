from pathlib import Path

from flash_attn.cute.indexed_policy import IndexedPath, choose_indexed_plan


def _plan(*, batch: int, q: int, topk: int, backend: str = "auto"):
    return choose_indexed_plan(
        batch_size=batch,
        query_length=q,
        kv_length=q,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        topk=topk,
        backend=backend,
    )


def test_public_dtype_packer_avoids_specialized_path_cast():
    root = Path(__file__).parents[2]
    packer = (root / "flash_attn" / "cute" / "indexed_bitmask_sm90.py").read_text()
    interface = (root / "flash_attn" / "cute" / "interface.py").read_text()

    assert "torch.int32, torch.int64, torch.uint16" in packer
    assert "index_align = max(4, indices.element_size())" in packer
    block = interface[interface.index("if is_block_sparse_indexed:"):]
    special = block[: block.index("elif is_dense_indexed:")]
    assert special.count("kernel_indices = gather_kv_indices") == 2
    assert "cast_indexed_kv_indices" not in special


def test_large_batch_backends_follow_measured_promotion():
    assert _plan(batch=4, q=2048, topk=512).path is IndexedPath.BLOCK_SPARSE_INDEXED
    assert _plan(batch=8, q=2048, topk=512, backend="block_sparse").path is IndexedPath.BLOCK_SPARSE_INDEXED
    assert _plan(batch=16, q=512, topk=256).path is IndexedPath.BLOCK_SPARSE_INDEXED
    assert _plan(batch=16, q=512, topk=256, backend="fa4_bitmask").path is IndexedPath.FA4_BITMASK_INDEXED


def test_batch_suite_and_byte_bounded_workspaces_exist():
    root = Path(__file__).parents[2]
    benchmark = (root / "benchmarks" / "benchmark_indexed_sm90.py").read_text()
    runner = (root / "benchmarks" / "run_glm_moe_dsa_45_batch_overnight.sh").read_text()
    short = (root / "flash_attn" / "cute" / "indexed_bitmask.py").read_text()
    sparse = (root / "flash_attn" / "cute" / "indexed_block_sparse.py").read_text()

    assert "def _glm_moe_dsa_45_batch_cases" in benchmark
    assert '"glm-moe-dsa-45-batch"' in benchmark
    assert "for batch in (1, 2, 4, 8, 16)" in benchmark
    assert "bitmask_build_cute_int32_only" in benchmark
    assert "indexed_index_cast_only" in benchmark
    assert "--suite glm-moe-dsa-45-batch" in runner
    assert "FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB" in short
    assert "FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB" in sparse
    assert "indexed_bitmask_workspace_cache_stats" in short
    assert "indexed_block_sparse_workspace_cache_stats" in sparse
