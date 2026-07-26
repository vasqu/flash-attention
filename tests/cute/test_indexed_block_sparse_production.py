from pathlib import Path

from flash_attn.cute.indexed_policy import IndexedPath, choose_indexed_plan


def test_production_sparse_module_is_native_cute_and_cached():
    root = Path(__file__).parents[2]
    source = (root / "flash_attn" / "cute" / "indexed_block_sparse.py").read_text()
    interface = (root / "flash_attn" / "cute" / "interface.py").read_text()
    assert "triton" not in source.lower()
    assert "build_indexed_block_sparse_tensors_cute_cached" in source
    assert "current_stream" in source
    assert "_MAX_CACHED_WORKSPACES = 8" in source
    assert "indexed_block_sparse_lab" not in interface
    assert "topk_bitmask_score_mod" in interface
    assert "block_sparse_tensors=sparse_blocks" in interface


def test_production_policy_keeps_short_prefill_dense():
    common = dict(
        batch_size=1,
        kv_length=1024,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        topk=256,
        sm_count=132,
    )
    short = choose_indexed_plan(query_length=1024, **common)
    long = choose_indexed_plan(
        query_length=2048,
        **{**common, "kv_length": 2048, "topk": 512},
    )
    assert short.path is IndexedPath.DENSE_INDEXED
    assert long.path is IndexedPath.BLOCK_SPARSE_INDEXED
