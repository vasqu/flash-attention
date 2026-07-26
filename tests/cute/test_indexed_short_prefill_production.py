from pathlib import Path

from flash_attn.cute.indexed_policy import IndexedPath, choose_indexed_plan


def _plan(*, q: int, k: int | None = None, topk: int = 256, batch: int = 1):
    return choose_indexed_plan(
        batch_size=batch,
        query_length=q,
        kv_length=q if k is None else k,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        topk=topk,
        sm_count=132,
    )


def test_glm_prefill_uses_measured_three_regime_policy():
    unmeasured_tiny = _plan(q=256, topk=128)
    short_512 = _plan(q=512, topk=256)
    short_1024 = _plan(q=1024, topk=1024)
    long_2048 = _plan(q=2048, topk=512)

    assert unmeasured_tiny.path not in (
        IndexedPath.FA4_BITMASK_INDEXED, IndexedPath.BLOCK_SPARSE_INDEXED
    )
    assert short_512.path is IndexedPath.FA4_BITMASK_INDEXED
    assert short_1024.path is IndexedPath.FA4_BITMASK_INDEXED
    assert long_2048.path is IndexedPath.BLOCK_SPARSE_INDEXED


def test_short_fa4_specialization_is_exactly_gated():
    assert _plan(q=1024, topk=256, batch=2).path is IndexedPath.DENSE_INDEXED
    non_true_prefill = _plan(q=1024, k=2048, topk=256)
    assert non_true_prefill.path is not IndexedPath.FA4_BITMASK_INDEXED


def test_short_fa4_backend_alias_and_shape_validation():
    explicit = choose_indexed_plan(
        batch_size=1,
        query_length=1024,
        kv_length=1024,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        topk=256,
        backend="fa4_bitmask",
    )
    assert explicit.path is IndexedPath.FA4_BITMASK_INDEXED

    try:
        choose_indexed_plan(
            batch_size=1,
            query_length=2048,
            kv_length=2048,
            query_heads=64,
            kv_heads=64,
            qk_head_dim=256,
            value_head_dim=256,
            topk=512,
            backend="fa4_bitmask",
        )
    except ValueError as exc:
        assert "short GLM true prefill" in str(exc)
    else:
        raise AssertionError("expected short FA4 shape validation")


def test_short_prefill_workspace_is_native_cached_fa4():
    root = Path(__file__).parents[2]
    source = (root / "flash_attn" / "cute" / "indexed_bitmask.py").read_text()
    interface = (root / "flash_attn" / "cute" / "interface.py").read_text()

    assert "triton" not in source.lower()
    assert "build_topk_bitmask" in source
    assert "current_stream" in source
    assert "_MAX_CACHED_WORKSPACES = 8" in source
    assert "build_indexed_topk_bitmask_cached" in interface
    assert "IndexedPath.FA4_BITMASK_INDEXED" in interface
    assert "tile_mn=(128, 64)" in interface
    assert "score_mod=topk_bitmask_score_mod" in interface
