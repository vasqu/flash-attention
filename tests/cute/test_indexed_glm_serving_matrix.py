from pathlib import Path

from flash_attn.cute.indexed_policy import IndexedPath, choose_indexed_plan


def _glm(*, batch: int, q: int, topk: int):
    return choose_indexed_plan(
        batch_size=batch,
        query_length=q,
        kv_length=q,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        topk=topk,
        backend="auto",
    )


def test_measured_glm_batch_prefill_promotion_matrix():
    assert _glm(batch=1, q=512, topk=256).path is IndexedPath.FA4_BITMASK_INDEXED
    assert _glm(batch=2, q=512, topk=256).path is IndexedPath.FA4_BITMASK_INDEXED
    for batch in (4, 8, 16):
        assert _glm(batch=batch, q=512, topk=256).path is IndexedPath.BLOCK_SPARSE_INDEXED

    assert _glm(batch=1, q=1024, topk=512).path is IndexedPath.FA4_BITMASK_INDEXED
    for batch in (2, 4, 8, 16):
        assert _glm(batch=batch, q=1024, topk=512).path is IndexedPath.BLOCK_SPARSE_INDEXED

    for q, topk in ((2048, 512), (4096, 2048)):
        for batch in (1, 2, 4, 8):
            assert _glm(batch=batch, q=q, topk=topk).path is IndexedPath.BLOCK_SPARSE_INDEXED
        assert _glm(batch=16, q=q, topk=topk).path is not IndexedPath.BLOCK_SPARSE_INDEXED


def test_unmeasured_batches_and_non_glm_shapes_remain_general():
    assert _glm(batch=3, q=512, topk=256).path is not IndexedPath.BLOCK_SPARSE_INDEXED
    assert _glm(batch=3, q=1024, topk=512).path is IndexedPath.BLOCK_SPARSE_INDEXED
    non_glm = choose_indexed_plan(
        batch_size=8,
        query_length=2048,
        kv_length=2048,
        query_heads=32,
        kv_heads=32,
        qk_head_dim=128,
        value_head_dim=128,
        topk=1024,
        backend="auto",
    )
    assert non_glm.path is not IndexedPath.BLOCK_SPARSE_INDEXED
    assert non_glm.path is not IndexedPath.FA4_BITMASK_INDEXED


def test_explicit_sparse_accepts_measured_short_glm_shape():
    plan = choose_indexed_plan(
        batch_size=16,
        query_length=512,
        kv_length=512,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        topk=256,
        backend="block_sparse",
    )
    assert plan.path is IndexedPath.BLOCK_SPARSE_INDEXED


def test_unified_serving_matrix_and_runner_exist():
    root = Path(__file__).parents[2]
    benchmark = (root / "benchmarks" / "benchmark_indexed_sm90.py").read_text()
    runner = (root / "benchmarks" / "run_glm_moe_dsa_45_serving_matrix.sh").read_text()
    summary = (root / "benchmarks" / "summarize_glm_moe_dsa_45_serving_matrix.py").read_text()

    assert "def _glm_moe_dsa_45_serving_matrix_cases" in benchmark
    assert '"glm-moe-dsa-45-serving-matrix"' in benchmark
    assert "guard_deepseek_decode_b1" in benchmark
    assert "guard_mha_prefill_b4_s2048" in benchmark
    assert "guard_gqa_chunk_b4_q128" in benchmark
    assert "guard_mqa_chunk_b4_q128" in benchmark
    assert "--suite glm-moe-dsa-45-serving-matrix" in runner
    assert "GLM dispatch by batch" in summary
    assert "General-shape guardrails" in summary
