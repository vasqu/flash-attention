from flash_attn.cute.indexed_policy import (
    IndexedPath,
    choose_indexed_plan,
    normalize_indexed_backend,
)


def _plan(**overrides):
    args = dict(
        batch_size=1,
        query_length=1,
        kv_length=32768,
        query_heads=32,
        kv_heads=8,
        qk_head_dim=128,
        value_head_dim=128,
        topk=2048,
        sm_count=120,
    )
    args.update(overrides)
    return choose_indexed_plan(**args)


def test_q1_decode_uses_row_sparse_and_keeps_scalar_head_mapping():
    plan = _plan()
    assert plan.path is IndexedPath.ROW_SPARSE
    assert plan.decode_splits > 1
    assert plan.decode_rows_per_cta == 8
    assert plan.decode_heads_per_warp == 1


def test_one_sixteenth_density_q128_prefill_stays_dense_at_4096_head_rows():
    plan = _plan(query_length=128, kv_length=32768, topk=2048)
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_sparse_long_context_grouped_gqa_uses_rows_through_8192_head_rows():
    for overrides in (
        dict(query_length=128, kv_length=32768, topk=1024),
        dict(query_length=128, kv_length=32768, topk=1536),
        dict(query_length=128, kv_length=65536, topk=2048),
        dict(query_length=128, kv_length=131072, topk=2048),
        dict(query_length=256, kv_length=65536, topk=2048),
        dict(batch_size=2, query_length=128, kv_length=65536, topk=2048),
        dict(batch_size=4, query_length=64, kv_length=65536, topk=2048),
        dict(query_length=128, kv_length=65536, topk=2048, query_heads=64),
    ):
        plan = _plan(**overrides)
        assert plan.path is IndexedPath.ROW_SPARSE
        assert plan.decode_heads_per_warp > 1


def test_sparse_8192_row_rule_keeps_adjacent_dense_controls():
    for overrides in (
        dict(query_length=128, kv_length=16384, topk=2048),
        dict(query_length=128, kv_length=32768, topk=2048),
        dict(query_length=128, kv_length=32768, topk=3072),
        dict(query_length=512, kv_length=65536, topk=2048),
        dict(batch_size=4, query_length=128, kv_length=65536, topk=2048),
        dict(query_length=128, kv_length=65536, topk=2048, qk_head_dim=192),
    ):
        assert _plan(**overrides).path is IndexedPath.DENSE_INDEXED


def test_long_context_mha_q128_uses_rows_only_beyond_measured_k_crossover():
    short_context = _plan(
        query_length=128, kv_length=32768, topk=2048, kv_heads=32,
    )
    measured_long_context = _plan(
        query_length=128, kv_length=65536, topk=2048, kv_heads=32,
    )
    longer_context = _plan(
        query_length=128, kv_length=131072, topk=2048, kv_heads=32,
    )
    true_prefill = _plan(
        query_length=8192, kv_length=8192, topk=2048, kv_heads=32,
    )
    assert short_context.path is IndexedPath.DENSE_INDEXED
    assert measured_long_context.path is IndexedPath.ROW_SPARSE
    assert measured_long_context.decode_heads_per_warp == 1
    assert longer_context.path is IndexedPath.ROW_SPARSE
    assert true_prefill.path is IndexedPath.BLOCK_SPARSE_INDEXED


def test_small_and_batched_speculative_decode_use_grouped_rows():
    for overrides in (
        dict(query_length=4),
        dict(batch_size=4, query_length=16),
    ):
        plan = _plan(**overrides)
        assert plan.path is IndexedPath.ROW_SPARSE
        assert plan.decode_heads_per_warp == 4


def test_speculative_decode_crosses_to_dense_at_4096_head_rows():
    plan = _plan(batch_size=8, query_length=16)
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_total_head_rows_guard_high_batch_sparse_prefill():
    plan = _plan(batch_size=8, query_length=16, kv_length=65536, topk=2048)
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_saturated_short_query_batches_return_to_row_sparse():
    for batch_size, query_length in ((64, 4), (32, 8)):
        plan = _plan(batch_size=batch_size, query_length=query_length)
        assert plan.path is IndexedPath.ROW_SPARSE
        assert plan.decode_heads_per_warp == 4

    # The measured B16/Q8 point is still a small dense win.
    assert _plan(batch_size=16, query_length=8).path is IndexedPath.DENSE_INDEXED


def test_wide_near_dense_short_query_uses_dense_indexed():
    plan = _plan(
        batch_size=3,
        query_length=5,
        kv_length=5003,
        topk=5003,
        query_heads=64,
        kv_heads=8,
        qk_head_dim=96,
        value_head_dim=256,
    )
    assert plan.path is IndexedPath.DENSE_INDEXED

    # Nearby full-density speculative attention with ordinary D128/DV128 is
    # still a strong row-sparse win, as is the wide case below the long-topk gate.
    assert _plan(query_length=8, kv_length=4097, topk=4096).path is IndexedPath.ROW_SPARSE
    assert _plan(
        query_length=3,
        kv_length=8191,
        topk=2047,
        query_heads=16,
        kv_heads=4,
        qk_head_dim=192,
        value_head_dim=256,
    ).path is IndexedPath.ROW_SPARSE


def test_low_inflation_hint_selects_general_gqa_union():
    hinted = _plan(
        query_length=128,
        union_compute_inflation_hint=1.0,
    )
    random_control = _plan(
        query_length=128,
        union_compute_inflation_hint=16.0,
    )
    assert hinted.path is IndexedPath.PACKED_UNION_FA4
    assert random_control.path is IndexedPath.DENSE_INDEXED


def test_union_hint_does_not_override_shape_safety_or_validate_bad_values():
    assert _plan(
        query_length=31,
        union_compute_inflation_hint=1.0,
    ).path is IndexedPath.ROW_SPARSE
    assert _plan(
        query_length=128,
        qk_head_dim=96,
        value_head_dim=96,
        union_compute_inflation_hint=1.0,
    ).path is IndexedPath.DENSE_INDEXED
    for bad in (0.99, float("inf"), float("nan")):
        try:
            _plan(query_length=128, union_compute_inflation_hint=bad)
        except ValueError as exc:
            assert "finite and >= 1" in str(exc)
        else:
            raise AssertionError(f"expected invalid union hint {bad!r} to fail")


def test_large_wide_quarter_density_prefill_uses_dense_indexed():
    plan = _plan(
        batch_size=3,
        query_length=257,
        kv_length=509,
        topk=127,
        query_heads=64,
        kv_heads=8,
        qk_head_dim=192,
        value_head_dim=256,
    )
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_wide_near_dense_q1_decode_uses_dense_indexed():
    plan = _plan(
        query_length=1,
        kv_length=8191,
        topk=8190,
        query_heads=28,
        kv_heads=4,
        qk_head_dim=256,
        value_head_dim=192,
    )
    assert plan.path is IndexedPath.DENSE_INDEXED

    # Keep the proven ordinary D128 decode mapping unchanged, even at full K.
    assert _plan(query_length=1, kv_length=4097, topk=4096).path is IndexedPath.ROW_SPARSE

def test_backend_aliases_are_cute_only():
    assert normalize_indexed_backend(None) == "auto"
    assert normalize_indexed_backend("decode") == "row_sparse"
    assert normalize_indexed_backend("grouped-warp") == "row_grouped"
    assert normalize_indexed_backend("scalar-warp") == "row_scalar"
    assert normalize_indexed_backend("dense") == "dense_indexed"
    assert normalize_indexed_backend("block-sparse") == "block_sparse_indexed"
    assert normalize_indexed_backend("sparse-prefill") == "block_sparse_indexed"
    assert normalize_indexed_backend("wgmma") == "union_fa4"


def test_external_backend_aliases_are_rejected():
    for backend in ("native", "bitmask", "sdpa", "torch_sdpa", "direct", "direct_mqa"):
        try:
            normalize_indexed_backend(backend)
        except ValueError as exc:
            assert "unknown indexed backend" in str(exc)
        else:
            raise AssertionError(f"expected {backend!r} to be rejected")


def test_forced_cute_backends_are_respected():
    assert _plan(backend="warp").path is IndexedPath.ROW_SPARSE
    assert _plan(query_length=16, backend="warp_grouped").decode_heads_per_warp == 4
    assert _plan(query_length=16, backend="warp_scalar").decode_heads_per_warp == 1
    assert _plan(backend="dense").path is IndexedPath.DENSE_INDEXED
    assert _plan(
        query_length=4096,
        kv_length=4096,
        topk=2048,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        backend="block_sparse",
    ).path is IndexedPath.BLOCK_SPARSE_INDEXED
    assert _plan(query_length=64, backend="union").path is IndexedPath.PACKED_UNION_FA4


def test_forced_union_rejects_short_query_tiles():
    try:
        _plan(query_length=7, backend="union")
    except ValueError as exc:
        assert "query_length >= 32" in str(exc)
    else:
        raise AssertionError("expected short-Q forced union to be rejected")


def test_forced_union_rejects_non_64_aligned_dimensions():
    try:
        _plan(query_length=64, qk_head_dim=96, value_head_dim=96, backend="union")
    except ValueError as exc:
        assert "divisible by 64" in str(exc)
    else:
        raise AssertionError("expected non-64-aligned forced union to be rejected")


def test_dense_indexed_rejects_user_score_mod():
    try:
        _plan(backend="dense", has_score_mod=True)
    except ValueError as exc:
        assert "cannot compose" in str(exc)
    else:
        raise AssertionError("expected dense_indexed + score_mod to fail")


def test_auto_user_score_mod_uses_direct_index_coordinates():
    plan = _plan(query_length=128, has_score_mod=True)
    assert plan.path is IndexedPath.ROW_SPARSE


def test_short_odd_shapes_follow_measured_total_row_crossover():
    sparse = _plan(query_length=65, kv_length=2501, topk=257)
    batched = _plan(batch_size=2, query_length=127, kv_length=4097, topk=513)
    assert sparse.path is IndexedPath.ROW_SPARSE
    assert sparse.decode_heads_per_warp == 4
    assert batched.path is IndexedPath.DENSE_INDEXED


def test_small_mqa_prefill_uses_grouped_rows_before_union_setup():
    plan = _plan(
        query_length=129,
        kv_length=5003,
        topk=257,
        query_heads=64,
        kv_heads=1,
    )
    assert plan.path is IndexedPath.ROW_SPARSE


def test_high_ratio_mqa_middle_tiny_topk_uses_scalar_rows_only_in_measured_band():
    lower = _plan(
        query_length=129, kv_length=5003, topk=73,
        query_heads=64, kv_heads=1,
    )
    middle = _plan(
        query_length=129, kv_length=5003, topk=129,
        query_heads=64, kv_heads=1,
    )
    upper = _plan(
        query_length=129, kv_length=5003, topk=257,
        query_heads=64, kv_heads=1,
    )
    assert lower.path is IndexedPath.ROW_SPARSE
    assert lower.decode_heads_per_warp > 1
    assert middle.path is IndexedPath.ROW_SPARSE
    assert middle.decode_heads_per_warp == 1
    assert upper.path is IndexedPath.ROW_SPARSE
    assert upper.decode_heads_per_warp > 1


def test_high_ratio_mqa_decode_keeps_q1_row_kernel():
    plan = _plan(query_heads=64, kv_heads=1)
    assert plan.path is IndexedPath.ROW_SPARSE
    assert plan.decode_heads_per_warp == 1


def test_high_density_q64_prefill_uses_dense_indexed_cute():
    plan = _plan(query_length=64, kv_length=4097, topk=2049)
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_quarter_density_q64_prefill_stays_row_sparse_below_head_row_guard():
    plan = _plan(query_length=64, kv_length=4097, topk=1025)
    assert plan.path is IndexedPath.ROW_SPARSE


def test_small_speculative_decode_remains_row_sparse_even_near_dense():
    plan = _plan(query_length=8, kv_length=4097, topk=4096)
    assert plan.path is IndexedPath.ROW_SPARSE


def test_mqa_density_crossovers_stay_inside_cute():
    tiny = _plan(
        query_length=129, kv_length=5003, topk=129,
        query_heads=64, kv_heads=1,
    )
    medium = _plan(
        query_length=129, kv_length=5003, topk=513,
        query_heads=64, kv_heads=1,
    )
    denser = _plan(
        query_length=129, kv_length=5003, topk=1025,
        query_heads=64, kv_heads=1,
    )
    assert tiny.path is IndexedPath.ROW_SPARSE
    assert medium.path is IndexedPath.PACKED_UNION_FA4
    assert denser.path is IndexedPath.DENSE_INDEXED


def test_batched_mqa_union_path_covers_measured_one_sixteenth_density():
    plan = _plan(
        batch_size=4,
        query_length=256,
        kv_length=65536,
        topk=4096,
        query_heads=64,
        kv_heads=1,
    )
    assert plan.path is IndexedPath.PACKED_UNION_FA4


def test_mqa_one_eighth_and_one_quarter_density_use_dense_indexed():
    for topk in (4096, 8192):
        plan = _plan(
            query_length=128,
            kv_length=32768,
            topk=topk,
            query_heads=64,
            kv_heads=1,
        )
        assert plan.path is IndexedPath.DENSE_INDEXED


def test_non_64_aligned_mqa_dimensions_use_grouped_rows():
    plan = _plan(
        query_length=129,
        kv_length=5003,
        topk=257,
        query_heads=64,
        kv_heads=1,
        qk_head_dim=96,
        value_head_dim=96,
    )
    assert plan.path is IndexedPath.ROW_SPARSE
    assert plan.decode_heads_per_warp == 8


def test_large_asymmetric_dimension_can_use_register_shared_rows():
    plan = _plan(
        query_length=31,
        kv_length=2501,
        topk=625,
        query_heads=64,
        kv_heads=1,
        qk_head_dim=256,
        value_head_dim=96,
    )
    assert plan.path is IndexedPath.ROW_SPARSE
    assert plan.decode_heads_per_warp == 4


def test_near_dense_wide_value_uses_dense_indexed_cute():
    plan = _plan(
        batch_size=4,
        query_length=95,
        kv_length=8191,
        topk=8190,
        qk_head_dim=128,
        value_head_dim=256,
    )
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_grouped_head_sharing_uses_exact_gqa_divisors():
    ratio4 = _plan(query_length=64, query_heads=32, kv_heads=8, backend="grouped")
    ratio8 = _plan(query_length=64, query_heads=64, kv_heads=8, backend="grouped")
    ratio7 = _plan(query_length=64, query_heads=28, kv_heads=4, backend="grouped")
    wide = _plan(
        query_length=64,
        query_heads=64,
        kv_heads=1,
        qk_head_dim=256,
        value_head_dim=256,
        topk=257,
        kv_length=4097,
        backend="grouped",
    )
    assert ratio4.decode_heads_per_warp == 4
    assert ratio8.decode_heads_per_warp == 8
    assert ratio7.decode_heads_per_warp == 7
    assert wide.decode_heads_per_warp == 4
    for plan in (ratio4, ratio8, ratio7, wide):
        assert plan.qhead_per_kvhead % plan.decode_heads_per_warp == 0


def test_near_dense_grouped_prefill_uses_dense_indexed_compute_guard():
    plan = _plan(
        query_length=63,
        kv_length=5003,
        topk=5002,
        query_heads=16,
        kv_heads=4,
    )
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_underfilled_mqa_with_long_selected_set_uses_dense_indexed():
    plan = _plan(
        query_length=16,
        kv_length=32768,
        topk=4096,
        query_heads=64,
        kv_heads=1,
    )
    assert plan.path is IndexedPath.DENSE_INDEXED


def test_general_prefill_total_head_row_and_topk_crossover():
    assert _plan(batch_size=1, query_length=64, topk=1024).path is IndexedPath.ROW_SPARSE
    assert _plan(batch_size=4, query_length=64, topk=1024).path is IndexedPath.DENSE_INDEXED
    assert _plan(batch_size=1, query_length=128, topk=2048).path is IndexedPath.DENSE_INDEXED
    assert _plan(batch_size=1, query_length=64, kv_length=4097, topk=1025).path is IndexedPath.ROW_SPARSE
    assert _plan(batch_size=1, query_length=64, kv_length=4097, topk=2049).path is IndexedPath.DENSE_INDEXED


def test_tiny_selected_sets_override_large_output_row_count():
    plan = _plan(
        batch_size=2,
        query_length=127,
        kv_length=127,
        topk=32,
        query_heads=64,
        kv_heads=1,
        qk_head_dim=256,
        value_head_dim=192,
    )
    assert plan.path is IndexedPath.ROW_SPARSE


def test_deepseek_v32_dsa_decode_uses_measured_split16_schedule():
    plan = _plan(
        query_length=1,
        kv_length=163840,
        topk=2048,
        query_heads=128,
        kv_heads=128,
        qk_head_dim=192,
        value_head_dim=128,
    )
    assert plan.path is IndexedPath.ROW_SPARSE
    assert plan.decode_rows_per_cta == 4
    assert plan.decode_splits == 16
    assert "DeepSeek" in plan.reason


def test_glm_moe_dsa_decode_uses_measured_cta8_split16_schedule():
    plan = _plan(
        query_length=1,
        kv_length=202752,
        topk=2048,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
    )
    assert plan.path is IndexedPath.ROW_SPARSE
    assert plan.decode_rows_per_cta == 8
    assert plan.decode_splits == 16
    assert "GLM" in plan.reason


def test_dsa_schedule_overrides_are_narrow():
    deepseek_nearby = _plan(
        query_length=1,
        kv_length=163840,
        topk=2048,
        query_heads=128,
        kv_heads=128,
        qk_head_dim=192,
        value_head_dim=192,
    )
    glm_nearby = _plan(
        query_length=1,
        kv_length=202752,
        topk=2048,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=192,
    )
    assert deepseek_nearby.decode_rows_per_cta == 4
    assert deepseek_nearby.decode_splits < 16
    assert glm_nearby.decode_rows_per_cta == 4


def test_qualified_common_true_prefill_uses_exact_fa4_paths():
    profiles = (
        (128, 128, 192, 128),
        (32, 32, 128, 128),
        (32, 8, 128, 128),
        (64, 8, 128, 128),
        (64, 1, 128, 128),
        (32, 32, 64, 64),
    )
    for hq, hkv, dq, dv in profiles:
        short = _plan(
            query_length=1024, kv_length=1024, topk=512,
            query_heads=hq, kv_heads=hkv, qk_head_dim=dq, value_head_dim=dv,
        )
        long = _plan(
            batch_size=4, query_length=4096, kv_length=4096, topk=2048,
            query_heads=hq, kv_heads=hkv, qk_head_dim=dq, value_head_dim=dv,
        )
        assert short.path is IndexedPath.FA4_BITMASK_INDEXED
        assert long.path is IndexedPath.BLOCK_SPARSE_INDEXED


def test_common_prefill_promotion_keeps_unmeasured_boundaries_dense():
    # Ratio-2 GQA is in the next qualification suite but is not yet promoted.
    assert _plan(
        query_length=1024, kv_length=1024, topk=512,
        query_heads=32, kv_heads=16,
    ).path is IndexedPath.DENSE_INDEXED
    # Long B16 was added as a guardrail and remains outside the measured limit.
    assert _plan(
        batch_size=16, query_length=2048, kv_length=2048, topk=512,
        query_heads=32, kv_heads=32,
    ).path is IndexedPath.DENSE_INDEXED
    # Intermediate sequence lengths were not part of the promotion matrix.
    assert _plan(
        query_length=1536, kv_length=1536, topk=512,
        query_heads=32, kv_heads=32,
    ).path is IndexedPath.DENSE_INDEXED


def test_glm_true_prefill_keeps_measured_hybrid_batch_policy():
    base = dict(
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
    )
    short = _plan(query_length=1024, kv_length=1024, topk=256, **base)
    measured = _plan(query_length=2048, kv_length=2048, topk=512, **base)
    high_batch = _plan(
        batch_size=4, query_length=4096, kv_length=4096, topk=2048, **base
    )
    unmeasured = _plan(
        batch_size=16, query_length=4096, kv_length=4096, topk=2048, **base
    )
    assert short.path is IndexedPath.FA4_BITMASK_INDEXED
    assert measured.path is IndexedPath.BLOCK_SPARSE_INDEXED
    assert high_batch.path is IndexedPath.BLOCK_SPARSE_INDEXED
    assert unmeasured.path is IndexedPath.DENSE_INDEXED
    assert "GLM" in measured.reason


def test_deepseek_high_batch_q4_uses_measured_row_crossover():
    common = dict(
        query_length=4, kv_length=16384, topk=1024,
        query_heads=128, kv_heads=128, qk_head_dim=192, value_head_dim=128,
    )
    assert _plan(batch_size=4, **common).path is IndexedPath.ROW_SPARSE
    measured = _plan(batch_size=8, **common)
    assert measured.path is IndexedPath.ROW_SPARSE
    assert "DeepSeek Q4" in measured.reason


def test_glm_sparse_prefill_user_score_mod_falls_back_to_rows():
    plan = _plan(
        query_length=4096,
        kv_length=4096,
        topk=2048,
        query_heads=64,
        kv_heads=64,
        qk_head_dim=256,
        value_head_dim=256,
        has_score_mod=True,
    )
    assert plan.path is IndexedPath.ROW_SPARSE


def test_forced_sparse_prefill_accepts_general_supported_true_prefill():
    assert _plan(
        query_length=1024, kv_length=1024, topk=512, backend="block_sparse"
    ).path is IndexedPath.BLOCK_SPARSE_INDEXED
    try:
        _plan(query_length=256, kv_length=256, backend="block_sparse")
    except ValueError as exc:
        assert "Q=K>=512" in str(exc)
    else:
        raise AssertionError("expected short forced sparse prefill validation")
