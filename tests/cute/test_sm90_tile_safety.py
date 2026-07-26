from flash_attn.cute.sm90_tile_safety import (
    SM90_MAX_DYNAMIC_SMEM_BYTES,
    choose_safe_sm90_fwd_tile,
    estimate_sm90_fwd_smem_bytes,
)


def test_d192_dv256_rs_caps_n_below_h100_limit():
    assert estimate_sm90_fwd_smem_bytes(
        tile_m=128,
        tile_n=112,
        head_dim=192,
        value_head_dim=256,
        mma_pv_is_rs=True,
    ) == 250_880
    safe = choose_safe_sm90_fwd_tile(
        tile_m=128,
        tile_n=112,
        head_dim=192,
        value_head_dim=256,
        mma_pv_is_rs=True,
    )
    assert (safe.tile_m, safe.tile_n) == (128, 96)
    assert safe.estimated_smem_bytes <= SM90_MAX_DYNAMIC_SMEM_BYTES


def test_d96_dv192_non_rs_reduces_m_and_stays_launch_safe():
    assert estimate_sm90_fwd_smem_bytes(
        tile_m=192,
        tile_n=144,
        head_dim=96,
        value_head_dim=192,
        mma_pv_is_rs=False,
    ) == 259_072
    safe = choose_safe_sm90_fwd_tile(
        tile_m=192,
        tile_n=144,
        head_dim=96,
        value_head_dim=192,
        mma_pv_is_rs=False,
    )
    assert (safe.tile_m, safe.tile_n) == (128, 144)
    assert safe.estimated_smem_bytes <= SM90_MAX_DYNAMIC_SMEM_BYTES


def test_wide_v_uses_m128_to_avoid_m192_register_exhaustion():
    safe = choose_safe_sm90_fwd_tile(
        tile_m=192,
        tile_n=128,
        head_dim=64,
        value_head_dim=256,
        mma_pv_is_rs=True,
    )
    assert (safe.tile_m, safe.tile_n) == (128, 128)


def test_common_d128_tile_is_unchanged():
    safe = choose_safe_sm90_fwd_tile(
        tile_m=128,
        tile_n=128,
        head_dim=128,
        value_head_dim=128,
        mma_pv_is_rs=True,
    )
    assert (safe.tile_m, safe.tile_n) == (128, 128)
    assert not safe.changed


def test_d256_tile_is_already_at_h100_limit_but_safe():
    safe = choose_safe_sm90_fwd_tile(
        tile_m=128,
        tile_n=80,
        head_dim=256,
        value_head_dim=256,
        mma_pv_is_rs=True,
    )
    assert (safe.tile_m, safe.tile_n) == (128, 80)
    assert safe.estimated_smem_bytes == 230_400
    assert safe.estimated_smem_bytes <= SM90_MAX_DYNAMIC_SMEM_BYTES


def test_all_supported_qk_v_dimension_pairs_have_a_safe_default_tile():
    dims = (64, 96, 128, 192, 256)
    for head_dim in dims:
        for value_head_dim in dims:
            if head_dim <= 64:
                tile_m, tile_n, rs = 192, 128, True
            elif head_dim <= 96:
                tile_m, tile_n, rs = 192, 144, False
            elif head_dim <= 128:
                tile_m, tile_n, rs = 128, 128, True
            elif head_dim <= 192:
                tile_m = 128
                tile_n = 128 if value_head_dim <= 128 else 112
                rs = True
            else:
                tile_m, tile_n, rs = 128, 80, True
            safe = choose_safe_sm90_fwd_tile(
                tile_m=tile_m,
                tile_n=tile_n,
                head_dim=head_dim,
                value_head_dim=value_head_dim,
                mma_pv_is_rs=rs,
            )
            assert safe.estimated_smem_bytes <= SM90_MAX_DYNAMIC_SMEM_BYTES
