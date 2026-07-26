import torch

from flash_attn.cute.indexed_prepare import prepare_indexed_kv_indices
from flash_attn.cute.indexed_score_mod import build_topk_bitmask, build_topk_boolean_mask


def _bit_is_set(bitmask: torch.Tensor, b: int, q: int, k: int) -> bool:
    word = int(bitmask[b, q, k // 32].item()) & 0xFFFFFFFF
    return bool((word >> (k % 32)) & 1)


def test_prepare_union_indices_is_internal_and_order_agnostic():
    indices = torch.tensor(
        [[[2, 7, 7, -1, 4], [65535, 31, 63, 32, 31]]],
        dtype=torch.int64,
    )
    prepared = prepare_indexed_kv_indices(indices, 64)
    assert prepared.dtype == torch.int32
    assert prepared.tolist() == [[[7, 7, 4, 2, -1], [63, 32, 31, 31, -1]]]


def test_build_topk_bitmask_accepts_unsorted_duplicates():
    indices = torch.tensor(
        [[[2, 5, -1, 5], [63, 31, 32, 31]]],
        dtype=torch.int32,
    )
    bitmask = build_topk_bitmask(indices, 64, assume_unique=False)
    assert bitmask.shape == (1, 2, 2)
    assert bitmask.dtype == torch.int32
    assert bitmask.__leading_dim__ == 2
    assert bitmask.__assumed_align__ == 4
    expected = ({2, 5}, {31, 32, 63})
    for q, selected in enumerate(expected):
        for k in range(64):
            assert _bit_is_set(bitmask, 0, q, k) == (k in selected)


def test_build_topk_bitmask_fast_path_accepts_unsorted_unique_rows():
    indices = torch.tensor([[[63, 2, 31, 5, 32]]], dtype=torch.int32)
    bitmask = build_topk_bitmask(indices, 64, assume_unique=True)
    selected = {2, 5, 31, 32, 63}
    for k in range(64):
        assert _bit_is_set(bitmask, 0, 0, k) == (k in selected)


def test_int64_values_are_sanitized_before_narrowing():
    indices = torch.tensor([[[2**32 + 5, 3, -4]]], dtype=torch.int64)
    prepared = prepare_indexed_kv_indices(indices, 16)
    assert prepared.tolist() == [[[3, -1, -1]]]


def test_build_topk_boolean_mask_is_broadcastable_and_exact():
    indices = torch.tensor(
        [[[5, -1, 2, 5, 99], [0, 7, 3, -4, 3]]], dtype=torch.int64
    )
    mask = build_topk_boolean_mask(indices, 8)
    assert mask.shape == (1, 1, 2, 8)
    assert mask.dtype == torch.bool
    assert set(mask[0, 0, 0].nonzero().flatten().tolist()) == {2, 5}
    assert set(mask[0, 0, 1].nonzero().flatten().tolist()) == {0, 3, 7}


def test_build_topk_bitmask_can_reuse_output_buffer():
    indices = torch.tensor([[[1, 7, 33], [0, 31, 63]]], dtype=torch.int32)
    expected = build_topk_bitmask(indices, 64)
    workspace = torch.empty_like(expected)
    actual = build_topk_bitmask(indices, 64, out=workspace)
    assert actual.data_ptr() == workspace.data_ptr()
    assert torch.equal(actual, expected)
