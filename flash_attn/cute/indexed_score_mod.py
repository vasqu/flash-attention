# Copyright (c) 2026
#
# Score-mod helpers for exact selected-index attention on SM90.
#
# The custom sparse path uses apply_indexed_score_mod_inner to pass original
# selected K/V coordinates to an ordinary FA4 score_mod. The packed-bitmask helpers express selection as an ordinary score_mod and
# are used by the dense indexed CuTe specialization plus benchmark/correctness
# references. Production dispatch never leaves the indexed CuTe API.

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Uint32

from flash_attn.cute import utils
from flash_attn.cute.indexed_prepare import cast_indexed_kv_indices, prepare_indexed_kv_indices
from flash_attn.cute.softmax import call_score_mod
from flash_attn.cute.utils import AuxData


@cute.jit
def apply_indexed_score_mod_inner(
    score_tensor: cute.Tensor,
    coord_tensor: cute.Tensor,
    sToken: cute.Tensor,
    stage: Int32,
    valid_count: Int32,
    score_mod: cutlass.Constexpr,
    batch_idx: Int32,
    head_idx: Int32,
    m_block: Int32,
    softmax_scale: Float32,
    vec_size: cutlass.Constexpr[int],
    qk_acc_dtype: cutlass.Constexpr,
    aux_data: AuxData,
    seqlen_info,
    *,
    tile_m: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    """Apply an ordinary FA4 score_mod to an indexed union score tile.

    ``sToken[column, stage]`` is the original absolute K/V token position.
    Padded accumulator elements receive safe zero coordinates before invoking
    the callback; the hard selected-set membership mask is applied afterwards.
    """

    n_vals = cutlass.const_expr(cute.size(score_tensor.shape))
    score_vec = cute.make_rmem_tensor(vec_size, qk_acc_dtype)
    q_idx_vec = cute.make_rmem_tensor(vec_size, Int32)
    kv_idx_vec = cute.make_rmem_tensor(vec_size, Int32)
    head_idx_vec = cute.make_rmem_tensor(vec_size, Int32)
    batch_idx_ssa = utils.scalar_to_ssa(batch_idx, Int32).broadcast_to((vec_size,))

    for i in cutlass.range(0, n_vals, vec_size, unroll_full=True):
        for j in cutlass.range(vec_size, unroll_full=True):
            packed_row = Int32(coord_tensor[i + j][0])
            column = Int32(coord_tensor[i + j][1])
            packed_q_idx = m_block * tile_m + packed_row

            if cutlass.const_expr(qhead_per_kvhead > 1):
                logical_q_idx = packed_q_idx // qhead_per_kvhead
                head_offset = packed_q_idx - logical_q_idx * qhead_per_kvhead
                logical_head_idx = head_idx * qhead_per_kvhead + head_offset
            else:
                logical_q_idx = packed_q_idx
                logical_head_idx = head_idx

            q_valid = Boolean(logical_q_idx >= 0 and logical_q_idx < seqlen_info.seqlen_q)
            kv_valid = Boolean(column >= 0 and column < valid_count)
            token = Int32(0)
            if kv_valid:
                token = Int32(sToken[column, stage])
                kv_valid = Boolean(token >= 0 and token < seqlen_info.seqlen_k)

            q_idx_vec[j] = logical_q_idx if q_valid else Int32(0)
            kv_idx_vec[j] = token if kv_valid else Int32(0)
            head_idx_vec[j] = logical_head_idx if q_valid else Int32(0)
            score_vec[j] = score_tensor[i + j] * softmax_scale

        post_mod_scores = call_score_mod(
            score_mod,
            score_vec.load(),
            batch_idx_ssa,
            head_idx_vec.load(),
            q_idx_vec.load(),
            kv_idx_vec.load(),
            seqlen_info,
            aux_data,
        )
        score_vec.store(post_mod_scores)
        for j in cutlass.range(vec_size, unroll_full=True):
            score_tensor[i + j] = score_vec[j]


@cute.jit
def topk_bitmask_score_mod(
    scores: cute.TensorSSA,
    batch_idx: cute.TensorSSA,
    head_idx: cute.TensorSSA,
    q_idx: cute.TensorSSA,
    kv_idx: cute.TensorSSA,
    seqlen_info,
    aux_tensors: list[cute.Tensor],
) -> cute.TensorSSA:
    """Native dense FA4 membership mask backed by packed 32-bit words.

    This is exact, but native FA4 still loads dense K/V tiles and executes dense
    QK/PV tensor-core work before/after the callback.
    """

    bitmask = aux_tensors[0]
    b = Int32(batch_idx[0])
    q = Int32(q_idx[0])
    k = Int32(kv_idx[0])
    word_idx = k // 32
    bit_idx = k - word_idx * 32
    word = Uint32(bitmask[b, q, word_idx])
    bit = utils.shl_u32(Uint32(1), Uint32(bit_idx))
    selected = Boolean((word & bit) != Uint32(0))
    bias = Float32(0.0) if selected else -Float32.inf
    return scores + bias


topk_bitmask_score_mod.__vec_size__ = 1


def build_topk_bitmask(
    gather_kv_indices: torch.Tensor,
    seqlen_k: int,
    *,
    assume_unique: bool = True,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack arbitrary-order selected tokens into ``[B,Q,ceil(K/32)]`` words.

    Normal top-k operators return unique token IDs, so the fast path sanitizes
    and scatters the unsorted rows directly.  Distinct powers of two may be
    summed within a word to form the same bit pattern as bitwise OR.  Set
    ``assume_unique=False`` only for defensive/reference use; that path sorts
    and removes duplicates before packing.
    """

    if gather_kv_indices.ndim != 3:
        raise ValueError("gather_kv_indices must have shape [batch, seqlen_q, topk]")
    if seqlen_k < 0 or seqlen_k >= 2**31:
        raise ValueError("seqlen_k must fit in signed int32")

    if assume_unique:
        indices = cast_indexed_kv_indices(gather_kv_indices, seqlen_k)
        valid = (indices >= 0) & (indices < seqlen_k)
    else:
        indices = prepare_indexed_kv_indices(gather_kv_indices, seqlen_k)
        valid = (indices >= 0) & (indices < seqlen_k)
        duplicate = torch.zeros_like(valid)
        if indices.shape[-1] > 1:
            duplicate[..., 1:] = valid[..., 1:] & (indices[..., 1:] == indices[..., :-1])
        valid &= ~duplicate

    # Keep bit arithmetic in int32.  ``scatter_add_`` still requires an int64
    # index tensor, but avoiding int64 safe/bit/value temporaries materially
    # reduces preparation bandwidth for large B*Q sweeps.  A shift by 31
    # intentionally produces the signed two's-complement high bit.
    safe = torch.where(valid, indices, torch.zeros_like(indices))
    word_idx = torch.bitwise_right_shift(safe, 5).to(torch.int64)
    bit_idx = torch.bitwise_and(safe, 31)
    bit_values = torch.bitwise_left_shift(torch.ones_like(bit_idx), bit_idx)
    bit_values = torch.where(valid, bit_values, torch.zeros_like(bit_values))

    words = (seqlen_k + 31) // 32
    expected_shape = (*indices.shape[:-1], words)
    if out is None:
        packed = torch.zeros(
            expected_shape,
            dtype=torch.int32,
            device=indices.device,
        )
    else:
        if out.shape != expected_shape:
            raise ValueError(f"out shape {tuple(out.shape)} != expected {expected_shape}")
        if out.dtype != torch.int32:
            raise TypeError("out must have dtype torch.int32")
        if out.device != indices.device:
            raise ValueError("out must be on the same device as gather_kv_indices")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        packed = out
        packed.zero_()
    packed.scatter_add_(-1, word_idx, bit_values)

    # ``to_cute_aux_tensor`` needs an explicit static unit-stride dimension.
    # Without this metadata, a compile first seen with singleton Q can
    # canonicalize Q as unit-stride and later reject an otherwise contiguous
    # [B, Q, words] bitmask with ``strides[1] expected to be 1``.
    packed.__assumed_align__ = 4
    packed.__leading_dim__ = 2
    return packed


def build_topk_boolean_mask(
    gather_kv_indices: torch.Tensor,
    seqlen_k: int,
) -> torch.Tensor:
    """Build an exact broadcastable ``[B,1,Q,K]`` selected-set mask.

    PyTorch SDPA interprets ``True`` as an allowed attention position.  Invalid
    and out-of-range entries are ignored, arbitrary index order is preserved,
    and duplicate IDs are harmless because boolean scatter is idempotent.
    """

    if gather_kv_indices.ndim != 3:
        raise ValueError("gather_kv_indices must have shape [batch, seqlen_q, topk]")
    if seqlen_k < 1:
        raise ValueError("seqlen_k must be positive")

    indices = gather_kv_indices.to(torch.int64)
    valid = (indices >= 0) & (indices < seqlen_k)
    safe = indices.clamp(0, seqlen_k - 1)
    # Boolean scatter is last-writer-wins: an invalid entry clamped to zero can
    # overwrite a valid selected token 0. Accumulate integer membership and
    # convert to bool so duplicates and invalid entries are exactly idempotent.
    counts = torch.zeros(
        (*indices.shape[:2], seqlen_k),
        dtype=torch.int32,
        device=indices.device,
    )
    counts.scatter_add_(2, safe, valid.to(torch.int32))
    return (counts != 0).unsqueeze(1)


def native_topk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gather_kv_indices: torch.Tensor,
    **kwargs,
):
    """Run dense FA4 with selected-set membership expressed via score_mod."""

    from flash_attn.cute.interface import _flash_attn_fwd

    if kwargs.get("score_mod") is not None or kwargs.get("aux_tensors") is not None:
        raise ValueError("native_topk_attention owns score_mod and aux_tensors")
    bitmask = build_topk_bitmask(gather_kv_indices, k.shape[1])
    return _flash_attn_fwd(
        q,
        k,
        v,
        score_mod=topk_bitmask_score_mod,
        aux_tensors=[bitmask],
        **kwargs,
    )
