"""Reference and benchmark helpers for exact indexed block-sparse prefill.

The generic PyTorch layout builder remains here as a readable correctness
reference.  The native CuTe implementation used by production dispatch lives in
``indexed_block_sparse`` and is re-exported for existing benchmark callers.
Every active K/V tile remains a masked tile; the ordinary exact top-k bitmask
enforces selected-token membership inside scheduled tiles.
"""

from __future__ import annotations

import math

import torch


def build_indexed_block_layout(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int,
    tile_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact active-tile lists for fixed-batch FA4 block sparsity.

    Returns ``mask_cnt, mask_idx, full_cnt, full_idx``.  The head dimension is
    broadcast (size one) because indexed DSA membership is shared by all heads.
    Active indices are stored in ascending order; SM90 consumes the list in
    reverse, matching FA4's normal descending K/V traversal.
    """

    if indices.ndim != 3:
        raise ValueError("indices must have shape [batch, seqlen_q, topk]")
    if indices.shape[1] != seqlen_q:
        raise ValueError("indices query dimension must match seqlen_q")
    if indices.dtype not in (torch.int32, torch.int64, torch.uint16):
        raise TypeError("indices must be int32, int64, or uint16")
    if seqlen_q < 1 or seqlen_k < 1:
        raise ValueError("seqlen_q and seqlen_k must be positive")
    if tile_m not in (64, 128) or tile_n != 64:
        raise ValueError("the initial SM90 lab supports 64x64 and 128x64 tiles")

    batch = indices.shape[0]
    num_m_blocks = math.ceil(seqlen_q / tile_m)
    num_n_blocks = math.ceil(seqlen_k / tile_n)

    selected = indices.to(torch.int64)
    valid = (selected >= 0) & (selected < seqlen_k)
    n_blocks = torch.div(selected.clamp(0, seqlen_k - 1), tile_n, rounding_mode="floor")

    q_blocks = torch.div(
        torch.arange(seqlen_q, device=indices.device, dtype=torch.int64),
        tile_m,
        rounding_mode="floor",
    ).view(1, seqlen_q, 1)
    batch_ids = torch.arange(batch, device=indices.device, dtype=torch.int64).view(batch, 1, 1)
    flat_ids = (batch_ids * num_m_blocks + q_blocks) * num_n_blocks + n_blocks

    active_flat = torch.zeros(
        batch * num_m_blocks * num_n_blocks,
        dtype=torch.bool,
        device=indices.device,
    )
    active_flat[flat_ids[valid]] = True
    active = active_flat.view(batch, num_m_blocks, num_n_blocks)

    mask_cnt = active.sum(dim=-1, dtype=torch.int32).unsqueeze(1).contiguous()
    block_ids = torch.arange(num_n_blocks, device=indices.device, dtype=torch.int32)
    block_ids = block_ids.view(1, 1, num_n_blocks).expand(batch, num_m_blocks, num_n_blocks)
    sentinel = torch.full_like(block_ids, num_n_blocks)
    mask_idx = torch.where(active, block_ids, sentinel)
    mask_idx = torch.sort(mask_idx, dim=-1).values.unsqueeze(1).contiguous()

    # Materialize an empty full-block list instead of using None.  This is a
    # compatibility guard for the older SM90 block-sparse implementation used
    # by the baseline commit.
    full_cnt = torch.zeros_like(mask_cnt)
    full_idx = torch.zeros_like(mask_idx)
    return mask_cnt, mask_idx, full_cnt, full_idx


def build_indexed_block_sparse_tensors(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int,
    tile_n: int,
):
    """Return a fully materialized ``BlockSparseTensorsTorch`` instance."""

    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

    mask_cnt, mask_idx, full_cnt, full_idx = build_indexed_block_layout(
        indices,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    return BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt,
        mask_block_idx=mask_idx,
        full_block_cnt=full_cnt,
        full_block_idx=full_idx,
        block_size=(tile_m, tile_n),
    )


def indexed_block_density(mask_block_cnt: torch.Tensor, num_n_blocks: int) -> float:
    """Return the mean scheduled-K/V-tile fraction for diagnostics."""

    if num_n_blocks < 1:
        raise ValueError("num_n_blocks must be positive")
    if mask_block_cnt.numel() == 0:
        return 0.0
    return float(mask_block_cnt.float().mean().item()) / float(num_n_blocks)

from flash_attn.cute.indexed_block_sparse import (
    IndexedBlockSparseCuteWorkspace,
    build_indexed_block_sparse_tensors_cute,
    create_indexed_block_sparse_cute_workspace,
)
