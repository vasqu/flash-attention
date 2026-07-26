# Copyright (c) 2026
#
# Dense exact selected-set backend for the indexed SM90 API.
#
# This is deliberately a CuTe kernel specialization, not a recursive call into
# the public FA4 wrapper and not a PyTorch SDPA fallback. It uses the ordinary
# FA4 tensor-core mainloop with an indexed membership score modifier.

from __future__ import annotations

from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_attn.cute.indexed_score_mod import topk_bitmask_score_mod


class IndexedDenseSm90(FlashAttentionForwardSm90):
    """FA4 SM90 forward specialized for an exact packed selected-set mask."""

    def __init__(
        self,
        dtype,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        *,
        pack_gqa: bool,
        tile_m: int,
        tile_n: int,
        mma_pv_is_rs: bool,
        intra_wg_overlap: bool,
    ):
        super().__init__(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=False,
            is_local=False,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=2,
            num_threads=384,
            Q_in_regs=False,
            intra_wg_overlap=intra_wg_overlap,
            mma_pv_is_rs=mma_pv_is_rs,
            mask_mod=None,
            score_mod=topk_bitmask_score_mod,
            has_aux_tensors=True,
        )
