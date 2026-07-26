# Copyright (c) 2026
#
# Pure-CuTe exact selected-index decode / sparse-row path for SM90.
#
# The public API accepts selected indices in arbitrary order.  For GQA
# prefill, one warp can carry several query heads that share the same KV head
# and selected-token row, loading each K/V element only once for the subgroup.

from __future__ import annotations

import math
from typing import Optional, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32

from flash_attn.cute import utils
from flash_attn.cute.softmax import call_score_mod
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.utils import AuxData


class IndexedRowAttentionSm90:
    """Exact selected-token attention with one warp per GQA head subgroup.

    ``heads_per_warp == 1`` is the original independent-row kernel.  For GQA
    prefill, a warp can instead carry several query heads that share the same
    KV head and selected-token row.  K and V are then loaded once per token and
    reused from registers across all carried heads.  This remains exact for
    arbitrary index order and avoids the cross-query union required by WGMMA.

    As with ordinary top-k output, entries are expected to be unique; repeated
    token IDs intentionally contribute repeated probability mass rather than
    paying for a per-row deduplication structure.
    """

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        qk_head_dim: int,
        value_head_dim: int,
        topk: int,
        *,
        rows_per_cta: int = 4,
        heads_per_warp: int = 1,
        invalid_sentinel: int = -1,
        direct_output: bool = False,
        score_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
    ):
        if qk_head_dim > 256 or value_head_dim > 256:
            raise ValueError("indexed decode supports dimensions <= 256")
        if rows_per_cta not in (4, 8):
            raise ValueError("rows_per_cta must be 4 or 8")
        if heads_per_warp < 1 or heads_per_warp > 8:
            raise ValueError("heads_per_warp must be in [1, 8]")
        self.dtype = dtype
        self.qk_head_dim = qk_head_dim
        self.value_head_dim = value_head_dim
        self.topk = topk
        self.rows_per_cta = rows_per_cta
        self.heads_per_warp = heads_per_warp
        self.invalid_sentinel = invalid_sentinel
        self.direct_output = direct_output
        self.score_mod = score_mod
        self.score_vec_size = getattr(score_mod, "__vec_size__", 1 if has_aux_tensors else 2)
        if self.score_mod is not None and self.score_vec_size != 1:
            raise ValueError("indexed warp decode requires score_mod.__vec_size__ == 1")
        self.num_threads = rows_per_cta * 32
        self.q_values_per_lane = cute.ceil_div(qk_head_dim, 32)
        self.v_values_per_lane = cute.ceil_div(value_head_dim, 32)
        # Grouped prefill amortizes online-softmax output rescaling over a small
        # selected-token microtile. Q=1/scalar decode keeps the proven one-token
        # step and therefore its existing register footprint and scheduling.
        self.tokens_per_step = 4 if heads_per_warp > 1 else 1

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mIndices: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        num_splits: Int32,
        aux_data: AuxData,
        stream: cuda.CUstream,
    ):
        query_heads = mQ.shape[2]
        kv_heads = mK.shape[2]
        qhead_per_kvhead = query_heads // kv_heads
        head_groups_per_kvhead = qhead_per_kvhead // self.heads_per_warp
        total_groups = (
            mQ.shape[0]
            * mQ.shape[1]
            * kv_heads
            * head_groups_per_kvhead
        )
        self.kernel(
            mQ,
            mK,
            mV,
            mIndices,
            mO,
            mLSE,
            softmax_scale,
            num_splits,
            total_groups,
            aux_data,
        ).launch(
            grid=(cute.ceil_div(total_groups, self.rows_per_cta), num_splits, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mIndices: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        num_splits: Int32,
        total_groups: Int32,
        aux_data: AuxData,
    ):
        thread_idx = cute.arch.thread_idx()[0]
        lane = thread_idx % 32
        warp = thread_idx // 32
        group_block = cute.arch.block_idx()[0]
        split_idx = cute.arch.block_idx()[1]

        group = group_block * self.rows_per_cta + warp
        if group < total_groups:
            query_heads = mQ.shape[2]
            query_length = mQ.shape[1]
            kv_heads = mK.shape[2]
            qhead_per_kvhead = query_heads // kv_heads
            head_groups_per_kvhead = qhead_per_kvhead // self.heads_per_warp
            groups_per_query = kv_heads * head_groups_per_kvhead

            batch_idx = group // (query_length * groups_per_query)
            remainder = group - batch_idx * (query_length * groups_per_query)
            query_idx = remainder // groups_per_query
            head_group = remainder - query_idx * groups_per_query
            kv_head = head_group // head_groups_per_kvhead
            subgroup = head_group - kv_head * head_groups_per_kvhead
            query_head_base = (
                kv_head * qhead_per_kvhead + subgroup * self.heads_per_warp
            )
            seqlen_info = SeqlenInfoQK.create(
                batch_idx,
                mQ.shape[1],
                mK.shape[1],
                tile_m=1,
                tile_n=1,
            )

            keys_per_split = cute.ceil_div(self.topk, num_splits)
            key_begin = split_idx * keys_per_split
            key_end = cutlass.min(key_begin + keys_per_split, self.topk)

            # Q and O are carried for several heads, while every K/V element is
            # loaded only once for the whole subgroup.  The head axis is a
            # compile-time register dimension, requiring no shared exchange.
            rQ = cute.make_rmem_tensor(
                (self.heads_per_warp, self.q_values_per_lane), Float32
            )
            rO = cute.make_rmem_tensor(
                (self.heads_per_warp, self.v_values_per_lane), Float32
            )
            rRunningMax = cute.make_rmem_tensor((self.heads_per_warp,), Float32)
            rRunningSum = cute.make_rmem_tensor((self.heads_per_warp,), Float32)
            rDot = cute.make_rmem_tensor((self.heads_per_warp,), Float32)
            rNextMax = cute.make_rmem_tensor((self.heads_per_warp,), Float32)
            rOldScale = cute.make_rmem_tensor((self.heads_per_warp,), Float32)
            rProbability = cute.make_rmem_tensor((self.heads_per_warp,), Float32)
            rBlockMax = cute.make_rmem_tensor((self.heads_per_warp,), Float32)
            rToken = cute.make_rmem_tensor((self.tokens_per_step,), Int32)
            rValid = cute.make_rmem_tensor((self.tokens_per_step,), Boolean)
            rScore = cute.make_rmem_tensor(
                (self.heads_per_warp, self.tokens_per_step), Float32
            )

            for head_local in cutlass.range_constexpr(self.heads_per_warp):
                query_head = query_head_base + head_local
                for item in cutlass.range_constexpr(self.q_values_per_lane):
                    dim = lane + item * 32
                    rQ[head_local, item] = (
                        Float32(mQ[batch_idx, query_idx, query_head, dim])
                        if dim < self.qk_head_dim
                        else Float32(0.0)
                    )
                for item in cutlass.range_constexpr(self.v_values_per_lane):
                    rO[head_local, item] = Float32(0.0)
                rRunningMax[head_local] = -Float32.inf
                rRunningSum[head_local] = Float32(0.0)

            # Preserve the proven Q=1 / one-head-per-warp loop exactly. The
            # grouped path below uses a four-token online-softmax microtile.
            if cutlass.const_expr(self.tokens_per_step == 1):
                for selected_idx in cutlass.range(key_begin, key_end, unroll=1):
                    token = Int32(mIndices[batch_idx, query_idx, selected_idx])
                    valid = Boolean(
                        token != self.invalid_sentinel
                        and token >= 0
                        and token < mK.shape[1]
                    )
                    if valid:
                        rDot.fill(0.0)
                        for item in cutlass.range_constexpr(self.q_values_per_lane):
                            dim = lane + item * 32
                            k_value = (
                                Float32(mK[batch_idx, token, kv_head, dim])
                                if dim < self.qk_head_dim
                                else Float32(0.0)
                            )
                            for head_local in cutlass.range_constexpr(self.heads_per_warp):
                                rDot[head_local] += rQ[head_local, item] * k_value

                        for head_local in cutlass.range_constexpr(self.heads_per_warp):
                            query_head = query_head_base + head_local
                            score = (
                                cute.arch.warp_reduction_sum(rDot[head_local])
                                * softmax_scale
                            )
                            if cutlass.const_expr(self.score_mod is not None):
                                if lane == 0:
                                    score_ssa = utils.scalar_to_ssa(
                                        score, Float32
                                    ).broadcast_to((1,))
                                    batch_ssa = utils.scalar_to_ssa(
                                        batch_idx, Int32
                                    ).broadcast_to((1,))
                                    head_ssa = utils.scalar_to_ssa(
                                        query_head, Int32
                                    ).broadcast_to((1,))
                                    q_ssa = utils.scalar_to_ssa(
                                        query_idx, Int32
                                    ).broadcast_to((1,))
                                    kv_ssa = utils.scalar_to_ssa(
                                        token, Int32
                                    ).broadcast_to((1,))
                                    score = Float32(
                                        call_score_mod(
                                            self.score_mod,
                                            score_ssa,
                                            batch_ssa,
                                            head_ssa,
                                            q_ssa,
                                            kv_ssa,
                                            seqlen_info,
                                            aux_data,
                                        )[0]
                                    )
                                score = utils.shuffle_sync(score, offset=0)

                            next_max = cutlass.max(rRunningMax[head_local], score)
                            old_scale = (
                                Float32(0.0)
                                if rRunningMax[head_local] == -Float32.inf
                                else cute.math.exp2(
                                    (rRunningMax[head_local] - next_max)
                                    * math.log2(math.e),
                                    fastmath=True,
                                )
                            )
                            probability = cute.math.exp2(
                                (score - next_max) * math.log2(math.e),
                                fastmath=True,
                            )
                            rNextMax[head_local] = next_max
                            rOldScale[head_local] = old_scale
                            rProbability[head_local] = probability

                        for item in cutlass.range_constexpr(self.v_values_per_lane):
                            dim = lane + item * 32
                            v_value = (
                                Float32(mV[batch_idx, token, kv_head, dim])
                                if dim < self.value_head_dim
                                else Float32(0.0)
                            )
                            for head_local in cutlass.range_constexpr(self.heads_per_warp):
                                rO[head_local, item] = (
                                    rO[head_local, item] * rOldScale[head_local]
                                    + rProbability[head_local] * v_value
                                )

                        for head_local in cutlass.range_constexpr(self.heads_per_warp):
                            rRunningSum[head_local] = (
                                rRunningSum[head_local] * rOldScale[head_local]
                                + rProbability[head_local]
                            )
                            rRunningMax[head_local] = rNextMax[head_local]
            else:
                # Order does not affect exact softmax attention over a unique set.
                # Invalid entries may appear anywhere and are skipped independently.
                # Grouped prefill computes a four-token score microtile, updates the
                # online-softmax maximum once, and rescales O once per microtile
                # rather than once per token. This is exact and materially reduces
                # the per-head PV bookkeeping that dominated wider sparse rows.
                block_count = cute.ceil_div(
                    cutlass.max(key_end - key_begin, Int32(0)), self.tokens_per_step
                )
                for block_idx in cutlass.range(block_count, unroll=1):
                    selected_base = key_begin + block_idx * self.tokens_per_step
                    for head_local in cutlass.range_constexpr(self.heads_per_warp):
                        rBlockMax[head_local] = -Float32.inf

                    # QK for the selected-token microtile. K is still loaded only
                    # once per subgroup and score_mod sees the original absolute K
                    # coordinate.
                    for token_local in cutlass.range_constexpr(self.tokens_per_step):
                        selected_idx = selected_base + token_local
                        token = Int32(-1)
                        valid = Boolean(False)
                        if selected_idx < key_end:
                            token = Int32(mIndices[batch_idx, query_idx, selected_idx])
                            valid = Boolean(
                                token != self.invalid_sentinel
                                and token >= 0
                                and token < mK.shape[1]
                            )
                        rToken[token_local] = token
                        rValid[token_local] = valid

                        for head_local in cutlass.range_constexpr(self.heads_per_warp):
                            rScore[head_local, token_local] = -Float32.inf
                        if valid:
                            rDot.fill(0.0)
                            for item in cutlass.range_constexpr(self.q_values_per_lane):
                                dim = lane + item * 32
                                k_value = (
                                    Float32(mK[batch_idx, token, kv_head, dim])
                                    if dim < self.qk_head_dim
                                    else Float32(0.0)
                                )
                                for head_local in cutlass.range_constexpr(self.heads_per_warp):
                                    rDot[head_local] += rQ[head_local, item] * k_value

                            for head_local in cutlass.range_constexpr(self.heads_per_warp):
                                query_head = query_head_base + head_local
                                score = (
                                    cute.arch.warp_reduction_sum(rDot[head_local])
                                    * softmax_scale
                                )
                                if cutlass.const_expr(self.score_mod is not None):
                                    if lane == 0:
                                        score_ssa = utils.scalar_to_ssa(
                                            score, Float32
                                        ).broadcast_to((1,))
                                        batch_ssa = utils.scalar_to_ssa(
                                            batch_idx, Int32
                                        ).broadcast_to((1,))
                                        head_ssa = utils.scalar_to_ssa(
                                            query_head, Int32
                                        ).broadcast_to((1,))
                                        q_ssa = utils.scalar_to_ssa(
                                            query_idx, Int32
                                        ).broadcast_to((1,))
                                        kv_ssa = utils.scalar_to_ssa(
                                            token, Int32
                                        ).broadcast_to((1,))
                                        score = Float32(
                                            call_score_mod(
                                                self.score_mod,
                                                score_ssa,
                                                batch_ssa,
                                                head_ssa,
                                                q_ssa,
                                                kv_ssa,
                                                seqlen_info,
                                                aux_data,
                                            )[0]
                                        )
                                    score = utils.shuffle_sync(score, offset=0)
                                rScore[head_local, token_local] = score
                                rBlockMax[head_local] = cutlass.max(
                                    rBlockMax[head_local], score
                                )

                    # One online-softmax rescale for the complete microtile.
                    for head_local in cutlass.range_constexpr(self.heads_per_warp):
                        next_max = cutlass.max(
                            rRunningMax[head_local], rBlockMax[head_local]
                        )
                        old_scale = (
                            Float32(0.0)
                            if rRunningMax[head_local] == -Float32.inf
                            else cute.math.exp2(
                                (rRunningMax[head_local] - next_max)
                                * math.log2(math.e),
                                fastmath=True,
                            )
                        )
                        rNextMax[head_local] = next_max
                        rOldScale[head_local] = old_scale
                        rRunningSum[head_local] *= old_scale
                    for item in cutlass.range_constexpr(self.v_values_per_lane):
                        for head_local in cutlass.range_constexpr(self.heads_per_warp):
                            rO[head_local, item] *= rOldScale[head_local]

                    # PV for each valid token. V is loaded once and reused across
                    # every query head in the subgroup.
                    for token_local in cutlass.range_constexpr(self.tokens_per_step):
                        if rValid[token_local]:
                            token = rToken[token_local]
                            for head_local in cutlass.range_constexpr(self.heads_per_warp):
                                score = rScore[head_local, token_local]
                                probability = (
                                    Float32(0.0)
                                    if score == -Float32.inf
                                    else cute.math.exp2(
                                        (score - rNextMax[head_local])
                                        * math.log2(math.e),
                                        fastmath=True,
                                    )
                                )
                                rProbability[head_local] = probability
                                rRunningSum[head_local] += probability

                            for item in cutlass.range_constexpr(self.v_values_per_lane):
                                dim = lane + item * 32
                                v_value = (
                                    Float32(mV[batch_idx, token, kv_head, dim])
                                    if dim < self.value_head_dim
                                    else Float32(0.0)
                                )
                                for head_local in cutlass.range_constexpr(self.heads_per_warp):
                                    rO[head_local, item] += (
                                        rProbability[head_local] * v_value
                                    )

                    for head_local in cutlass.range_constexpr(self.heads_per_warp):
                        rRunningMax[head_local] = rNextMax[head_local]

            for head_local in cutlass.range_constexpr(self.heads_per_warp):
                query_head = query_head_base + head_local
                inverse_sum = (
                    Float32(0.0)
                    if rRunningSum[head_local] == 0.0
                    else Float32(1.0) / rRunningSum[head_local]
                )
                lse_value = (
                    -Float32.inf
                    if rRunningSum[head_local] == 0.0
                    else rRunningMax[head_local]
                    + cute.math.log(rRunningSum[head_local], fastmath=True)
                )

                for item in cutlass.range_constexpr(self.v_values_per_lane):
                    dim = lane + item * 32
                    if dim < self.value_head_dim:
                        value = rO[head_local, item] * inverse_sum
                        if cutlass.const_expr(self.direct_output):
                            mO[batch_idx, query_idx, query_head, dim] = (
                                mO.element_type(value)
                            )
                        else:
                            mO[
                                split_idx,
                                batch_idx,
                                query_idx,
                                query_head,
                                dim,
                            ] = value

                if cutlass.const_expr(mLSE is not None):
                    if lane == 0:
                        if cutlass.const_expr(self.direct_output):
                            mLSE[batch_idx, query_head, query_idx] = lse_value
                        else:
                            mLSE[
                                split_idx, batch_idx, query_idx, query_head
                            ] = lse_value


# Backward-compatible internal name used by older benchmark scripts.
IndexedDecodeSm90 = IndexedRowAttentionSm90
