# Copyright (c) 2026
#
# CuTeDSL helpers for exact selected-index K/V traversal inside the existing
# FA4 SM90 forward mainloop.
#
# This file is not a standalone attention kernel. Public callers may pass
# indices in any order; interface.py prepares the descending representation
# consumed by this streaming union only when the union path is selected.

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Uint16, Uint32
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase

from flash_attn.cute import utils


@dataclass
class IndexedUnionStage(ParamsBase):
    valid_count: Int32
    first_token: Int32
    is_contiguous_descending: Boolean
    all_columns_full: Boolean


@dataclass
class QueryTileUnionSm90(ParamsBase):
    """Streaming union over internally prepared descending index rows.

    One producer warp owns up to six query streams per lane, supporting up to
    192 logical query positions. Packed GQA reduces the number of independent
    streams from ``tile_m`` to ``tile_m / qhead_per_kvhead``.

    The public API does *not* require sorted input. The host wrapper sorts only
    for this WGMMA union path. The low-Q decode path consumes the original order
    directly and therefore pays no preparation cost.
    """

    mIndices: cute.Tensor
    batch_idx: Int32
    m_block: Int32
    seqlen_q: Int32
    seqlen_k: Int32

    topk: cutlass.Constexpr[int]
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    qhead_per_kvhead: cutlass.Constexpr[int]
    logical_q_per_tile: cutlass.Constexpr[int]
    streams_per_lane: cutlass.Constexpr[int]
    mask_words: cutlass.Constexpr[int]
    invalid_sentinel: cutlass.Constexpr[int]

    rCursor: cute.Tensor
    rValue: cute.Tensor
    rQueryIdx: cute.Tensor
    rActive: cute.Tensor
    rActiveWord: cute.Tensor

    @staticmethod
    @cute.jit
    def create(
        mIndices: cute.Tensor,
        batch_idx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        seqlen_k: Int32,
        *,
        topk: cutlass.Constexpr[int],
        tile_m: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        qhead_per_kvhead: cutlass.Constexpr[int],
        invalid_sentinel: cutlass.Constexpr[int] = -1,
    ):
        assert tile_m % qhead_per_kvhead == 0
        logical_q = tile_m // qhead_per_kvhead
        assert logical_q <= 192
        streams_per_lane = cute.ceil_div(logical_q, 32)
        mask_words = streams_per_lane

        rCursor = cute.make_rmem_tensor((streams_per_lane,), Int32)
        rValue = cute.make_rmem_tensor((streams_per_lane,), Int32)
        rQueryIdx = cute.make_rmem_tensor((streams_per_lane,), Int32)
        rActive = cute.make_rmem_tensor((streams_per_lane,), Boolean)
        rActiveWord = cute.make_rmem_tensor((streams_per_lane,), Uint32)

        lane = cute.arch.lane_idx()
        query_start = (m_block * tile_m) // qhead_per_kvhead

        # Query coordinates and active masks are invariant for every emitted
        # column. Cache both in registers instead of recomputing a ballot for
        # every union token.
        for stream in cutlass.range_constexpr(streams_per_lane):
            local_query = lane + stream * 32
            query_idx = query_start + local_query
            active = Boolean(local_query < logical_q and query_idx < seqlen_q)
            active_word = Uint32(cute.arch.vote_ballot_sync(active))

            rQueryIdx[stream] = query_idx
            rActive[stream] = active
            rActiveWord[stream] = active_word
            rCursor[stream] = Int32(0)

            value = Int32(-1)
            if active:
                raw = mIndices[batch_idx, query_idx, 0]
                value = Int32(raw)
                if value == invalid_sentinel or value < 0 or value >= seqlen_k:
                    value = Int32(-1)
            rValue[stream] = value

        return QueryTileUnionSm90(
            mIndices=mIndices,
            batch_idx=batch_idx,
            m_block=m_block,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            qhead_per_kvhead=qhead_per_kvhead,
            logical_q_per_tile=logical_q,
            streams_per_lane=streams_per_lane,
            mask_words=mask_words,
            invalid_sentinel=invalid_sentinel,
            rCursor=rCursor,
            rValue=rValue,
            rQueryIdx=rQueryIdx,
            rActive=rActive,
            rActiveWord=rActiveWord,
        )

    @cute.jit
    def _read(self, query_idx: Int32, cursor: Int32) -> Int32:
        value = Int32(-1)
        if cursor < self.topk and query_idx < self.seqlen_q:
            raw = self.mIndices[self.batch_idx, query_idx, cursor]
            value = Int32(raw)
            if value == self.invalid_sentinel or value < 0 or value >= self.seqlen_k:
                value = Int32(-1)
        return value

    @cute.jit
    def _advance_past(
        self,
        stream: cutlass.Constexpr[int],
        emitted_token: Int32,
    ):
        """Advance through adjacent duplicates in the prepared row."""

        value = self.rValue[stream]
        cursor = self.rCursor[stream]
        query_idx = self.rQueryIdx[stream]
        while value == emitted_token:
            cursor += 1
            value = self._read(query_idx, cursor)
        self.rCursor[stream] = cursor
        self.rValue[stream] = value

    @cute.jit
    def _current_max(self) -> Int32:
        lane_max = Int32(-1)
        for stream in cutlass.range_constexpr(self.streams_per_lane):
            lane_max = cutlass.max(lane_max, self.rValue[stream])
        return Int32(cute.arch.warp_redux_sync(lane_max, kind="max"))

    @cute.jit
    def emit(
        self,
        sToken: cute.Tensor,
        sMembership: cute.Tensor,
        sColumnFull: cute.Tensor,
        sValidCount: cute.Tensor,
        sFirstToken: cute.Tensor,
        sContiguous: cute.Tensor,
        sStageFull: cute.Tensor,
        stage: Int32,
    ) -> IndexedUnionStage:
        """Emit one compact union stage and exact logical-query membership.

        Empty terminal stages return immediately instead of executing a full
        ``tile_n`` worth of reductions. Partially filled final stages likewise
        stop as soon as all streams are exhausted.
        """

        lane = cute.arch.lane_idx()
        warp_in_group = cute.arch.warp_idx() % 4

        valid_count = Int32(0)
        first_token = Int32(-1)
        stage_contiguous = Boolean(True)
        stage_all_selected = Boolean(True)

        if warp_in_group == 0:
            token = self._current_max()
            token_valid = Boolean(token >= 0 and token < self.seqlen_k)

            while valid_count < self.tile_n and token_valid:
                column = valid_count
                column_full = Boolean(True)

                for stream in cutlass.range_constexpr(self.streams_per_lane):
                    match = Boolean(
                        self.rActive[stream]
                        and self.rValue[stream] == token
                    )
                    match_word = Uint32(cute.arch.vote_ballot_sync(match))

                    if lane == 0:
                        sMembership[column, stream, stage] = match_word
                        column_full = Boolean(
                            column_full
                            and match_word == self.rActiveWord[stream]
                        )

                    if match:
                        self._advance_past(stream, token)

                if lane == 0:
                    sToken[column, stage] = token
                    sColumnFull[column, stage] = Uint16(1 if column_full else 0)
                    stage_all_selected = Boolean(stage_all_selected and column_full)
                    if valid_count == 0:
                        first_token = token
                    else:
                        stage_contiguous = Boolean(
                            stage_contiguous and token == first_token - valid_count
                        )

                valid_count += 1
                if valid_count < self.tile_n:
                    token = self._current_max()
                    token_valid = Boolean(token >= 0 and token < self.seqlen_k)

            if lane == 0:
                stage_full = Boolean(
                    valid_count == self.tile_n and stage_all_selected
                )
                sValidCount[stage] = valid_count
                sFirstToken[stage] = first_token
                sContiguous[stage] = Uint16(1 if stage_contiguous else 0)
                sStageFull[stage] = Uint16(1 if stage_full else 0)

        # Existing non-TMA K/V production uses one 128-thread warpgroup.
        cute.arch.barrier(barrier_id=7, number_of_threads=128)

        return IndexedUnionStage(
            valid_count=Int32(sValidCount[stage]),
            first_token=Int32(sFirstToken[stage]),
            is_contiguous_descending=Boolean(sContiguous[stage] != Uint16(0)),
            all_columns_full=Boolean(sStageFull[stage] != Uint16(0)),
        )


@cute.jit
def apply_membership_mask(
    acc_S: cute.Tensor,
    thr_mma_qk,
    sMembership: cute.Tensor,
    sColumnFull: cute.Tensor,
    valid_count: Int32,
    stage: Int32,
    *,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    qhead_per_kvhead: cutlass.Constexpr[int],
):
    """Apply only nontrivial per-query membership predicates."""

    cS = cute.make_identity_tensor((tile_m, tile_n))
    tScS = thr_mma_qk.partition_C(cS)
    tScS_mn = layout_utils.reshape_acc_to_mn(tScS)
    acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)

    for element in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
        packed_row = Int32(tScS_mn[element][0])
        column = Int32(tScS_mn[element][1])

        selected = Boolean(False)
        if column < valid_count:
            if sColumnFull[column, stage] != Uint16(0):
                selected = Boolean(True)
            else:
                logical_query = packed_row // qhead_per_kvhead
                word_idx = logical_query // 32
                bit_idx = logical_query - word_idx * 32
                word = Uint32(sMembership[column, word_idx, stage])
                bit = utils.shl_u32(Uint32(1), Uint32(bit_idx))
                selected = Boolean((word & bit) != Uint32(0))

        if not selected:
            acc_S_mn[element] = -Float32.inf
