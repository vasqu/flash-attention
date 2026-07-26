"""CuTe metadata compaction for exact indexed block-sparse prefill on SM90.

This kernel consumes the already-packed exact selected-token bitmask produced by
``build_topk_bitmask`` and derives FA4's masked K/V block lists.  One CTA owns
one ``(batch, query-tile)`` row, so no global atomics or temporary tensors are
required.  The active-block flags live in shared memory and thread 0 compacts
ascending block IDs into persistent output buffers.

The module is imported lazily by the benchmark-only lab path.  Production
indexed dispatch remains unchanged.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32

from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.testing import is_fake_mode


class IndexedBlockSparseMetadataSm90:
    """Build sorted active 64-token K/V block lists from an exact bitmask."""

    def __init__(
        self,
        *,
        seqlen_q: int,
        seqlen_k: int,
        tile_m: int = 128,
        tile_n: int = 64,
        num_threads: int = 128,
    ) -> None:
        if seqlen_q < 1 or seqlen_k < 1:
            raise ValueError("sequence lengths must be positive")
        if tile_m != 128 or tile_n != 64:
            raise ValueError("CuTe metadata currently supports tile_mn=(128,64)")
        if num_threads != 128:
            raise ValueError("CuTe metadata currently uses 128 threads")
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_m_blocks = cute.ceil_div(seqlen_q, tile_m)
        self.num_n_blocks = cute.ceil_div(seqlen_k, tile_n)
        self.bitmask_words = cute.ceil_div(seqlen_k, 32)

    @cute.jit
    def __call__(
        self,
        mBitmask: cute.Tensor,
        mMaskCnt: cute.Tensor,
        mMaskIdx: cute.Tensor,
        stream: cuda.CUstream,
    ):
        total_rows = mBitmask.shape[0] * self.num_m_blocks
        self.kernel(mBitmask, mMaskCnt, mMaskIdx, total_rows).launch(
            grid=(total_rows, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mBitmask: cute.Tensor,
        mMaskCnt: cute.Tensor,
        mMaskIdx: cute.Tensor,
        total_rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        if row < total_rows:
            batch_idx = row // self.num_m_blocks
            m_block = row - batch_idx * self.num_m_blocks
            q_begin = m_block * self.tile_m

            smem = cute.arch.alloc_smem(Int32, self.num_n_blocks)
            sActive = cute.make_tensor(
                smem,
                cute.make_layout((self.num_n_blocks,)),
            )

            # Each thread owns one or more K/V blocks.  A block is active when
            # either packed word for that 64-token block is non-zero in any
            # query row covered by this 128-row query tile.
            for n_block in cutlass.range(
                tidx,
                self.num_n_blocks,
                self.num_threads,
                unroll=1,
            ):
                word0 = n_block * 2
                word1 = word0 + 1
                active = Boolean(False)
                for q_local in cutlass.range(0, self.tile_m, unroll=1):
                    q_idx = q_begin + q_local
                    if q_idx < self.seqlen_q and active == Boolean(False):
                        lo = Int32(0)
                        hi = Int32(0)
                        if word0 < self.bitmask_words:
                            lo = Int32(mBitmask[batch_idx, q_idx, word0])
                        if word1 < self.bitmask_words:
                            hi = Int32(mBitmask[batch_idx, q_idx, word1])
                        active = Boolean((lo | hi) != Int32(0))
                sActive[n_block] = Int32(1) if active else Int32(0)

            cute.arch.sync_threads()

            # One CTA owns this output row, so deterministic serial compaction
            # is cheaper than a global scan and preserves ascending block IDs.
            if tidx == 0:
                count = Int32(0)
                for n_block in cutlass.range(0, self.num_n_blocks, unroll=1):
                    if sActive[n_block] != Int32(0):
                        mMaskIdx[batch_idx, 0, m_block, count] = Int32(n_block)
                        count = count + 1
                mMaskCnt[batch_idx, 0, m_block] = count
                for slot in cutlass.range(count, self.num_n_blocks, unroll=1):
                    mMaskIdx[batch_idx, 0, m_block, slot] = Int32(
                        self.num_n_blocks
                    )


_metadata_compile_cache = get_jit_cache("indexed_block_sparse_metadata_sm90")


def run_indexed_block_sparse_metadata_sm90(
    bitmask,
    mask_cnt,
    mask_idx,
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 64,
) -> None:
    """Launch the cached CuTe active-block compaction kernel."""

    bitmask_cute = to_cute_tensor(bitmask, assumed_align=4, leading_dim=2)
    mask_cnt_cute = to_cute_tensor(mask_cnt, assumed_align=4, leading_dim=2)
    mask_idx_cute = to_cute_tensor(mask_idx, assumed_align=4, leading_dim=3)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compile_key = (
        bitmask.dtype,
        tuple(bitmask.shape),
        tuple(mask_cnt.shape),
        tuple(mask_idx.shape),
        seqlen_q,
        seqlen_k,
        tile_m,
        tile_n,
    )
    if compile_key not in _metadata_compile_cache:
        kernel = IndexedBlockSparseMetadataSm90(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            tile_m=tile_m,
            tile_n=tile_n,
        )
        _metadata_compile_cache[compile_key] = cute.compile(
            kernel,
            bitmask_cute,
            mask_cnt_cute,
            mask_idx_cute,
            stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        _metadata_compile_cache[compile_key](
            bitmask.detach(),
            mask_cnt.detach(),
            mask_idx.detach(),
        )
