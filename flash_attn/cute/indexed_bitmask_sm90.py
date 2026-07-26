"""Native CuTe packing of exact indexed top-k bitmasks on SM90.

The previous production paths reused persistent output buffers but still built
those buffers with a sequence of PyTorch elementwise kernels plus
``scatter_add_``.  This module packs arbitrary-order int32, int64, or uint16
token indices in one CuTe launch.  One CTA owns one ``(batch, query)`` row,
clears its output words, and atomically ORs selected bits into that row.
Duplicate indices are idempotent and invalid entries are ignored.

Accepting the public index dtype directly is important for short prefill and
large batches: it removes the otherwise separate full-tensor int64-to-int32
conversion before bitmask construction.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint32
import torch

from flash_attn.cute import utils
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from flash_attn.cute.testing import is_fake_mode


class IndexedTopKBitmaskSm90:
    """Pack ``[B,Q,T]`` int32 token indices into ``[B,Q,ceil(K/32)]`` words."""

    def __init__(
        self,
        *,
        seqlen_q: int,
        seqlen_k: int,
        topk: int,
        num_threads: int = 256,
    ) -> None:
        if seqlen_q < 1 or seqlen_k < 1:
            raise ValueError("sequence lengths must be positive")
        if topk < 1:
            raise ValueError("topk must be positive")
        if num_threads not in (128, 256):
            raise ValueError("native bitmask packing uses 128 or 256 threads")
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.topk = topk
        self.num_threads = num_threads
        self.bitmask_words = cute.ceil_div(seqlen_k, 32)

    @cute.jit
    def __call__(
        self,
        mIndices: cute.Tensor,
        mBitmask: cute.Tensor,
        stream: cuda.CUstream,
    ):
        total_rows = mIndices.shape[0] * self.seqlen_q
        self.kernel(mIndices, mBitmask, total_rows).launch(
            grid=(total_rows, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIndices: cute.Tensor,
        mBitmask: cute.Tensor,
        total_rows: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        if row < total_rows:
            batch_idx = row // self.seqlen_q
            q_idx = row - batch_idx * self.seqlen_q

            # The same CTA owns all writes to this output row.  Clear first,
            # synchronize, then use relaxed atomic OR for bits that share a
            # packed word.  Atomic OR also makes duplicate indices idempotent.
            for word_idx in cutlass.range(
                tidx,
                self.bitmask_words,
                self.num_threads,
                unroll=1,
            ):
                mBitmask[batch_idx, q_idx, word_idx] = Int32(0)

            cute.arch.sync_threads()

            row_offset = (
                (batch_idx * self.seqlen_q + q_idx) * self.bitmask_words
            )
            for slot in cutlass.range(
                tidx,
                self.topk,
                self.num_threads,
                unroll=1,
            ):
                token = Int32(mIndices[batch_idx, q_idx, slot])
                valid = Boolean(token >= 0 and token < self.seqlen_k)
                if valid:
                    word_idx = token // 32
                    bit_idx = token - word_idx * 32
                    bit = Int32(
                        utils.shl_u32(Uint32(1), Uint32(bit_idx))
                    )
                    cute.arch.atomic_or(
                        mBitmask.iterator + row_offset + word_idx,
                        bit,
                    )


_bitmask_compile_cache = get_jit_cache("indexed_topk_bitmask_sm90")


def build_topk_bitmask_cute(
    indices: torch.Tensor,
    seqlen_k: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build an exact packed top-k bitmask with one native CuTe launch.

    ``indices`` may be contiguous int32, int64, or uint16.  The kernel casts
    each loaded token to signed int32 after bounds validation, avoiding a
    separate full-tensor conversion.  Passing ``out`` avoids all hot-path
    allocation; allocation is supported for focused benchmarking and explicit
    use.
    """

    if indices.ndim != 3:
        raise ValueError("indices must have shape [batch, seqlen_q, topk]")
    if not indices.is_cuda:
        raise ValueError("native indexed bitmask packing requires CUDA indices")
    if indices.dtype not in (torch.int32, torch.int64, torch.uint16):
        raise TypeError(
            "native indexed bitmask packing requires int32, int64, or uint16 indices"
        )
    if not indices.is_contiguous():
        raise ValueError("native indexed bitmask packing requires contiguous indices")
    if seqlen_k < 1 or seqlen_k >= 2**31:
        raise ValueError("seqlen_k must be positive and fit signed int32")

    batch, seqlen_q, topk = indices.shape
    words = math.ceil(seqlen_k / 32)
    expected_shape = (batch, seqlen_q, words)
    if out is None:
        out = torch.empty(expected_shape, dtype=torch.int32, device=indices.device)
    else:
        if out.shape != expected_shape:
            raise ValueError(f"out shape {tuple(out.shape)} != expected {expected_shape}")
        if out.dtype != torch.int32:
            raise TypeError("out must have dtype torch.int32")
        if out.device != indices.device:
            raise ValueError("out must be on the same device as indices")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    out.__assumed_align__ = 4
    out.__leading_dim__ = 2
    index_align = max(4, indices.element_size())
    indices_cute = to_cute_tensor(
        indices, assumed_align=index_align, leading_dim=2
    )
    out_cute = to_cute_tensor(out, assumed_align=4, leading_dim=2)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    num_threads = 128 if topk <= 512 else 256
    compile_key = (
        indices.dtype,
        tuple(indices.shape),
        tuple(out.shape),
        seqlen_k,
        num_threads,
    )
    if compile_key not in _bitmask_compile_cache:
        kernel = IndexedTopKBitmaskSm90(
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            topk=topk,
            num_threads=num_threads,
        )
        _bitmask_compile_cache[compile_key] = cute.compile(
            kernel,
            indices_cute,
            out_cute,
            stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        _bitmask_compile_cache[compile_key](indices.detach(), out.detach())
    return out
