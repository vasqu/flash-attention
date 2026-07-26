"""Production exact block-sparse indexed prefill for SM90.

The implementation preserves exact selected-token semantics by combining FA4's
ordinary top-k bitmask score modifier with a compact list of active 64-token K/V
blocks for each 128-query tile.  Metadata is prepared entirely with native
PyTorch/CuTe code and consumed by the existing FA4 SM90 block-sparse forward.

Automatic dispatch is intentionally restricted to the measured GLM MoE DSA
prefill family.  Workspaces are cached per CUDA stream so repeated model calls
do not allocate metadata buffers in the hot path.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import os
from threading import RLock, get_ident, local
import weakref

import torch


class IndexedBlockSparseCuteWorkspace:
    """Persistent exact metadata buffers for exact CuTe 128x64 sparse prefill."""

    def __init__(
        self,
        *,
        batch: int,
        seqlen_q: int,
        seqlen_k: int,
        tile_m: int,
        tile_n: int,
        device: torch.device,
    ) -> None:
        if tile_m != 128 or tile_n != 64:
            raise ValueError("indexed block-sparse metadata supports tile_mn=(128,64)")
        if batch < 1 or seqlen_q < 1 or seqlen_k < 1:
            raise ValueError("batch and sequence lengths must be positive")

        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch

        self.batch = batch
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.device = device
        self.num_m_blocks = math.ceil(seqlen_q / tile_m)
        self.num_n_blocks = math.ceil(seqlen_k / tile_n)
        self.bitmask_words = math.ceil(seqlen_k / 32)

        self.bitmask = torch.empty(
            (batch, seqlen_q, self.bitmask_words),
            dtype=torch.int32,
            device=device,
        )
        self.mask_cnt = torch.empty(
            (batch, 1, self.num_m_blocks),
            dtype=torch.int32,
            device=device,
        )
        self.mask_idx = torch.empty(
            (batch, 1, self.num_m_blocks, self.num_n_blocks),
            dtype=torch.int32,
            device=device,
        )
        # Every scheduled tile remains a masked tile.  Materialize immutable
        # empty full-block tensors for compatibility with older FA4 SM90 paths.
        self.full_cnt = torch.zeros_like(self.mask_cnt)
        self.full_idx = torch.zeros_like(self.mask_idx)
        self.block_sparse_tensors = BlockSparseTensorsTorch(
            mask_block_cnt=self.mask_cnt,
            mask_block_idx=self.mask_idx,
            full_block_cnt=self.full_cnt,
            full_block_idx=self.full_idx,
            block_size=(tile_m, tile_n),
        )

        self.bitmask.__assumed_align__ = 4
        self.bitmask.__leading_dim__ = 2

    @property
    def nbytes(self) -> int:
        tensors = (
            self.bitmask,
            self.mask_cnt,
            self.mask_idx,
            self.full_cnt,
            self.full_idx,
        )
        return sum(t.numel() * t.element_size() for t in tensors)

    def validate(self, indices: torch.Tensor) -> None:
        if indices.device != self.device:
            raise ValueError("workspace and indices must be on the same device")
        if indices.shape[:2] != (self.batch, self.seqlen_q):
            raise ValueError("workspace batch/query shape does not match indices")
        if indices.dtype not in (torch.int32, torch.int64, torch.uint16):
            raise TypeError("CuTe metadata requires int32, int64, or uint16 indices")
        if not indices.is_contiguous():
            raise ValueError("CuTe metadata requires contiguous indices")


def create_indexed_block_sparse_cute_workspace(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 64,
) -> IndexedBlockSparseCuteWorkspace:
    """Allocate persistent outputs for exact CuTe metadata preparation."""

    if indices.ndim != 3:
        raise ValueError("indices must have shape [batch, seqlen_q, topk]")
    if indices.shape[1] != seqlen_q:
        raise ValueError("indices query dimension must match seqlen_q")
    return IndexedBlockSparseCuteWorkspace(
        batch=indices.shape[0],
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
        device=indices.device,
    )


def build_indexed_block_sparse_tensors_cute(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 64,
    workspace: IndexedBlockSparseCuteWorkspace | None = None,
):
    """Build exact bitmask and active-block metadata with persistent buffers.

    A native CuTe kernel packs the exact selected-token bitmask directly from
    int32 indices.  The existing CuTe compactor then scans that bitmask and
    compacts active 64-token K/V
    blocks for each 128-query tile.  The timed path performs no allocations,
    sorting, large boolean materialization, or host synchronization.

    Returns ``(block_sparse_tensors, bitmask, workspace)``.
    """

    if not indices.is_cuda:
        raise ValueError("CuTe metadata requires CUDA indices")
    if indices.dtype not in (torch.int32, torch.int64, torch.uint16):
        raise TypeError("CuTe metadata requires int32, int64, or uint16 indices")
    if tile_m != 128 or tile_n != 64:
        raise ValueError("indexed block-sparse metadata supports tile_mn=(128,64)")
    if workspace is None:
        workspace = create_indexed_block_sparse_cute_workspace(
            indices,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            tile_m=tile_m,
            tile_n=tile_n,
        )
    workspace.validate(indices)
    if workspace.seqlen_k != seqlen_k:
        raise ValueError("workspace KV length does not match seqlen_k")

    from flash_attn.cute.indexed_block_sparse_metadata_sm90 import (
        run_indexed_block_sparse_metadata_sm90,
    )
    from flash_attn.cute.indexed_bitmask_sm90 import build_topk_bitmask_cute

    build_topk_bitmask_cute(
        indices,
        seqlen_k,
        out=workspace.bitmask,
    )
    run_indexed_block_sparse_metadata_sm90(
        workspace.bitmask,
        workspace.mask_cnt,
        workspace.mask_idx,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    return workspace.block_sparse_tensors, workspace.bitmask, workspace


_MAX_CACHED_WORKSPACES = 8
_MAX_CACHED_WORKSPACE_BYTES = int(
    float(os.getenv("FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB", "1024"))
    * 1024
    * 1024
)
_workspace_cache: OrderedDict[tuple[int, int, int, int, int, int], IndexedBlockSparseCuteWorkspace] = OrderedDict()
_workspace_cache_lock = RLock()
_workspace_hot = local()
_workspace_cache_generation = 0
_workspace_hot_hits = 0
_workspace_cache_hits = 0
_workspace_cache_misses = 0


def _workspace_cache_key(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
) -> tuple[int, int, int, int, int, int]:
    if not indices.is_cuda:
        raise ValueError("indexed block-sparse workspace caching requires CUDA indices")
    device_index = indices.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(indices.device).cuda_stream)
    return device_index, stream_id, get_ident(), indices.shape[0], seqlen_q, seqlen_k


def get_cached_indexed_block_sparse_cute_workspace(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 64,
) -> IndexedBlockSparseCuteWorkspace:
    """Return a stream-local reusable workspace for a fixed indexed shape.

    CUDA work submitted by one host thread to one stream is ordered, so a
    workspace can be safely reused by successive calls.  Different streams and
    host threads receive independent buffers.  The small LRU bound prevents
    shape exploration from retaining unbounded bitmask storage.
    """

    global _workspace_hot_hits, _workspace_cache_hits, _workspace_cache_misses

    if tile_m != 128 or tile_n != 64:
        raise ValueError("indexed block-sparse metadata supports tile_mn=(128,64)")
    key = _workspace_cache_key(indices, seqlen_q=seqlen_q, seqlen_k=seqlen_k)
    hot = getattr(_workspace_hot, "entry", None)
    if hot is not None and hot[0] == _workspace_cache_generation and hot[1] == key:
        workspace = hot[2]()
        if workspace is not None:
            workspace.validate(indices)
            _workspace_hot_hits += 1
            return workspace

    with _workspace_cache_lock:
        workspace = _workspace_cache.pop(key, None)
        if workspace is None:
            _workspace_cache_misses += 1
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm up indexed block-sparse attention before CUDA graph capture"
                )
            workspace = create_indexed_block_sparse_cute_workspace(
                indices,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                tile_m=tile_m,
                tile_n=tile_n,
            )
        else:
            _workspace_cache_hits += 1
        workspace.validate(indices)
        _workspace_cache[key] = workspace
        total_bytes = sum(item.nbytes for item in _workspace_cache.values())
        while (
            len(_workspace_cache) > _MAX_CACHED_WORKSPACES
            or (
                total_bytes > _MAX_CACHED_WORKSPACE_BYTES
                and len(_workspace_cache) > 1
            )
        ):
            _, evicted = _workspace_cache.popitem(last=False)
            total_bytes -= evicted.nbytes
        _workspace_hot.entry = (
            _workspace_cache_generation,
            key,
            weakref.ref(workspace),
        )
        return workspace


def indexed_block_sparse_workspace_cache_stats() -> dict[str, int]:
    """Return bounded-cache occupancy for batch-scaling diagnostics."""

    with _workspace_cache_lock:
        return {
            "entries": len(_workspace_cache),
            "bytes": sum(item.nbytes for item in _workspace_cache.values()),
            "limit_bytes": _MAX_CACHED_WORKSPACE_BYTES,
            "hot_hits": _workspace_hot_hits,
            "cache_hits": _workspace_cache_hits,
            "cache_misses": _workspace_cache_misses,
        }


def clear_indexed_block_sparse_workspace_cache() -> None:
    """Release all cached indexed block-sparse metadata buffers."""

    global _workspace_cache_generation

    with _workspace_cache_lock:
        _workspace_cache.clear()
        _workspace_cache_generation += 1
    _workspace_hot.entry = None


def build_indexed_block_sparse_tensors_cute_cached(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
    tile_m: int = 128,
    tile_n: int = 64,
):
    """Prepare exact sparse metadata using a persistent stream-local workspace."""

    workspace = get_cached_indexed_block_sparse_cute_workspace(
        indices,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
    )
    return build_indexed_block_sparse_tensors_cute(
        indices,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        tile_m=tile_m,
        tile_n=tile_n,
        workspace=workspace,
    )
