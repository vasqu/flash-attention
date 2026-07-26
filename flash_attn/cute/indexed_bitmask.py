"""Persistent exact indexed bitmasks for short SM90 FA4 prefill.

The short GLM true-prefill specialization uses FA4's ordinary 128x64 forward
with the existing exact top-k bitmask score modifier.  A small stream-local LRU
cache keeps the bitmask output buffer alive across calls, so the hot path only
launches one native CuTe bitmask packer and the FA4 kernel.
"""

from __future__ import annotations

from collections import OrderedDict
import math
import os
from threading import RLock, get_ident, local
import weakref

import torch


class IndexedBitmaskWorkspace:
    """Persistent exact bitmask storage for one fixed indexed shape."""

    def __init__(
        self,
        *,
        batch: int,
        seqlen_q: int,
        seqlen_k: int,
        device: torch.device,
    ) -> None:
        if batch < 1 or seqlen_q < 1 or seqlen_k < 1:
            raise ValueError("batch and sequence lengths must be positive")
        self.batch = batch
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.device = device
        self.bitmask_words = math.ceil(seqlen_k / 32)
        self.bitmask = torch.empty(
            (batch, seqlen_q, self.bitmask_words),
            dtype=torch.int32,
            device=device,
        )
        self.bitmask.__assumed_align__ = 4
        self.bitmask.__leading_dim__ = 2

    @property
    def nbytes(self) -> int:
        return self.bitmask.numel() * self.bitmask.element_size()

    def validate(self, indices: torch.Tensor) -> None:
        if indices.device != self.device:
            raise ValueError("workspace and indices must be on the same device")
        if indices.shape[:2] != (self.batch, self.seqlen_q):
            raise ValueError("workspace batch/query shape does not match indices")
        if indices.dtype not in (torch.int32, torch.int64, torch.uint16):
            raise TypeError(
                "cached indexed bitmask requires int32, int64, or uint16 indices"
            )
        if not indices.is_contiguous():
            raise ValueError("cached indexed bitmask requires contiguous indices")


_MAX_CACHED_WORKSPACES = 8
_MAX_CACHED_WORKSPACE_BYTES = int(
    float(os.getenv("FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB", "1024"))
    * 1024
    * 1024
)
_workspace_cache: OrderedDict[
    tuple[int, int, int, int, int, int], IndexedBitmaskWorkspace
] = OrderedDict()
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
        raise ValueError("indexed bitmask workspace caching requires CUDA indices")
    device_index = indices.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream_id = int(torch.cuda.current_stream(indices.device).cuda_stream)
    return device_index, stream_id, get_ident(), indices.shape[0], seqlen_q, seqlen_k


def get_cached_indexed_bitmask_workspace(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
) -> IndexedBitmaskWorkspace:
    """Return a stream-local reusable bitmask workspace for a fixed shape."""

    global _workspace_hot_hits, _workspace_cache_hits, _workspace_cache_misses

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
                    "warm up indexed FA4 bitmask attention before CUDA graph capture"
                )
            workspace = IndexedBitmaskWorkspace(
                batch=indices.shape[0],
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                device=indices.device,
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


def build_indexed_topk_bitmask_cached(
    indices: torch.Tensor,
    *,
    seqlen_q: int,
    seqlen_k: int,
) -> torch.Tensor:
    """Build the exact top-k bitmask into persistent stream-local storage."""

    if indices.ndim != 3:
        raise ValueError("indices must have shape [batch, seqlen_q, topk]")
    if indices.shape[1] != seqlen_q:
        raise ValueError("indices query dimension must match seqlen_q")
    workspace = get_cached_indexed_bitmask_workspace(
        indices,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
    )
    from flash_attn.cute.indexed_bitmask_sm90 import build_topk_bitmask_cute

    build_topk_bitmask_cute(
        indices,
        seqlen_k,
        out=workspace.bitmask,
    )
    return workspace.bitmask


def indexed_bitmask_workspace_cache_stats() -> dict[str, int]:
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


def clear_indexed_bitmask_workspace_cache() -> None:
    """Release all cached short-prefill exact bitmask buffers."""

    global _workspace_cache_generation

    with _workspace_cache_lock:
        _workspace_cache.clear()
        _workspace_cache_generation += 1
    _workspace_hot.entry = None
