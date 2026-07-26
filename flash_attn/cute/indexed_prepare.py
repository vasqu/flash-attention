"""Host-side preparation helpers for exact indexed SM90 attention.

The public indexed API accepts top-k indices in arbitrary order. Only the
cross-query WGMMA union path needs descending rows for its streaming merge, so
that preparation is performed lazily after path selection. The warp-decode path
uses the original order and only sanitizes/casts when the input is not int32.
"""

from __future__ import annotations

import torch


_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64, torch.uint16)


def cast_indexed_kv_indices(
    indices: torch.Tensor,
    seqlen_k: int | None = None,
) -> torch.Tensor:
    """Return contiguous int32 indices without reordering them.

    Native int32 input is returned without an extra kernel; bounds are checked
    in the attention kernel. int64/uint16 input is sanitized before narrowing so
    large values cannot wrap into an apparently valid int32 token ID. uint16
    value 65535 is reserved as the invalid sentinel.
    """

    if indices.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices must be int32, int64, or uint16")
    if indices.dtype == torch.int32 and indices.is_contiguous():
        return indices

    work64 = indices.to(torch.int64)
    if seqlen_k is not None:
        valid = (work64 >= 0) & (work64 < seqlen_k)
        if indices.dtype == torch.uint16:
            valid &= work64 != 65_535
        work64 = torch.where(valid, work64, torch.full_like(work64, -1))
    return work64.to(dtype=torch.int32).contiguous()


def prepare_indexed_kv_indices(
    indices: torch.Tensor,
    seqlen_k: int,
) -> torch.Tensor:
    """Prepare arbitrary-order rows for the streaming union mainloop.

    Invalid or out-of-range values become ``-1`` and are moved to the tail.
    Valid entries are sorted by physical token position in descending order.
    Duplicate entries become adjacent and are skipped by the in-kernel merge.

    This function is called internally by the public indexed API. It is exposed
    only so benchmarks can report preparation cost separately from attention.
    """

    if indices.ndim != 3:
        raise ValueError("indices must have shape [batch, seqlen_q, topk]")
    if indices.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices must be int32, int64, or uint16")
    if seqlen_k < 0 or seqlen_k >= 2**31:
        raise ValueError("seqlen_k must fit in signed int32")

    if indices.dtype == torch.int32:
        work = indices.contiguous()
        valid = (work >= 0) & (work < seqlen_k)
        work = torch.where(valid, work, torch.full_like(work, -1))
    else:
        work = cast_indexed_kv_indices(indices, seqlen_k)

    if work.shape[-1] > 1:
        work = torch.sort(work, dim=-1, descending=True).values
    return work.contiguous()
