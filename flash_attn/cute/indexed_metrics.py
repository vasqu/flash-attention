"""Offline union/load metrics for real selected-index tensors.

This is a profiling utility, not part of the attention hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class IndexedTileMetrics:
    tile_m: int
    qhead_per_kvhead: int
    logical_queries_per_tile: int
    mean_valid_per_query: float
    mean_union: float
    p50_union: float
    p90_union: float
    p99_union: float
    mean_compute_inflation: float
    mean_kv_reuse: float
    mean_dense_fraction: float
    mean_contiguous_edge_fraction: float


@torch.no_grad()
def analyze_index_tiles(
    indices: torch.Tensor,
    *,
    kv_length: int,
    tile_m: int,
    query_heads: int,
    kv_heads: int,
    pack_gqa: bool | None = None,
) -> IndexedTileMetrics:
    """Measure exact union behavior for a packed or unpacked FA4 M tile.

    The function performs a profiling-only sort over each flattened tile. It
    must not be called from the production attention path.
    """

    if indices.ndim != 3:
        raise ValueError("indices must be [B, Q, topk]")
    if query_heads % kv_heads != 0:
        raise ValueError(
            "query_heads must be divisible by kv_heads"
        )
    ratio = query_heads // kv_heads
    use_pack_gqa = (ratio > 1 and tile_m % ratio == 0) if pack_gqa is None else bool(pack_gqa)
    if use_pack_gqa and tile_m % ratio != 0:
        raise ValueError("tile_m must be divisible by qhead_per_kvhead when pack_gqa=True")

    effective_ratio = ratio if use_pack_gqa else 1
    q_per_tile = tile_m // effective_ratio
    batch, query_length, topk = indices.shape
    padded_q = (
        (query_length + q_per_tile - 1)
        // q_per_tile
        * q_per_tile
    )
    padded = F.pad(
        indices,
        (0, 0, 0, padded_q - query_length),
        value=-1,
    )
    tiles = padded.view(
        batch,
        padded_q // q_per_tile,
        q_per_tile,
        topk,
    )

    valid = (tiles >= 0) & (tiles < kv_length)
    valid_count = valid.sum(dim=-1)
    total_valid = valid_count.sum(dim=-1)

    flat = tiles.masked_fill(~valid, -1).flatten(-2)
    sorted_flat = flat.sort(
        dim=-1,
        descending=True,
    ).values
    sorted_valid = sorted_flat >= 0

    first_valid = sorted_valid[..., :1].to(torch.int64)
    transitions = (
        sorted_valid[..., 1:]
        & (sorted_flat[..., 1:] != sorted_flat[..., :-1])
    ).sum(dim=-1)
    union = first_valid.squeeze(-1) + transitions

    # Compute inflation relative to exact selected QK/PV work. The factor from
    # grouped Q heads cancels because every logical query position has `ratio`
    # packed Q rows.
    safe_total_valid = total_valid.clamp_min(1)
    inflation = q_per_tile * union / safe_total_valid
    kv_reuse = safe_total_valid / union.clamp_min(1)
    dense_fraction = union / max(kv_length, 1)

    # Adjacent union values differing by one indicate a descending contiguous
    # run edge. This estimates how often the base+stride loader can be used.
    adjacent_valid = sorted_valid[..., 1:]
    contiguous_edges = (
        adjacent_valid
        & (sorted_flat[..., :-1] - sorted_flat[..., 1:] == 1)
    ).sum(dim=-1)
    contiguous_fraction = (
        contiguous_edges
        / (union - 1).clamp_min(1)
    )

    union_float = union.float()
    quantiles = torch.quantile(
        union_float.flatten(),
        torch.tensor(
            [0.5, 0.9, 0.99],
            device=union.device,
        ),
    )

    return IndexedTileMetrics(
        tile_m=tile_m,
        qhead_per_kvhead=effective_ratio,
        logical_queries_per_tile=q_per_tile,
        mean_valid_per_query=float(
            valid_count.float().mean().item()
        ),
        mean_union=float(union_float.mean().item()),
        p50_union=float(quantiles[0].item()),
        p90_union=float(quantiles[1].item()),
        p99_union=float(quantiles[2].item()),
        mean_compute_inflation=float(
            inflation.float().mean().item()
        ),
        mean_kv_reuse=float(
            kv_reuse.float().mean().item()
        ),
        mean_dense_fraction=float(
            dense_fraction.float().mean().item()
        ),
        mean_contiguous_edge_fraction=float(
            contiguous_fraction.float().mean().item()
        ),
    )
