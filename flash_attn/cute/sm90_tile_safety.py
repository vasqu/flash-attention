"""Pure-Python safety helpers for SM90 FA4 forward tile selection.

The SM90 forward kernel stores Q, staged K/V, and optionally P in dynamic
shared memory.  The upstream tile heuristic is tuned for common equal-head-dim
shapes; arbitrary QK/V dimension pairs can exceed Hopper's 232,448-byte launch
limit or create excessive register pressure when M=192 and V is wide.

This module keeps the original tile whenever it is safe and only shrinks the
M/N tile for uncommon asymmetric shapes.  It deliberately has no torch/CuTe
imports so policy tests can run on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass


SM90_MAX_DYNAMIC_SMEM_BYTES = 232_448
# The observed CuTe allocation is exactly the Q/K/V/P payload plus one 1 KiB
# alignment/barrier page for the affected kernels.
SM90_SMEM_OVERHEAD_BYTES = 1_024
SM90_TILE_N_CANDIDATES = (192, 176, 160, 144, 128, 112, 96, 80, 64)


@dataclass(frozen=True)
class Sm90SafeTile:
    tile_m: int
    tile_n: int
    estimated_smem_bytes: int
    changed: bool


def estimate_sm90_fwd_smem_bytes(
    *,
    tile_m: int,
    tile_n: int,
    head_dim: int,
    value_head_dim: int,
    mma_pv_is_rs: bool,
    num_stages: int = 2,
    element_bytes: int = 2,
) -> int:
    """Estimate the dynamic shared-memory allocation of FA4 SM90 forward.

    The formula matches the failing H100 allocations from the validation run:
    128x112 D192/DV256 RS -> 250,880 bytes and
    192x144 D96/DV192 non-RS -> 259,072 bytes.
    """

    if min(tile_m, tile_n, head_dim, value_head_dim, num_stages, element_bytes) <= 0:
        raise ValueError("tile and dimension values must be positive")
    q_elements = tile_m * head_dim
    staged_kv_elements = num_stages * tile_n * (head_dim + value_head_dim)
    p_elements = 0 if mma_pv_is_rs else tile_m * tile_n
    return (
        element_bytes * (q_elements + staged_kv_elements + p_elements)
        + SM90_SMEM_OVERHEAD_BYTES
    )


def choose_safe_sm90_fwd_tile(
    *,
    tile_m: int,
    tile_n: int,
    head_dim: int,
    value_head_dim: int,
    mma_pv_is_rs: bool,
    num_stages: int = 2,
    max_smem_bytes: int = SM90_MAX_DYNAMIC_SMEM_BYTES,
) -> Sm90SafeTile:
    """Return the largest launch-safe tile no larger than the tuned tile.

    Wide V with M=192 also triggered ptxas register-allocation failures.  M=128
    has the same numerical semantics and is already the standard FA4 tile for
    larger head dimensions, so asymmetric wide-V shapes use that conservative
    M tile before the N cap is applied.
    """

    safe_m = 128 if tile_m > 128 and value_head_dim > 128 else tile_m
    candidates = sorted(
        {tile_n, *(candidate for candidate in SM90_TILE_N_CANDIDATES if candidate <= tile_n)},
        reverse=True,
    )
    for candidate_n in candidates:
        allocated = estimate_sm90_fwd_smem_bytes(
            tile_m=safe_m,
            tile_n=candidate_n,
            head_dim=head_dim,
            value_head_dim=value_head_dim,
            mma_pv_is_rs=mma_pv_is_rs,
            num_stages=num_stages,
        )
        if allocated <= max_smem_bytes:
            return Sm90SafeTile(
                tile_m=safe_m,
                tile_n=candidate_n,
                estimated_smem_bytes=allocated,
                changed=(safe_m, candidate_n) != (tile_m, tile_n),
            )
    raise ValueError(
        "no supported SM90 tile fits the dynamic shared-memory limit for "
        f"D={head_dim}, DV={value_head_dim}, M={tile_m}, N={tile_n}"
    )
