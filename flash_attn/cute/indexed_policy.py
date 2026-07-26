"""CuTe-only workload policy for exact indexed attention on SM90.

The public indexed API always stays inside the FA4 CuTe extension. Automatic
selection chooses among six internal kernels:

* ``row_sparse``: direct arbitrary-order selected-token traversal, with K/V
  register reuse across GQA heads for multi-token workloads;
* ``union_fa4`` / ``packed_union_fa4``: cross-query WGMMA union for overlap or
  high-ratio MQA cases where tensor-core reuse outweighs union inflation;
* ``fa4_bitmask_indexed``: ordinary exact FA4 128x64 traversal with a cached
  indexed bitmask for measured short true-prefill families;
* ``block_sparse_indexed``: exact 128x64 FA4 block-sparse traversal for
  measured long true-prefill families, with native CuTe metadata;
* ``dense_indexed``: ordinary FA4 tensor-core traversal with exact indexed
  membership, compiled and launched directly by the indexed API.

There is no PyTorch SDPA dispatch. Row, union, and dense-indexed kernels stay
inside the indexed CuTe extension. Measured GLM and common-profile prefill
specializations reuse FA4's exact SM90 forward with either a cached indexed
bitmask or native block-sparse metadata. SDPA remains a benchmark reference only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import math


_MAX_DECODE_SPLITS = 16
_PRODUCTION_TOPK = 2048
_LONG_CONTEXT = 65_536
_UINT16_INDEX_LIMIT = 65_535

# Complete multi-hour qualification families whose every true-prefill case
# selected an exact FA4 bitmask or block-sparse candidate by at least 1.46x,
# with no correctness failures. GLM D256 keeps its separately tuned batch
# crossover. Ratio-8 GQA was promoted independently by the follow-up 252-case
# run; its worst measured best-path improvement was 1.494x.
_COMMON_EXACT_PREFILL_PROFILES = frozenset({
    (128, 128, 192, 128),  # DeepSeek-style MHA
    (32, 32, 128, 128),    # ordinary D128 MHA
    (32, 8, 128, 128),     # ratio-4 GQA
    (64, 8, 128, 128),     # ratio-8 GQA
    (64, 1, 128, 128),     # ratio-64 MQA
    (32, 32, 64, 64),      # small-head MHA
})


class IndexedPath(str, Enum):
    ROW_SPARSE = "row_sparse"
    UNION_FA4 = "union_fa4"
    PACKED_UNION_FA4 = "packed_union_fa4"
    FA4_BITMASK_INDEXED = "fa4_bitmask_indexed"
    BLOCK_SPARSE_INDEXED = "block_sparse_indexed"
    DENSE_INDEXED = "dense_indexed"


@dataclass(frozen=True)
class IndexedPlan:
    path: IndexedPath
    tile_m: int
    tile_n: int
    num_stages: int
    pack_gqa: bool
    qhead_per_kvhead: int
    logical_queries_per_tile: int
    mask_words: int
    decode_splits: int
    decode_rows_per_cta: int
    decode_heads_per_warp: int
    use_uint16_indices: bool
    cache_recent_tail_in_l1: bool
    reason: str


@dataclass(frozen=True)
class _Workload:
    batch_size: int
    query_length: int
    kv_length: int
    query_heads: int
    kv_heads: int
    qk_head_dim: int
    value_head_dim: int
    topk: int

    @property
    def ratio(self) -> int:
        return self.query_heads // self.kv_heads

    @property
    def max_head_dim(self) -> int:
        return max(self.qk_head_dim, self.value_head_dim)

    @property
    def batch_query_rows(self) -> int:
        return self.batch_size * self.query_length

    @property
    def total_head_rows(self) -> int:
        return self.batch_query_rows * self.query_heads

    @property
    def is_decode(self) -> bool:
        return self.query_length == 1

    @property
    def is_large_true_prefill(self) -> bool:
        return self.query_length == self.kv_length and self.query_length >= 1024

    @property
    def has_64_aligned_dims(self) -> bool:
        return self.qk_head_dim % 64 == 0 and self.value_head_dim % 64 == 0


_BACKEND_ALIASES = {
    "auto": "auto",
    "row": "row_sparse",
    "sparse": "row_sparse",
    "warp": "row_sparse",
    "decode": "row_sparse",
    "row_sparse": "row_sparse",
    "warp_decode": "row_sparse",
    "grouped": "row_grouped",
    "warp_grouped": "row_grouped",
    "grouped_warp": "row_grouped",
    "scalar": "row_scalar",
    "warp_scalar": "row_scalar",
    "scalar_warp": "row_scalar",
    "dense": "dense_indexed",
    "indexed_dense": "dense_indexed",
    "dense_indexed": "dense_indexed",
    "fa4": "fa4_bitmask_indexed",
    "fa4_bitmask": "fa4_bitmask_indexed",
    "fa4_bitmask_indexed": "fa4_bitmask_indexed",
    "fa4_bitmask_general": "fa4_bitmask_indexed",
    "general_fa4_bitmask": "fa4_bitmask_indexed",
    "short_prefill": "fa4_bitmask_indexed",
    "block_sparse": "block_sparse_indexed",
    "block_sparse_indexed": "block_sparse_indexed",
    "block_sparse_general": "block_sparse_indexed",
    "general_block_sparse": "block_sparse_indexed",
    "sparse_prefill": "block_sparse_indexed",
    "union": "union_fa4",
    "wgmma": "union_fa4",
    "union_fa4": "union_fa4",
    "packed_union": "union_fa4",
    "packed_union_fa4": "union_fa4",
}


def normalize_indexed_backend(backend: str | None) -> str:
    value = "auto" if backend is None else str(backend).strip().lower().replace("-", "_")
    if value not in _BACKEND_ALIASES:
        allowed = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(f"unknown indexed backend {backend!r}; expected one of: {allowed}")
    return _BACKEND_ALIASES[value]


def _validate_workload(workload: _Workload) -> None:
    if workload.query_heads % workload.kv_heads != 0:
        raise ValueError("query_heads must be divisible by kv_heads")
    if workload.query_length < 1 or workload.kv_length < 1 or workload.topk < 1:
        raise ValueError("query_length, kv_length, and topk must be positive")


def _normalize_union_hint(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value < 1.0:
        raise ValueError("union_compute_inflation_hint must be finite and >= 1")
    return value


def _union_tile(qk_head_dim: int, value_head_dim: int) -> tuple[int, int, int]:
    max_dim = max(qk_head_dim, value_head_dim)
    if max_dim <= 96:
        return 192, 128, 2
    if max_dim <= 128:
        return 128, 128, 2
    if max_dim <= 192:
        return 128, 96, 2
    return 128, 64, 2


def _decode_splits(
    *,
    batch_size: int,
    query_length: int,
    warp_groups_per_query: int,
    topk: int,
    sm_count: int,
    rows_per_cta: int = 4,
    target_ctas_per_sm: int = 2,
    min_keys_per_split: int = 128,
    max_splits: int = _MAX_DECODE_SPLITS,
) -> int:
    groups = batch_size * query_length * warp_groups_per_query
    group_ctas = max(1, math.ceil(groups / rows_per_cta))
    occupancy_splits = math.ceil(max(1, sm_count * target_ctas_per_sm) / group_ctas)
    work_splits = max(1, math.ceil(topk / min_keys_per_split))
    return max(1, min(max_splits, occupancy_splits, work_splits))


def _grouped_heads_per_warp(
    *,
    query_length: int,
    qhead_per_kvhead: int,
    qk_head_dim: int,
    value_head_dim: int,
) -> int:
    if query_length <= 1 or qhead_per_kvhead <= 1:
        return 1
    values_per_head_per_lane = math.ceil(qk_head_dim / 32) + math.ceil(value_head_dim / 32)
    register_budget_heads = max(1, 96 // max(1, values_per_head_per_lane + 3))
    cap = max(1, min(qhead_per_kvhead, 8, register_budget_heads))
    for heads_per_warp in range(cap, 0, -1):
        if qhead_per_kvhead % heads_per_warp == 0:
            return heads_per_warp
    return 1


def _is_deepseek_v32_dsa_decode(workload: _Workload) -> bool:
    """Exact measured DeepSeek-V3.2 DSA decode family."""

    return (
        workload.is_decode
        and workload.query_heads == workload.kv_heads == 128
        and workload.qk_head_dim == 192
        and workload.value_head_dim == 128
        and workload.kv_length >= 32_768
        and 1024 <= workload.topk <= 4096
    )


def _is_glm_moe_dsa_decode(workload: _Workload) -> bool:
    """Exact measured GLM MoE DSA decode family."""

    return (
        workload.is_decode
        and workload.query_heads == workload.kv_heads == 64
        and workload.qk_head_dim == 256
        and workload.value_head_dim == 256
        and workload.kv_length >= 32_768
        and 1024 <= workload.topk <= 4096
    )


def _is_glm_moe_dsa_prefill_shape(workload: _Workload) -> bool:
    return (
        workload.query_length == workload.kv_length
        and workload.query_length >= 512
        and workload.query_heads == workload.kv_heads == 64
        and workload.qk_head_dim == 256
        and workload.value_head_dim == 256
        and workload.topk <= workload.kv_length
    )


def _is_glm_moe_dsa_short_prefill(workload: _Workload) -> bool:
    """Measured low-batch GLM true-prefill family for exact FA4 bitmasks.

    The B1--B16 sweep keeps ordinary FA4 only where it was the measured winner:
    S=512 at B1/B2 and S=1024 at B1.  Larger batches cross to the exact sparse
    scheduler even at short sequence lengths.
    """

    if not _is_glm_moe_dsa_prefill_shape(workload):
        return False
    return (
        workload.query_length == 512 and workload.batch_size <= 2
    ) or (
        workload.query_length == 1024 and workload.batch_size == 1
    )


def _is_glm_moe_dsa_sparse_prefill(workload: _Workload) -> bool:
    """Measured exact GLM MoE DSA sparse-prefill family across batch sizes.

    The batch sweep promotes only directly measured regions:

    * S=512: B4/B8/B16;
    * S=1024: B2/B4/B8/B16;
    * S>=2048: B1/B2/B4/B8.

    Long B16 shapes remain unmeasured and therefore retain the general policy.
    """

    if not _is_glm_moe_dsa_prefill_shape(workload):
        return False
    if workload.query_length == 512:
        return 4 <= workload.batch_size <= 16
    if workload.query_length == 1024:
        return 2 <= workload.batch_size <= 16
    return workload.query_length >= 2048 and workload.batch_size <= 8


def _is_common_exact_prefill_shape(workload: _Workload) -> bool:
    """Measured non-GLM true-prefill profiles from the 3h qualification."""

    profile = (
        workload.query_heads,
        workload.kv_heads,
        workload.qk_head_dim,
        workload.value_head_dim,
    )
    return (
        workload.query_length == workload.kv_length
        and workload.topk <= workload.kv_length
        and profile in _COMMON_EXACT_PREFILL_PROFILES
    )


def _is_common_short_prefill(workload: _Workload) -> bool:
    """Short common profiles where exact native-bitmask FA4 won end to end."""

    return (
        _is_common_exact_prefill_shape(workload)
        and workload.query_length in (512, 1024)
        and workload.batch_size <= 16
    )


def _is_common_sparse_prefill(workload: _Workload) -> bool:
    """Long common profiles with directly measured exact sparse promotion."""

    if not _is_common_exact_prefill_shape(workload):
        return False
    measured_batch_limit = {2048: 8, 4096: 4, 8192: 2}.get(
        workload.query_length
    )
    return (
        measured_batch_limit is not None
        and workload.batch_size <= measured_batch_limit
    )


def _is_deepseek_high_batch_q4_row(workload: _Workload) -> bool:
    """Measured DeepSeek Q4/B8 chunk crossover from dense to direct rows."""

    return (
        workload.batch_size >= 8
        and workload.query_length == 4
        and workload.kv_length >= 16_384
        and workload.topk <= 1024
        and workload.query_heads == workload.kv_heads == 128
        and workload.qk_head_dim == 192
        and workload.value_head_dim == 128
    )


def _row_geometry(
    workload: _Workload,
    *,
    heads_per_warp: int,
    sm_count: int,
) -> tuple[int, int]:
    """Return stable row CTA geometry and split count.

    Most workloads use the occupancy heuristic. Two exact DSA decode families
    use schedules that won consistently in both BF16 and FP16 sweeps:

    * DeepSeek-V3.2 DSA: four rows per CTA and 16 splits;
    * GLM MoE DSA: eight rows per CTA and 16 splits.
    """

    rows_per_cta = 8 if workload.max_head_dim <= 128 else 4
    minimum_splits = 1
    if _is_deepseek_v32_dsa_decode(workload):
        minimum_splits = _MAX_DECODE_SPLITS
    elif _is_glm_moe_dsa_decode(workload):
        rows_per_cta = 8
        minimum_splits = _MAX_DECODE_SPLITS

    groups_per_query = workload.kv_heads * math.ceil(workload.ratio / heads_per_warp)
    splits = _decode_splits(
        batch_size=workload.batch_size,
        query_length=workload.query_length,
        warp_groups_per_query=groups_per_query,
        topk=workload.topk,
        sm_count=sm_count,
        rows_per_cta=rows_per_cta,
    )
    return rows_per_cta, max(splits, minimum_splits)


def _wide_near_dense_decode(workload: _Workload) -> bool:
    return (
        workload.is_decode
        and workload.qk_head_dim >= 192
        and workload.value_head_dim >= 192
        and workload.topk * 8 >= workload.kv_length * 7
    )


def _hinted_general_union(workload: _Workload, hint: float | None) -> bool:
    return (
        hint is not None
        and hint <= 1.25
        and 2 <= workload.ratio < 32
        and 64 <= workload.query_length <= 256
        and workload.batch_query_rows <= 256
        and workload.kv_length >= 16_384
        and 512 <= workload.topk <= _PRODUCTION_TOPK
        and workload.max_head_dim <= 128
        and workload.has_64_aligned_dims
    )


def _wide_near_dense_short_query(workload: _Workload) -> bool:
    return (
        workload.ratio < 32
        and 2 <= workload.query_length <= 12
        and workload.total_head_rows >= 512
        and workload.value_head_dim >= 192
        and workload.topk > _PRODUCTION_TOPK
        and workload.topk * 8 >= workload.kv_length * 7
    )


def _saturated_short_query(workload: _Workload) -> bool:
    return (
        workload.ratio < 32
        and workload.query_length <= 8
        and workload.batch_query_rows >= 256
        and workload.max_head_dim <= 128
        and workload.topk <= _PRODUCTION_TOPK
    )


def _large_wide_dense_work(workload: _Workload) -> bool:
    return (
        workload.ratio < 32
        and workload.total_head_rows >= 32_768
        and workload.topk * 5 >= workload.kv_length
        and workload.max_head_dim >= 192
    )


def _sparse_long_context_gqa_row(workload: _Workload, grouped: int) -> bool:
    return (
        2 <= workload.ratio < 32
        and grouped > 1
        and workload.batch_query_rows <= 256
        and 64 <= workload.query_length <= 256
        and workload.max_head_dim <= 128
        and workload.topk * 16 < workload.kv_length
        and (
            (workload.total_head_rows == 4096 and workload.topk <= _PRODUCTION_TOPK)
            or (
                4096 < workload.total_head_rows <= 8192
                and workload.topk == _PRODUCTION_TOPK
                and workload.kv_length >= _LONG_CONTEXT
            )
        )
    )


def _sparse_long_context_mha_row(workload: _Workload) -> bool:
    return (
        workload.ratio == 1
        and workload.batch_size == 1
        and workload.query_length == 128
        and workload.query_heads == workload.kv_heads == 32
        and workload.kv_length >= _LONG_CONTEXT
        and workload.topk == _PRODUCTION_TOPK
        and workload.topk * 16 < workload.kv_length
        and workload.max_head_dim <= 128
    )


@lru_cache(maxsize=1024)
def choose_indexed_plan(
    *,
    batch_size: int,
    query_length: int,
    kv_length: int,
    query_heads: int,
    kv_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    topk: int,
    sm_count: int = 120,
    backend: str | None = "auto",
    has_score_mod: bool = False,
    union_compute_inflation_hint: float | None = None,
) -> IndexedPlan:
    workload = _Workload(
        batch_size=batch_size,
        query_length=query_length,
        kv_length=kv_length,
        query_heads=query_heads,
        kv_heads=kv_heads,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
        topk=topk,
    )
    _validate_workload(workload)
    requested = normalize_indexed_backend(backend)
    union_hint = _normalize_union_hint(union_compute_inflation_hint)

    ratio = workload.ratio
    tile_m, tile_n, stages = _union_tile(qk_head_dim, value_head_dim)
    pack_gqa = ratio > 1 and tile_m % ratio == 0
    logical_q = tile_m // ratio if pack_gqa else tile_m
    mask_words = math.ceil(logical_q / 32)
    grouped = _grouped_heads_per_warp(
        query_length=query_length,
        qhead_per_kvhead=ratio,
        qk_head_dim=qk_head_dim,
        value_head_dim=value_head_dim,
    )
    use_uint16 = kv_length < _UINT16_INDEX_LIMIT

    def row(reason: str, heads_per_warp: int | None = None) -> IndexedPlan:
        hpw = grouped if heads_per_warp is None else heads_per_warp
        if hpw < 1 or hpw > 8 or hpw > ratio or ratio % hpw:
            raise ValueError("invalid row-sparse heads_per_warp")
        rows_per_cta, splits = _row_geometry(
            workload,
            heads_per_warp=hpw,
            sm_count=sm_count,
        )
        return IndexedPlan(
            IndexedPath.ROW_SPARSE,
            0,
            0,
            0,
            False,
            ratio,
            1,
            0,
            splits,
            rows_per_cta,
            hpw,
            use_uint16,
            True,
            reason,
        )

    def dense(reason: str) -> IndexedPlan:
        rows_per_cta = 8 if workload.max_head_dim <= 128 else 4
        return IndexedPlan(
            IndexedPath.DENSE_INDEXED,
            0,
            0,
            2,
            False,
            ratio,
            1,
            0,
            1,
            rows_per_cta,
            1,
            use_uint16,
            False,
            reason,
        )

    def fa4_bitmask(reason: str) -> IndexedPlan:
        return IndexedPlan(
            IndexedPath.FA4_BITMASK_INDEXED,
            128,
            64,
            2,
            False,
            ratio,
            128,
            4,
            1,
            4,
            1,
            False,
            False,
            reason,
        )

    def block_sparse(reason: str) -> IndexedPlan:
        return IndexedPlan(
            IndexedPath.BLOCK_SPARSE_INDEXED,
            128,
            64,
            2,
            False,
            ratio,
            128,
            4,
            1,
            4,
            1,
            False,
            False,
            reason,
        )

    def union(reason: str) -> IndexedPlan:
        path = IndexedPath.PACKED_UNION_FA4 if pack_gqa else IndexedPath.UNION_FA4
        rows_per_cta = 8 if workload.max_head_dim <= 128 else 4
        return IndexedPlan(
            path,
            tile_m,
            tile_n,
            stages,
            pack_gqa,
            ratio,
            logical_q,
            mask_words,
            1,
            rows_per_cta,
            1,
            use_uint16,
            False,
            reason,
        )

    if requested == "row_sparse":
        return row("row-sparse CuTe backend explicitly requested")
    if requested == "row_grouped":
        if grouped <= 1:
            raise ValueError("row_grouped requires multi-token GQA/MQA with ratio > 1")
        return row("grouped row-sparse CuTe backend explicitly requested", grouped)
    if requested == "row_scalar":
        return row("scalar row-sparse CuTe backend explicitly requested", 1)
    if requested == "dense_indexed":
        if has_score_mod:
            raise ValueError("dense_indexed cannot compose a user score_mod; use row or union")
        return dense("dense indexed CuTe backend explicitly requested")
    if requested == "fa4_bitmask_indexed":
        if has_score_mod:
            raise ValueError("fa4_bitmask_indexed cannot compose a user score_mod")
        if not (
            workload.query_length == workload.kv_length
            and workload.query_length >= 128
            and workload.topk <= workload.kv_length
            and workload.qk_head_dim in {64, 96, 128, 192, 256}
            and workload.value_head_dim in {64, 96, 128, 192, 256}
        ):
            raise ValueError(
                "fa4_bitmask_indexed explicit exploration requires true prefill: "
                "Q=K>=128, topk<=K, and supported SM90 head dimensions"
            )
        return fa4_bitmask(
            "exact native-bitmask FA4 128x64 backend explicitly requested"
        )
    if requested == "block_sparse_indexed":
        if has_score_mod:
            raise ValueError("block_sparse_indexed cannot compose a user score_mod")
        if not (
            workload.query_length == workload.kv_length
            and workload.query_length >= 512
            and workload.topk <= workload.kv_length
            and workload.qk_head_dim in {64, 96, 128, 192, 256}
            and workload.value_head_dim in {64, 96, 128, 192, 256}
        ):
            raise ValueError(
                "block_sparse_indexed explicit exploration requires true prefill: "
                "Q=K>=512, topk<=K, and supported SM90 head dimensions"
            )
        return block_sparse("exact 128x64 block-sparse CuTe backend explicitly requested")
    if requested == "union_fa4":
        if query_length < 32:
            raise ValueError("union_fa4 requires query_length >= 32; use row for short query tiles")
        if not workload.has_64_aligned_dims:
            raise ValueError("union_fa4 requires QK and V dimensions divisible by 64")
        return union("cross-query indexed WGMMA backend explicitly requested")

    # User callbacks must receive absolute selected coordinates; the row and
    # union kernels compose them directly without a secondary framework.
    if has_score_mod:
        return row("user score_mod requires direct indexed-coordinate execution")

    # Q=1 is the primary serving decode manifold. Two model families have
    # measured row schedules encoded in _row_geometry; all other Q=1 shapes
    # retain the occupancy-derived schedule.
    if _wide_near_dense_decode(workload):
        return dense("wide near-dense Q=1 row crosses to indexed tensor cores")
    if _is_deepseek_v32_dsa_decode(workload):
        return row("DeepSeek-V3.2 DSA decode uses the measured split-16 row schedule")
    if _is_glm_moe_dsa_decode(workload):
        return row("GLM MoE DSA decode uses the measured 8-row split-16 schedule")
    if workload.is_decode:
        return row("Q=1 direct selected-token traversal")

    if _is_deepseek_high_batch_q4_row(workload):
        return row(
            "DeepSeek Q4 high-batch chunk uses the measured direct-row crossover"
        )

    if _is_glm_moe_dsa_short_prefill(workload):
        return fa4_bitmask(
            "GLM MoE DSA low-batch short prefill uses measured native-bitmask FA4 128x64"
        )

    if _is_glm_moe_dsa_sparse_prefill(workload):
        return block_sparse(
            "GLM MoE DSA measured batch/sequence region uses exact 128x64 block sparsity"
        )

    if _is_common_short_prefill(workload):
        return fa4_bitmask(
            "qualified common short prefill uses exact native-bitmask FA4 128x64"
        )

    if _is_common_sparse_prefill(workload):
        return block_sparse(
            "qualified common long prefill uses exact 128x64 block sparsity"
        )

    # Other large Q=K prefill remains on the stable tensor-core path.
    # This explicit gate makes the serving assumption visible and prevents a
    # tiny-topk or overlap heuristic from accidentally selecting a row kernel
    # for thousands of query rows.
    if workload.is_large_true_prefill:
        return dense("large Q=K prefill uses stable indexed tensor cores")

    if _hinted_general_union(workload, union_hint):
        return union("caller overlap hint selects low-inflation cross-query WGMMA")

    # High-ratio MQA has three measured regimes: tiny selected sets use rows,
    # sparse overlapping Q32-Q256 work uses packed union, and the remainder
    # uses dense indexed tensor cores.
    if ratio >= 32:
        if workload.batch_query_rows <= 256 and topk <= 257:
            scalar_middle_topk = (
                batch_size == 1
                and 96 <= query_length <= 192
                and 97 <= topk <= 192
                and workload.max_head_dim <= 128
            )
            if scalar_middle_topk:
                return row("middle-tiny-topk MQA favors one head per warp", 1)
            return row("tiny-topk high-ratio MQA grouped-row path")

        union_shape_ok = (
            32 <= query_length <= 256
            and workload.batch_query_rows <= 1024
            and workload.max_head_dim <= 128
            and workload.has_64_aligned_dims
        )
        union_work_ok = (
            topk * 16 <= kv_length
            or (
                workload.batch_query_rows <= 256
                and topk <= 513
                and topk * 8 <= kv_length
            )
        )
        if union_shape_ok and union_work_ok:
            return union("packed indexed WGMMA in the measured MQA union regime")
        if query_length >= 32:
            return dense("dense indexed MQA beyond the row/union crossover")

    if _wide_near_dense_short_query(workload):
        return dense("wide near-dense short-query work crosses to indexed tensor cores")
    if _saturated_short_query(workload):
        return row("saturated short-query batch favors direct selected rows")
    if _large_wide_dense_work(workload):
        return dense("wide high-row selected work crosses to indexed tensor cores")

    small_row_work = workload.total_head_rows < 4096 and topk <= _PRODUCTION_TOPK
    tiny_selected_set = topk <= 219
    short_query_underfilled = query_length <= 12 and workload.total_head_rows < 4096
    if (
        small_row_work
        or _sparse_long_context_gqa_row(workload, grouped)
        or _sparse_long_context_mha_row(workload)
        or tiny_selected_set
        or short_query_underfilled
    ):
        return row("row-sparse work below measured total-row/density crossover")

    return dense("dense indexed CuTe safety path for high-density/high-row prefill")


def indexed_plan_cache_info():
    """Return the bounded scalar-plan cache statistics for diagnostics."""

    return choose_indexed_plan.cache_info()


def clear_indexed_plan_cache() -> None:
    """Clear cached immutable indexed plans, primarily for benchmark isolation."""

    choose_indexed_plan.cache_clear()
