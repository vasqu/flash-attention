"""FA4-style indexed-attention correctness tests against eager and torch SDPA.

The sequence-length matrix is copied from upstream FA4's fixed-length forward
correctness test. Indexed attention is forward-only, non-causal, and fixed-batch,
so the matrix is extended along the relevant axes instead: top-k width, MHA/GQA/
MQA, QK/V dimensions, index dtype, invalid entries, forced backend, and odd
non-aligned shapes.

The numerical criterion mirrors upstream FA4 testing: float32 eager attention is
the reference and the indexed kernel's maximum error must be bounded relative to
a PyTorch SDPA implementation run at input precision.

Environment controls:

* ``FA4_INDEXED_CORRECTNESS_LEVEL=smoke|standard|full`` (default: standard)
* ``FA4_RANDOM_SHAPE_CASES=<N>`` (default: 16)
* ``FA4_RANDOM_SHAPE_SEED=<seed>``
"""

from __future__ import annotations

import contextlib
import math
import os
import random
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from flash_attn.cute.interface import _flash_attn_fwd

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older torch fallback
    SDPBackend = None
    sdpa_kernel = None


# Same fixed-length sequence pairs used by upstream tests/cute/test_flash_attn.py.
UPSTREAM_SEQLEN_SHAPES = (
    (1, 1),
    (3, 3),
    (64, 32),
    (64, 128),
    (64, 1),
    (128, 128),
    (128, 192),
    (256, 256),
    (255, 256),
    (239, 1),
    (799, 3),
    (113, 203),
    (113, 128),
    (128, 217),
    (113, 211),
    (108, 256),
    (256, 512),
    (384, 256),
    (640, 128),
    (512, 256),
    (1024, 1024),
    (1023, 1024),
    (1024, 1023),
    (2048, 2048),
    (4096, 4096),
    (4224, 4224),
)
SUPPORTED_DIMS = (64, 96, 128, 192, 256)
MHA_TYPES = ("mha", "gqa", "mqa")
INDEX_DTYPES = (torch.int32, torch.int64, torch.uint16)


@dataclass(frozen=True)
class CorrectnessCase:
    batch: int
    q_len: int
    k_len: int
    topk: int
    hq: int
    hkv: int
    dim: int
    value_dim: int
    dtype: torch.dtype = torch.bfloat16
    index_dtype: torch.dtype = torch.int32
    backend: str = "auto"
    invalid_fraction: float = 0.0
    seed: int = 0

    @property
    def id(self) -> str:
        dtype_name = str(self.dtype).removeprefix("torch.")
        index_name = str(self.index_dtype).removeprefix("torch.")
        return (
            f"b{self.batch}-q{self.q_len}-k{self.k_len}-t{self.topk}-"
            f"h{self.hq}x{self.hkv}-d{self.dim}x{self.value_dim}-"
            f"{dtype_name}-{index_name}-{self.backend}"
        )


def _requires_sm90() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90 CUDA GPU")


def _dedupe(values: list[int]) -> list[int]:
    result: list[int] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _topk_bank(k_len: int) -> list[int]:
    candidates = [
        1,
        2,
        3,
        7,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        257,
        math.ceil(k_len / 32),
        math.ceil(k_len / 16),
        math.ceil(k_len / 8),
        math.ceil(k_len / 4),
        max(1, k_len - 1),
        k_len,
    ]
    return _dedupe([value for value in candidates if 1 <= value <= k_len])


def _heads(mha_type: str) -> tuple[int, int]:
    if mha_type == "mha":
        return 6, 6
    if mha_type == "gqa":
        return 6, 3
    if mha_type == "mqa":
        return 6, 1
    raise ValueError(mha_type)


def _value_dim(dim: int, variant: int) -> int:
    if dim == 128 and variant % 2:
        return 64
    if dim == 192:
        return 128
    return dim


def _matrix_cases() -> list[CorrectnessCase]:
    level = os.environ.get("FA4_INDEXED_CORRECTNESS_LEVEL", "standard").lower()
    if level not in {"smoke", "standard", "full"}:
        raise ValueError("FA4_INDEXED_CORRECTNESS_LEVEL must be smoke, standard, or full")

    if level == "smoke":
        shape_indices = (0, 1, 3, 8, 10, 11, 16, 20, 21, 22, 24, 25)
        shapes = [UPSTREAM_SEQLEN_SHAPES[i] for i in shape_indices]
        combinations = [(i % len(SUPPORTED_DIMS), i % len(MHA_TYPES)) for i in range(len(shapes))]
    elif level == "standard":
        shapes = list(UPSTREAM_SEQLEN_SHAPES)
        # Every upstream sequence pair is exercised as MHA, GQA, and MQA.
        # Head dimensions rotate so the complete suite covers all supported D/DV.
        combinations = [
            ((shape_idx * 3 + mha_idx) % len(SUPPORTED_DIMS), mha_idx)
            for shape_idx in range(len(shapes))
            for mha_idx in range(len(MHA_TYPES))
        ]
    else:
        shapes = list(UPSTREAM_SEQLEN_SHAPES)
        combinations = [
            (dim_idx, mha_idx)
            for _shape_idx in range(len(shapes))
            for dim_idx in range(len(SUPPORTED_DIMS))
            for mha_idx in range(len(MHA_TYPES))
        ]

    cases: list[CorrectnessCase] = []
    combo_idx = 0
    if level == "smoke":
        iterable = [(shape_idx, *combo) for shape_idx, combo in enumerate(combinations)]
    elif level == "standard":
        iterable = []
        for shape_idx in range(len(shapes)):
            for mha_idx in range(len(MHA_TYPES)):
                iterable.append((shape_idx, *combinations[combo_idx]))
                combo_idx += 1
    else:
        iterable = []
        combo_idx = 0
        for shape_idx in range(len(shapes)):
            for _dim_idx in range(len(SUPPORTED_DIMS)):
                for _mha_idx in range(len(MHA_TYPES)):
                    iterable.append((shape_idx, *combinations[combo_idx]))
                    combo_idx += 1

    for case_idx, (shape_idx, dim_idx, mha_idx) in enumerate(iterable):
        q_len, k_len = shapes[shape_idx]
        dim = SUPPORTED_DIMS[dim_idx]
        mha_type = MHA_TYPES[mha_idx]
        hq, hkv = _heads(mha_type)
        bank = _topk_bank(k_len)
        # Rotate through tiny, boundary, sparse, medium, and nearly-dense top-k
        # values across the upstream shape matrix without a prohibitive Cartesian
        # product. A dedicated sweep below tests the complete bank.
        topk = bank[(case_idx * 7 + shape_idx) % len(bank)]
        batch = 9 if k_len <= 2048 else 2
        cases.append(
            CorrectnessCase(
                batch=batch,
                q_len=q_len,
                k_len=k_len,
                topk=topk,
                hq=hq,
                hkv=hkv,
                dim=dim,
                value_dim=_value_dim(dim, case_idx),
                dtype=torch.bfloat16 if case_idx % 4 else torch.float16,
                index_dtype=INDEX_DTYPES[case_idx % len(INDEX_DTYPES)],
                seed=10_000 + case_idx,
            )
        )
    return cases


def _topk_sweep_cases() -> list[CorrectnessCase]:
    level = os.environ.get("FA4_INDEXED_CORRECTNESS_LEVEL", "standard").lower()
    representatives = [
        (1, 127),
        (64, 128),
        (113, 203),
        (255, 256),
        (256, 512),
        (1024, 1023),
        (4096, 4096),
    ]
    if level == "smoke":
        representatives = representatives[:4]
    cases: list[CorrectnessCase] = []
    for shape_idx, (q_len, k_len) in enumerate(representatives):
        bank = _topk_bank(k_len)
        if level == "smoke":
            bank = [bank[i] for i in sorted({0, len(bank) // 3, 2 * len(bank) // 3, len(bank) - 1})]
        for topk_idx, topk in enumerate(bank):
            dim = SUPPORTED_DIMS[(shape_idx + topk_idx) % len(SUPPORTED_DIMS)]
            mha_type = MHA_TYPES[(shape_idx + topk_idx) % len(MHA_TYPES)]
            hq, hkv = _heads(mha_type)
            cases.append(
                CorrectnessCase(
                    batch=1 if k_len > 2048 else 2,
                    q_len=q_len,
                    k_len=k_len,
                    topk=topk,
                    hq=hq,
                    hkv=hkv,
                    dim=dim,
                    value_dim=_value_dim(dim, topk_idx),
                    dtype=torch.bfloat16,
                    index_dtype=INDEX_DTYPES[topk_idx % len(INDEX_DTYPES)],
                    seed=20_000 + shape_idx * 100 + topk_idx,
                )
            )
    return cases


def _make_indices(case: CorrectnessCase, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(case.seed)
    invalid_count = min(case.topk - 1, round(case.topk * case.invalid_fraction))
    valid_count = case.topk - invalid_count
    sentinel = 65_535 if case.index_dtype == torch.uint16 else -1
    result = torch.full(
        (case.batch, case.q_len, case.topk),
        sentinel,
        dtype=torch.int64,
        device=device,
    )
    for batch_idx in range(case.batch):
        for query_idx in range(case.q_len):
            row = result[batch_idx, query_idx]
            row[:valid_count] = torch.randperm(
                case.k_len, generator=generator, device=device
            )[:valid_count]
            for invalid_idx in range(invalid_count):
                slot = valid_count + invalid_idx
                if invalid_idx % 3 == 1:
                    row[slot] = min(65_534, case.k_len + 7) if case.index_dtype == torch.uint16 else case.k_len + 7
                elif invalid_idx % 3 == 2 and case.index_dtype == torch.int64:
                    row[slot] = 2**40 + invalid_idx
            result[batch_idx, query_idx] = row[
                torch.randperm(case.topk, generator=generator, device=device)
            ]
    return result.to(case.index_dtype)


def _additive_topk_mask(indices: torch.Tensor, k_len: int) -> torch.Tensor:
    """Dense float32 [B,1,Q,K] additive mask containing exactly 0 or -inf."""

    indices64 = indices.to(torch.int64)
    valid = (indices64 >= 0) & (indices64 < k_len)
    if indices.dtype == torch.uint16:
        valid &= indices64 != 65_535
    mask = torch.full(
        (*indices.shape[:2], k_len),
        -float("inf"),
        dtype=torch.float32,
        device=indices.device,
    )
    locations = valid.nonzero(as_tuple=False)
    if locations.numel():
        mask[locations[:, 0], locations[:, 1], indices64[valid]] = 0.0
    return mask[:, None]


def _expand_kv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    ratio = q.shape[2] // k.shape[2]
    return (
        k.repeat_interleave(ratio, dim=2),
        v.repeat_interleave(ratio, dim=2),
    )


def _eager_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    additive_mask: torch.Tensor,
    *,
    q_chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Float32 eager attention with the same dense additive selected-set mask."""

    k_expanded, v_expanded = _expand_kv(q, k, v)
    qf = q.float()
    kf = k_expanded.float()
    vf = v_expanded.float()
    scale = 1.0 / math.sqrt(q.shape[-1])
    out_chunks = []
    lse_chunks = []
    for q_start in range(0, q.shape[1], q_chunk):
        q_end = min(q.shape[1], q_start + q_chunk)
        scores = torch.einsum(
            "bqhd,bkhd->bhqk", qf[:, q_start:q_end], kf
        ) * scale
        scores = scores + additive_mask[:, :, q_start:q_end]
        lse_chunks.append(torch.logsumexp(scores, dim=-1))
        probabilities = torch.softmax(scores, dim=-1)
        out_chunks.append(torch.einsum("bhqk,bkhd->bqhd", probabilities, vf))
    return torch.cat(out_chunks, dim=1), torch.cat(lse_chunks, dim=2)


def _math_sdpa_context():
    if sdpa_kernel is None or SDPBackend is None:
        return contextlib.nullcontext()
    return sdpa_kernel(backends=[SDPBackend.MATH])


def _sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    additive_mask: torch.Tensor,
) -> torch.Tensor:
    """PyTorch SDPA baseline at input precision, matching upstream methodology."""

    k_expanded, v_expanded = _expand_kv(q, k, v)
    with _math_sdpa_context():
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_expanded.transpose(1, 2),
            v_expanded.transpose(1, 2),
            attn_mask=additive_mask.to(q.dtype),
            dropout_p=0.0,
            is_causal=False,
            scale=1.0 / math.sqrt(q.shape[-1]),
        )
    return out.transpose(1, 2)


def _assert_fa4_style_close(
    actual: torch.Tensor,
    eager: torch.Tensor,
    sdpa: torch.Tensor,
    *,
    label: str,
) -> None:
    actual_f = actual.float()
    sdpa_f = sdpa.float()
    indexed_error = (actual_f - eager).abs().max().item()
    sdpa_error = (sdpa_f - eager).abs().max().item()
    roundoff = 2 * (eager + 0.3 - 0.3 - eager).abs().max().item()
    assert indexed_error <= 2.0 * sdpa_error + roundoff + 1e-5, (
        f"{label}: indexed max error {indexed_error} exceeds FA4-style bound "
        f"2*SDPA_error({sdpa_error}) + roundoff({roundoff})"
    )


def _run_and_check(case: CorrectnessCase) -> None:
    _requires_sm90()
    torch.manual_seed(case.seed)
    device = torch.device("cuda")
    q = torch.randn(
        (case.batch, case.q_len, case.hq, case.dim),
        device=device,
        dtype=case.dtype,
    )
    k = torch.randn(
        (case.batch, case.k_len, case.hkv, case.dim),
        device=device,
        dtype=case.dtype,
    )
    v = torch.randn(
        (case.batch, case.k_len, case.hkv, case.value_dim),
        device=device,
        dtype=case.dtype,
    )
    indices = _make_indices(case, device)
    additive_mask = _additive_topk_mask(indices, case.k_len)

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        gather_kv_indices=indices,
        indexed_backend=case.backend,
        return_lse=True,
        _arch=90,
    )
    eager_out, eager_lse = _eager_reference(q, k, v, additive_mask)
    sdpa_out = _sdpa_reference(q, k, v, additive_mask)

    _assert_fa4_style_close(out, eager_out, sdpa_out, label=case.id)
    torch.testing.assert_close(out.float(), eager_out, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(lse, eager_lse, rtol=3e-2, atol=7e-2)


MATRIX_CASES = _matrix_cases()
TOPK_SWEEP_CASES = _topk_sweep_cases()


@pytest.mark.parametrize("case", MATRIX_CASES, ids=lambda case: case.id)
def test_upstream_fa4_shape_matrix_matches_eager_and_sdpa(case: CorrectnessCase):
    _run_and_check(case)


@pytest.mark.parametrize("case", TOPK_SWEEP_CASES, ids=lambda case: case.id)
def test_topk_width_sweep_matches_eager_and_sdpa(case: CorrectnessCase):
    _run_and_check(case)


@pytest.mark.parametrize(
    "backend", ["warp", "warp_grouped", "warp_scalar", "dense", "union"]
)
@pytest.mark.parametrize("dim,value_dim", [(64, 64), (96, 96), (128, 64), (192, 128), (256, 256)])
def test_forced_backends_match_references(backend: str, dim: int, value_dim: int):
    if backend == "union" and (dim % 64 != 0 or value_dim % 64 != 0):
        pytest.skip("indexed union cp.async loader requires D and DV divisible by 64")
    _run_and_check(
        CorrectnessCase(
            batch=1,
            q_len=33,
            k_len=997,
            topk=73,
            hq=8,
            hkv=2,
            dim=dim,
            value_dim=value_dim,
            index_dtype=torch.int32,
            backend=backend,
            seed=30_000 + dim,
        )
    )


def test_mqa_union_matches_references_with_unsorted_invalid_indices():
    _run_and_check(
        CorrectnessCase(
            batch=1,
            q_len=65,
            k_len=5003,
            topk=513,
            hq=64,
            hkv=1,
            dim=128,
            value_dim=128,
            dtype=torch.bfloat16,
            index_dtype=torch.int64,
            backend="union",
            invalid_fraction=0.07,
            seed=38_513,
        )
    )


def test_dense_indexed_aux_layout_reuses_compile_cache_across_q_strides():
    # The first singleton-Q call must not make CuTe assume that Q is the
    # bitmask's unit-stride dimension.  The following B9/Q64 call shares the
    # same score-mod ABI but has stride(Q)=words rather than one.
    for batch, q_len in ((1, 1), (9, 64)):
        _run_and_check(
            CorrectnessCase(
                batch=batch,
                q_len=q_len,
                k_len=128,
                topk=17,
                hq=6,
                hkv=3,
                dim=64,
                value_dim=64,
                dtype=torch.bfloat16,
                index_dtype=torch.int64,
                backend="dense",
                seed=39_000 + q_len,
            )
        )


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_unsorted_interspersed_invalid_entries_match_references(index_dtype: torch.dtype):
    _run_and_check(
        CorrectnessCase(
            batch=2,
            q_len=17,
            k_len=2501,
            topk=257,
            hq=8,
            hkv=2,
            dim=96,
            value_dim=128,
            index_dtype=index_dtype,
            invalid_fraction=0.11,
            seed=40_000 + INDEX_DTYPES.index(index_dtype),
        )
    )


def test_randomized_shapes_match_eager_and_sdpa():
    _requires_sm90()
    count = int(os.environ.get("FA4_RANDOM_SHAPE_CASES", "16"))
    seed = int(os.environ.get("FA4_RANDOM_SHAPE_SEED", "20260721"))
    rng = random.Random(seed)
    q_choices = (1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 97, 127, 129, 257)
    k_choices = (1, 3, 127, 203, 255, 509, 997, 1025, 1537, 2049, 2500, 2501, 4093, 4097, 5003)
    head_choices = ((4, 4), (8, 2), (8, 1), (32, 8), (64, 1))
    for case_idx in range(count):
        k_len = rng.choice(k_choices)
        topk = rng.choice(_topk_bank(k_len))
        hq, hkv = rng.choice(head_choices)
        dim = rng.choice(SUPPORTED_DIMS)
        value_dim = rng.choice(SUPPORTED_DIMS)
        # Keep the heaviest random reference cases bounded; the fixed/full
        # upstream matrix covers large dense attention separately.
        batch = rng.choice((1, 1, 2)) if max(k_len, rng.choice(q_choices)) <= 2501 else 1
        q_len = rng.choice(q_choices)
        _run_and_check(
            CorrectnessCase(
                batch=batch,
                q_len=q_len,
                k_len=k_len,
                topk=topk,
                hq=hq,
                hkv=hkv,
                dim=dim,
                value_dim=value_dim,
                dtype=rng.choice((torch.bfloat16, torch.float16)),
                index_dtype=rng.choice(INDEX_DTYPES),
                invalid_fraction=rng.choice((0.0, 0.0, 0.05)),
                seed=seed + case_idx,
            )
        )
