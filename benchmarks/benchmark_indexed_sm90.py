"""Benchmark the arbitrary-order indexed FA4 SM90 implementation.

The script supports both single-shape benchmarking and an expanded curated
suite covering decode, large-batch scaling, speculative decode, chunked
prefill, GQA head layouts, index overlap, and sparsity crossover points.

Three baselines are deliberately separated:

* fa4_dense_full: unmodified FA4 full attention; fast but different semantics.
* fa4_native_topk: unmodified FA4 plus an ordinary score_mod bitmask; exact
  selected-index semantics, but still dense K/V and tensor-core work.
* torch_sdpa_topk: PyTorch SDPA with a dense broadcast boolean top-k mask.
* indexed_*: the modified sparse warp/union implementations.

Recommended full run:
    PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py \
        --suite common --dtype bf16 --output-dir benchmark_results

Fast smoke run:
    PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py \
        --suite quick --rounds 2 --target-round-ms 40

Serving-focused policy iteration (Q=1 decode and Q=K prefill):
    benchmarks/run_indexed_sm90_iteration.sh

DeepSeek-V3.2 and GLM MoE DSA final validation:
    PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py \
        --suite dsa-models --dtype bf16 --compare-backends all

Exact 45-layer GLM MoE DSA profile from the deployment config:
    benchmarks/run_glm_moe_dsa_45_final.sh

Legacy mixed-Q policy iteration:
    SUITE=iteration-medium benchmarks/run_indexed_sm90_iteration.sh

List the exact cases without allocating GPU tensors:
    PYTHONPATH=. python benchmarks/benchmark_indexed_sm90.py \
        --suite common --list-cases

The output JSON is intended to be shared for kernel-policy tuning. A flat CSV
is written alongside it for spreadsheet analysis.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.machinery
import importlib.metadata
import gc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import traceback
import types
import sys
import time
from typing import Any, Callable, Iterable
import zlib
from functools import lru_cache

import torch
import torch.nn.functional as F


_FA4_ROOT_OVERRIDE: Path | None = None
_FA4_IMPORT_INFO: dict[str, Any] | None = None


def _normalize_fa4_package_dir(path: Path) -> Path:
    """Return the ``flash_attn`` package directory for a root or package path."""

    path = path.expanduser().resolve()
    if path.name == "flash_attn":
        return path
    return path / "flash_attn"


def _discover_fa4_package_dirs(explicit_root: Path | None = None) -> list[Path]:
    """Find FA4 source/package directories without importing ``flash_attn``.

    Importing the repository's top-level ``flash_attn`` package can eagerly
    import the optional FA2 extension.  FA4 only needs ``flash_attn/cute``;
    therefore discovery is filesystem/metadata based and does not execute the
    parent package's ``__init__.py``.
    """

    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(_normalize_fa4_package_dir(explicit_root))
    env_root = os.environ.get("FLASH_ATTN_FA4_ROOT")
    if env_root:
        candidates.append(_normalize_fa4_package_dir(Path(env_root)))

    # Source checkout containing this benchmark, then current working tree.
    candidates.extend(
        [
            Path(__file__).resolve().parents[1] / "flash_attn",
            Path.cwd() / "flash_attn",
        ]
    )

    # Installed flash-attn-4 distribution.  Do not import it: another
    # ``flash_attn`` distribution may own the parent package name.
    for distribution_name in ("flash-attn-4", "flash_attn_4"):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for entry in distribution.files or ():
            parts = tuple(entry.parts)
            if len(parts) >= 3 and parts[-3:] == ("flash_attn", "cute", "interface.py"):
                candidates.append(Path(distribution.locate_file(entry)).resolve().parents[1])
                break

    # Last-resort sys.path scan, still without importing the parent package.
    for entry in sys.path:
        if not entry:
            entry = str(Path.cwd())
        candidates.append(Path(entry).expanduser().resolve() / "flash_attn")

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "cute" / "interface.py").is_file():
            unique.append(candidate)
    return unique


def ensure_fa4_namespace(explicit_root: Path | None = None) -> dict[str, Any]:
    """Expose ``flash_attn.cute`` without requiring ``flash_attn_2_cuda``.

    The function installs or extends a lightweight namespace parent in
    ``sys.modules``.  Python can then import FA4's ``flash_attn.cute`` modules
    directly while skipping the legacy top-level ``flash_attn/__init__.py``.
    """

    global _FA4_IMPORT_INFO
    package_dirs = _discover_fa4_package_dirs(explicit_root)
    if not package_dirs:
        requested = explicit_root or os.environ.get("FLASH_ATTN_FA4_ROOT")
        hint = f" under {requested}" if requested else ""
        raise ModuleNotFoundError(
            "Could not locate FA4's flash_attn/cute/interface.py"
            f"{hint}. Run from the flash-attention repository, install "
            "flash-attn-4, or pass --fa4-root /path/to/repo."
        )

    search_paths = [str(path) for path in package_dirs]
    parent = sys.modules.get("flash_attn")
    bypassed_parent_init = parent is None
    if parent is None:
        parent = types.ModuleType("flash_attn")
        spec = importlib.machinery.ModuleSpec(
            name="flash_attn",
            loader=None,
            is_package=True,
        )
        spec.submodule_search_locations = list(search_paths)
        parent.__spec__ = spec
        parent.__loader__ = None
        parent.__package__ = "flash_attn"
        parent.__path__ = list(search_paths)
        parent.__file__ = None
        try:
            parent.__version__ = importlib.metadata.version("flash-attn-4")
        except importlib.metadata.PackageNotFoundError:
            parent.__version__ = "fa4-source"
        sys.modules["flash_attn"] = parent
    else:
        existing = list(getattr(parent, "__path__", []))
        parent.__path__ = search_paths + [path for path in existing if path not in search_paths]
        if getattr(parent, "__spec__", None) is not None:
            parent.__spec__.submodule_search_locations = list(parent.__path__)

    importlib.invalidate_caches()
    _FA4_IMPORT_INFO = {
        "mode": "fa4-only-namespace",
        "package_dirs": search_paths,
        "primary_package_dir": search_paths[0],
        "bypassed_parent_init": bypassed_parent_init,
        "flash_attn_2_cuda_required": False,
    }
    return dict(_FA4_IMPORT_INFO)


@lru_cache(maxsize=1)
def load_fa4_modules() -> dict[str, Any]:
    """Import FA4 only when an actual GPU benchmark is requested."""

    import_info = ensure_fa4_namespace(_FA4_ROOT_OVERRIDE)
    try:
        from flash_attn.cute.indexed_bitmask_sm90 import build_topk_bitmask_cute
        from flash_attn.cute.indexed_block_sparse import (
            build_indexed_block_sparse_tensors_cute,
            create_indexed_block_sparse_cute_workspace,
        )
        from flash_attn.cute.indexed_block_sparse_lab import indexed_block_density
        from flash_attn.cute.indexed_metrics import analyze_index_tiles
        from flash_attn.cute.indexed_policy import IndexedPath, choose_indexed_plan
        from flash_attn.cute.indexed_prepare import (
            cast_indexed_kv_indices,
            prepare_indexed_kv_indices,
        )
        from flash_attn.cute.indexed_score_mod import (
            build_topk_bitmask,
            topk_bitmask_score_mod,
        )
        from flash_attn.cute.interface import _flash_attn_fwd
    except ModuleNotFoundError as exc:
        if exc.name == "flash_attn_2_cuda":
            raise RuntimeError(
                "FA2 was imported while loading FA4. This benchmark is FA4-only; "
                "ensure the patched benchmark is running and do not import "
                "flash_attn before it."
            ) from exc
        raise

    return {
        "build_topk_bitmask_cute": build_topk_bitmask_cute,
        "build_indexed_block_sparse_tensors_cute": build_indexed_block_sparse_tensors_cute,
        "create_indexed_block_sparse_cute_workspace": create_indexed_block_sparse_cute_workspace,
        "indexed_block_density": indexed_block_density,
        "analyze_index_tiles": analyze_index_tiles,
        "IndexedPath": IndexedPath,
        "choose_indexed_plan": choose_indexed_plan,
        "cast_indexed_kv_indices": cast_indexed_kv_indices,
        "prepare_indexed_kv_indices": prepare_indexed_kv_indices,
        "build_topk_bitmask": build_topk_bitmask,
        "topk_bitmask_score_mod": topk_bitmask_score_mod,
        "flash_attn_fwd": _flash_attn_fwd,
        "import_info": import_info,
    }


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    batch: int
    query_length: int
    kv_length: int
    topk: int
    query_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    value_head_dim: int = 128
    pattern: str = "tail-random"
    tail: int | None = None
    union_compute_inflation_hint: float | None = None
    benchmark_union: bool = True

    def resolved_tail(self) -> int:
        if self.pattern not in ("tail-random", "causal-mixed"):
            return 0
        requested = self.tail
        if requested is None:
            requested = min(512, max(64, self.topk // 4))
        return min(requested, self.topk, self.kv_length)

    def shape_key(self) -> tuple[Any, ...]:
        return (
            self.batch,
            self.query_length,
            self.kv_length,
            self.topk,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.value_head_dim,
            self.pattern,
            self.resolved_tail(),
            self.union_compute_inflation_hint,
            self.benchmark_union,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tail"] = self.resolved_tail()
        result["selection_density"] = self.topk / self.kv_length
        return result


@dataclass(frozen=True)
class AttentionModelProfile:
    """Model metadata attached to a focused attention benchmark suite.

    Only heads, QK/V dimensions, selected-set size, batch, and sequence lengths
    affect the indexed attention kernels directly. The remaining fields keep
    the benchmark tied to the deployment configuration and allow model-level
    latency projections without pretending to benchmark projections or MoE.
    """

    name: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    n_shared_experts: int
    n_routed_experts: int
    routed_scaling_factor: float
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    v_head_dim: int
    qk_nope_head_dim: int
    n_group: int
    topk_group: int
    num_experts_per_tok: int
    norm_topk_prob: bool
    hidden_act: str
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    benchmark_max_context: int

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["qk_head_dim"] = self.qk_head_dim
        return result


_GLM_MOE_DSA_45_PROFILE = AttentionModelProfile(
    name="glm_moe_dsa_45",
    vocab_size=154_880,
    hidden_size=4_096,
    intermediate_size=12_288,
    moe_intermediate_size=2_048,
    num_hidden_layers=45,
    num_attention_heads=64,
    num_key_value_heads=64,
    n_shared_experts=1,
    n_routed_experts=288,
    routed_scaling_factor=2.5,
    kv_lora_rank=512,
    q_lora_rank=1_536,
    qk_rope_head_dim=0,
    v_head_dim=256,
    qk_nope_head_dim=256,
    n_group=1,
    topk_group=1,
    num_experts_per_tok=8,
    norm_topk_prob=True,
    hidden_act="silu",
    # DSA/indexer assumptions used by the final attention benchmark. They are
    # explicit so results are not accidentally interpreted as MoE top-k=8.
    index_topk=2_048,
    index_n_heads=32,
    index_head_dim=128,
    benchmark_max_context=202_752,
)


def model_profile_for_suite(suite: str) -> AttentionModelProfile | None:
    return (
        _GLM_MOE_DSA_45_PROFILE
        if suite in (
            "glm-moe-dsa-45",
            "glm-moe-dsa-45-overnight",
            "glm-moe-dsa-45-batch",
            "glm-moe-dsa-45-serving-matrix",
        )
        else None
    )


@dataclass
class CaseRuntime:
    case: BenchmarkCase
    plan: Any
    backend_plans: dict[str, Any]
    indices: torch.Tensor
    cast_indices: torch.Tensor
    union_indices: torch.Tensor
    bitmask: torch.Tensor | None
    sdpa_mask: torch.Tensor | None
    outputs: dict[str, torch.Tensor]
    calls: dict[str, Callable[[], object]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class TimingConfig:
    warmup: int
    rounds: int
    fixed_iters: int
    target_round_ms: float
    min_iters: int
    max_iters: int


def _coprime_stride(modulus: int) -> int:
    if modulus <= 1:
        return 1
    stride = max(1, modulus // 2 - 1)
    while math.gcd(stride, modulus) != 1:
        stride -= 1
    return stride


def _stable_case_seed(base_seed: int, case_name: str) -> int:
    return (base_seed + zlib.crc32(case_name.encode("utf-8"))) & 0x7FFFFFFF


def make_indices(
    *,
    pattern: str,
    batch: int,
    query_length: int,
    kv_length: int,
    topk: int,
    tail: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create unique, arbitrary-order top-k rows without a B*Q*K router tensor."""

    if topk <= 0:
        raise ValueError("topk must be positive")
    if topk > kv_length:
        raise ValueError("topk must be <= kv_length for unique selected sets")

    rows = batch * query_length
    slot = torch.arange(topk, device=device, dtype=torch.int64)
    row = torch.arange(rows, device=device, dtype=torch.int64)[:, None]

    if pattern == "identical":
        stride = _coprime_stride(kv_length)
        values = (slot * stride + 17) % kv_length
        values = values[None].expand(rows, -1).clone()
    elif pattern == "random":
        stride = _coprime_stride(kv_length)
        offsets = (row * 104729 + 17) % kv_length
        values = (slot[None] * stride + offsets) % kv_length
    elif pattern == "tail-random":
        tail = min(tail, topk, kv_length)
        sparse = topk - tail
        sparse_domain = kv_length - tail
        if sparse > sparse_domain:
            raise ValueError("topk-tail exceeds the non-tail KV domain")
        if sparse:
            stride = _coprime_stride(sparse_domain)
            offsets = (row * 104729 + 17) % sparse_domain
            sparse_values = (
                torch.arange(sparse, device=device, dtype=torch.int64)[None] * stride
                + offsets
            ) % sparse_domain
        else:
            sparse_values = torch.empty((rows, 0), device=device, dtype=torch.int64)
        tail_values = torch.arange(
            kv_length - tail,
            kv_length,
            device=device,
            dtype=torch.int64,
        )[None].expand(rows, -1)
        values = torch.cat([sparse_values, tail_values], dim=-1)
    elif pattern in ("causal-window", "causal-mixed"):
        if query_length != kv_length:
            raise ValueError(f"{pattern} requires query_length == kv_length")
        # Prefix-valid prefill rows. Early queries have fewer than top-k valid
        # predecessors, so unused slots carry the public -1 invalid sentinel.
        query_pos = torch.arange(rows, device=device, dtype=torch.int64)[:, None] % query_length
        prefix = query_pos + 1
        valid = torch.minimum(prefix, torch.full_like(prefix, topk))
        if pattern == "causal-window":
            values = prefix - valid + slot[None]
            values = torch.where(slot[None] < valid, values, -1)
        else:
            tail_count = torch.minimum(valid, torch.full_like(valid, min(tail, topk)))
            sparse_count = valid - tail_count
            old_domain = prefix - tail_count
            sparse_values = (slot[None] * old_domain) // sparse_count.clamp_min(1)
            tail_values = prefix - tail_count + (slot[None] - sparse_count)
            values = torch.where(
                slot[None] < sparse_count,
                sparse_values,
                torch.where(slot[None] < valid, tail_values, -1),
            )
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    # The public API accepts arbitrary ordering. Apply a slot permutation after
    # constructing unique sets so no benchmark accidentally relies on sorting.
    order = torch.randperm(topk, device=device, generator=generator)
    values = values[:, order]
    return values.view(batch, query_length, topk).to(torch.int32)


def _add_unique(cases: list[BenchmarkCase], case: BenchmarkCase) -> None:
    if all(existing.shape_key() != case.shape_key() for existing in cases):
        cases.append(case)


def _decode_grid() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for kv_length, topks in (
        (4096, (256, 512)),
        (16384, (512, 1024, 2048)),
        (32768, (1024, 2048, 4096)),
        (65536, (2048, 4096, 8192)),
        (131072, (4096, 8192, 16384)),
    ):
        for topk in topks:
            cases.append(
                BenchmarkCase(
                    name=f"decode_b1_q1_k{kv_length}_t{topk}",
                    category="decode-density",
                    batch=1,
                    query_length=1,
                    kv_length=kv_length,
                    topk=topk,
                )
            )
    return cases


def _batching_cases() -> list[BenchmarkCase]:
    """Continuous-batching sweeps, including H100 saturation regimes."""

    cases: list[BenchmarkCase] = []
    for kv_length, topk, batches in (
        (32768, 2048, (2, 4, 8, 16, 32, 64, 128)),
        (65536, 4096, (4, 16, 64, 128)),
        (131072, 8192, (4, 16, 32)),
    ):
        for batch in batches:
            cases.append(
                BenchmarkCase(
                    name=f"decode_batch_b{batch}_q1_k{kv_length}_t{topk}",
                    category="decode-batching",
                    batch=batch,
                    query_length=1,
                    kv_length=kv_length,
                    topk=topk,
                )
            )
    return cases

def _speculative_cases() -> list[BenchmarkCase]:
    """Speculative/multi-token decode across small and saturated batches."""

    cases: list[BenchmarkCase] = []
    for query_length, batches in (
        (4, (1, 4, 16, 64)),
        (8, (1, 4, 16, 32)),
        (16, (1, 4, 16, 32)),
    ):
        for batch in batches:
            cases.append(
                BenchmarkCase(
                    name=f"spec_b{batch}_q{query_length}_k32768_t2048",
                    category="speculative-decode",
                    batch=batch,
                    query_length=query_length,
                    kv_length=32768,
                    topk=2048,
                )
            )
    return cases

def _regression_cases() -> list[BenchmarkCase]:
    """Shapes that exposed the Q-contiguous split-LSE regression on H100."""

    return [
        BenchmarkCase("regress_spec_b1_q4", "regression", 1, 4, 32768, 2048),
        BenchmarkCase("regress_spec_b4_q4", "regression", 4, 4, 32768, 2048),
        BenchmarkCase("regress_spec_b16_q4", "regression", 16, 4, 32768, 2048),
        BenchmarkCase("regress_spec_b1_q8", "regression", 1, 8, 32768, 2048),
        BenchmarkCase("regress_spec_b4_q8", "regression", 4, 8, 32768, 2048),
        BenchmarkCase("regress_spec_b1_q16", "regression", 1, 16, 32768, 2048),
        BenchmarkCase("regress_spec_b4_q16", "regression", 4, 16, 32768, 2048),
        BenchmarkCase("regress_prefill_b1_q64", "regression", 1, 64, 16384, 1024),
        BenchmarkCase("regress_decode_b1_q1", "regression", 1, 1, 32768, 2048),
        BenchmarkCase("regress_native_b1_q128", "regression", 1, 128, 32768, 2048),
    ]


def _prefill_cases() -> list[BenchmarkCase]:
    """Chunked-prefill scaling without creating a full Cartesian product."""

    shapes: list[tuple[int, int, int, int]] = []
    for batch in (1, 4, 16):
        shapes.append((batch, 64, 16384, 1024))
    for batch in (1, 2, 4, 8, 16):
        shapes.append((batch, 128, 32768, 2048))
    for batch in (1, 2, 4):
        shapes.append((batch, 512, 32768, 2048))
    for batch in (1, 4, 8):
        shapes.append((batch, 128, 65536, 4096))
    for batch in (1, 2):
        shapes.append((batch, 512, 65536, 4096))

    return [
        BenchmarkCase(
            name=f"prefill_b{batch}_q{query_length}_k{kv_length}_t{topk}",
            category="chunked-prefill",
            batch=batch,
            query_length=query_length,
            kv_length=kv_length,
            topk=topk,
        )
        for batch, query_length, kv_length, topk in shapes
    ]

def _head_layout_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for query_length, label in ((1, "decode"), (128, "union")):
        for query_heads, kv_heads in ((64, 8), (28, 4), (64, 1)):
            cases.append(
                BenchmarkCase(
                    name=(
                        f"heads_{label}_b1_q{query_length}_k32768_t2048_"
                        f"hq{query_heads}_hkv{kv_heads}"
                    ),
                    category="head-layout",
                    batch=1,
                    query_length=query_length,
                    kv_length=32768,
                    topk=2048,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                )
            )
    return cases


def _pattern_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            name="pattern_union_identical_b1_q128_k32768_t2048",
            category="index-pattern",
            batch=1,
            query_length=128,
            kv_length=32768,
            topk=2048,
            pattern="identical",
            union_compute_inflation_hint=1.0,
        ),
        BenchmarkCase(
            name="pattern_union_random_b1_q128_k32768_t2048",
            category="index-pattern",
            batch=1,
            query_length=128,
            kv_length=32768,
            topk=2048,
            pattern="random",
            union_compute_inflation_hint=16.0,
        ),
        BenchmarkCase(
            name="pattern_decode_random_b1_q1_k32768_t2048",
            category="index-pattern",
            batch=1,
            query_length=1,
            kv_length=32768,
            topk=2048,
            pattern="random",
        ),
    ]


def _iteration_cases() -> list[BenchmarkCase]:
    """Small policy-crossover suite for fast edit/benchmark cycles.

    The overnight suites remain the coverage source of truth.  This set keeps
    only the measured dispatch boundaries and regressions needed to answer:
    did auto choose the fastest eligible backend, and did a policy edit move a
    nearby control in the wrong direction?
    """

    return [
        BenchmarkCase("iter_decode_row", "iteration", 1, 1, 32768, 2048),
        BenchmarkCase(
            "iter_decode_wide_near_dense",
            "iteration",
            1,
            1,
            8191,
            8190,
            query_heads=28,
            kv_heads=4,
            head_dim=256,
            value_head_dim=192,
            pattern="random",
        ),
        BenchmarkCase("iter_spec_dense_b16_q8", "iteration", 16, 8, 32768, 2048),
        BenchmarkCase("iter_spec_row_b32_q8", "iteration", 32, 8, 32768, 2048),
        BenchmarkCase("iter_prefill_row_q64_t1025", "iteration", 1, 64, 4097, 1025),
        BenchmarkCase("iter_prefill_dense_q64_t2049", "iteration", 1, 64, 4097, 2049),
        BenchmarkCase(
            "iter_union_identical",
            "iteration",
            1,
            128,
            32768,
            2048,
            pattern="identical",
            union_compute_inflation_hint=1.0,
        ),
        BenchmarkCase(
            "iter_union_random_control",
            "iteration",
            1,
            128,
            32768,
            2048,
            pattern="random",
            union_compute_inflation_hint=16.0,
        ),
        BenchmarkCase(
            "iter_mqa_row",
            "iteration",
            1,
            129,
            5003,
            257,
            query_heads=64,
            kv_heads=1,
            pattern="random",
        ),
        BenchmarkCase(
            "iter_mqa_union",
            "iteration",
            1,
            129,
            5003,
            513,
            query_heads=64,
            kv_heads=1,
            pattern="random",
        ),
        BenchmarkCase(
            "iter_mqa_dense",
            "iteration",
            1,
            129,
            5003,
            1025,
            query_heads=64,
            kv_heads=1,
            pattern="random",
        ),
        BenchmarkCase(
            "iter_regress_wide_short_near_dense",
            "iteration",
            3,
            5,
            5003,
            5003,
            query_heads=64,
            kv_heads=8,
            head_dim=96,
            value_head_dim=256,
            pattern="random",
        ),
        BenchmarkCase(
            "iter_regress_wide_high_rows",
            "iteration",
            3,
            257,
            509,
            127,
            query_heads=64,
            kv_heads=8,
            head_dim=192,
            value_head_dim=256,
            pattern="random",
        ),
    ]



def _iteration_medium_cases() -> list[BenchmarkCase]:
    """Medium BF16 policy suite, centered on production DSA top-k=2048.

    This is the default human-in-the-loop tuning set: broad enough to expose
    dispatch boundaries across serving regimes, but intentionally far smaller
    than the overnight correctness/performance matrix.  Neighboring non-2K
    points are retained only as crossover controls.
    """

    cases = list(_iteration_cases())

    def add(case: BenchmarkCase) -> None:
        _add_unique(cases, case)

    # Long-context decode at the common DSA selection width.
    for kv_length in (4096, 16384, 65536, 131072):
        add(
            BenchmarkCase(
                f"iter2k_decode_context_k{kv_length}",
                "iteration-2k-decode-context",
                1,
                1,
                kv_length,
                2048,
            )
        )

    # Continuous batching at one representative production context.
    for batch in (8, 32, 128):
        add(
            BenchmarkCase(
                f"iter2k_decode_batch_b{batch}",
                "iteration-2k-decode-batch",
                batch,
                1,
                32768,
                2048,
            )
        )

    # Speculative decode around the measured dense/row occupancy crossover.
    for batch, query_length in (
        (1, 4),
        (16, 4),
        (64, 4),
        (1, 8),
        (16, 8),
        (32, 8),
        (1, 16),
        (16, 16),
    ):
        add(
            BenchmarkCase(
                f"iter2k_spec_b{batch}_q{query_length}",
                "iteration-2k-speculative",
                batch,
                query_length,
                32768,
                2048,
            )
        )

    # Chunked-prefill/query-row scaling, including extra batch occupancy.
    for batch, query_length in (
        (1, 32),
        (1, 64),
        (1, 128),
        (1, 256),
        (1, 512),
        (4, 64),
        (4, 128),
        (4, 512),
    ):
        add(
            BenchmarkCase(
                f"iter2k_prefill_b{batch}_q{query_length}",
                "iteration-2k-prefill",
                batch,
                query_length,
                32768,
                2048,
            )
        )

    # Context-length scaling at a fixed prefill tile.  Q128 brackets the
    # measured 4096-output-row density crossover; adjacent Q64/Q256 and batched
    # controls guard against extending the rule too aggressively.
    for batch, query_length, kv_length in (
        (1, 128, 16384),
        (1, 128, 65536),
        (1, 128, 131072),
        (1, 64, 65536),
        (2, 64, 65536),
        (4, 64, 65536),
        (1, 256, 65536),
        (2, 128, 65536),
        (1, 512, 65536),
        (4, 128, 65536),
    ):
        name = (
            f"iter2k_prefill_context_q128_k{kv_length}"
            if batch == 1 and query_length == 128
            else f"iter2k_prefill_context_b{batch}_q{query_length}_k{kv_length}"
        )
        add(
            BenchmarkCase(
                name,
                "iteration-2k-prefill-context",
                batch,
                query_length,
                kv_length,
                2048,
            )
        )

    # Same total-row boundary with different head sharing.  H64/8 is another
    # grouped-GQA 8192-row point; H32/32 is the no-sharing MHA control and must
    # not inherit a GQA-specific sparse dispatch rule without measurements.
    for query_heads, kv_heads in ((64, 8), (32, 32)):
        add(
            BenchmarkCase(
                f"iter2k_prefill_context_heads_h{query_heads}_{kv_heads}",
                "iteration-2k-prefill-context",
                1,
                128,
                65536,
                2048,
                query_heads=query_heads,
                kv_heads=kv_heads,
            )
        )

    # GQA, MQA and MHA controls at the same B/Q/K/T point.
    for query_heads, kv_heads in ((64, 8), (28, 4), (64, 1), (32, 32)):
        add(
            BenchmarkCase(
                f"iter2k_heads_q128_h{query_heads}_{kv_heads}",
                "iteration-2k-head-layout",
                1,
                128,
                32768,
                2048,
                query_heads=query_heads,
                kv_heads=kv_heads,
            )
        )

    # Released-model shape proxies.  These benchmark the attention dimensions
    # and 2K selected width, not the upstream indexer or MLA projection cost.
    add(
        BenchmarkCase(
            "iter2k_deepseek_v32_decode_long",
            "iteration-2k-model-proxy",
            1,
            1,
            163840,
            2048,
            query_heads=128,
            kv_heads=128,
            head_dim=192,
            value_head_dim=128,
            pattern="random",
        )
    )
    add(
        BenchmarkCase(
            "iter2k_deepseek_v32_spec_q8",
            "iteration-2k-model-proxy",
            1,
            8,
            32768,
            2048,
            query_heads=128,
            kv_heads=128,
            head_dim=192,
            value_head_dim=128,
            pattern="random",
        )
    )
    add(
        BenchmarkCase(
            "iter2k_glm_moe_dsa_decode_long",
            "iteration-2k-model-proxy",
            1,
            1,
            202752,
            2048,
            query_heads=64,
            kv_heads=64,
            head_dim=256,
            value_head_dim=256,
            pattern="random",
        )
    )
    add(
        BenchmarkCase(
            "iter2k_glm_moe_dsa_spec_q8",
            "iteration-2k-model-proxy",
            1,
            8,
            32768,
            2048,
            query_heads=64,
            kv_heads=64,
            head_dim=256,
            value_head_dim=256,
            pattern="random",
        )
    )

    # A narrow density bracket around 2K catches row/dense policy movement.
    for topk in (1024, 1536, 3072, 4096):
        add(
            BenchmarkCase(
                f"iter2k_density_q128_t{topk}",
                "iteration-2k-density-control",
                1,
                128,
                32768,
                topk,
            )
        )

    return cases



def _serving_decode_cases() -> list[BenchmarkCase]:
    """Production decode manifold: Q=1 with 2K-centered context/batch sweeps."""

    cases: list[BenchmarkCase] = []
    for kv_length in (4096, 8192, 16384, 32768, 65536, 131072):
        cases.append(
            BenchmarkCase(
                f"serving_decode_k{kv_length}_t2048",
                "serving-decode",
                1, 1, kv_length, 2048,
            )
        )
    for batch in (8, 32, 128):
        cases.append(
            BenchmarkCase(
                f"serving_decode_b{batch}_k32768_t2048",
                "serving-decode",
                batch, 1, 32768, 2048,
            )
        )
    for topk in (512, 1024, 1536, 3072, 4096):
        cases.append(
            BenchmarkCase(
                f"serving_decode_k32768_t{topk}",
                "serving-decode",
                1, 1, 32768, topk,
            )
        )
    for query_heads, kv_heads in ((64, 8), (28, 4), (64, 1), (32, 32)):
        cases.append(
            BenchmarkCase(
                f"serving_decode_heads_h{query_heads}_{kv_heads}",
                "serving-decode",
                1, 1, 65536, 2048,
                query_heads=query_heads, kv_heads=kv_heads,
            )
        )
    cases.append(
        BenchmarkCase(
            "serving_decode_k65536_t2048_random",
            "serving-decode",
            1, 1, 65536, 2048, pattern="random",
        )
    )
    cases.extend(
        [
            BenchmarkCase(
                "serving_decode_deepseek_v32",
                "serving-decode",
                1, 1, 163840, 2048,
                query_heads=128, kv_heads=128,
                head_dim=192, value_head_dim=128, pattern="random",
            ),
            BenchmarkCase(
                "serving_decode_glm_moe_dsa",
                "serving-decode",
                1, 1, 202752, 2048,
                query_heads=64, kv_heads=64,
                head_dim=256, value_head_dim=256, pattern="random",
            ),
        ]
    )
    return cases


def _serving_prefill_cases() -> list[BenchmarkCase]:
    """Production prefill manifold: Q=K with causal prefix-valid selections."""

    cases: list[BenchmarkCase] = []
    for seqlen in (2048, 4096, 8192, 16384):
        cases.append(
            BenchmarkCase(
                f"serving_prefill_s{seqlen}_t2048",
                "serving-prefill",
                1, seqlen, seqlen, 2048, pattern="causal-mixed",
            )
        )
    for topk in (512, 1024, 1536, 3072, 4096):
        cases.append(
            BenchmarkCase(
                f"serving_prefill_s8192_t{topk}",
                "serving-prefill",
                1, 8192, 8192, topk, pattern="causal-mixed",
            )
        )
    for batch in (2, 4):
        cases.append(
            BenchmarkCase(
                f"serving_prefill_b{batch}_s4096_t2048",
                "serving-prefill",
                batch, 4096, 4096, 2048, pattern="causal-mixed",
            )
        )
    for query_heads, kv_heads in ((64, 8), (28, 4), (64, 1), (32, 32)):
        cases.append(
            BenchmarkCase(
                f"serving_prefill_s8192_heads_h{query_heads}_{kv_heads}",
                "serving-prefill",
                1, 8192, 8192, 2048,
                query_heads=query_heads, kv_heads=kv_heads, pattern="causal-mixed",
            )
        )
    cases.extend(
        [
            BenchmarkCase(
                "serving_prefill_s8192_causal_window",
                "serving-prefill",
                1, 8192, 8192, 2048, pattern="causal-window",
                union_compute_inflation_hint=(2048 + 127) / 2048,
            ),
            BenchmarkCase(
                "serving_prefill_s8192_random_control",
                "serving-prefill",
                1, 8192, 8192, 2048, pattern="random",
                union_compute_inflation_hint=16.0,
            ),
            BenchmarkCase(
                "serving_prefill_deepseek_v32_s4096",
                "serving-prefill",
                1, 4096, 4096, 2048,
                query_heads=128, kv_heads=128,
                head_dim=192, value_head_dim=128, pattern="causal-mixed",
            ),
            BenchmarkCase(
                "serving_prefill_glm_moe_dsa_s4096",
                "serving-prefill",
                1, 4096, 4096, 2048,
                query_heads=64, kv_heads=64,
                head_dim=256, value_head_dim=256, pattern="causal-mixed",
            ),
        ]
    )
    return cases



def _dsa_model_cases() -> list[BenchmarkCase]:
    """DeepSeek-V3.2 and GLM MoE DSA serving validation.

    The suite is intentionally limited to the two production manifolds:
    Q=1 decode and Q=K prefill. It brackets context length, batch, and the
    2K selected-set operating point without reintroducing a broad mixed-Q
    Cartesian sweep.
    """

    cases: list[BenchmarkCase] = []
    models = (
        (
            "deepseek_v32",
            128,
            128,
            192,
            128,
            163840,
        ),
        (
            "glm_moe_dsa",
            64,
            64,
            256,
            256,
            202752,
        ),
    )

    for name, query_heads, kv_heads, head_dim, value_head_dim, long_context in models:
        for kv_length in (32768, 65536, long_context):
            cases.append(
                BenchmarkCase(
                    f"dsa_{name}_decode_k{kv_length}_t2048",
                    "dsa-model-decode",
                    1,
                    1,
                    kv_length,
                    2048,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    value_head_dim=value_head_dim,
                    pattern="random",
                )
            )
        cases.append(
            BenchmarkCase(
                f"dsa_{name}_decode_b8_k65536_t2048",
                "dsa-model-decode",
                8,
                1,
                65536,
                2048,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                value_head_dim=value_head_dim,
                pattern="random",
            )
        )
        for topk in (1024, 4096):
            cases.append(
                BenchmarkCase(
                    f"dsa_{name}_decode_k{long_context}_t{topk}",
                    "dsa-model-decode",
                    1,
                    1,
                    long_context,
                    topk,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    value_head_dim=value_head_dim,
                    pattern="random",
                )
            )

        for seqlen in (2048, 4096, 8192):
            cases.append(
                BenchmarkCase(
                    f"dsa_{name}_prefill_s{seqlen}_t2048",
                    "dsa-model-prefill",
                    1,
                    seqlen,
                    seqlen,
                    2048,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    value_head_dim=value_head_dim,
                    pattern="causal-mixed",
                )
            )
        for topk in (1024, 3072):
            cases.append(
                BenchmarkCase(
                    f"dsa_{name}_prefill_s4096_t{topk}",
                    "dsa-model-prefill",
                    1,
                    4096,
                    4096,
                    topk,
                    query_heads=query_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    value_head_dim=value_head_dim,
                    pattern="causal-mixed",
                )
            )
        cases.append(
            BenchmarkCase(
                f"dsa_{name}_prefill_s4096_window",
                "dsa-model-prefill",
                1,
                4096,
                4096,
                2048,
                query_heads=query_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                value_head_dim=value_head_dim,
                pattern="causal-window",
            )
        )

    return cases

def _glm_moe_dsa_45_cases() -> list[BenchmarkCase]:
    """Final BF16 matrix for the supplied 45-layer GLM MoE DSA profile.

    Decode stays on Q=1 and brackets context, selected-set size, batch, and
    index locality. Prefill stays on Q=K and brackets the production top-k
    plus density controls. Forced-union timing is disabled for these wide MHA
    cases; prior sweeps found no useful 64/64 D256 union regime and its compile
    cost obscures the row-vs-dense decision that matters here.
    """

    profile = _GLM_MOE_DSA_45_PROFILE

    def case(
        name: str,
        category: str,
        batch: int,
        query_length: int,
        kv_length: int,
        topk: int,
        pattern: str,
    ) -> BenchmarkCase:
        return BenchmarkCase(
            name,
            category,
            batch,
            query_length,
            kv_length,
            topk,
            query_heads=profile.num_attention_heads,
            kv_heads=profile.num_key_value_heads,
            head_dim=profile.qk_head_dim,
            value_head_dim=profile.v_head_dim,
            pattern=pattern,
            benchmark_union=False,
        )

    t = profile.index_topk
    max_context = profile.benchmark_max_context
    return [
        # Decode: context and DSA selected-set brackets.
        case("glm45_decode_b1_k4096_t512", "glm45-decode", 1, 1, 4_096, 512, "random"),
        case("glm45_decode_b1_k16384_t1024", "glm45-decode", 1, 1, 16_384, 1_024, "random"),
        case("glm45_decode_b1_k32768_t2048", "glm45-decode", 1, 1, 32_768, t, "random"),
        case("glm45_decode_b1_k65536_t2048_random", "glm45-decode", 1, 1, 65_536, t, "random"),
        case("glm45_decode_b1_k65536_t2048_tail", "glm45-decode", 1, 1, 65_536, t, "tail-random"),
        case("glm45_decode_b1_k131072_t2048", "glm45-decode", 1, 1, 131_072, t, "random"),
        case("glm45_decode_b1_k202752_t1024", "glm45-decode", 1, 1, max_context, 1_024, "random"),
        case("glm45_decode_b1_k202752_t2048", "glm45-decode", 1, 1, max_context, t, "random"),
        case("glm45_decode_b1_k202752_t4096", "glm45-decode", 1, 1, max_context, 4_096, "random"),
        case("glm45_decode_b8_k65536_t2048", "glm45-decode", 8, 1, 65_536, t, "random"),
        case("glm45_decode_b32_k65536_t2048", "glm45-decode", 32, 1, 65_536, t, "random"),

        # True prefill: causal selected sets only.
        case("glm45_prefill_s1024_t512", "glm45-prefill", 1, 1_024, 1_024, 512, "causal-mixed"),
        case("glm45_prefill_s1024_t1024", "glm45-prefill", 1, 1_024, 1_024, 1_024, "causal-window"),
        case("glm45_prefill_s2048_t1024", "glm45-prefill", 1, 2_048, 2_048, 1_024, "causal-mixed"),
        case("glm45_prefill_s2048_t2048", "glm45-prefill", 1, 2_048, 2_048, t, "causal-window"),
        case("glm45_prefill_s4096_t1024", "glm45-prefill", 1, 4_096, 4_096, 1_024, "causal-mixed"),
        case("glm45_prefill_s4096_t2048_mixed", "glm45-prefill", 1, 4_096, 4_096, t, "causal-mixed"),
        case("glm45_prefill_s4096_t2048_window", "glm45-prefill", 1, 4_096, 4_096, t, "causal-window"),
        case("glm45_prefill_s4096_t3072", "glm45-prefill", 1, 4_096, 4_096, 3_072, "causal-mixed"),
        case("glm45_prefill_s4096_t4096", "glm45-prefill", 1, 4_096, 4_096, 4_096, "causal-window"),
        case("glm45_prefill_b2_s4096_t2048", "glm45-prefill", 2, 4_096, 4_096, t, "causal-mixed"),
        case("glm45_prefill_s8192_t2048", "glm45-prefill", 1, 8_192, 8_192, t, "causal-mixed"),
        case("glm45_prefill_s16384_t2048", "glm45-prefill", 1, 16_384, 16_384, t, "causal-mixed"),
    ]


def _glm_moe_dsa_45_overnight_cases() -> list[BenchmarkCase]:
    """Long BF16 matrix for final GLM DSA decode and prefill decisions.

    High-batch decode cases are capped so fixed-batch K/V tensors fit on an
    80 GB H100. Paged-cache scaling belongs in a separate integration test.
    """

    profile = _GLM_MOE_DSA_45_PROFILE

    def case(
        name: str,
        category: str,
        batch: int,
        query_length: int,
        kv_length: int,
        topk: int,
        pattern: str,
    ) -> BenchmarkCase:
        return BenchmarkCase(
            name,
            category,
            batch,
            query_length,
            kv_length,
            topk,
            query_heads=profile.num_attention_heads,
            kv_heads=profile.num_key_value_heads,
            head_dim=profile.qk_head_dim,
            value_head_dim=profile.v_head_dim,
            pattern=pattern,
            benchmark_union=False,
        )

    t = profile.index_topk
    decode = [
        case("glm45o_decode_b1_k4096_t512", "glm45o-decode", 1, 1, 4_096, 512, "random"),
        case("glm45o_decode_b1_k16384_t1024", "glm45o-decode", 1, 1, 16_384, 1_024, "random"),
        case("glm45o_decode_b1_k32768_t2048", "glm45o-decode", 1, 1, 32_768, t, "random"),
        case("glm45o_decode_b1_k65536_t1024", "glm45o-decode", 1, 1, 65_536, 1_024, "random"),
        case("glm45o_decode_b1_k65536_t2048_random", "glm45o-decode", 1, 1, 65_536, t, "random"),
        case("glm45o_decode_b1_k65536_t2048_tail", "glm45o-decode", 1, 1, 65_536, t, "tail-random"),
        case("glm45o_decode_b1_k65536_t4096", "glm45o-decode", 1, 1, 65_536, 4_096, "random"),
        case("glm45o_decode_b1_k131072_t2048", "glm45o-decode", 1, 1, 131_072, t, "random"),
        case("glm45o_decode_b1_k202752_t1024", "glm45o-decode", 1, 1, 202_752, 1_024, "random"),
        case("glm45o_decode_b1_k202752_t2048", "glm45o-decode", 1, 1, 202_752, t, "random"),
        case("glm45o_decode_b1_k202752_t4096", "glm45o-decode", 1, 1, 202_752, 4_096, "random"),
        case("glm45o_decode_b4_k65536_t2048", "glm45o-decode", 4, 1, 65_536, t, "random"),
        case("glm45o_decode_b8_k65536_t2048", "glm45o-decode", 8, 1, 65_536, t, "random"),
        case("glm45o_decode_b16_k16384_t2048", "glm45o-decode", 16, 1, 16_384, t, "random"),
        case("glm45o_decode_b32_k8192_t2048", "glm45o-decode", 32, 1, 8_192, t, "random"),
    ]

    prefill_specs = [
        (512, 256), (512, 512),
        (1_024, 256), (1_024, 512), (1_024, 768), (1_024, 1_024),
        (2_048, 512), (2_048, 1_024), (2_048, 1_536), (2_048, 2_048),
        (4_096, 512), (4_096, 1_024), (4_096, 1_536), (4_096, 2_048),
        (4_096, 3_072), (4_096, 4_096),
        (8_192, 1_024), (8_192, 2_048), (8_192, 4_096),
        (16_384, 1_024), (16_384, 2_048), (16_384, 4_096),
    ]
    prefill: list[BenchmarkCase] = []
    for seq, topk in prefill_specs:
        for pattern in ("causal-window", "causal-mixed"):
            suffix = "window" if pattern == "causal-window" else "mixed"
            prefill.append(
                case(
                    f"glm45o_prefill_s{seq}_t{topk}_{suffix}",
                    "glm45o-prefill",
                    1,
                    seq,
                    seq,
                    topk,
                    pattern,
                )
            )
    prefill.extend([
        case("glm45o_prefill_b2_s2048_t2048_window", "glm45o-prefill", 2, 2_048, 2_048, t, "causal-window"),
        case("glm45o_prefill_b2_s4096_t2048_mixed", "glm45o-prefill", 2, 4_096, 4_096, t, "causal-mixed"),
    ])
    return decode + prefill


def _glm_moe_dsa_45_batch_cases() -> list[BenchmarkCase]:
    """Batch-scaling matrix for exact GLM DSA true prefill.

    The production policy remains conservative while this suite measures the
    explicit native-bitmask and exact block-sparse paths at B=2/4/8/16.  Long
    shapes stop at B=8 to keep the many benchmark output tensors comfortably
    inside an 80 GB H100.
    """

    profile = _GLM_MOE_DSA_45_PROFILE

    def case(
        name: str,
        batch: int,
        seq: int,
        topk: int,
        pattern: str,
    ) -> BenchmarkCase:
        return BenchmarkCase(
            name=name,
            category="glm45b-prefill",
            batch=batch,
            query_length=seq,
            kv_length=seq,
            topk=topk,
            query_heads=profile.num_attention_heads,
            kv_heads=profile.num_key_value_heads,
            head_dim=profile.qk_head_dim,
            value_head_dim=profile.v_head_dim,
            pattern=pattern,
            benchmark_union=False,
        )

    cases: list[BenchmarkCase] = []
    short_specs = (
        (512, 256, "causal-window"),
        (512, 256, "causal-mixed"),
        (1024, 512, "causal-window"),
        (1024, 512, "causal-mixed"),
    )
    for seq, topk, pattern in short_specs:
        suffix = "window" if pattern == "causal-window" else "mixed"
        for batch in (1, 2, 4, 8, 16):
            cases.append(
                case(
                    f"glm45b_prefill_b{batch}_s{seq}_t{topk}_{suffix}",
                    batch,
                    seq,
                    topk,
                    pattern,
                )
            )

    long_specs = (
        (2048, 512, "causal-window"),
        (2048, 2048, "causal-mixed"),
        (4096, 1024, "causal-window"),
        (4096, 2048, "causal-mixed"),
    )
    for seq, topk, pattern in long_specs:
        suffix = "window" if pattern == "causal-window" else "mixed"
        for batch in (1, 2, 4, 8):
            cases.append(
                case(
                    f"glm45b_prefill_b{batch}_s{seq}_t{topk}_{suffix}",
                    batch,
                    seq,
                    topk,
                    pattern,
                )
            )
    return cases


def _glm_moe_dsa_45_serving_matrix_cases() -> list[BenchmarkCase]:
    """Unified GLM-first serving matrix with cross-shape guardrails.

    The matrix combines the measured GLM decode, speculative/chunked-query,
    true-prefill, and B1--B16 scaling manifolds.  A smaller DeepSeek/MHA/GQA/MQA
    slice is included to ensure GLM-specific promotion does not regress the
    general indexed policy.
    """

    profile = _GLM_MOE_DSA_45_PROFILE

    def glm_case(
        name: str,
        category: str,
        batch: int,
        query_length: int,
        kv_length: int,
        topk: int,
        pattern: str = "random",
    ) -> BenchmarkCase:
        return BenchmarkCase(
            name=name,
            category=category,
            batch=batch,
            query_length=query_length,
            kv_length=kv_length,
            topk=topk,
            query_heads=profile.num_attention_heads,
            kv_heads=profile.num_key_value_heads,
            head_dim=profile.qk_head_dim,
            value_head_dim=profile.v_head_dim,
            pattern=pattern,
            benchmark_union=False,
        )

    cases: list[BenchmarkCase] = [
        # GLM decode: context and occupancy scaling while bounding K/V memory.
        glm_case("glm45m_decode_b1_k4096_t512", "glm45m-decode", 1, 1, 4096, 512),
        glm_case("glm45m_decode_b1_k32768_t2048", "glm45m-decode", 1, 1, 32768, 2048),
        glm_case("glm45m_decode_b1_k65536_t2048_tail", "glm45m-decode", 1, 1, 65536, 2048, "tail-random"),
        glm_case("glm45m_decode_b1_k202752_t2048", "glm45m-decode", 1, 1, 202752, 2048),
        glm_case("glm45m_decode_b4_k65536_t2048", "glm45m-decode", 4, 1, 65536, 2048),
        glm_case("glm45m_decode_b8_k32768_t2048", "glm45m-decode", 8, 1, 32768, 2048),
        glm_case("glm45m_decode_b16_k16384_t2048", "glm45m-decode", 16, 1, 16384, 2048),
        glm_case("glm45m_decode_b32_k8192_t2048", "glm45m-decode", 32, 1, 8192, 2048),

        # Speculative/chunked queries between decode and true prefill.
        glm_case("glm45m_chunk_b1_q4_k32768_t2048", "glm45m-chunk", 1, 4, 32768, 2048, "tail-random"),
        glm_case("glm45m_chunk_b8_q4_k32768_t2048", "glm45m-chunk", 8, 4, 32768, 2048, "tail-random"),
        glm_case("glm45m_chunk_b1_q16_k32768_t2048", "glm45m-chunk", 1, 16, 32768, 2048, "tail-random"),
        glm_case("glm45m_chunk_b8_q16_k32768_t2048", "glm45m-chunk", 8, 16, 32768, 2048, "tail-random"),
        glm_case("glm45m_chunk_b1_q64_k32768_t2048", "glm45m-chunk", 1, 64, 32768, 2048, "tail-random"),
        glm_case("glm45m_chunk_b4_q64_k32768_t2048", "glm45m-chunk", 4, 64, 32768, 2048, "tail-random"),
        glm_case("glm45m_chunk_b1_q128_k65536_t2048", "glm45m-chunk", 1, 128, 65536, 2048, "tail-random"),
        glm_case("glm45m_chunk_b4_q128_k32768_t2048", "glm45m-chunk", 4, 128, 32768, 2048, "tail-random"),
        glm_case("glm45m_chunk_b1_q512_k65536_t4096", "glm45m-chunk", 1, 512, 65536, 4096, "tail-random"),
        glm_case("glm45m_chunk_b2_q512_k32768_t2048", "glm45m-chunk", 2, 512, 32768, 2048, "tail-random"),
    ]

    # Reuse the full measured B1--B16 GLM true-prefill matrix.
    cases.extend(_glm_moe_dsa_45_batch_cases())

    # General-policy guardrails.  These remain outside the GLM specialization.
    cases.extend([
        # DeepSeek-V3.2-style MHA, including its asymmetric D192/D128 decode.
        BenchmarkCase("guard_deepseek_decode_b1", "guard-decode", 1, 1, 32768, 2048,
                      query_heads=128, kv_heads=128, head_dim=192, value_head_dim=128),
        BenchmarkCase("guard_deepseek_decode_b8", "guard-decode", 8, 1, 16384, 2048,
                      query_heads=128, kv_heads=128, head_dim=192, value_head_dim=128),
        BenchmarkCase("guard_deepseek_chunk_q128", "guard-chunk", 1, 128, 32768, 2048,
                      query_heads=128, kv_heads=128, head_dim=192, value_head_dim=128, pattern="tail-random"),
        BenchmarkCase("guard_deepseek_prefill_s1024", "guard-prefill", 1, 1024, 1024, 512,
                      query_heads=128, kv_heads=128, head_dim=192, value_head_dim=128, pattern="causal-mixed"),
        BenchmarkCase("guard_deepseek_prefill_s2048", "guard-prefill", 1, 2048, 2048, 1024,
                      query_heads=128, kv_heads=128, head_dim=192, value_head_dim=128, pattern="causal-window"),

        # Ordinary MHA D128 controls.
        BenchmarkCase("guard_mha_decode_b1", "guard-decode", 1, 1, 32768, 2048,
                      query_heads=32, kv_heads=32, head_dim=128, value_head_dim=128),
        BenchmarkCase("guard_mha_decode_b16", "guard-decode", 16, 1, 32768, 2048,
                      query_heads=32, kv_heads=32, head_dim=128, value_head_dim=128),
        BenchmarkCase("guard_mha_chunk_q128", "guard-chunk", 1, 128, 32768, 2048,
                      query_heads=32, kv_heads=32, head_dim=128, value_head_dim=128, pattern="tail-random"),
        BenchmarkCase("guard_mha_prefill_b1_s1024", "guard-prefill", 1, 1024, 1024, 512,
                      query_heads=32, kv_heads=32, head_dim=128, value_head_dim=128, pattern="causal-mixed"),
        BenchmarkCase("guard_mha_prefill_b4_s2048", "guard-prefill", 4, 2048, 2048, 1024,
                      query_heads=32, kv_heads=32, head_dim=128, value_head_dim=128, pattern="causal-window"),

        # Ratio-4 GQA and ratio-64 MQA controls exercise non-GLM dispatch.
        BenchmarkCase("guard_gqa_decode_b1", "guard-decode", 1, 1, 32768, 2048,
                      query_heads=32, kv_heads=8, head_dim=128, value_head_dim=128),
        BenchmarkCase("guard_gqa_decode_b16", "guard-decode", 16, 1, 32768, 2048,
                      query_heads=32, kv_heads=8, head_dim=128, value_head_dim=128),
        BenchmarkCase("guard_gqa_chunk_b1_q128", "guard-chunk", 1, 128, 32768, 2048,
                      query_heads=32, kv_heads=8, head_dim=128, value_head_dim=128, pattern="tail-random"),
        BenchmarkCase("guard_gqa_chunk_b4_q128", "guard-chunk", 4, 128, 32768, 2048,
                      query_heads=32, kv_heads=8, head_dim=128, value_head_dim=128, pattern="tail-random"),
        BenchmarkCase("guard_gqa_prefill_b1_s1024", "guard-prefill", 1, 1024, 1024, 512,
                      query_heads=32, kv_heads=8, head_dim=128, value_head_dim=128, pattern="causal-mixed"),
        BenchmarkCase("guard_gqa_prefill_b4_s1024", "guard-prefill", 4, 1024, 1024, 512,
                      query_heads=32, kv_heads=8, head_dim=128, value_head_dim=128, pattern="causal-window"),
        BenchmarkCase("guard_mqa_chunk_b1_q128", "guard-chunk", 1, 128, 32768, 2048,
                      query_heads=64, kv_heads=1, head_dim=128, value_head_dim=128, pattern="tail-random"),
        BenchmarkCase("guard_mqa_chunk_b4_q128", "guard-chunk", 4, 128, 32768, 2048,
                      query_heads=64, kv_heads=1, head_dim=128, value_head_dim=128, pattern="tail-random"),
    ])
    return cases



_COMMON_INDEXED_3H_PROFILES = (
    ("glm", 64, 64, 256, 256),
    ("deepseek", 128, 128, 192, 128),
    ("mha128", 32, 32, 128, 128),
    ("gqa4", 32, 8, 128, 128),
    ("gqa8", 64, 8, 128, 128),
    ("mqa64", 64, 1, 128, 128),
    ("mha64", 32, 32, 64, 64),
)

_COMMON_INDEXED_5H_PROFILES = _COMMON_INDEXED_3H_PROFILES + (
    # Independent guardrails for common but previously extrapolated layouts.
    ("gqa2", 32, 16, 128, 128),
    ("gqa7", 28, 4, 128, 128),
    ("mha96", 32, 32, 96, 96),
)

_COMMON_INDEXED_PREFILL_SPECS = (
    (1, 512, 256, "causal-mixed"),
    (4, 512, 256, "causal-window"),
    (1, 1024, 512, "causal-mixed"),
    (4, 1024, 512, "causal-window"),
    (1, 2048, 512, "causal-window"),
    (4, 2048, 1024, "causal-mixed"),
    (1, 4096, 1024, "causal-window"),
    (4, 4096, 2048, "causal-mixed"),
    (1, 8192, 1024, "causal-window"),
    (2, 8192, 2048, "causal-mixed"),
    (1, 2048, 2048, "causal-mixed"),
    (2, 4096, 4096, "causal-window"),
    (1, 1024, 1024, "causal-mixed"),
    (2, 2048, 1536, "causal-window"),
    (1, 4096, 512, "causal-window"),
    (2, 8192, 4096, "causal-mixed"),
    # High-batch prefill guardrails across every profile.
    (8, 512, 256, "causal-window"),
    (16, 512, 256, "causal-mixed"),
    (8, 1024, 512, "causal-window"),
    (16, 1024, 512, "causal-mixed"),
    (8, 2048, 512, "causal-window"),
    (8, 2048, 1024, "causal-mixed"),
    # Long-B16 boundary is deliberately measured without extrapolating it.
    (16, 2048, 512, "causal-window"),
)

_COMMON_INDEXED_5H_CROSSOVER_SPECS = (
    # Complete both locality patterns around the remaining short-prefill
    # bitmask-to-block-sparse crossover.
    (8, 512, 256, "causal-mixed"),
    (16, 512, 256, "causal-window"),
    (4, 1024, 512, "causal-mixed"),
    (8, 1024, 512, "causal-mixed"),
    (16, 1024, 512, "causal-window"),
    # Pair the existing sparse B16 window case with a denser mixed control.
    (16, 2048, 1024, "causal-mixed"),
)


def _common_indexed_cases(
    *,
    prefix: str,
    profiles: tuple[tuple[str, int, int, int, int], ...],
    extra_prefill_specs: tuple[tuple[int, int, int, str], ...] = (),
) -> list[BenchmarkCase]:
    """Build a cross-profile decode, chunk, and exact-prefill matrix."""

    cases: list[BenchmarkCase] = []

    def add(
        profile: str,
        regime: str,
        batch: int,
        q: int,
        k: int,
        topk: int,
        hq: int,
        hkv: int,
        dq: int,
        dv: int,
        pattern: str,
    ) -> None:
        cases.append(
            BenchmarkCase(
                name=(
                    f"{prefix}_{profile}_{regime}_b{batch}_q{q}_k{k}_t{topk}_"
                    f"{pattern.replace('-', '_')}"
                ),
                category=f"{prefix}-{profile}-{regime}",
                batch=batch,
                query_length=q,
                kv_length=k,
                topk=topk,
                query_heads=hq,
                kv_heads=hkv,
                head_dim=dq,
                value_head_dim=dv,
                pattern=pattern,
                benchmark_union=True,
            )
        )

    for profile, hq, hkv, dq, dv in profiles:
        # Decode occupancy, context, selected-set, and batch scaling.
        for batch, k, topk, pattern in (
            (1, 4096, 256, "random"),
            (1, 32768, 2048, "tail-random"),
            (4, 65536, 4096, "tail-random"),
            (8, 16384, 1024, "random"),
            (16, 8192, 512, "random"),
            (32, 4096, 512, "tail-random"),
        ):
            add(profile, "decode", batch, 1, k, min(topk, k), hq, hkv, dq, dv, pattern)

        # Speculative and chunked-query manifold.
        for batch, q, k, topk, pattern in (
            (1, 4, 32768, 2048, "tail-random"),
            (4, 4, 16384, 1024, "random"),
            (8, 4, 16384, 1024, "random"),
            (1, 16, 32768, 2048, "tail-random"),
            (4, 64, 32768, 2048, "tail-random"),
            (4, 128, 32768, 2048, "tail-random"),
            (2, 512, 8192, 1024, "random"),
        ):
            add(profile, "chunk", batch, q, k, min(topk, k), hq, hkv, dq, dv, pattern)

        # True-prefill density, locality, sequence, and batch scaling.
        for batch, q, topk, pattern in (
            _COMMON_INDEXED_PREFILL_SPECS + extra_prefill_specs
        ):
            add(profile, "prefill", batch, q, q, min(topk, q), hq, hkv, dq, dv, pattern)

    return cases


def _common_indexed_3h_cases() -> list[BenchmarkCase]:
    """The stable 252-case GLM-first/common-profile qualification matrix."""

    return _common_indexed_cases(
        prefix="common3h",
        profiles=_COMMON_INDEXED_3H_PROFILES,
    )


def _common_indexed_5h_cases() -> list[BenchmarkCase]:
    """Expanded 420-case promotion and crossover qualification matrix.

    In addition to every 3h regime, this suite measures ratio-2 and ratio-7
    GQA plus D96 MHA. It completes paired locality coverage at S512/S1024 high
    batch and at the S2048/B16 boundary. With the companion runner's 12 x
    1000 ms timing defaults and all exact candidates enabled, this is intended
    to run for at least five hours on an H100.
    """

    return _common_indexed_cases(
        prefix="common5h",
        profiles=_COMMON_INDEXED_5H_PROFILES,
        extra_prefill_specs=_COMMON_INDEXED_5H_CROSSOVER_SPECS,
    )


def _serving_guardrail_cases() -> list[BenchmarkCase]:
    """Small compatibility slice for speculative/chunked and known regressions."""

    return [
        BenchmarkCase("serving_guard_spec_b1_q8", "serving-guardrail", 1, 8, 32768, 2048),
        BenchmarkCase("serving_guard_spec_b16_q8", "serving-guardrail", 16, 8, 32768, 2048),
        BenchmarkCase("serving_guard_chunk_q128_k32768", "serving-guardrail", 1, 128, 32768, 2048),
        BenchmarkCase("serving_guard_chunk_q128_k65536", "serving-guardrail", 1, 128, 65536, 2048),
        BenchmarkCase(
            "serving_guard_chunk_mha_q128_k65536", "serving-guardrail",
            1, 128, 65536, 2048, query_heads=32, kv_heads=32,
        ),
        BenchmarkCase(
            "serving_guard_mqa_scalar_q129_t129", "serving-guardrail",
            1, 129, 5003, 129, query_heads=64, kv_heads=1, pattern="random",
        ),
        BenchmarkCase("serving_guard_chunk_q512_k65536", "serving-guardrail", 1, 512, 65536, 2048),
        BenchmarkCase(
            "serving_guard_wide_short_near_dense", "serving-guardrail",
            3, 5, 5003, 5003, query_heads=64, kv_heads=8,
            head_dim=96, value_head_dim=256, pattern="random",
        ),
        BenchmarkCase(
            "serving_guard_wide_high_rows", "serving-guardrail",
            3, 257, 509, 127, query_heads=64, kv_heads=8,
            head_dim=192, value_head_dim=256, pattern="random",
        ),
    ]


def _serving_medium_cases() -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for group in (_serving_decode_cases(), _serving_prefill_cases(), _serving_guardrail_cases()):
        for case in group:
            _add_unique(cases, case)
    return cases


def _odd_shape_cases() -> list[BenchmarkCase]:
    """Non-aligned sequence lengths and top-k widths for robustness/perf."""

    specs = [
        (1, 1, 127, 17, 4, 4, 64, 64),
        (1, 1, 2500, 257, 32, 8, 128, 128),
        (2, 3, 2501, 333, 32, 8, 128, 128),
        (1, 7, 4093, 511, 32, 8, 64, 128),
        (3, 5, 1025, 129, 32, 8, 128, 64),
        (1, 17, 8191, 777, 32, 8, 128, 128),
        (1, 63, 2500, 257, 32, 8, 128, 128),
        (1, 65, 2501, 257, 32, 8, 128, 128),
        (2, 127, 4097, 513, 32, 8, 128, 128),
        (1, 129, 5003, 257, 64, 1, 128, 128),
        (1, 33, 997, 73, 28, 4, 64, 64),
    ]
    return [
        BenchmarkCase(
            name=(
                f"odd_b{batch}_q{query_length}_k{kv_length}_t{topk}_"
                f"hq{hq}_hkv{hkv}_d{head_dim}_dv{value_head_dim}"
            ),
            category="odd-shape",
            batch=batch,
            query_length=query_length,
            kv_length=kv_length,
            topk=topk,
            query_heads=hq,
            kv_heads=hkv,
            head_dim=head_dim,
            value_head_dim=value_head_dim,
            pattern="random",
        )
        for (
            batch, query_length, kv_length, topk, hq, hkv, head_dim, value_head_dim
        ) in specs
    ]


def _random_shape_cases(*, seed: int, count: int, max_kv_length: int) -> list[BenchmarkCase]:
    """Deterministic random shapes, biased toward awkward non-aligned sizes."""

    if count < 1:
        raise ValueError("random-cases must be positive")
    if max_kv_length < 127:
        raise ValueError("random-max-k must be at least 127")
    rng = random.Random(seed)
    q_choices = (1, 2, 3, 5, 7, 9, 15, 17, 31, 33, 63, 65, 95, 127, 129, 257)
    k_pool = (127, 255, 509, 997, 1025, 1537, 2049, 2500, 2501, 4093, 4097, 5003, 6143, 8191)
    k_choices = tuple(value for value in k_pool if value <= max_kv_length)
    head_choices = ((4, 4), (8, 2), (16, 4), (28, 4), (32, 8), (64, 8), (64, 1))
    cases: list[BenchmarkCase] = []
    attempts = 0
    max_attempts = count * 100
    while len(cases) < count:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("could not generate enough launch-safe random shapes")
        batch = rng.choice((1, 1, 1, 2, 3, 4))
        query_length = rng.choice(q_choices)
        kv_length = rng.choice(k_choices)
        topk_candidates = (
            1, 2, 3, 7, 17, 31, 32, 33, 63, 64, 65, 73, 127, 129,
            257, 333, 511, 513, 777, 1025,
            max(1, kv_length // 16), max(1, kv_length // 8),
            max(1, kv_length // 4), max(1, kv_length - 1), kv_length,
        )
        topk = rng.choice(tuple(dict.fromkeys(v for v in topk_candidates if v <= kv_length)))
        hq, hkv = rng.choice(head_choices)
        head_dim = rng.choice((64, 96, 128, 192, 256))
        value_head_dim = rng.choice((64, 96, 128, 192, 256))
        if _known_unsafe_dense_fa4_random_shape(
            query_length=query_length,
            head_dim=head_dim,
            value_head_dim=value_head_dim,
        ):
            continue
        index = len(cases)
        cases.append(
            BenchmarkCase(
                name=(
                    f"random_{index:03d}_b{batch}_q{query_length}_k{kv_length}_t{topk}_"
                    f"hq{hq}_hkv{hkv}_d{head_dim}_dv{value_head_dim}"
                ),
                category="random-shape",
                batch=batch,
                query_length=query_length,
                kv_length=kv_length,
                topk=topk,
                query_heads=hq,
                kv_heads=hkv,
                head_dim=head_dim,
                value_head_dim=value_head_dim,
                pattern=rng.choice(("random", "tail-random")),
            )
        )
    return cases



def _topk_sweep_cases() -> list[BenchmarkCase]:
    """Representative top-k crossover sweep, including odd widths."""

    specs = [
        ("decode", 1, 1, 4097, 32, 8, 128, 128, (1, 17, 73, 129, 257, 513, 1025, 2049, 4096)),
        ("spec", 1, 8, 4097, 32, 8, 128, 128, (1, 17, 73, 129, 257, 513, 1025, 2049, 4096)),
        ("prefill", 1, 64, 4097, 32, 8, 128, 128, (1, 17, 73, 129, 257, 513, 1025, 2049, 4096)),
        ("mqa", 1, 129, 5003, 64, 1, 128, 128, (1, 17, 73, 129, 257, 513, 1025, 2049, 4097)),
    ]
    cases: list[BenchmarkCase] = []
    for label, batch, query_length, kv_length, hq, hkv, dim, value_dim, topks in specs:
        for topk in topks:
            if topk > kv_length:
                continue
            cases.append(
                BenchmarkCase(
                    name=(
                        f"topk_{label}_b{batch}_q{query_length}_k{kv_length}_t{topk}_"
                        f"hq{hq}_hkv{hkv}_d{dim}_dv{value_dim}"
                    ),
                    category="topk-sweep",
                    batch=batch,
                    query_length=query_length,
                    kv_length=kv_length,
                    topk=topk,
                    query_heads=hq,
                    kv_heads=hkv,
                    head_dim=dim,
                    value_head_dim=value_dim,
                    pattern="random",
                )
            )
    return cases

def _union_candidate_cases() -> list[BenchmarkCase]:
    """Candidate-only overlap/MQA shapes; auto dispatch remains conservative."""

    cases: list[BenchmarkCase] = []
    for batch in (1, 4):
        for query_length in (64, 128, 256):
            for kv_length, topk in ((32768, 2048), (65536, 4096)):
                cases.append(
                    BenchmarkCase(
                        name=(
                            f"union_candidate_mqa_b{batch}_q{query_length}_"
                            f"k{kv_length}_t{topk}"
                        ),
                        category="union-candidate",
                        batch=batch,
                        query_length=query_length,
                        kv_length=kv_length,
                        topk=topk,
                        query_heads=64,
                        kv_heads=1,
                        pattern="tail-random",
                    )
                )
    return cases



def _prefill_focus_cases() -> list[BenchmarkCase]:
    """Focused sparse-prefill crossovers for grouped rows, union, SDPA, and FA4."""

    cases: list[BenchmarkCase] = []
    # Ordinary GQA: batch/query crossover at production context lengths.
    for batch, query_length, kv_length, topk in (
        (1, 64, 16384, 1024),
        (4, 64, 16384, 1024),
        (16, 64, 16384, 1024),
        (1, 128, 32768, 2048),
        (4, 128, 32768, 2048),
        (16, 128, 32768, 2048),
        (1, 512, 32768, 2048),
        (2, 512, 32768, 2048),
        (4, 512, 32768, 2048),
    ):
        cases.append(
            BenchmarkCase(
                f"prefill_focus_gqa_b{batch}_q{query_length}_k{kv_length}_t{topk}",
                "prefill-focus",
                batch,
                query_length,
                kv_length,
                topk,
            )
        )

    # Head-sharing controls: ratio-8 and ratio-7 GQA should benefit from
    # register-shared K/V, while MHA (ratio 1) is the no-sharing control.
    for batch, query_length, query_heads, kv_heads, kv_length, topk in (
        (1, 128, 64, 8, 32768, 2048),
        (4, 128, 64, 8, 32768, 2048),
        (1, 64, 28, 4, 16384, 1024),
        (4, 64, 28, 4, 16384, 1024),
        (1, 64, 32, 32, 16384, 1024),
        (1, 128, 32, 32, 32768, 2048),
    ):
        cases.append(
            BenchmarkCase(
                f"prefill_focus_heads_b{batch}_q{query_length}_h{query_heads}_{kv_heads}",
                "prefill-focus-head-sharing",
                batch,
                query_length,
                kv_length,
                topk,
                query_heads=query_heads,
                kv_heads=kv_heads,
            )
        )

    # Odd-K density sweep around the grouped-warp/SDPA crossover.
    for topk in (17, 129, 513, 1025, 2049, 4096):
        cases.append(
            BenchmarkCase(
                f"prefill_focus_q64_k4097_t{topk}",
                "prefill-focus",
                1,
                64,
                4097,
                topk,
                pattern="random",
            )
        )

    # High-ratio MQA: measure packed union against grouped rows and references.
    # expose the transition to masked SDPA at denser selections.
    for batch, query_length in ((1, 64), (1, 128), (1, 256), (4, 64), (4, 128)):
        for kv_length, topk in ((32768, 2048), (32768, 4096), (32768, 8192)):
            cases.append(
                BenchmarkCase(
                    f"prefill_focus_mqa_b{batch}_q{query_length}_k{kv_length}_t{topk}",
                    "prefill-focus-mqa",
                    batch,
                    query_length,
                    kv_length,
                    topk,
                    query_heads=64,
                    kv_heads=1,
                    pattern="tail-random",
                )
            )
    return cases

def build_suite(
    name: str,
    single: BenchmarkCase,
    *,
    seed: int = 0,
    random_cases: int = 24,
    random_max_k: int = 8192,
) -> list[BenchmarkCase]:
    if name == "single":
        return [single]

    batching = _batching_cases()
    speculative = _speculative_cases()
    prefill = _prefill_cases()
    groups = {
        "decode": _decode_grid() + batching + speculative,
        "batch": batching + speculative + [case for case in prefill if case.batch > 1],
        "prefill": prefill,
        "heads": _head_layout_cases(),
        "patterns": _pattern_cases(),
        "regression": _regression_cases(),
        "odd": _odd_shape_cases(),
        "random": _random_shape_cases(
            seed=seed, count=random_cases, max_kv_length=random_max_k
        ),
        "topk-sweep": _topk_sweep_cases(),
        "union-candidates": _union_candidate_cases(),
        "prefill-focus": _prefill_focus_cases(),
        "iteration": _iteration_cases(),
        "iteration-medium": _iteration_medium_cases(),
        "serving-decode": _serving_decode_cases(),
        "serving-prefill": _serving_prefill_cases(),
        "serving-medium": _serving_medium_cases(),
        "dsa-models": _dsa_model_cases(),
        "glm-moe-dsa-45": _glm_moe_dsa_45_cases(),
        "glm-moe-dsa-45-overnight": _glm_moe_dsa_45_overnight_cases(),
        "glm-moe-dsa-45-batch": _glm_moe_dsa_45_batch_cases(),
        "glm-moe-dsa-45-serving-matrix": _glm_moe_dsa_45_serving_matrix_cases(),
        "common-indexed-3h": _common_indexed_3h_cases(),
        "common-indexed-5h": _common_indexed_5h_cases(),
    }

    if name == "quick":
        return [
            BenchmarkCase("quick_decode_16k", "quick", 1, 1, 16384, 1024),
            BenchmarkCase("quick_decode_32k", "quick", 1, 1, 32768, 2048),
            BenchmarkCase("quick_decode_batch8", "quick", 8, 1, 32768, 2048),
            BenchmarkCase("quick_spec_q4", "quick", 1, 4, 32768, 2048),
            BenchmarkCase("quick_prefill_q128", "quick", 1, 128, 32768, 2048),
            BenchmarkCase("quick_prefill_q512", "quick", 1, 512, 65536, 4096),
            BenchmarkCase(
                "quick_union_identical", "quick", 1, 128, 32768, 2048, pattern="identical"
            ),
            BenchmarkCase(
                "quick_union_random", "quick", 1, 128, 32768, 2048, pattern="random"
            ),
        ]

    if name in groups:
        return groups[name]
    if name not in ("common", "all"):
        raise ValueError(f"unknown suite: {name}")

    common_groups = ("decode", "prefill", "heads", "patterns")
    all_groups = common_groups + (
        "regression",
        "odd",
        "random",
        "topk-sweep",
        "union-candidates",
        "prefill-focus",
    )
    cases: list[BenchmarkCase] = []
    for group_name in common_groups if name == "common" else all_groups:
        for case in groups[group_name]:
            _add_unique(cases, case)
    return cases


def should_run_native(case: BenchmarkCase, mode: str, suite: str) -> bool:
    """Whether to run the exact upstream-compatible FA4 score-mod baseline."""

    if mode == "none":
        return False
    if mode == "all" or suite == "single":
        return True
    # Subset mode keeps one point from every important execution regime.
    if suite in ("serving-medium", "serving-prefill") and case.category == "serving-prefill":
        return (
            case.batch == 1
            and case.kv_length <= 8192
            and case.topk in (1024, 2048, 4096)
        )
    return (
        (case.query_length == 1 and case.batch in (1, 8, 32, 128))
        or (case.query_length in (4, 16) and case.batch in (1, 16))
        or (case.query_length in (128, 512) and case.batch in (1, 4))
        or case.category in ("head-layout", "index-pattern")
    )


def _sdpa_mask_bytes(case: BenchmarkCase) -> int:
    # A [B, 1, Q, K] boolean mask broadcasts across query heads.
    return case.batch * case.query_length * case.kv_length


def should_run_sdpa(
    case: BenchmarkCase,
    mode: str,
    suite: str,
    max_mask_bytes: int,
) -> bool:
    """Whether to run the dense masked PyTorch SDPA context baseline."""

    if mode == "none" or _sdpa_mask_bytes(case) > max_mask_bytes:
        return False
    if mode == "all" or suite == "single":
        return True
    # Subset mode provides a different implementation context without roughly
    # doubling every large sweep. It samples decode, batching, speculative
    # decode, prefill, MQA, awkward shapes, and top-k crossover points.
    if suite in ("serving-medium", "serving-prefill") and case.category == "serving-prefill":
        return case.batch == 1 and case.kv_length <= 4096 and case.topk == 2048
    return (
        case.category in ("odd-shape", "union-candidate")
        or (
            case.category == "random-shape"
            and int(case.name.split("_", 2)[1]) % 8 == 0
        )
        or (case.category == "topk-sweep" and case.topk in (1, 73, 257, 513, 2049, case.kv_length - 1))
        or (case.query_length == 1 and case.batch in (1, 32, 128))
        or (case.query_length in (8, 64, 128) and case.batch == 1)
        or case.name in (
            "spec_b4_q8_k32768_t2048",
            "prefill_b4_q128_k32768_t2048",
            "heads_decode_b1_q1_k32768_t2048_hq64_hkv1",
        )
    )


def build_sdpa_topk_mask(indices: torch.Tensor, kv_length: int) -> torch.Tensor:
    """Build a broadcastable boolean selected-set mask for PyTorch SDPA."""

    valid = (indices >= 0) & (indices < kv_length)
    safe = indices.to(torch.int64).clamp_(0, kv_length - 1)
    counts = torch.zeros(
        (*indices.shape[:2], kv_length),
        dtype=torch.int32,
        device=indices.device,
    )
    counts.scatter_add_(2, safe, valid.to(torch.int32))
    return counts.ne(0).unsqueeze(1)


def _known_unsafe_dense_fa4_random_shape(
    *, query_length: int, head_dim: int, value_head_dim: int
) -> bool:
    """Exclude a reproducible upstream dense-FA4 H100 launch corner.

    D64/DV256 produced CUDA illegal-address failures in the ordinary dense
    FA4 reference for Q=1 as well as multi-token MQA/GQA.  The indexed
    correctness matrix covers these dimensions; random timing excludes the
    unsafe reference shape so one baseline launch cannot poison the entire
    suite CUDA context.
    """

    del query_length  # The upstream launch corner reproduces for Q=1 and Q>1.
    return head_dim == 64 and value_head_dim == 256


def _union_benchmark_eligible(case: BenchmarkCase) -> bool:
    """Only force union where the cross-query kernel has a real query tile.

    Auto dispatch already requires Q>=32.  Forcing union on tiny-Q random
    shapes adds no useful crossover data and a Q7/high-ratio stress case was
    the last operation before a repeatable CUDA-context corruption.  Keep the
    stress sweep inside the production-valid cross-query domain.
    """

    return (
        case.query_length >= 32
        and case.head_dim % 64 == 0
        and case.value_head_dim % 64 == 0
    )


def should_compare_backends(case: BenchmarkCase, mode: str) -> bool:
    """Whether to time forced row/dense controls in addition to auto."""

    if mode == "none":
        return False
    if mode == "row-dense":
        return True
    if mode in ("all", "qgt1"):
        return mode == "all" or case.query_length > 1
    # Focused subset: enough points to tune the warp/native crossover and
    # detect whether extra batch occupancy rescues the union path.
    return (
        case.category == "regression"
        or (case.category == "speculative-decode" and case.batch in (1, 4, 16))
        or (
            case.category == "chunked-prefill"
            and case.query_length in (64, 128)
            and case.batch in (1, 4)
        )
        or case.category in ("head-layout", "index-pattern")
    )


def should_compare_union(case: BenchmarkCase, mode: str) -> bool:
    """Whether forced-backend comparison should include the union kernel."""

    return (
        case.benchmark_union
        and mode != "row-dense"
        and should_compare_backends(case, mode)
    )


def _cuda_event_elapsed(fn: Callable[[], object], iterations: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def bench_operation(fn: Callable[[], object], config: TimingConfig) -> dict[str, Any]:
    for _ in range(config.warmup):
        fn()
    torch.cuda.synchronize()

    if config.fixed_iters > 0:
        iterations = config.fixed_iters
    else:
        pilot_iterations = min(5, config.max_iters)
        pilot_ms = _cuda_event_elapsed(fn, pilot_iterations)
        per_iter_ms = max(pilot_ms / pilot_iterations, 1e-6)
        iterations = int(round(config.target_round_ms / per_iter_ms))
        iterations = max(config.min_iters, min(config.max_iters, iterations))

    samples_ms = [
        _cuda_event_elapsed(fn, iterations) / iterations
        for _ in range(config.rounds)
    ]
    return {
        "iterations_per_round": iterations,
        "rounds": config.rounds,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "p90_ms": _percentile_nearest_rank(samples_ms, 0.90),
        "mean_ms": statistics.fmean(samples_ms),
        "stddev_ms": statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def _relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    abs_error = (ref - cand).abs()
    denominator = ref.abs().clamp_min(1e-6)
    return {
        "max_abs": float(abs_error.max().item()),
        "mean_abs": float(abs_error.mean().item()),
        "max_rel": float((abs_error / denominator).max().item()),
    }


def create_runtime(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    device: torch.device,
    sm_count: int,
    seed: int,
    include_native: bool,
    include_sdpa: bool,
    sdpa_max_mask_bytes: int,
    compare_forced_backends: bool,
    compare_union: bool,
    prefill_lab: bool = False,
    block_sparse_lab: bool = False,
    cute_block_metadata: bool = False,
) -> CaseRuntime:
    fa4 = load_fa4_modules()
    choose_indexed_plan = fa4["choose_indexed_plan"]
    IndexedPath = fa4["IndexedPath"]
    cast_indexed_kv_indices = fa4["cast_indexed_kv_indices"]
    prepare_indexed_kv_indices = fa4["prepare_indexed_kv_indices"]
    build_topk_bitmask = fa4["build_topk_bitmask"]
    build_topk_bitmask_cute = fa4["build_topk_bitmask_cute"]
    topk_bitmask_score_mod = fa4["topk_bitmask_score_mod"]
    flash_attn_fwd = fa4["flash_attn_fwd"]
    build_indexed_block_sparse_tensors_cute = fa4["build_indexed_block_sparse_tensors_cute"]
    create_indexed_block_sparse_cute_workspace = fa4["create_indexed_block_sparse_cute_workspace"]
    indexed_block_density = fa4["indexed_block_density"]

    case_seed = _stable_case_seed(seed, case.name)
    torch.manual_seed(case_seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(case_seed)

    q = torch.randn(
        (case.batch, case.query_length, case.query_heads, case.head_dim),
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        (case.batch, case.kv_length, case.kv_heads, case.head_dim),
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        (case.batch, case.kv_length, case.kv_heads, case.value_head_dim),
        device=device,
        dtype=dtype,
    )
    indices = make_indices(
        pattern=case.pattern,
        batch=case.batch,
        query_length=case.query_length,
        kv_length=case.kv_length,
        topk=case.topk,
        tail=case.resolved_tail(),
        device=device,
        generator=generator,
    )

    plan_kwargs = dict(
        batch_size=case.batch,
        query_length=case.query_length,
        kv_length=case.kv_length,
        query_heads=case.query_heads,
        kv_heads=case.kv_heads,
        qk_head_dim=case.head_dim,
        value_head_dim=case.value_head_dim,
        topk=case.topk,
        sm_count=sm_count,
    )
    backend_plans = {
        "auto": choose_indexed_plan(
            **plan_kwargs,
            backend="auto",
            union_compute_inflation_hint=case.union_compute_inflation_hint,
        ),
        **{
            name: choose_indexed_plan(**plan_kwargs, backend=name)
            for name in ("warp", "warp_scalar", "dense")
        },
    }
    plan = backend_plans["auto"]
    auto_uses_union = plan.path in (
        IndexedPath.UNION_FA4,
        IndexedPath.PACKED_UNION_FA4,
    )
    union_eligible = _union_benchmark_eligible(case)
    need_union = union_eligible and (auto_uses_union or compare_union)
    if need_union:
        backend_plans["union"] = choose_indexed_plan(
            **plan_kwargs, backend="union"
        )
    grouped_warp_eligible = backend_plans["warp"].decode_heads_per_warp > 1

    prefill_lab_eligible = (
        prefill_lab
        and case.query_length == case.kv_length
        and case.query_length >= 512
        and case.topk <= case.kv_length
        and case.query_heads % case.kv_heads == 0
        and case.head_dim in {64, 96, 128, 192, 256}
        and case.value_head_dim in {64, 96, 128, 192, 256}
    )
    block_sparse_lab_eligible = prefill_lab_eligible and block_sparse_lab

    # These preparations are intentionally retained separately so the report
    # can distinguish policy choice from preparation and kernel costs.
    cast_indices = cast_indexed_kv_indices(indices, case.kv_length)
    union_indices = (
        prepare_indexed_kv_indices(indices, case.kv_length)
        if need_union
        else cast_indices
    )
    need_bitmask = (
        include_native
        or plan.path in (
            IndexedPath.DENSE_INDEXED,
            IndexedPath.FA4_BITMASK_INDEXED,
            IndexedPath.BLOCK_SPARSE_INDEXED,
        )
        or prefill_lab_eligible
    )
    bitmask = (
        build_topk_bitmask(indices, case.kv_length, assume_unique=True)
        if need_bitmask
        else None
    )
    sdpa_mask = (
        build_sdpa_topk_mask(indices, case.kv_length)
        if include_sdpa
        else None
    )

    output_names = ["fa4_dense_full", "indexed_auto", "indexed_prepared"]
    if compare_forced_backends:
        output_names.extend(["indexed_warp", "indexed_dense"])
        if compare_union and union_eligible:
            output_names.append("indexed_union")
        if grouped_warp_eligible:
            output_names.append("indexed_warp_scalar")
    if bitmask is not None:
        output_names.append("fa4_native_topk")
    if prefill_lab_eligible:
        output_names.extend([
            "indexed_dense_128x64",
            "fa4_topk_128x64",
            "fa4_topk_reuse_128x64",
            "fa4_topk_cute_128x64",
        ])
    if block_sparse_lab_eligible and cute_block_metadata:
        output_names.append("fa4_block_sparse_cute_128x64")
    outputs = {
        name: torch.empty(
            (case.batch, case.query_length, case.query_heads, case.value_head_dim),
            device=device,
            dtype=dtype,
        )
        for name in output_names
    }

    def indexed_call(
        *,
        backend: str,
        selected: torch.Tensor,
        output_name: str,
        prepared: bool,
        union_compute_inflation_hint: float | None = None,
        tile_mn: tuple[int, int] | None = None,
    ):
        return flash_attn_fwd(
            q,
            k,
            v,
            gather_kv_indices=selected,
            gather_kv_indices_prepared=prepared,
            indexed_backend=backend,
            indexed_union_compute_inflation_hint=union_compute_inflation_hint,
            tile_mn=tile_mn,
            out=outputs[output_name],
            _arch=90,
        )

    calls: dict[str, Callable[[], object]] = {
        "fa4_dense_full_kernel": lambda: flash_attn_fwd(q, k, v, out=outputs["fa4_dense_full"], _arch=90),
        "indexed_auto_end_to_end": lambda: indexed_call(
            backend="auto",
            selected=indices,
            output_name="indexed_auto",
            prepared=False,
            union_compute_inflation_hint=case.union_compute_inflation_hint,
        ),
    }

    if plan.path in (IndexedPath.UNION_FA4, IndexedPath.PACKED_UNION_FA4):
        calls["indexed_auto_prepared_kernel"] = lambda: indexed_call(
            backend="union", selected=union_indices, output_name="indexed_prepared", prepared=True
        )
        calls["indexed_auto_preparation_only"] = lambda: prepare_indexed_kv_indices(
            indices, case.kv_length
        )
    elif plan.path is IndexedPath.FA4_BITMASK_INDEXED:
        assert bitmask is not None
        auto_bitmask = torch.empty_like(bitmask)
        build_topk_bitmask_cute(indices, case.kv_length, out=auto_bitmask)
        calls["indexed_auto_prepared_kernel"] = lambda: flash_attn_fwd(
            q,
            k,
            v,
            score_mod=topk_bitmask_score_mod,
            aux_tensors=[auto_bitmask],
            tile_mn=(128, 64),
            mma_pv_is_rs=True,
            intra_wg_overlap=False,
            num_threads=384,
            num_splits=1,
            pack_gqa=False,
            out=outputs["indexed_prepared"],
            _arch=90,
        )
        calls["indexed_auto_preparation_only"] = lambda: build_topk_bitmask_cute(
            indices, case.kv_length, out=auto_bitmask
        )
    elif plan.path is IndexedPath.BLOCK_SPARSE_INDEXED:
        auto_sparse_workspace = create_indexed_block_sparse_cute_workspace(
            indices,
            seqlen_q=case.query_length,
            seqlen_k=case.kv_length,
            tile_m=128,
            tile_n=64,
        )
        auto_sparse_blocks, auto_sparse_bitmask, auto_sparse_workspace = (
            build_indexed_block_sparse_tensors_cute(
                indices,
                seqlen_q=case.query_length,
                seqlen_k=case.kv_length,
                tile_m=128,
                tile_n=64,
                workspace=auto_sparse_workspace,
            )
        )
        calls["indexed_auto_prepared_kernel"] = lambda: flash_attn_fwd(
            q,
            k,
            v,
            score_mod=topk_bitmask_score_mod,
            aux_tensors=[auto_sparse_bitmask],
            block_sparse_tensors=auto_sparse_blocks,
            tile_mn=(128, 64),
            pack_gqa=False,
            num_splits=1,
            intra_wg_overlap=False,
            out=outputs["indexed_prepared"],
            _arch=90,
        )
        calls["indexed_auto_preparation_only"] = lambda: (
            build_indexed_block_sparse_tensors_cute(
                indices,
                seqlen_q=case.query_length,
                seqlen_k=case.kv_length,
                tile_m=128,
                tile_n=64,
                workspace=auto_sparse_workspace,
            )
        )
    elif plan.path is IndexedPath.DENSE_INDEXED:
        assert bitmask is not None
        calls["indexed_auto_prepared_kernel"] = lambda: flash_attn_fwd(
            q,
            k,
            v,
            score_mod=topk_bitmask_score_mod,
            aux_tensors=[bitmask],
            out=outputs["indexed_prepared"],
            _arch=90,
        )
        calls["indexed_auto_preparation_only"] = lambda: build_topk_bitmask(
            indices, case.kv_length, assume_unique=True
        )
    else:
        calls["indexed_auto_prepared_kernel"] = lambda: indexed_call(
            backend="warp", selected=cast_indices, output_name="indexed_prepared", prepared=True
        )
        calls["indexed_auto_preparation_only"] = lambda: cast_indexed_kv_indices(
            indices, case.kv_length
        )

    if compare_forced_backends:
        calls.update(
            {
                "indexed_warp_kernel": lambda: indexed_call(
                    backend="warp", selected=cast_indices, output_name="indexed_warp", prepared=True
                ),
                "indexed_warp_end_to_end": lambda: indexed_call(
                    backend="warp", selected=indices, output_name="indexed_warp", prepared=False
                ),
                "indexed_warp_preparation_only": lambda: cast_indexed_kv_indices(
                    indices, case.kv_length
                ),
                "indexed_dense_end_to_end": lambda: indexed_call(
                    backend="dense", selected=indices, output_name="indexed_dense", prepared=False
                ),
                "indexed_dense_preparation_only": lambda: build_topk_bitmask(
                    indices, case.kv_length, assume_unique=True
                ),
            }
        )
        if compare_union and union_eligible:
            calls.update(
                {
                    "indexed_union_kernel": lambda: indexed_call(
                        backend="union",
                        selected=union_indices,
                        output_name="indexed_union",
                        prepared=True,
                    ),
                    "indexed_union_end_to_end": lambda: indexed_call(
                        backend="union",
                        selected=indices,
                        output_name="indexed_union",
                        prepared=False,
                    ),
                    "indexed_union_preparation_only": lambda: prepare_indexed_kv_indices(
                        indices, case.kv_length
                    ),
                }
            )
        if grouped_warp_eligible:
            calls.update(
                {
                    "indexed_warp_scalar_kernel": lambda: indexed_call(
                        backend="warp_scalar",
                        selected=cast_indices,
                        output_name="indexed_warp_scalar",
                        prepared=True,
                    ),
                    "indexed_warp_scalar_end_to_end": lambda: indexed_call(
                        backend="warp_scalar",
                        selected=indices,
                        output_name="indexed_warp_scalar",
                        prepared=False,
                    ),
                }
            )


    if prefill_lab_eligible:
        assert bitmask is not None
        reusable_bitmask = torch.empty_like(bitmask)
        cute_bitmask = torch.empty_like(bitmask)
        cute_bitmask_int32 = torch.empty_like(bitmask)

        def fa4_topk_candidate(*, reuse_workspace: bool):
            current_bitmask = build_topk_bitmask(
                indices,
                case.kv_length,
                assume_unique=True,
                out=reusable_bitmask if reuse_workspace else None,
            )
            return flash_attn_fwd(
                q,
                k,
                v,
                score_mod=topk_bitmask_score_mod,
                aux_tensors=[current_bitmask],
                tile_mn=(128, 64),
                out=outputs[
                    "fa4_topk_reuse_128x64" if reuse_workspace else "fa4_topk_128x64"
                ],
                _arch=90,
            )

        def fa4_cute_bitmask_candidate():
            current_bitmask = build_topk_bitmask_cute(
                indices,
                case.kv_length,
                out=cute_bitmask,
            )
            return flash_attn_fwd(
                q,
                k,
                v,
                score_mod=topk_bitmask_score_mod,
                aux_tensors=[current_bitmask],
                tile_mn=(128, 64),
                out=outputs["fa4_topk_cute_128x64"],
                _arch=90,
            )

        calls.update({
            "indexed_dense_128x64_end_to_end": lambda: indexed_call(
                backend="dense", selected=indices,
                output_name="indexed_dense_128x64", prepared=False,
                tile_mn=(128, 64),
            ),
            "fa4_topk_128x64_end_to_end": lambda: fa4_topk_candidate(
                reuse_workspace=False
            ),
            "fa4_topk_reuse_128x64_end_to_end": lambda: fa4_topk_candidate(
                reuse_workspace=True
            ),
            "fa4_topk_cute_128x64_end_to_end": fa4_cute_bitmask_candidate,
            "bitmask_build_cute_only": lambda: build_topk_bitmask_cute(
                indices, case.kv_length, out=cute_bitmask
            ),
            "bitmask_build_cute_int32_only": lambda: build_topk_bitmask_cute(
                cast_indices, case.kv_length, out=cute_bitmask_int32
            ),
            "indexed_index_cast_only": lambda: cast_indexed_kv_indices(
                indices, case.kv_length
            ),
            "bitmask_build_reuse_only": lambda: build_topk_bitmask(
                indices, case.kv_length, assume_unique=True, out=reusable_bitmask
            ),
            "fa4_topk_128x64_prepared_kernel": lambda: flash_attn_fwd(
                q, k, v, score_mod=topk_bitmask_score_mod,
                aux_tensors=[bitmask], tile_mn=(128, 64),
                out=outputs["fa4_topk_128x64"], _arch=90,
            ),
        })

    diagnostics: dict[str, Any] = {}
    if block_sparse_lab_eligible and cute_block_metadata:
        assert bitmask is not None
        cute_workspace = create_indexed_block_sparse_cute_workspace(
            indices,
            seqlen_q=case.query_length,
            seqlen_k=case.kv_length,
            tile_m=128,
            tile_n=64,
        )
        cute_blocks, cute_bitmask, cute_workspace = (
            build_indexed_block_sparse_tensors_cute(
                indices,
                seqlen_q=case.query_length,
                seqlen_k=case.kv_length,
                tile_m=128,
                tile_n=64,
                workspace=cute_workspace,
            )
        )
        num_n_blocks = math.ceil(case.kv_length / 64)
        diagnostics["block_sparse_cute_128x64_active_fraction"] = (
            indexed_block_density(cute_blocks.mask_block_cnt, num_n_blocks)
        )
        diagnostics["block_sparse_cute_workspace_bytes"] = cute_workspace.nbytes
        diagnostics["public_index_dtype"] = str(indices.dtype)

        def cute_metadata_only():
            return build_indexed_block_sparse_tensors_cute(
                indices,
                seqlen_q=case.query_length,
                seqlen_k=case.kv_length,
                tile_m=128,
                tile_n=64,
                workspace=cute_workspace,
            )

        def cute_block_sparse_end_to_end():
            current_blocks, current_bitmask, _ = cute_metadata_only()
            return flash_attn_fwd(
                q,
                k,
                v,
                score_mod=topk_bitmask_score_mod,
                aux_tensors=[current_bitmask],
                block_sparse_tensors=current_blocks,
                tile_mn=(128, 64),
                pack_gqa=False,
                num_splits=1,
                intra_wg_overlap=False,
                out=outputs["fa4_block_sparse_cute_128x64"],
                _arch=90,
            )

        calls.update({
            "fa4_block_sparse_cute_128x64_end_to_end": cute_block_sparse_end_to_end,
            "fa4_block_sparse_cute_128x64_prepared_kernel": lambda: flash_attn_fwd(
                q, k, v,
                score_mod=topk_bitmask_score_mod,
                aux_tensors=[cute_bitmask],
                block_sparse_tensors=cute_blocks,
                tile_mn=(128, 64),
                pack_gqa=False,
                num_splits=1,
                intra_wg_overlap=False,
                out=outputs["fa4_block_sparse_cute_128x64"],
                _arch=90,
            ),
            "fa4_block_sparse_cute_128x64_metadata_only": cute_metadata_only,
        })

    if sdpa_mask is not None:
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        enable_gqa = case.query_heads != case.kv_heads

        def torch_sdpa_kernel(mask: torch.Tensor = sdpa_mask):
            return F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=enable_gqa,
            ).transpose(1, 2)

        def torch_sdpa_end_to_end():
            current_mask = build_sdpa_topk_mask(indices, case.kv_length)
            return torch_sdpa_kernel(current_mask)

        calls.update(
            {
                "torch_sdpa_topk_kernel": torch_sdpa_kernel,
                "torch_sdpa_topk_mask_build_only": lambda: build_sdpa_topk_mask(
                    indices, case.kv_length
                ),
                "torch_sdpa_topk_end_to_end": torch_sdpa_end_to_end,
            }
        )

    if bitmask is not None:
        def fa4_native_topk_end_to_end():
            # This path requires no FA4 source modification: build an external
            # packed membership tensor and pass an ordinary score_mod/aux tensor
            # to the upstream FA4 forward API.
            current_bitmask = build_topk_bitmask(
                indices, case.kv_length, assume_unique=True
            )
            return flash_attn_fwd(
                q,
                k,
                v,
                score_mod=topk_bitmask_score_mod,
                aux_tensors=[current_bitmask],
                out=outputs["fa4_native_topk"],
                _arch=90,
            )

        calls.update(
            {
                "fa4_native_topk_kernel": lambda: flash_attn_fwd(
                    q,
                    k,
                    v,
                    score_mod=topk_bitmask_score_mod,
                    aux_tensors=[bitmask],
                    out=outputs["fa4_native_topk"],
                    _arch=90,
                ),
                "fa4_native_topk_bitmask_build_only": lambda: build_topk_bitmask(
                    indices, case.kv_length, assume_unique=True
                ),
                "fa4_native_topk_end_to_end": fa4_native_topk_end_to_end,
            }
        )

    return CaseRuntime(
        case=case,
        plan=plan,
        backend_plans=backend_plans,
        indices=indices,
        cast_indices=cast_indices,
        union_indices=union_indices,
        bitmask=bitmask,
        sdpa_mask=sdpa_mask,
        outputs=outputs,
        calls=calls,
        diagnostics=diagnostics,
    )


def compile_runtime(runtime: CaseRuntime) -> dict[str, str]:
    required = {
        "fa4_dense_full_kernel",
        "indexed_auto_end_to_end",
        "indexed_auto_prepared_kernel",
        "indexed_auto_preparation_only",
    }
    if "fa4_native_topk_kernel" in runtime.calls:
        required.update(
            {
                "fa4_native_topk_kernel",
                "fa4_native_topk_bitmask_build_only",
                "fa4_native_topk_end_to_end",
            }
        )
    errors: dict[str, str] = {}
    for name, fn in list(runtime.calls.items()):
        try:
            fn()
            torch.cuda.synchronize()
        except Exception as exc:
            message = f"compilation/warm call failed for {name}: {exc}"
            if name in required:
                raise RuntimeError(message) from exc
            errors[name] = traceback.format_exc()
            del runtime.calls[name]
    sdpa_names = {
        "torch_sdpa_topk_kernel",
        "torch_sdpa_topk_mask_build_only",
        "torch_sdpa_topk_end_to_end",
    }
    if sdpa_names.intersection(errors):
        for name in sdpa_names:
            runtime.calls.pop(name, None)
    return errors


def _safe_empty_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        # A failed launch may poison only this process's CUDA context.  Preserve
        # the original per-case error instead of replacing it during cleanup.
        print(f"  CUDA cache cleanup skipped after prior failure: {exc}", file=sys.stderr)


def release_runtime(runtime: CaseRuntime | None) -> None:
    if runtime is not None:
        # Q/K/V are captured by the call closures. Clear those first, then drop
        # direct tensor references so a suite never retains the previous shape.
        runtime.calls.clear()
        runtime.outputs.clear()
        runtime.bitmask = None
        runtime.sdpa_mask = None
        runtime.indices = torch.empty(0)
        runtime.cast_indices = torch.empty(0)
        runtime.union_indices = torch.empty(0)
    gc.collect()
    _safe_empty_cache()


def _median(timings: dict[str, dict[str, Any]], name: str) -> float:
    return float(timings[name]["median_ms"])


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _rough_speedup_text(reference_ms: float, candidate_ms: float) -> str:
    """Human-readable median comparison for live benchmark progress."""

    ratio = _safe_ratio(reference_ms, candidate_ms)
    if ratio is None:
        return "n/a"
    if ratio >= 1.0:
        return f"~{ratio:.2f}x faster"
    return f"~{1.0 / ratio:.2f}x slower"


def _geometric_mean(values: Iterable[float]) -> float | None:
    positive = [float(value) for value in values if value is not None and value > 0]
    if not positive:
        return None
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def summarize_exact_speedups(
    results: Iterable[dict[str, Any]],
    *,
    parity_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Summarize same-semantics E2E speedups.

    Native fallback and the exact FA4 baseline execute the same implementation.
    Treat sub-two-percent differences as timing parity rather than a real win/loss.
    """

    rows = []
    for result in results:
        if result.get("status") != "ok":
            continue
        speedup = (result.get("derived") or {}).get(
            "indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e"
        )
        if speedup is None:
            continue
        rows.append(
            {
                "name": result["case"]["name"],
                "category": result["case"]["category"],
                "path": result["plan"]["path"],
                "speedup": float(speedup),
            }
        )

    by_path: dict[str, dict[str, Any]] = {}
    for path in sorted({row["path"] for row in rows}):
        values = [row["speedup"] for row in rows if row["path"] == path]
        by_path[path] = {
            "cases": len(values),
            "geomean_speedup": _geometric_mean(values),
            "min_speedup": min(values),
            "max_speedup": max(values),
            "faster_cases": sum(value > 1.0 + parity_tolerance for value in values),
            "parity_cases": sum(
                1.0 - parity_tolerance <= value <= 1.0 + parity_tolerance
                for value in values
            ),
            "slower_cases": sum(value < 1.0 - parity_tolerance for value in values),
        }

    ordered = sorted(rows, key=lambda row: row["speedup"], reverse=True)
    return {
        "cases": len(rows),
        "parity_tolerance": parity_tolerance,
        "geomean_speedup": _geometric_mean(row["speedup"] for row in rows),
        "faster_cases": sum(row["speedup"] > 1.0 + parity_tolerance for row in rows),
        "parity_cases": sum(
            1.0 - parity_tolerance <= row["speedup"] <= 1.0 + parity_tolerance
            for row in rows
        ),
        "slower_cases": sum(row["speedup"] < 1.0 - parity_tolerance for row in rows),
        "by_path": by_path,
        "top_speedups": ordered[:5],
        "slowest": list(reversed(ordered[-5:])),
    }


def summarize_sdpa_speedups(
    results: Iterable[dict[str, Any]],
    *,
    parity_tolerance: float = 0.02,
) -> dict[str, Any]:
    rows = []
    for result in results:
        if result.get("status") != "ok":
            continue
        speedup = (result.get("derived") or {}).get(
            "indexed_e2e_speedup_vs_torch_sdpa_topk_exact_e2e"
        )
        if speedup is None:
            continue
        rows.append(
            {
                "name": result["case"]["name"],
                "category": result["case"]["category"],
                "path": result["plan"]["path"],
                "speedup": float(speedup),
            }
        )
    ordered = sorted(rows, key=lambda row: row["speedup"], reverse=True)
    return {
        "cases": len(rows),
        "parity_tolerance": parity_tolerance,
        "geomean_speedup": _geometric_mean(row["speedup"] for row in rows),
        "faster_cases": sum(row["speedup"] > 1.0 + parity_tolerance for row in rows),
        "parity_cases": sum(
            1.0 - parity_tolerance <= row["speedup"] <= 1.0 + parity_tolerance
            for row in rows
        ),
        "slower_cases": sum(row["speedup"] < 1.0 - parity_tolerance for row in rows),
        "top_speedups": ordered[:5],
        "slowest": list(reversed(ordered[-5:])),
    }


def print_sdpa_context_summary(summary: dict[str, Any]) -> None:
    if not summary.get("cases"):
        return
    print(
        "PyTorch SDPA context: "
        f"{summary['faster_cases']} materially faster, "
        f"{summary['parity_cases']} within ±{summary['parity_tolerance'] * 100:.0f}% parity, "
        f"{summary['slower_cases']} materially slower; "
        f"geomean ~{summary['geomean_speedup']:.2f}x vs masked torch SDPA.",
        flush=True,
    )


def print_speedup_summary(summary: dict[str, Any]) -> None:
    if not summary.get("cases"):
        return
    geomean = summary.get("geomean_speedup")
    print(
        "Same-semantics rough summary: "
        f"{summary['faster_cases']} materially faster, "
        f"{summary.get('parity_cases', 0)} within "
        f"±{summary.get('parity_tolerance', 0.02) * 100:.0f}% parity, "
        f"{summary['slower_cases']} materially slower; "
        f"geomean ~{geomean:.2f}x vs unmodified FA4 top-k score_mod.",
        flush=True,
    )
    for path, values in summary.get("by_path", {}).items():
        path_geomean = values.get("geomean_speedup")
        print(
            f"  {path}: {values['faster_cases']} faster, "
            f"{values.get('parity_cases', 0)} parity, "
            f"{values.get('slower_cases', 0)} slower; "
            f"geomean ~{path_geomean:.2f}x "
            f"(range {values['min_speedup']:.2f}x-{values['max_speedup']:.2f}x)",
            flush=True,
        )


def run_case(
    case: BenchmarkCase,
    *,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    sm_count: int,
    seed: int,
    include_native: bool,
    include_sdpa: bool,
    sdpa_max_mask_bytes: int,
    compare_forced_backends: bool,
    compare_union: bool,
    prefill_lab: bool,
    block_sparse_lab: bool,
    cute_block_metadata: bool,
    timing: TimingConfig,
    skip_metrics: bool,
    operation_seed: int,
) -> dict[str, Any]:
    runtime: CaseRuntime | None = None
    started = time.perf_counter()
    try:
        runtime = create_runtime(
            case,
            dtype=dtype,
            device=device,
            sm_count=sm_count,
            seed=seed,
            include_native=include_native,
            include_sdpa=include_sdpa,
            sdpa_max_mask_bytes=sdpa_max_mask_bytes,
            compare_forced_backends=compare_forced_backends,
            compare_union=compare_union,
            prefill_lab=prefill_lab,
            block_sparse_lab=block_sparse_lab,
            cute_block_metadata=cute_block_metadata,
        )
        backend_errors = compile_runtime(runtime)

        operation_names = list(runtime.calls)
        random.Random(operation_seed).shuffle(operation_names)
        timings: dict[str, dict[str, Any]] = {}
        for name in operation_names:
            timings[name] = bench_operation(runtime.calls[name], timing)

        runtime.calls["indexed_auto_end_to_end"]()
        runtime.calls["indexed_auto_prepared_kernel"]()
        correctness: dict[str, Any] = {
            "indexed_auto_vs_prepared": _relative_error(
                runtime.outputs["indexed_auto"], runtime.outputs["indexed_prepared"]
            )
        }
        comparisons = {
            "indexed_warp_kernel": ("indexed_warp", "indexed_auto_vs_warp"),
            "indexed_warp_scalar_kernel": (
                "indexed_warp_scalar",
                "indexed_auto_vs_warp_scalar",
            ),
            "indexed_dense_end_to_end": ("indexed_dense", "indexed_auto_vs_dense"),
            "indexed_union_kernel": ("indexed_union", "indexed_auto_vs_union"),
            "fa4_native_topk_kernel": ("fa4_native_topk", "indexed_auto_vs_fa4_native_topk"),
            "indexed_dense_128x64_end_to_end": ("indexed_dense_128x64", "indexed_auto_vs_dense_128x64"),
            "fa4_topk_128x64_end_to_end": ("fa4_topk_128x64", "indexed_auto_vs_fa4_topk_128x64"),
            "fa4_topk_reuse_128x64_end_to_end": ("fa4_topk_reuse_128x64", "indexed_auto_vs_fa4_topk_reuse_128x64"),
            "fa4_topk_cute_128x64_end_to_end": ("fa4_topk_cute_128x64", "indexed_auto_vs_fa4_topk_cute_128x64"),
            "fa4_block_sparse_cute_128x64_end_to_end": (
                "fa4_block_sparse_cute_128x64",
                "indexed_auto_vs_fa4_block_sparse_cute_128x64",
            ),
        }
        for call_name, (output_name, comparison_name) in comparisons.items():
            if call_name in runtime.calls:
                runtime.calls[call_name]()
                correctness[comparison_name] = _relative_error(
                    runtime.outputs[output_name], runtime.outputs["indexed_auto"]
                )
        if "torch_sdpa_topk_kernel" in runtime.calls:
            sdpa_out = runtime.calls["torch_sdpa_topk_kernel"]()
            correctness["indexed_auto_vs_torch_sdpa_topk"] = _relative_error(
                sdpa_out, runtime.outputs["indexed_auto"]
            )

        metrics = None
        if not skip_metrics:
            analyze_index_tiles = load_fa4_modules()["analyze_index_tiles"]
            union_plan = runtime.backend_plans.get("union")
            if union_plan is not None:
                metrics = asdict(
                    analyze_index_tiles(
                        runtime.union_indices,
                        kv_length=case.kv_length,
                        tile_m=union_plan.tile_m,
                        query_heads=case.query_heads,
                        kv_heads=case.kv_heads,
                        pack_gqa=union_plan.pack_gqa,
                    )
                )

        dense_full_ms = _median(timings, "fa4_dense_full_kernel")
        indexed_prepared_ms = _median(timings, "indexed_auto_prepared_kernel")
        indexed_e2e_ms = _median(timings, "indexed_auto_end_to_end")
        indexed_preparation_ms = _median(timings, "indexed_auto_preparation_only")
        dense_flops = (
            2
            * case.batch
            * case.query_length
            * case.query_heads
            * case.kv_length
            * (case.head_dim + case.value_head_dim)
        )
        selected_flops = (
            2
            * case.batch
            * case.query_length
            * case.query_heads
            * case.topk
            * (case.head_dim + case.value_head_dim)
        )
        logical_tokens = case.batch * case.query_length
        derived: dict[str, Any] = {
            # This comparison is intentionally labeled non-equivalent: full
            # dense attention does not apply the selected-set mask.
            "indexed_kernel_speedup_vs_fa4_dense_full_non_equivalent": _safe_ratio(
                dense_full_ms, indexed_prepared_ms
            ),
            "indexed_e2e_speedup_vs_fa4_dense_full_non_equivalent": _safe_ratio(
                dense_full_ms, indexed_e2e_ms
            ),
            "indexed_preparation_fraction_of_e2e": _safe_ratio(
                indexed_preparation_ms, indexed_e2e_ms
            ),
            "selection_density": case.topk / case.kv_length,
            "fa4_dense_full_effective_tflops": dense_flops / (dense_full_ms * 1e9),
            "indexed_useful_topk_tflops": selected_flops / (indexed_prepared_ms * 1e9),
            "fa4_dense_full_tokens_per_second": logical_tokens * 1000.0 / dense_full_ms,
            "indexed_auto_tokens_per_second": logical_tokens * 1000.0 / indexed_e2e_ms,
            "bitmask_bytes": 0
            if runtime.bitmask is None
            else runtime.bitmask.numel() * runtime.bitmask.element_size(),
            "torch_sdpa_mask_bytes": 0
            if runtime.sdpa_mask is None
            else runtime.sdpa_mask.numel() * runtime.sdpa_mask.element_size(),
            "indices_bytes": runtime.indices.numel() * runtime.indices.element_size(),
            "cast_indices_bytes": runtime.cast_indices.numel()
            * runtime.cast_indices.element_size(),
            "union_indices_bytes": runtime.union_indices.numel()
            * runtime.union_indices.element_size(),
        }

        native_topk_ms = None
        if "fa4_native_topk_kernel" in timings:
            native_topk_ms = _median(timings, "fa4_native_topk_kernel")
            native_topk_e2e_ms = _median(timings, "fa4_native_topk_end_to_end")
            derived.update(
                {
                    "indexed_kernel_speedup_vs_fa4_native_topk_exact": _safe_ratio(
                        native_topk_ms, indexed_prepared_ms
                    ),
                    "indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e": _safe_ratio(
                        native_topk_e2e_ms, indexed_e2e_ms
                    ),
                    "fa4_native_topk_slowdown_vs_dense_full": _safe_ratio(
                        native_topk_ms, dense_full_ms
                    ),
                    "fa4_native_topk_e2e_slowdown_vs_dense_full": _safe_ratio(
                        native_topk_e2e_ms, dense_full_ms
                    ),
                    "fa4_native_topk_tokens_per_second": (
                        logical_tokens * 1000.0 / native_topk_e2e_ms
                    ),
                }
            )

        if (
            "torch_sdpa_topk_kernel" in timings
            and "torch_sdpa_topk_end_to_end" in timings
        ):
            sdpa_kernel_ms = _median(timings, "torch_sdpa_topk_kernel")
            sdpa_e2e_ms = _median(timings, "torch_sdpa_topk_end_to_end")
            derived.update(
                {
                    "indexed_kernel_speedup_vs_torch_sdpa_topk_exact": _safe_ratio(
                        sdpa_kernel_ms, indexed_prepared_ms
                    ),
                    "indexed_e2e_speedup_vs_torch_sdpa_topk_exact_e2e": _safe_ratio(
                        sdpa_e2e_ms, indexed_e2e_ms
                    ),
                    "torch_sdpa_topk_slowdown_vs_dense_full": _safe_ratio(
                        sdpa_kernel_ms, dense_full_ms
                    ),
                    "torch_sdpa_topk_e2e_slowdown_vs_dense_full": _safe_ratio(
                        sdpa_e2e_ms, dense_full_ms
                    ),
                    "torch_sdpa_topk_tokens_per_second": (
                        logical_tokens * 1000.0 / sdpa_e2e_ms
                    ),
                }
            )

        for backend_name, kernel_timing_name, e2e_timing_name in (
            ("indexed_warp", "indexed_warp_kernel", "indexed_warp_end_to_end"),
            (
                "indexed_warp_scalar",
                "indexed_warp_scalar_kernel",
                "indexed_warp_scalar_end_to_end",
            ),
            ("indexed_union", "indexed_union_kernel", "indexed_union_end_to_end"),
        ):
            if kernel_timing_name in timings:
                backend_ms = _median(timings, kernel_timing_name)
                derived[
                    f"{backend_name}_speedup_vs_fa4_dense_full_non_equivalent"
                ] = _safe_ratio(dense_full_ms, backend_ms)
                if native_topk_ms is not None:
                    derived[f"{backend_name}_speedup_vs_fa4_native_topk_exact"] = (
                        _safe_ratio(native_topk_ms, backend_ms)
                    )
            if e2e_timing_name in timings:
                backend_e2e_ms = _median(timings, e2e_timing_name)
                derived[f"{backend_name}_end_to_end_ms"] = backend_e2e_ms
                derived[
                    f"{backend_name}_e2e_speedup_vs_fa4_dense_full_non_equivalent"
                ] = _safe_ratio(dense_full_ms, backend_e2e_ms)
                if "fa4_native_topk_end_to_end" in timings:
                    native_topk_e2e_ms = _median(timings, "fa4_native_topk_end_to_end")
                    derived[
                        f"{backend_name}_e2e_speedup_vs_fa4_native_topk_exact_e2e"
                    ] = _safe_ratio(native_topk_e2e_ms, backend_e2e_ms)

        return {
            "status": "ok",
            "case": case.to_dict(),
            "dtype": dtype_name,
            "plan": asdict(runtime.plan),
            "backend_plans": {
                name: asdict(plan) for name, plan in runtime.backend_plans.items()
            },
            "metrics": metrics,
            "timings": timings,
            "correctness": correctness,
            "derived": derived,
            "backend_errors": backend_errors,
            "diagnostics": runtime.diagnostics,
            "wall_time_seconds": time.perf_counter() - started,
        }
    finally:
        release_runtime(runtime)


def _run_command(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=None if cwd is None else str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def collect_environment(device: torch.device, repo_root: Path) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    device_index = torch.cuda.current_device()
    smi_query = _run_command(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=name,driver_version,pstate,clocks.current.sm,"
            "clocks.current.memory,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    git_commit = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_dirty = _run_command(["git", "status", "--porcelain"], cwd=repo_root)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fa4_import": dict(_FA4_IMPORT_INFO or {}),
        "gpu": {
            "index": device_index,
            "name": props.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "sm_count": props.multi_processor_count,
            "total_memory_bytes": props.total_memory,
            "nvidia_smi": smi_query,
        },
        "repository": {
            "root": str(repo_root.resolve()),
            "git_commit": git_commit,
            "git_dirty": bool(git_dirty) if git_dirty is not None else None,
            "git_status_porcelain": git_dirty,
        },
    }


def _json_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    return value


def flatten_result(result: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"status": result.get("status")}
    case = result.get("case", {})
    for key, value in case.items():
        row[key] = _json_cell(value)
    row["dtype"] = result.get("dtype")
    if result.get("error"):
        row["error"] = result["error"]
        return row
    if result.get("backend_errors"):
        row["backend_errors"] = _json_cell(result["backend_errors"])

    plan = result.get("plan") or {}
    for key, value in plan.items():
        row[f"plan_{key}"] = _json_cell(value)
    for backend, backend_plan in (result.get("backend_plans") or {}).items():
        for key, value in backend_plan.items():
            row[f"backend_{backend}_{key}"] = _json_cell(value)
    metrics = result.get("metrics") or {}
    for key, value in metrics.items():
        row[f"metrics_{key}"] = _json_cell(value)
    diagnostics = result.get("diagnostics") or {}
    for key, value in diagnostics.items():
        row[f"diagnostics_{key}"] = _json_cell(value)
    for operation, stats in (result.get("timings") or {}).items():
        for key in ("median_ms", "min_ms", "p90_ms", "mean_ms", "stddev_ms", "iterations_per_round"):
            row[f"{operation}_{key}"] = stats.get(key)
    for comparison, values in (result.get("correctness") or {}).items():
        for key, value in values.items():
            row[f"correctness_{comparison}_{key}"] = value
    for key, value in (result.get("derived") or {}).items():
        row[key] = _json_cell(value)
    row["wall_time_seconds"] = result.get("wall_time_seconds")
    return row


def write_csv(path: Path, results: Iterable[dict[str, Any]]) -> None:
    rows = [flatten_result(result) for result in results]
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _case_line(
    index: int,
    total: int,
    case: BenchmarkCase,
    include_native: bool,
    include_sdpa: bool,
    compare_backends: bool,
) -> str:
    return (
        f"[{index:02d}/{total:02d}] {case.name}: "
        f"B={case.batch} Q={case.query_length} K={case.kv_length} T={case.topk} "
        f"H={case.query_heads}/{case.kv_heads} D={case.head_dim}/{case.value_head_dim} "
        f"pattern={case.pattern} union-hint={case.union_compute_inflation_hint} "
        f"exact-fa4-topk={'yes' if include_native else 'no'} "
        f"torch-sdpa={'yes' if include_sdpa else 'no'} "
        f"compare={'yes' if compare_backends else 'no'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--suite",
        choices=(
            "single", "quick", "iteration", "iteration-medium", "serving-medium",
            "serving-decode", "serving-prefill", "dsa-models", "glm-moe-dsa-45", "glm-moe-dsa-45-overnight", "glm-moe-dsa-45-batch", "glm-moe-dsa-45-serving-matrix", "common-indexed-3h", "common-indexed-5h", "regression", "common", "all", "decode", "batch",
            "prefill", "prefill-focus", "heads", "patterns", "odd", "random", "topk-sweep", "union-candidates",
        ),
        default="single",
        help="benchmark case collection",
    )

    # Single-shape compatibility options.
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--q", type=int, default=128)
    parser.add_argument("--k", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--hq", type=int, default=32)
    parser.add_argument("--hkv", type=int, default=8)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--dv", type=int, default=None)
    parser.add_argument(
        "--pattern",
        choices=("identical", "random", "tail-random", "causal-window", "causal-mixed"),
        default="tail-random",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        help="tail size; default is min(512, max(64, topk/4))",
    )

    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-cases", type=int, default=24)
    parser.add_argument("--random-max-k", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--iters",
        type=int,
        default=0,
        help="fixed iterations per round; zero enables auto-calibration",
    )
    parser.add_argument("--target-round-ms", type=float, default=100.0)
    parser.add_argument("--min-iters", type=int, default=10)
    parser.add_argument("--max-iters", type=int, default=500)
    parser.add_argument("--native", choices=("none", "subset", "all"), default="all", help="run the exact upstream-compatible FA4 top-k score_mod baseline")
    parser.add_argument(
        "--sdpa",
        choices=("none", "subset", "all"),
        default="subset",
        help="run PyTorch SDPA with an exact dense boolean top-k mask",
    )
    parser.add_argument(
        "--sdpa-max-mask-mib",
        type=float,
        default=256.0,
        help="skip SDPA when its broadcast boolean mask would exceed this size",
    )
    parser.add_argument(
        "--compare-backends",
        choices=("none", "subset", "qgt1", "row-dense", "all"),
        default="subset",
        help="time forced controls; row-dense avoids expensive union compilation for wide model suites",
    )
    parser.add_argument(
        "--prefill-lab",
        action="store_true",
        help="benchmark safe GLM D256 prefill tile and bitmask-workspace candidates",
    )
    parser.add_argument(
        "--block-sparse-lab",
        action="store_true",
        help="benchmark exact FA4 block-sparse scheduling as an isolated prefill candidate",
    )
    parser.add_argument(
        "--cute-block-metadata",
        action="store_true",
        help="use persistent CuTe active-block compaction for the exact 128x64 sparse candidate",
    )
    parser.add_argument("--skip-native", action="store_true", help="alias for --native none")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--precompile-all", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--case-regex", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--fail-below-speedup",
        type=float,
        default=None,
        help=(
            "after writing results, exit nonzero if any same-semantics auto case "
            "is below this speedup (for example 0.98 to allow a two-percent timing band)"
        ),
    )

    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--fa4-root",
        type=Path,
        default=None,
        help="FA4 repository root or flash_attn package directory; auto-detected by default",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--print-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    global _FA4_ROOT_OVERRIDE
    args = parse_args()
    _FA4_ROOT_OVERRIDE = args.fa4_root
    if args.skip_native:
        args.native = "none"
    if args.rounds <= 0 or args.warmup < 0:
        raise ValueError("rounds must be positive and warmup must be non-negative")
    if args.iters < 0 or args.min_iters <= 0 or args.max_iters < args.min_iters:
        raise ValueError("invalid iteration configuration")
    if args.hq % args.hkv:
        raise ValueError("hq must be divisible by hkv")
    if args.sdpa_max_mask_mib <= 0:
        raise ValueError("sdpa-max-mask-mib must be positive")
    sdpa_max_mask_bytes = int(args.sdpa_max_mask_mib * 1024 * 1024)

    single = BenchmarkCase(
        name="single",
        category="single",
        batch=args.batch,
        query_length=args.q,
        kv_length=args.k,
        topk=args.topk,
        query_heads=args.hq,
        kv_heads=args.hkv,
        head_dim=args.d,
        value_head_dim=args.d if args.dv is None else args.dv,
        pattern=args.pattern,
        tail=args.tail,
    )
    cases = build_suite(
        args.suite,
        single,
        seed=args.seed,
        random_cases=args.random_cases,
        random_max_k=args.random_max_k,
    )
    if args.case_regex:
        matcher = re.compile(args.case_regex)
        cases = [case for case in cases if matcher.search(case.name)]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if not cases:
        raise ValueError("no benchmark cases remain after filtering")
    for case in cases:
        if case.query_heads % case.kv_heads:
            raise ValueError(f"{case.name}: query heads must be divisible by KV heads")
        if case.topk <= 0 or case.topk > case.kv_length:
            raise ValueError(f"{case.name}: topk must be in [1, kv_length]")

    if args.list_cases or args.dry_run:
        for index, case in enumerate(cases, 1):
            print(
                _case_line(
                    index,
                    len(cases),
                    case,
                    should_run_native(case, args.native, args.suite),
                    should_run_sdpa(case, args.sdpa, args.suite, sdpa_max_mask_bytes),
                    should_compare_backends(case, args.compare_backends),
                )
            )
        return

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("benchmark requires an SM90 GPU")

    # One import preflight prevents a missing dependency from being repeated as
    # a misleading failure for every benchmark case.
    try:
        fa4_import_info = load_fa4_modules()["import_info"]
    except Exception as exc:
        raise RuntimeError(
            "FA4-only import preflight failed before allocating benchmark tensors: "
            f"{exc}"
        ) from exc
    print(
        "FA4 import: "
        f"{fa4_import_info['primary_package_dir']} "
        "(FA2 extension not required)",
        flush=True,
    )

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    props = torch.cuda.get_device_properties(device)
    timing = TimingConfig(
        warmup=args.warmup,
        rounds=args.rounds,
        fixed_iters=args.iters,
        target_round_ms=args.target_round_ms,
        min_iters=args.min_iters,
        max_iters=args.max_iters,
    )
    precompile_all = args.precompile_all if args.precompile_all is not None else args.suite != "single"
    shuffle = args.shuffle if args.shuffle is not None else args.suite != "single"

    failed_precompile: dict[str, str] = {}
    if precompile_all:
        print(f"Precompiling {len(cases)} cases...")
        for index, case in enumerate(cases, 1):
            include_native = should_run_native(case, args.native, args.suite)
            include_sdpa = should_run_sdpa(
                case, args.sdpa, args.suite, sdpa_max_mask_bytes
            )
            compare_backends = should_compare_backends(case, args.compare_backends)
            compare_union = should_compare_union(case, args.compare_backends)
            print(
                _case_line(
                    index, len(cases), case, include_native, include_sdpa, compare_backends
                ),
                flush=True,
            )
            runtime: CaseRuntime | None = None
            try:
                runtime = create_runtime(
                    case,
                    dtype=dtype,
                    device=device,
                    sm_count=props.multi_processor_count,
                    seed=args.seed,
                    include_native=include_native,
                    include_sdpa=include_sdpa,
                    sdpa_max_mask_bytes=sdpa_max_mask_bytes,
                    compare_forced_backends=compare_backends,
                    compare_union=compare_union,
                    prefill_lab=args.prefill_lab,
                    block_sparse_lab=args.block_sparse_lab,
                    cute_block_metadata=args.cute_block_metadata,
                )
                optional_errors = compile_runtime(runtime)
                for operation, error in optional_errors.items():
                    print(
                        f"  OPTIONAL BACKEND FAILED ({operation}): {error}",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                failed_precompile[case.name] = repr(exc)
                print(f"  PRECOMPILE FAILED: {exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    raise
            finally:
                release_runtime(runtime)

    execution_cases = list(cases)
    if shuffle:
        random.Random(args.seed).shuffle(execution_cases)

    print(f"Benchmarking {len(execution_cases)} cases...", flush=True)
    results: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for index, case in enumerate(execution_cases, 1):
        include_native = should_run_native(case, args.native, args.suite)
        include_sdpa = should_run_sdpa(
            case, args.sdpa, args.suite, sdpa_max_mask_bytes
        )
        compare_backends = should_compare_backends(case, args.compare_backends)
        compare_union = should_compare_union(case, args.compare_backends)
        print(
            _case_line(
                index, len(execution_cases), case, include_native, include_sdpa, compare_backends
            ),
            flush=True,
        )
        if case.name in failed_precompile:
            results.append(
                {
                    "status": "precompile-failed",
                    "case": case.to_dict(),
                    "dtype": args.dtype,
                    "error": failed_precompile[case.name],
                }
            )
            continue
        try:
            result = run_case(
                case,
                dtype=dtype,
                dtype_name=args.dtype,
                device=device,
                sm_count=props.multi_processor_count,
                seed=args.seed,
                include_native=include_native,
                include_sdpa=include_sdpa,
                sdpa_max_mask_bytes=sdpa_max_mask_bytes,
                compare_forced_backends=compare_backends,
                compare_union=compare_union,
                prefill_lab=args.prefill_lab,
                block_sparse_lab=args.block_sparse_lab,
                cute_block_metadata=args.cute_block_metadata,
                timing=timing,
                skip_metrics=args.skip_metrics,
                operation_seed=_stable_case_seed(args.seed + 1, case.name),
            )
            results.append(result)
            derived = result["derived"]
            timings = result["timings"]
            auto_ms = timings["indexed_auto_end_to_end"]["median_ms"]
            dense_ms = timings["fa4_dense_full_kernel"]["median_ms"]
            timing_summary = [
                f"auto({result['plan']['path']}) {auto_ms:.4f} ms",
            ]
            if "fa4_native_topk_end_to_end" in timings:
                exact_e2e_ms = timings["fa4_native_topk_end_to_end"]["median_ms"]
                timing_summary.append(
                    f"exact-FA4-topk e2e {exact_e2e_ms:.4f} ms "
                    f"=> {_rough_speedup_text(exact_e2e_ms, auto_ms)}"
                )
            if "torch_sdpa_topk_end_to_end" in timings:
                sdpa_e2e_ms = timings["torch_sdpa_topk_end_to_end"]["median_ms"]
                timing_summary.append(
                    f"torch-SDPA-topk e2e {sdpa_e2e_ms:.4f} ms "
                    f"=> {_rough_speedup_text(sdpa_e2e_ms, auto_ms)}"
                )
            timing_summary.append(
                f"dense-full* {dense_ms:.4f} ms "
                f"=> {_rough_speedup_text(dense_ms, auto_ms)}"
            )
            for label, timing_name in (
                ("grouped-warp e2e", "indexed_warp_end_to_end"),
                ("scalar-warp e2e", "indexed_warp_scalar_end_to_end"),
                ("union e2e", "indexed_union_end_to_end"),
                ("native-bitmask FA4 128x64 e2e", "fa4_topk_cute_128x64_end_to_end"),
                ("block-sparse CuTe 128x64 e2e", "fa4_block_sparse_cute_128x64_end_to_end"),
            ):
                if timing_name not in timings:
                    continue
                backend_ms = timings[timing_name]["median_ms"]
                if "fa4_native_topk_end_to_end" in timings:
                    exact_e2e_ms = timings["fa4_native_topk_end_to_end"]["median_ms"]
                    comparison = _rough_speedup_text(exact_e2e_ms, backend_ms)
                    timing_summary.append(f"{label} {backend_ms:.4f} ms ({comparison})")
                else:
                    timing_summary.append(f"{label} {backend_ms:.4f} ms")
            print("  rough: " + "; ".join(timing_summary), flush=True)
        except Exception as exc:
            result = {
                "status": "failed",
                "case": case.to_dict(),
                "dtype": args.dtype,
                "error": repr(exc),
            }
            results.append(result)
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            _safe_empty_cache()
            if args.fail_fast:
                raise

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"indexed_sm90_{args.suite}_{timestamp}"
    json_path = args.json_out or args.output_dir / f"{run_name}.json"
    csv_path = args.csv_out or args.output_dir / f"{run_name}.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    model_profile = model_profile_for_suite(args.suite)
    payload = {
        "schema_version": 12,
        "suite": args.suite,
        "model_profile": None if model_profile is None else model_profile.to_dict(),
        "configuration": {
            "dtype": args.dtype,
            "seed": args.seed,
            "timing": asdict(timing),
            "exact_fa4_topk_baseline": args.native,
            "torch_sdpa_topk_baseline": args.sdpa,
            "torch_sdpa_max_mask_mib": args.sdpa_max_mask_mib,
            "compare_backends": args.compare_backends,
            "prefill_lab": args.prefill_lab,
            "block_sparse_lab": args.block_sparse_lab,
            "cute_block_metadata": args.cute_block_metadata,
            "skip_metrics": args.skip_metrics,
            "precompile_all": precompile_all,
            "shuffle": shuffle,
            "case_regex": args.case_regex,
            "fa4_root": None if args.fa4_root is None else str(args.fa4_root),
            "fa4_only": True,
            "fail_below_speedup": args.fail_below_speedup,
        },
        "comparison_contract": {
            "fa4_dense_full_kernel": {
                "source_modification_required": False,
                "same_selected_topk_semantics": False,
                "description": "Unmodified FA4 full attention over all KV positions; timing-only throughput reference.",
            },
            "fa4_native_topk_kernel": {
                "source_modification_required": False,
                "same_selected_topk_semantics": True,
                "description": "Unmodified FA4 with exact selected-set membership expressed by an ordinary score_mod bitmask.",
            },
            "torch_sdpa_topk_kernel": {
                "source_modification_required": False,
                "same_selected_topk_semantics": True,
                "description": "PyTorch scaled_dot_product_attention with a dense broadcast boolean mask built from the selected indices.",
            },
            "indexed_auto_end_to_end": {
                "source_modification_required": True,
                "same_selected_topk_semantics": True,
                "description": "Modified SM90 indexed implementation including internal index preparation.",
            },
            "indexed_warp_end_to_end": {
                "source_modification_required": True,
                "same_selected_topk_semantics": True,
                "description": "Forced arbitrary-order grouped sparse-row backend; each warp reuses K/V across GQA heads.",
            },
            "indexed_warp_scalar_end_to_end": {
                "source_modification_required": True,
                "same_selected_topk_semantics": True,
                "description": "One-query-head-per-warp control path used to quantify grouped-head K/V reuse.",
            },
            "indexed_union_end_to_end": {
                "source_modification_required": True,
                "same_selected_topk_semantics": True,
                "description": "Forced WGMMA union backend including internal union-index preparation.",
            },
            "fa4_block_sparse_cute_128x64_end_to_end": {
                "source_modification_required": True,
                "same_selected_topk_semantics": True,
                "description": "Explicit exact 128x64 sparse FA4 path mirroring production GLM prefill dispatch, using persistent CuTe active-block compaction plus the ordinary fine selected-token bitmask.",
            },
        },
        "environment": collect_environment(device, args.repo_root),
        "summary": {
            "requested_cases": len(cases),
            "successful_cases": sum(result.get("status") == "ok" for result in results),
            "failed_cases": sum(result.get("status") != "ok" for result in results),
            "wall_time_seconds": time.perf_counter() - total_started,
            "same_semantics_speedups": summarize_exact_speedups(results),
            "torch_sdpa_context_speedups": summarize_sdpa_speedups(results),
        },
        "results": results,
    }
    text = json.dumps(payload, indent=2, default=str)
    json_path.write_text(text + "\n")
    write_csv(csv_path, results)

    if args.print_json:
        print(text)
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")
    print(
        f"Completed {payload['summary']['successful_cases']}/{payload['summary']['requested_cases']} "
        "cases successfully."
    )
    print_speedup_summary(payload["summary"]["same_semantics_speedups"])
    print_sdpa_context_summary(payload["summary"]["torch_sdpa_context_speedups"])
    print("* dense-full is a non-equivalent throughput reference; exact-FA4-topk and masked torch SDPA have selected-set semantics.")
    if args.fail_below_speedup is not None:
        regressions = []
        for result in results:
            if result.get("status") != "ok":
                continue
            speedup = (result.get("derived") or {}).get(
                "indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e"
            )
            if speedup is not None and speedup < args.fail_below_speedup:
                regressions.append((result["case"]["name"], float(speedup)))
        if regressions:
            print(
                "Material same-semantics regressions: "
                + ", ".join(f"{name}={speedup:.3f}x" for name, speedup in regressions),
                file=sys.stderr,
            )
            raise SystemExit(2)


if __name__ == "__main__":
    main()
