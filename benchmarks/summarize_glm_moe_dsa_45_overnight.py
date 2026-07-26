#!/usr/bin/env python3
"""Summarize the BF16 GLM MoE DSA prefill and CuTe sparse-metadata lab."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


CANDIDATES = {
    "dense_128x64": "indexed_dense_128x64_end_to_end",
    "fa4_128x64": "fa4_topk_128x64_end_to_end",
    "fa4_reuse_128x64": "fa4_topk_reuse_128x64_end_to_end",
    "fa4_cute_bitmask_128x64": "fa4_topk_cute_128x64_end_to_end",
    "block_sparse_cute_128x64": "fa4_block_sparse_cute_128x64_end_to_end",
}


def _median(result: dict[str, Any], operation: str) -> float | None:
    value = ((result.get("timings") or {}).get(operation) or {}).get("median_ms")
    return None if value is None else float(value)


def _ratio(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or candidate <= 0:
        return None
    return reference / candidate


def _geomean(values: Iterable[float | None]) -> float | None:
    positive = [float(value) for value in values if value is not None and value > 0]
    if not positive:
        return None
    return math.exp(statistics.fmean(math.log(value) for value in positive))


def _fmt(value: float | None, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.3f}{suffix}"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        if result.get("status") != "ok":
            continue
        case = result.get("case") or {}
        auto = _median(result, "indexed_auto_end_to_end")
        if auto is None:
            continue
        fa4 = _median(result, "fa4_native_topk_end_to_end")
        sdpa = _median(result, "torch_sdpa_topk_end_to_end")
        normal_build = _median(result, "fa4_native_topk_bitmask_build_only")
        reuse_build = _median(result, "bitmask_build_reuse_only")
        cute_build = _median(result, "bitmask_build_cute_only")
        cute_int32_build = _median(result, "bitmask_build_cute_int32_only")
        index_cast = _median(result, "indexed_index_cast_only")
        sparse_e2e = _median(result, "fa4_block_sparse_cute_128x64_end_to_end")
        sparse_prepared = _median(result, "fa4_block_sparse_cute_128x64_prepared_kernel")
        sparse_metadata = _median(result, "fa4_block_sparse_cute_128x64_metadata_only")
        metadata_budget = (
            auto - sparse_prepared
            if sparse_prepared is not None and auto > sparse_prepared
            else None
        )
        diagnostics = result.get("diagnostics") or {}
        row = {
            "name": case.get("name", "unknown"),
            "category": case.get("category", "unknown"),
            "pattern": case.get("pattern", "unknown"),
            "batch": int(case.get("batch", 0)),
            "q": int(case.get("query_length", 0)),
            "k": int(case.get("kv_length", 0)),
            "topk": int(case.get("topk", 0)),
            "density": (
                float(case.get("topk", 0)) / float(case.get("kv_length", 1))
                if int(case.get("kv_length", 0)) > 0 else None
            ),
            "auto": auto,
            "fa4": fa4,
            "sdpa": sdpa,
            "vs_fa4": _ratio(fa4, auto),
            "vs_sdpa": _ratio(sdpa, auto),
            "normal_build": normal_build,
            "reuse_build": reuse_build,
            "build_reuse_speedup": _ratio(normal_build, reuse_build),
            "cute_build": cute_build,
            "build_cute_speedup": _ratio(normal_build, cute_build),
            "cute_int32_build": cute_int32_build,
            "index_cast": index_cast,
            "direct_vs_cast_then_pack": _ratio(
                (index_cast + cute_int32_build)
                if index_cast is not None and cute_int32_build is not None
                else None,
                cute_build,
            ),
            "sparse_e2e": sparse_e2e,
            "sparse_prepared": sparse_prepared,
            "sparse_metadata": sparse_metadata,
            "metadata_budget": metadata_budget,
            "metadata_budget_ratio": _ratio(metadata_budget, sparse_metadata),
            "metadata_share": (
                sparse_metadata / sparse_e2e
                if sparse_metadata is not None and sparse_e2e is not None and sparse_e2e > 0
                else None
            ),
            "block_density": diagnostics.get(
                "block_sparse_cute_128x64_active_fraction"
            ),
            "candidates": {},
        }
        for label, operation in CANDIDATES.items():
            candidate = _median(result, operation)
            row["candidates"][label] = {
                "ms": candidate,
                "vs_auto": _ratio(auto, candidate),
                "vs_fa4": _ratio(fa4, candidate),
            }
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    values = [row["candidates"][label]["vs_auto"] for row in rows]
    values = [value for value in values if value is not None]
    return {
        "samples": len(values),
        "geo": _geomean(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "wins": sum(value > 1.03 for value in values),
        "losses": sum(value < 0.98 for value in values),
    }


def _decision(summary: dict[str, Any]) -> str:
    if not summary["samples"]:
        return "NO_DATA"
    if summary["geo"] is not None and summary["geo"] >= 1.03 and summary["minimum"] >= 0.98:
        return "PROMOTE"
    if summary["wins"] >= 3 and summary["losses"] == 0:
        return "SPECIALIZE"
    if summary["losses"]:
        return "KEEP_EXPLICIT_ONLY"
    return "NO_MATERIAL_GAIN"


def render(payload: dict[str, Any]) -> str:
    rows = _rows(payload)
    prefill = [row for row in rows if "prefill" in row["category"]]
    decode = [row for row in rows if "decode" in row["category"]]
    failed = [result for result in payload.get("results", []) if result.get("status") != "ok"]

    lines = [
        "# GLM MoE DSA 45-layer BF16 overnight report",
        "",
        "Production auto uses native CuTe bitmask packing for both measured GLM prefill "
        "regimes: exact FA4 128x64 for Q=K 512/1024 and exact block-sparse FA4 128x64 "
        "for Q=K>=2048. Generic PyTorch bitmask construction remains a benchmark reference.",
        "",
        "## Current production path",
        "",
        "| Regime | Cases | vs exact FA4 score_mod | vs torch SDPA |",
        "|---|---:|---:|---:|",
    ]
    for name, current in (("decode", decode), ("prefill", prefill), ("overall", rows)):
        lines.append(
            f"| {name} | {len(current)} | {_fmt(_geomean(row['vs_fa4'] for row in current), 'x')} | "
            f"{_fmt(_geomean(row['vs_sdpa'] for row in current), 'x')} |"
        )

    lines.extend([
        "",
        "## Prefill candidates versus production auto",
        "",
        "| Decision | Candidate | Samples | Geo | Min | Max | >3% wins | >2% losses |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label in CANDIDATES:
        summary = _summary(prefill, label)
        lines.append(
            f"| {_decision(summary)} | `{label}` | {summary['samples']} | {_fmt(summary['geo'], 'x')} | "
            f"{_fmt(summary['minimum'], 'x')} | {_fmt(summary['maximum'], 'x')} | "
            f"{summary['wins']} | {summary['losses']} |"
        )

    for pattern in ("causal-window", "causal-mixed"):
        current = [row for row in prefill if row["pattern"] == pattern]
        lines.extend([
            "",
            f"### {pattern}",
            "",
            "| Candidate | Geo vs auto | Min | Wins | Losses |",
            "|---|---:|---:|---:|---:|",
        ])
        for label in CANDIDATES:
            summary = _summary(current, label)
            lines.append(
                f"| `{label}` | {_fmt(summary['geo'], 'x')} | {_fmt(summary['minimum'], 'x')} | "
                f"{summary['wins']} | {summary['losses']} |"
            )

    sparse_rows = [row for row in prefill if row["block_density"] is not None]
    lines.extend([
        "",
        "## CuTe exact block-sparse diagnostics",
        "",
        "| Case | Pattern | Active blocks | Metadata ms | Prepared kernel ms | End-to-end ms | Metadata share | Metadata budget ms | Budget / cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(sparse_rows, key=lambda item: (item["q"], item["topk"], item["pattern"])):
        lines.append(
            f"| `{row['name']}` | {row['pattern']} | "
            f"{_fmt(None if row['block_density'] is None else row['block_density'] * 100.0, '%')} | "
            f"{_fmt(row['sparse_metadata'])} | {_fmt(row['sparse_prepared'])} | "
            f"{_fmt(row['sparse_e2e'])} | "
            f"{_fmt(None if row['metadata_share'] is None else row['metadata_share'] * 100.0, '%')} | "
            f"{_fmt(row['metadata_budget'])} | {_fmt(row['metadata_budget_ratio'], 'x')} |"
        )

    lines.extend([
        "",
        "## Bitmask preparation",
        "",
        "| Case | Q/K | Top-k density | Normal build ms | Reused-buffer build ms | Native direct ms | Native int32 ms | Avoided cast ms | Direct vs cast+pack |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted((row for row in prefill if row["normal_build"] is not None), key=lambda item: item["q"]):
        lines.append(
            f"| `{row['name']}` | {row['q']} | {_fmt(None if row['density'] is None else row['density'] * 100.0, '%')} | "
            f"{_fmt(row['normal_build'])} | {_fmt(row['reuse_build'])} | "
            f"{_fmt(row['cute_build'])} | {_fmt(row['cute_int32_build'])} | "
            f"{_fmt(row['index_cast'])} | {_fmt(row['direct_vs_cast_then_pack'], 'x')} |"
        )

    lines.extend([
        "",
        "## Largest measured prefill opportunities",
        "",
        "| Case | Pattern | Density | Active blocks | Auto ms | Best candidate | Candidate ms | vs auto | vs FA4 |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ])
    opportunities = []
    for row in prefill:
        available = [
            (label, values["ms"], values["vs_auto"], values["vs_fa4"])
            for label, values in row["candidates"].items()
            if values["ms"] is not None
        ]
        if not available:
            continue
        best = min(available, key=lambda item: item[1])
        opportunities.append((best[2] or 0.0, row, best))
    for _, row, best in sorted(opportunities, key=lambda item: item[0], reverse=True)[:24]:
        label, ms, speedup, vs_fa4 = best
        lines.append(
            f"| `{row['name']}` | {row['pattern']} | "
            f"{_fmt(None if row['density'] is None else row['density'] * 100.0, '%')} | "
            f"{_fmt(None if row['block_density'] is None else row['block_density'] * 100.0, '%')} | "
            f"{row['auto']:.4f} | `{label}` | {ms:.4f} | {_fmt(speedup, 'x')} | {_fmt(vs_fa4, 'x')} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `fa4_cute_bitmask_128x64` isolates the one-launch native bitmask packer on the short exact-FA4 path.",
        "- Native direct packing accepts the public int64/int32/uint16 indices and avoids a separate full-tensor cast; the bitmask table reports that saving explicitly.",
        "- `block_sparse_cute_128x64` uses the same native bitmask packer before the existing CuTe active-block compactor.",
        "- Metadata budget is `auto latency - prepared sparse latency`; budget/cost above 1.0 means metadata is cheap enough for sparse execution to beat auto in that case.",
        "- A prepared-kernel win without an end-to-end win still points to metadata or launch overhead, not a sparse-attention failure.",
        "- Promotion should use both sequence length and measured active-block fraction; random/mixed selections can activate almost every K/V block.",
        "- Production auto uses exact native-bitmask FA4 for GLM Q=K 512/1024 at batch 1, and exact sparse prefill for Q=K>=2048 at batch <=2; unmeasured shapes retain the prior indexed policy.",
        "",
        "## Decision rule",
        "",
        "Promote only when BF16 geomean improves by at least 3%, the worst measured regression is no more than 2%, correctness passes, and end-to-end latency beats exact FA4 score_mod—not only the prepared kernel.",
    ])
    if failed:
        lines.extend(["", "## Failures", ""])
        for result in failed:
            case = (result.get("case") or {}).get("name", "unknown")
            lines.append(f"- `{case}`: {result.get('error', 'unknown error')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.json.read_text())
    args.output.write_text(render(payload))
    print(args.output)


if __name__ == "__main__":
    main()
