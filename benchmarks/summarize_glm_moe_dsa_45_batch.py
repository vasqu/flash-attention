#!/usr/bin/env python3
"""Summarize GLM DSA true-prefill scaling from batch 1 through batch 16."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _median(result: dict[str, Any], operation: str) -> float | None:
    value = ((result.get("timings") or {}).get(operation) or {}).get("median_ms")
    return None if value is None else float(value)


def _ratio(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or candidate <= 0:
        return None
    return reference / candidate


def _geomean(values: Iterable[float | None]) -> float | None:
    values = [float(value) for value in values if value is not None and value > 0]
    if not values:
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


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
        short = _median(result, "fa4_topk_cute_128x64_end_to_end")
        sparse = _median(result, "fa4_block_sparse_cute_128x64_end_to_end")
        direct_pack = _median(result, "bitmask_build_cute_only")
        int32_pack = _median(result, "bitmask_build_cute_int32_only")
        index_cast = _median(result, "indexed_index_cast_only")
        diagnostics = result.get("diagnostics") or {}
        candidates = {
            "auto": auto,
            "fa4_direct": short,
            "sparse": sparse,
        }
        available = [(name, value) for name, value in candidates.items() if value is not None]
        best_name, best_ms = min(available, key=lambda item: item[1])
        rows.append({
            "name": case.get("name", "unknown"),
            "batch": int(case.get("batch", 0)),
            "q": int(case.get("query_length", 0)),
            "topk": int(case.get("topk", 0)),
            "pattern": case.get("pattern", "unknown"),
            "auto": auto,
            "fa4": fa4,
            "short": short,
            "sparse": sparse,
            "best_name": best_name,
            "best_ms": best_ms,
            "best_vs_auto": _ratio(auto, best_ms),
            "auto_vs_fa4": _ratio(fa4, auto),
            "best_vs_fa4": _ratio(fa4, best_ms),
            "direct_pack": direct_pack,
            "int32_pack": int32_pack,
            "index_cast": index_cast,
            "cast_plus_pack": (
                index_cast + int32_pack
                if index_cast is not None and int32_pack is not None
                else None
            ),
            "workspace_bytes": diagnostics.get("block_sparse_cute_workspace_bytes"),
            "active_fraction": diagnostics.get(
                "block_sparse_cute_128x64_active_fraction"
            ),
        })
    return rows


def render(payload: dict[str, Any]) -> str:
    rows = _rows(payload)
    failed = [r for r in payload.get("results", []) if r.get("status") != "ok"]
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["q"], row["topk"], row["pattern"])].append(row)

    lines = [
        "# GLM MoE DSA batch-scaling BF16 report",
        "",
        "This suite keeps production auto conservative and compares it with the explicit "
        "native-bitmask FA4 and exact CuTe block-sparse paths at B=1/2/4/8/16.",
        "",
        "## Batch scaling",
        "",
        "| Shape | B | Auto ms | Auto samples/s | Auto scaling | Best path | Best ms | Best samples/s | Best scaling | vs auto | vs FA4 | Workspace MiB |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(groups):
        current = sorted(groups[key], key=lambda item: item["batch"])
        base_auto = current[0]["auto"]
        base_best = current[0]["best_ms"]
        for row in current:
            batch = row["batch"]
            auto_throughput = batch * 1000.0 / row["auto"]
            best_throughput = batch * 1000.0 / row["best_ms"]
            auto_scaling = batch * base_auto / row["auto"]
            best_scaling = batch * base_best / row["best_ms"]
            pattern = "window" if row["pattern"] == "causal-window" else "mixed"
            shape = f"S{row['q']}/T{row['topk']} {pattern}"
            workspace_mib = (
                float(row["workspace_bytes"]) / (1024.0 * 1024.0)
                if row["workspace_bytes"] is not None
                else None
            )
            lines.append(
                f"| {shape} | {batch} | {row['auto']:.4f} | {auto_throughput:.1f} | "
                f"{auto_scaling:.3f}x | `{row['best_name']}` | {row['best_ms']:.4f} | "
                f"{best_throughput:.1f} | {best_scaling:.3f}x | "
                f"{_fmt(row['best_vs_auto'], 'x')} | {_fmt(row['best_vs_fa4'], 'x')} | "
                f"{_fmt(workspace_mib)} |"
            )

    lines.extend([
        "",
        "## Public-dtype packing and avoided cast",
        "",
        "| Shape | B | Direct pack ms | int32 pack ms | Separate cast ms | Cast+pack ms | Direct speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(rows, key=lambda item: (item["q"], item["topk"], item["pattern"], item["batch"])):
        direct_speedup = _ratio(row["cast_plus_pack"], row["direct_pack"])
        pattern = "window" if row["pattern"] == "causal-window" else "mixed"
        lines.append(
            f"| S{row['q']}/T{row['topk']} {pattern} | {row['batch']} | "
            f"{_fmt(row['direct_pack'])} | {_fmt(row['int32_pack'])} | "
            f"{_fmt(row['index_cast'])} | {_fmt(row['cast_plus_pack'])} | "
            f"{_fmt(direct_speedup, 'x')} |"
        )

    for batch in (1, 2, 4, 8, 16):
        current = [row for row in rows if row["batch"] == batch]
        if not current:
            continue
        lines.extend([
            "",
            f"## Batch {batch} summary",
            "",
            f"- Cases: {len(current)}",
            f"- Production auto vs exact FA4 score_mod: {_fmt(_geomean(row['auto_vs_fa4'] for row in current), 'x')}",
            f"- Best measured explicit path vs production auto: {_fmt(_geomean(row['best_vs_auto'] for row in current), 'x')}",
            f"- Worst best-path result vs auto: {_fmt(min(row['best_vs_auto'] for row in current if row['best_vs_auto'] is not None), 'x')}",
        ])

    lines.extend([
        "",
        "## Promotion guidance",
        "",
        "- Extend short native-bitmask FA4 auto-dispatch only for batch sizes whose full shape family improves by at least 3% with no result below 0.98x.",
        "- Extend long exact sparse auto-dispatch independently; B=4 and B=8 should not be inferred from B=2 without this matrix.",
        "- Native direct packing should beat `cast + int32 pack`; otherwise keep the public-dtype support for simplicity but do not attribute an end-to-end gain to it.",
        "- The workspace cache is byte bounded by `FLASH_ATTN_INDEXED_WORKSPACE_CACHE_MIB`; a single active workspace may exceed the limit to avoid immediate allocation thrash.",
    ])
    if failed:
        lines.extend(["", "## Failures", ""])
        for result in failed:
            name = (result.get("case") or {}).get("name", "unknown")
            lines.append(f"- `{name}`: {result.get('error', 'unknown error')}")
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
