#!/usr/bin/env python3
"""Summarize the GLM-first decode/prefill/batch serving matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
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
    valid = [float(v) for v in values if v is not None and v > 0]
    if not valid:
        return None
    return math.exp(statistics.fmean(math.log(v) for v in valid))


def _fmt(value: float | None, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.3f}{suffix}"


def _regime(category: str) -> str:
    if category == "glm45m-decode":
        return "GLM decode"
    if category == "glm45m-chunk":
        return "GLM chunked/speculative"
    if category == "glm45b-prefill":
        return "GLM true prefill"
    if category.startswith("guard-"):
        return "General guardrails"
    return category


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
        direct = _median(result, "fa4_topk_cute_128x64_end_to_end")
        sparse = _median(result, "fa4_block_sparse_cute_128x64_end_to_end")
        candidates = [("auto", auto)]
        if direct is not None:
            candidates.append(("fa4_direct", direct))
        if sparse is not None:
            candidates.append(("sparse", sparse))
        best_name, best_ms = min(candidates, key=lambda item: item[1])
        plan = result.get("plan") or {}
        rows.append({
            "name": case.get("name", "unknown"),
            "category": case.get("category", "unknown"),
            "regime": _regime(case.get("category", "unknown")),
            "batch": int(case.get("batch", 0)),
            "q": int(case.get("query_length", 0)),
            "k": int(case.get("kv_length", 0)),
            "topk": int(case.get("topk", 0)),
            "heads": f"{case.get('query_heads')}/{case.get('kv_heads')}",
            "dims": f"{case.get('head_dim')}/{case.get('value_head_dim')}",
            "path": str(plan.get("path", "unknown")),
            "auto": auto,
            "fa4": fa4,
            "sdpa": sdpa,
            "best_name": best_name,
            "best_ms": best_ms,
            "auto_vs_fa4": _ratio(fa4, auto),
            "auto_vs_sdpa": _ratio(sdpa, auto),
            "best_vs_auto": _ratio(auto, best_ms),
        })
    return rows


def render(payload: dict[str, Any]) -> str:
    rows = _rows(payload)
    failed = [r for r in payload.get("results", []) if r.get("status") != "ok"]
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_regime[row["regime"]].append(row)

    lines = [
        "# GLM MoE DSA unified serving-matrix BF16 report",
        "",
        "This matrix is GLM-first, but includes DeepSeek, ordinary MHA, GQA, and MQA guardrails so model-specific dispatch cannot silently regress general indexed attention.",
        "",
        "## Production summary",
        "",
        "| Regime | Cases | vs exact FA4 | vs torch SDPA | Best explicit vs auto | Worst best-path |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = ("GLM decode", "GLM chunked/speculative", "GLM true prefill", "General guardrails")
    for regime in order:
        current = by_regime.get(regime, [])
        if not current:
            continue
        best_values = [r["best_vs_auto"] for r in current]
        lines.append(
            f"| {regime} | {len(current)} | {_fmt(_geomean(r['auto_vs_fa4'] for r in current), 'x')} | "
            f"{_fmt(_geomean(r['auto_vs_sdpa'] for r in current), 'x')} | "
            f"{_fmt(_geomean(best_values), 'x')} | {_fmt(min(v for v in best_values if v is not None), 'x')} |"
        )

    lines.extend([
        "",
        "## GLM dispatch by batch",
        "",
        "| B | Cases | Auto paths | vs exact FA4 | Best explicit vs auto | Worst best-path |",
        "|---:|---:|---|---:|---:|---:|",
    ])
    glm_rows = [r for r in rows if r["regime"].startswith("GLM")]
    for batch in sorted({r["batch"] for r in glm_rows}):
        current = [r for r in glm_rows if r["batch"] == batch]
        paths = Counter(r["path"] for r in current)
        path_text = ", ".join(f"{name}:{count}" for name, count in sorted(paths.items()))
        best_values = [r["best_vs_auto"] for r in current]
        lines.append(
            f"| {batch} | {len(current)} | `{path_text}` | "
            f"{_fmt(_geomean(r['auto_vs_fa4'] for r in current), 'x')} | "
            f"{_fmt(_geomean(best_values), 'x')} | {_fmt(min(v for v in best_values if v is not None), 'x')} |"
        )

    lines.extend([
        "",
        "## Remaining opportunities",
        "",
        "| Case | Regime | B/Q/K/T | H/D | Auto path | Auto ms | Best | Best ms | vs auto |",
        "|---|---|---|---|---|---:|---|---:|---:|",
    ])
    opportunities = sorted(rows, key=lambda r: r["best_vs_auto"] or 0, reverse=True)[:20]
    for row in opportunities:
        lines.append(
            f"| `{row['name']}` | {row['regime']} | {row['batch']}/{row['q']}/{row['k']}/{row['topk']} | "
            f"{row['heads']} D{row['dims']} | `{row['path']}` | {row['auto']:.4f} | "
            f"`{row['best_name']}` | {row['best_ms']:.4f} | {_fmt(row['best_vs_auto'], 'x')} |"
        )

    guards = by_regime.get("General guardrails", [])
    if guards:
        lines.extend([
            "",
            "## General-shape guardrails",
            "",
            "| Case | B/Q/K/T | H/D | Auto path | vs exact FA4 | vs SDPA |",
            "|---|---|---|---|---:|---:|",
        ])
        for row in guards:
            lines.append(
                f"| `{row['name']}` | {row['batch']}/{row['q']}/{row['k']}/{row['topk']} | "
                f"{row['heads']} D{row['dims']} | `{row['path']}` | "
                f"{_fmt(row['auto_vs_fa4'], 'x')} | {_fmt(row['auto_vs_sdpa'], 'x')} |"
            )

    lines.extend([
        "",
        "## Promotion checks",
        "",
        "- GLM S=512 uses native-bitmask FA4 at B1/B2 and exact sparse FA4 at B4/B8/B16.",
        "- GLM S=1024 uses native-bitmask FA4 at B1 and exact sparse FA4 at B2/B4/B8/B16.",
        "- GLM S>=2048 uses exact sparse FA4 through measured B8; unmeasured long B16 remains on the general policy.",
        "- Decode and non-GLM profiles retain their prior row/union/dense policy.",
        "- Promote further only when geomean improves by at least 3%, the minimum is at least 0.98x, and correctness passes.",
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
