#!/usr/bin/env python3
"""Summarize the final 45-layer GLM MoE DSA attention benchmark."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _geomean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and float(value) > 0]
    if not values:
        return None
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _median_ms(result: dict[str, Any], name: str) -> float | None:
    value = ((result.get("timings") or {}).get(name) or {}).get("median_ms")
    return None if value is None else float(value)


def _ratio(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or candidate <= 0:
        return None
    return reference / candidate


def _fmt(value: float | None, suffix: str = "") -> str:
    return "-" if value is None else f"{value:.3f}{suffix}"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    profile = payload.get("model_profile") or {}
    layers = int(profile.get("num_hidden_layers", 45))
    rows: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        if result.get("status") != "ok":
            continue
        case = result.get("case") or {}
        auto_ms = _median_ms(result, "indexed_auto_end_to_end")
        if auto_ms is None:
            continue
        row_ms = _median_ms(result, "indexed_warp_end_to_end")
        dense_ms = _median_ms(result, "indexed_dense_end_to_end")
        fa4_ms = _median_ms(result, "fa4_native_topk_end_to_end")
        sdpa_ms = _median_ms(result, "torch_sdpa_topk_end_to_end")
        prep_ms = _median_ms(result, "indexed_auto_preparation_only")
        prepared_kernel_ms = _median_ms(result, "indexed_auto_prepared_kernel")
        forced = {name: value for name, value in (("row", row_ms), ("dense", dense_ms)) if value is not None}
        best_backend, best_ms = min(forced.items(), key=lambda item: item[1]) if forced else ("-", auto_ms)
        tokens = int(case.get("batch", 1)) * int(case.get("query_length", 1))
        stack_ms = layers * auto_ms
        prepared_stack_ms = (
            prep_ms + layers * prepared_kernel_ms
            if prep_ms is not None and prepared_kernel_ms is not None
            else None
        )
        rows.append(
            {
                "name": case.get("name", "unknown"),
                "category": case.get("category", "unknown"),
                "path": (result.get("plan") or {}).get("path", "unknown"),
                "batch": case.get("batch"),
                "q": case.get("query_length"),
                "k": case.get("kv_length"),
                "topk": case.get("topk"),
                "pattern": case.get("pattern"),
                "auto_ms": auto_ms,
                "row_ms": row_ms,
                "dense_ms": dense_ms,
                "fa4_ms": fa4_ms,
                "sdpa_ms": sdpa_ms,
                "vs_fa4": _ratio(fa4_ms, auto_ms),
                "vs_sdpa": _ratio(sdpa_ms, auto_ms),
                "best_backend": best_backend,
                "best_ms": best_ms,
                "auto_regret": auto_ms / best_ms if best_ms else None,
                "stack_ms": stack_ms,
                "stack_tokens_per_second": tokens * 1000.0 / stack_ms,
                "prepared_stack_ms": prepared_stack_ms,
                "prepared_stack_speedup": _ratio(stack_ms, prepared_stack_ms),
            }
        )
    return rows


def _group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = rows
    for row in rows:
        groups[row["category"]].append(row)
    summaries = []
    for name, current in groups.items():
        fa4 = [row["vs_fa4"] for row in current if row["vs_fa4"] is not None]
        sdpa = [row["vs_sdpa"] for row in current if row["vs_sdpa"] is not None]
        regrets = [row["auto_regret"] for row in current if row["auto_regret"] is not None]
        summaries.append(
            {
                "name": name,
                "cases": len(current),
                "fa4_geomean": _geomean(fa4),
                "fa4_losses": sum(value < 1.0 for value in fa4),
                "sdpa_geomean": _geomean(sdpa),
                "sdpa_losses": sum(value < 1.0 for value in sdpa),
                "regret_geomean": _geomean(regrets),
                "regret_max": max(regrets) if regrets else None,
            }
        )
    return summaries


def render(payload: dict[str, Any]) -> str:
    profile = payload.get("model_profile") or {}
    rows = _rows(payload)
    groups = _group_summary(rows)
    failed = [result for result in payload.get("results", []) if result.get("status") != "ok"]

    lines = [
        "# GLM MoE DSA 45-layer final benchmark",
        "",
        "## Profile",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    profile_fields = (
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "qk_head_dim",
        "v_head_dim",
        "q_lora_rank",
        "kv_lora_rank",
        "index_topk",
        "index_n_heads",
        "index_head_dim",
        "benchmark_max_context",
        "n_routed_experts",
        "num_experts_per_tok",
    )
    lines.extend(f"| `{field}` | {profile.get(field, '-')} |" for field in profile_fields)
    lines.extend(
        [
            "",
            "The projection is attention-only: it multiplies one attention call by 45 layers. "
            "It does not include Q/K/V projections, the DSA indexer, output projection, MoE, communication, or framework overhead.",
            "",
            "## Target summary",
            "",
            "| Regime | Cases | vs exact FA4 score_mod | Losses | vs torch SDPA | Losses | Auto/best geo | Max regret |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in groups:
        lines.append(
            f"| {group['name']} | {group['cases']} | {_fmt(group['fa4_geomean'], 'x')} | "
            f"{group['fa4_losses']} | {_fmt(group['sdpa_geomean'], 'x')} | {group['sdpa_losses']} | "
            f"{_fmt(group['regret_geomean'], 'x')} | {_fmt(group['regret_max'], 'x')} |"
        )

    lines.extend(
        [
            "",
            "## Per-case results",
            "",
            "| Case | Path | Auto ms | vs FA4 | vs SDPA | Best forced | Regret | 45-layer ms | Attention-only tok/s |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| `{row['name']}` | {row['path']} | {row['auto_ms']:.4f} | {_fmt(row['vs_fa4'], 'x')} | "
            f"{_fmt(row['vs_sdpa'], 'x')} | {row['best_backend']} ({row['best_ms']:.4f} ms) | "
            f"{_fmt(row['auto_regret'], 'x')} | {row['stack_ms']:.2f} | {row['stack_tokens_per_second']:.1f} |"
        )

    baseline_losses = sorted(
        (
            row
            for row in rows
            if (row["vs_fa4"] is not None and row["vs_fa4"] < 1.0)
            or (row["vs_sdpa"] is not None and row["vs_sdpa"] < 1.0)
        ),
        key=lambda row: min(row["vs_fa4"] or 99.0, row["vs_sdpa"] or 99.0),
    )
    dispatch_misses = sorted(
        (row for row in rows if row["auto_regret"] is not None and row["auto_regret"] > 1.03),
        key=lambda row: row["auto_regret"],
        reverse=True,
    )
    preparation_leads = sorted(
        (
            row
            for row in rows
            if row["prepared_stack_speedup"] is not None and row["prepared_stack_speedup"] > 1.03
        ),
        key=lambda row: row["prepared_stack_speedup"],
        reverse=True,
    )

    lines.extend(["", "## Optimization leads", ""])
    if dispatch_misses:
        lines.append("### Dispatch misses above 3%")
        lines.append("")
        for row in dispatch_misses:
            lines.append(
                f"- `{row['name']}`: auto `{row['path']}` is {row['auto_regret']:.3f}x slower than forced {row['best_backend']}."
            )
    else:
        lines.append("- No auto-dispatch miss exceeded 3% among successful cases.")

    if baseline_losses:
        lines.extend(["", "### Cases not yet beating both semantic baselines", ""])
        for row in baseline_losses:
            lines.append(
                f"- `{row['name']}`: vs FA4={_fmt(row['vs_fa4'], 'x')}, vs SDPA={_fmt(row['vs_sdpa'], 'x')}."
            )
    else:
        lines.append("- Every measured case beats both exact FA4 score_mod and masked SDPA.")

    if preparation_leads:
        lines.extend(
            [
                "",
                "### Reuse lower bound",
                "",
                "These rows would gain more than 3% if one prepared selected-set representation could be reused across all 45 layers. "
                "Treat this only as an architectural lead; layer-specific DSA indices invalidate that assumption.",
                "",
            ]
        )
        for row in preparation_leads[:10]:
            lines.append(
                f"- `{row['name']}`: hypothetical 45-layer prepared-state speedup {row['prepared_stack_speedup']:.3f}x."
            )

    if failed:
        lines.extend(["", "## Failures", ""])
        for result in failed:
            case = result.get("case") or {}
            lines.append(f"- `{case.get('name', 'unknown')}`: {result.get('error') or result.get('status')}")

    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "Keep a new optimization only when it improves the relevant decode or prefill geomean by at least 3%, "
            "keeps worst-case regression within 2%, passes correctness, and improves end-to-end latency rather than only prepared-kernel latency.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.json.read_text())
    if payload.get("suite") != "glm-moe-dsa-45":
        raise ValueError("expected a glm-moe-dsa-45 benchmark payload")
    text = render(payload)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")
        print(args.output.resolve())


if __name__ == "__main__":
    main()
