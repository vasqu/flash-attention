#!/usr/bin/env python3
"""Summarize the multi-hour cross-profile indexed-attention qualification run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


AUTO = "indexed_auto_end_to_end"
FA4 = "fa4_native_topk_end_to_end"
SDPA = "torch_sdpa_topk_end_to_end"
CANDIDATES = (
    "indexed_warp_end_to_end",
    "indexed_warp_scalar_end_to_end",
    "indexed_dense_end_to_end",
    "indexed_union_end_to_end",
    "fa4_topk_cute_128x64_end_to_end",
    "fa4_block_sparse_cute_128x64_end_to_end",
)
PREFILL_CANDIDATES = (
    (
        "native bitmask",
        "fa4_topk_cute_128x64_end_to_end",
        "fa4_bitmask_indexed",
        "indexed_auto_vs_fa4_topk_cute_128x64",
    ),
    (
        "block sparse",
        "fa4_block_sparse_cute_128x64_end_to_end",
        "block_sparse_indexed",
        "indexed_auto_vs_fa4_block_sparse_cute_128x64",
    ),
)
CORRECTNESS_MAX_ABS = 0.05
CORRECTNESS_MEAN_ABS = 0.01


def _median(row: dict[str, Any], op: str) -> float | None:
    value = ((row.get("timings") or {}).get(op) or {}).get("median_ms")
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


def _fmt(value: float | None, suffix: str = "x") -> str:
    return "-" if value is None else f"{value:.3f}{suffix}"


def _parts(row: dict[str, Any]) -> tuple[str, str]:
    category = str((row.get("case") or {}).get("category", "unknown"))
    pieces = category.split("-")
    if len(pieces) >= 3 and pieces[0] in ("common3h", "common5h"):
        return pieces[1], pieces[2]
    return "unknown", category


def _best(row: dict[str, Any]) -> tuple[str, float | None]:
    auto = _median(row, AUTO)
    options = [(AUTO, auto)]
    options.extend((op, _median(row, op)) for op in CANDIDATES)
    valid = [(name, ms) for name, ms in options if ms is not None and ms > 0]
    if not valid:
        return "-", None
    return min(valid, key=lambda item: item[1])


def _prefill_certificates(
    rows: list[dict[str, Any]],
) -> list[
    tuple[str, int, str, str, int, float | None, float | None, float | None, str]
]:
    """Return per-profile/sequence candidate certificates.

    A family qualifies only when every row has timing and correctness data,
    geomean improvement is at least 3%, and no row falls below 0.98x.
    Existing production paths are labeled separately so direct-wrapper
    differences are not mistaken for a dispatch promotion.
    """

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        profile, regime = _parts(row)
        case = row.get("case") or {}
        query_length = int(case.get("query_length", 0) or 0)
        batch = int(case.get("batch", 0) or 0)
        if (
            regime == "prefill"
            and query_length == int(case.get("kv_length", -1) or -1)
        ):
            batch_band = "B1-2" if batch <= 2 else "B4-8" if batch <= 8 else "B16+"
            grouped[(profile, query_length, batch_band)].append(row)

    certificates = []
    for (profile, query_length, batch_band), group in sorted(grouped.items()):
        for label, operation, production_path, comparison in PREFILL_CANDIDATES:
            ratios = [
                _ratio(_median(row, AUTO), _median(row, operation))
                for row in group
            ]
            valid = [ratio for ratio in ratios if ratio is not None]
            errors = [
                (row.get("correctness") or {}).get(comparison)
                for row in group
            ]
            error_rows = [error for error in errors if isinstance(error, dict)]
            correctness_complete = len(error_rows) == len(group)
            correctness_ok = correctness_complete and all(
                float(error.get("max_abs", math.inf)) <= CORRECTNESS_MAX_ABS
                and float(error.get("mean_abs", math.inf)) <= CORRECTNESS_MEAN_ABS
                for error in error_rows
            )
            worst_abs = (
                max(float(error.get("max_abs", math.inf)) for error in error_rows)
                if error_rows
                else None
            )
            production = all(
                str((row.get("plan") or {}).get("path")) == production_path
                for row in group
            )
            geo = _geomean(valid)
            minimum = min(valid) if valid else None
            if production:
                decision = "PRODUCTION"
            elif len(valid) != len(group) or not correctness_complete:
                decision = "INCOMPLETE"
            elif not correctness_ok:
                decision = "CORRECTNESS_FAIL"
            elif geo is not None and geo >= 1.03 and minimum is not None and minimum >= 0.98:
                decision = "QUALIFIES"
            elif minimum is not None and minimum < 0.98:
                decision = "REGRESSES"
            else:
                decision = "NO_GAIN"
            certificates.append(
                (
                    profile,
                    query_length,
                    batch_band,
                    label,
                    len(valid),
                    geo,
                    minimum,
                    worst_abs,
                    decision,
                )
            )
    return certificates


def summarize(payload: dict[str, Any]) -> str:
    rows = [r for r in payload.get("results", []) if not r.get("error")]
    expanded = payload.get("suite") == "common-indexed-5h"
    lines = [
        (
            "# Common indexed-attention 5h+ BF16 report"
            if expanded
            else "# Common indexed-attention 3h+ BF16 report"
        ),
        "",
        (
            "GLM remains the primary D256 target. Ratio-8 GQA now joins the "
            "qualified DeepSeek, MHA, ratio-4 GQA, MQA, and D64 MHA exact-FA4 "
            "prefill paths. Ratio-2 GQA, ratio-7 GQA, D96 MHA, and paired "
            "high-batch locality crossovers remain explicit qualification targets."
            if expanded
            else
            "GLM remains the primary D256 target. Qualified DeepSeek, ordinary "
            "MHA, ratio-4 and ratio-8 GQA, MQA, and D64 MHA prefill families "
            "use exact native-bitmask/block-sparse production paths."
        ),
        "",
        "## Production by profile and regime",
        "",
        "| Profile | Regime | Cases | Auto paths | vs exact FA4 | vs SDPA | Best explicit vs auto | Worst best-path |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_parts(row)].append(row)
    for (profile, regime), group in sorted(grouped.items()):
        paths = Counter(str((r.get("plan") or {}).get("path", "unknown")) for r in group)
        path_text = ", ".join(f"{k}:{v}" for k, v in sorted(paths.items()))
        auto_vs_fa4 = _geomean(_ratio(_median(r, FA4), _median(r, AUTO)) for r in group)
        auto_vs_sdpa = _geomean(_ratio(_median(r, SDPA), _median(r, AUTO)) for r in group)
        best_ratios = []
        for r in group:
            _, best_ms = _best(r)
            best_ratios.append(_ratio(_median(r, AUTO), best_ms))
        valid_best = [x for x in best_ratios if x is not None]
        lines.append(
            f"| `{profile}` | {regime} | {len(group)} | `{path_text}` | "
            f"{_fmt(auto_vs_fa4)} | {_fmt(auto_vs_sdpa)} | {_fmt(_geomean(valid_best))} | "
            f"{_fmt(min(valid_best) if valid_best else None)} |"
        )

    lines.extend([
        "",
        "## Prefill candidate certificates",
        "",
        "A candidate qualifies only with complete timing and correctness data, "
        "at least 1.03x geomean improvement, and no result below 0.98x. "
        "Production means every row in the family already dispatches to that path.",
        "",
        "| Profile | Sequence | Batch | Candidate | Samples | Geo vs auto | Min | Worst max-abs | Decision |",
        "|---|---:|---|---|---:|---:|---:|---:|---|",
    ])
    for (
        profile,
        sequence,
        batch_band,
        candidate,
        samples,
        geo,
        minimum,
        worst_abs,
        decision,
    ) in _prefill_certificates(rows):
        lines.append(
            f"| `{profile}` | {sequence} | {batch_band} | {candidate} | {samples} | "
            f"{_fmt(geo)} | {_fmt(minimum)} | "
            f"{'-' if worst_abs is None else f'{worst_abs:.5f}'} | `{decision}` |"
        )

    opportunities = []
    for row in rows:
        auto = _median(row, AUTO)
        best_name, best_ms = _best(row)
        speedup = _ratio(auto, best_ms)
        if speedup is not None and speedup >= 1.02 and best_name != AUTO:
            opportunities.append((speedup, row, best_name, best_ms))
    opportunities.sort(key=lambda item: item[0], reverse=True)
    lines.extend([
        "",
        "## Largest remaining opportunities",
        "",
        "| Case | Profile/regime | B/Q/K/T | Hq/Hkv Dq/Dv | Auto path | Auto ms | Best candidate | Best ms | vs auto |",
        "|---|---|---|---|---|---:|---|---:|---:|",
    ])
    for speedup, row, best_name, best_ms in opportunities[:40]:
        case = row.get("case") or {}
        profile, regime = _parts(row)
        plan = row.get("plan") or {}
        lines.append(
            f"| `{case.get('name')}` | `{profile}/{regime}` | "
            f"{case.get('batch')}/{case.get('query_length')}/{case.get('kv_length')}/{case.get('topk')} | "
            f"{case.get('query_heads')}/{case.get('kv_heads')} D{case.get('head_dim')}/{case.get('value_head_dim')} | "
            f"`{plan.get('path', '-')}` | {_median(row, AUTO):.4f} | `{best_name}` | "
            f"{best_ms:.4f} | {speedup:.3f}x |"
        )

    lines.extend([
        "",
        "## Promotion discipline",
        "",
        "- GLM automatic promotion remains limited to its measured batch/sequence regions.",
        "- DeepSeek, MHA128, GQA4, GQA8, MQA64, and MHA64 true-prefill are promoted only at measured lengths and batch limits.",
        "- Ratio-2 GQA, ratio-7 GQA, D96 MHA, and other unmeasured profiles remain explicit experiments.",
        "- S512/S1024 high-batch and S2048/B16 promotion requires both causal-window and causal-mixed locality certificates.",
        "- Promote a new profile only when its complete shape family improves by at least 3%, no result falls below 0.98x, and correctness passes.",
        "- Decode, chunked-query, prefill, and batch policies should be promoted independently.",
        "- Use the cache diagnostics to verify repeated serving calls hit the lock-free workspace hot entry and scalar plan cache.",
    ])
    failures = [r for r in payload.get("results", []) if r.get("error")]
    if failures:
        lines.extend(["", "## Failures", ""])
        for row in failures:
            lines.append(f"- `{(row.get('case') or {}).get('name', 'unknown')}`: {row.get('error')}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    args.output.write_text(summarize(payload))
    print(args.output)


if __name__ == "__main__":
    main()
