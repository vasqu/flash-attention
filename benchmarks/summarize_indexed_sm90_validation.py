#!/usr/bin/env python3
"""Summarize indexed SM90 correctness and benchmark results.

Produces a concise terminal report and a Markdown artifact with three buckets:
works well, acceptable/parity, and needs work. The primary same-semantics
baseline is unmodified FA4 with top-k membership expressed by score_mod. A
secondary context section compares against masked PyTorch SDPA.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_FORCED_BACKEND_TIMINGS = {
    "row": "indexed_warp_end_to_end",
    "row_scalar": "indexed_warp_scalar_end_to_end",
    "dense": "indexed_dense_end_to_end",
    "union": "indexed_union_end_to_end",
}


def geometric_mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and float(v) > 0]
    if not vals:
        return None
    return math.exp(statistics.fmean(math.log(v) for v in vals))


def load_junit(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    result = {"present": True, "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            result[key] += int(suite.attrib.get(key, 0))
        result["time"] += float(suite.attrib.get("time", 0.0))
    return result


def load_benchmarks(directory: Path) -> list[dict[str, Any]]:
    payloads = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            payload["_path"] = str(path)
            payloads.append(payload)
    return payloads


def payload_label(payload: dict[str, Any]) -> str:
    path_stem = Path(payload.get("_path", "unknown")).stem
    source = re.sub(r"_(?:bf16|fp16)$", "", path_stem).replace("_", "-")
    suite = str(payload.get("suite") or source).replace("_", "-")
    # Some benchmark families intentionally share a broad payload suite name
    # (for example random-shapes and random-all-backends both report "random").
    # Preserve the more specific filename label when it is a strict extension
    # of that suite so summary rows do not collapse into duplicates.
    if source.startswith(f"{suite}-"):
        suite = source
    dtype = (payload.get("configuration") or {}).get("dtype")
    return f"{suite}/{dtype}" if dtype else suite


def speedup_row(result: dict[str, Any], suite: str) -> dict[str, Any] | None:
    if result.get("status") != "ok":
        return None
    derived = result.get("derived") or {}
    speedup = derived.get("indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e")
    if speedup is None:
        return None
    case = result.get("case") or {}
    plan = result.get("plan") or {}
    timings = result.get("timings") or {}
    return {
        "suite": suite,
        "name": case.get("name", "unknown"),
        "category": case.get("category", "unknown"),
        "dtype": result.get("dtype", "unknown"),
        "path": plan.get("path", "unknown"),
        "speedup": float(speedup),
        "auto_ms": (timings.get("indexed_auto_end_to_end") or {}).get("median_ms"),
        "baseline_ms": (timings.get("fa4_native_topk_end_to_end") or {}).get("median_ms"),
        "batch": case.get("batch"),
        "q": case.get("query_length"),
        "k": case.get("kv_length"),
        "topk": case.get("topk"),
        "hq": case.get("query_heads"),
        "hkv": case.get("kv_heads"),
        "d": case.get("head_dim"),
        "dv": case.get("value_head_dim"),
    }


def sdpa_speedup_row(result: dict[str, Any], suite: str) -> dict[str, Any] | None:
    if result.get("status") != "ok":
        return None
    derived = result.get("derived") or {}
    speedup = derived.get("indexed_e2e_speedup_vs_torch_sdpa_topk_exact_e2e")
    if speedup is None:
        return None
    case = result.get("case") or {}
    plan = result.get("plan") or {}
    return {
        "suite": suite,
        "name": case.get("name", "unknown"),
        "category": case.get("category", "unknown"),
        "dtype": result.get("dtype", "unknown"),
        "path": plan.get("path", "unknown"),
        "speedup": float(speedup),
    }


def summarize_sdpa_context(payloads: list[dict[str, Any]], parity: float) -> dict[str, Any]:
    rows = []
    for payload in payloads:
        suite = payload_label(payload)
        rows.extend(
            row
            for result in payload["results"]
            if (row := sdpa_speedup_row(result, suite)) is not None
        )
    return {
        "rows": rows,
        "geomean": geometric_mean(row["speedup"] for row in rows),
        "good": sorted(
            (row for row in rows if row["speedup"] > 1 + parity),
            key=lambda row: row["speedup"],
            reverse=True,
        ),
        "okay": sorted(
            (row for row in rows if 1 - parity <= row["speedup"] <= 1 + parity),
            key=lambda row: abs(row["speedup"] - 1),
        ),
        "bad": sorted(
            (row for row in rows if row["speedup"] < 1 - parity),
            key=lambda row: row["speedup"],
        ),
    }


def summarize_kernel_experiments(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    grouped_vs_scalar: list[dict[str, Any]] = []
    for payload in payloads:
        suite = payload_label(payload)
        for result in payload.get("results", []):
            if result.get("status") != "ok":
                continue
            case = result.get("case") or {}
            timings = result.get("timings") or {}

            grouped = (timings.get("indexed_warp_end_to_end") or {}).get("median_ms")
            scalar = (timings.get("indexed_warp_scalar_end_to_end") or {}).get("median_ms")
            if grouped and scalar:
                grouped_vs_scalar.append(
                    {
                        "suite": suite,
                        "name": case.get("name", "unknown"),
                        "speedup": float(scalar) / float(grouped),
                    }
                )

    return {
        "grouped_vs_scalar": grouped_vs_scalar,
        "grouped_geomean": geometric_mean(row["speedup"] for row in grouped_vs_scalar),
    }


def fmt_speedup(value: float) -> str:
    if value >= 1:
        return f"{value:.2f}x faster"
    return f"{1 / value:.2f}x slower"


def case_label(row: dict[str, Any]) -> str:
    return (
        f"{row['name']} [{row.get('dtype', 'unknown')}, {row['path']}] "
        f"B{row['batch']} Q{row['q']} K{row['k']} T{row['topk']} "
        f"H{row['hq']}/{row['hkv']} D{row['d']}/{row['dv']}"
    )


def summarize(payloads: list[dict[str, Any]], parity: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failed_cases: list[dict[str, Any]] = []
    suite_summaries: list[dict[str, Any]] = []
    for payload in payloads:
        suite = payload_label(payload)
        current = [row for result in payload["results"] if (row := speedup_row(result, suite))]
        rows.extend(current)
        failures = [result for result in payload["results"] if result.get("status") != "ok"]
        for result in failures:
            failed_cases.append(
                {
                    "suite": suite,
                    "name": (result.get("case") or {}).get("name", "unknown"),
                    "status": result.get("status"),
                    "error": result.get("error") or result.get("backend_errors"),
                }
            )
        suite_summaries.append(
            {
                "suite": suite,
                "path": payload["_path"],
                "cases": len(current),
                "geomean": geometric_mean(row["speedup"] for row in current),
                "good": sum(row["speedup"] > 1 + parity for row in current),
                "parity": sum(1 - parity <= row["speedup"] <= 1 + parity for row in current),
                "bad": sum(row["speedup"] < 1 - parity for row in current),
                "failed": len(failures),
            }
        )

    by_path: dict[str, list[float]] = defaultdict(list)
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_path[row["path"]].append(row["speedup"])
        by_category[row["category"]].append(row["speedup"])

    good = sorted((r for r in rows if r["speedup"] > 1 + parity), key=lambda r: r["speedup"], reverse=True)
    okay = sorted((r for r in rows if 1 - parity <= r["speedup"] <= 1 + parity), key=lambda r: abs(r["speedup"] - 1))
    bad = sorted((r for r in rows if r["speedup"] < 1 - parity), key=lambda r: r["speedup"])
    return {
        "rows": rows,
        "good": good,
        "okay": okay,
        "bad": bad,
        "failed": failed_cases,
        "suite_summaries": suite_summaries,
        "geomean": geometric_mean(row["speedup"] for row in rows),
        "by_path": {
            key: {
                "cases": len(values),
                "geomean": geometric_mean(values),
                "min": min(values),
                "max": max(values),
            }
            for key, values in sorted(by_path.items())
        },
        "by_category": {
            key: {
                "cases": len(values),
                "geomean": geometric_mean(values),
                "min": min(values),
                "max": max(values),
            }
            for key, values in sorted(by_category.items())
        },
    }


def summarize_auto_dispatch(
    payloads: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        suite = payload_label(payload)
        for result in payload.get("results", []):
            if result.get("status") != "ok":
                continue
            timings = result.get("timings") or {}
            auto_ms = (timings.get("indexed_auto_end_to_end") or {}).get("median_ms")
            if auto_ms is None:
                continue
            forced = {
                backend: float(value)
                for backend, timing_name in _FORCED_BACKEND_TIMINGS.items()
                if (value := (timings.get(timing_name) or {}).get("median_ms")) is not None
            }
            if not forced:
                continue
            best_backend, best_ms = min(forced.items(), key=lambda item: item[1])
            auto_ms = float(auto_ms)
            raw_regret = auto_ms / best_ms if best_ms > 0 else math.inf
            case = result.get("case") or {}
            plan = result.get("plan") or {}
            rows.append(
                {
                    "suite": suite,
                    "name": case.get("name", "unknown"),
                    "category": case.get("category", "unknown"),
                    "dtype": result.get("dtype", "unknown"),
                    "path": plan.get("path", "unknown"),
                    "auto_ms": auto_ms,
                    "best_backend": best_backend,
                    "best_ms": best_ms,
                    "regret": max(1.0, raw_regret),
                    "batch": case.get("batch"),
                    "q": case.get("query_length"),
                    "k": case.get("kv_length"),
                    "topk": case.get("topk"),
                    "hq": case.get("query_heads"),
                    "hkv": case.get("kv_heads"),
                    "d": case.get("head_dim"),
                    "dv": case.get("value_head_dim"),
                }
            )
    rows.sort(key=lambda row: row["regret"], reverse=True)
    topk_2k = [row for row in rows if row["topk"] == 2048]
    return {
        "rows": rows,
        "geomean": geometric_mean(row["regret"] for row in rows),
        "max": max((row["regret"] for row in rows), default=None),
        "misses": [row for row in rows if row["regret"] > threshold],
        "threshold": threshold,
        "topk_2k_cases": len(topk_2k),
        "topk_2k_geomean": geometric_mean(row["regret"] for row in topk_2k),
        "topk_2k_max": max((row["regret"] for row in topk_2k), default=None),
    }


def auto_case_label(row: dict[str, Any]) -> str:
    return (
        f"{row['name']} [{row['dtype']}, auto={row['path']}, best={row['best_backend']}] "
        f"B{row['batch']} Q{row['q']} K{row['k']} T{row['topk']} "
        f"H{row['hq']}/{row['hkv']} D{row['d']}/{row['dv']}"
    )


def render_markdown(
    summary: dict[str, Any],
    auto_dispatch: dict[str, Any],
    sdpa: dict[str, Any],
    experiments: dict[str, Any],
    junit: dict[str, Any],
    parity: float,
) -> str:
    rows = summary["rows"]
    lines = ["# Indexed SM90 validation summary", ""]
    lines.append("## Correctness")
    if junit["present"]:
        passed = junit["tests"] - junit["failures"] - junit["errors"] - junit["skipped"]
        lines.append(
            f"- **{passed}/{junit['tests']} passed**, {junit['failures']} failures, "
            f"{junit['errors']} errors, {junit['skipped']} skipped in {junit['time']:.1f}s."
        )
    else:
        lines.append("- No JUnit correctness report was found.")
    lines.append("- References: float32 eager additive-mask attention and PyTorch SDPA at input precision.")
    lines.append("")

    lines.append("## Same-semantics performance")
    if rows:
        lines.append(
            f"- **{len(summary['good'])} materially faster**, **{len(summary['okay'])} within ±{parity*100:.0f}% parity**, "
            f"**{len(summary['bad'])} materially slower** across {len(rows)} measured cases."
        )
        lines.append(f"- Aggregate geometric-mean speedup: **{summary['geomean']:.2f}x** versus unmodified FA4 top-k `score_mod`.")
    else:
        lines.append("- No same-semantics benchmark rows were found.")
    lines.append("- Full dense FA4 is a different-semantics throughput reference and is excluded from these classifications.")
    lines.append("")

    lines.append("## Auto-dispatch quality")
    if auto_dispatch["rows"]:
        lines.append(
            f"- Auto/best-forced geometric-mean regret: **{auto_dispatch['geomean']:.4f}x**; "
            f"maximum **{auto_dispatch['max']:.4f}x** across {len(auto_dispatch['rows'])} comparable cases."
        )
        lines.append(
            f"- **{len(auto_dispatch['misses'])}** cases exceed "
            f"{auto_dispatch['threshold']:.3f}x regret."
        )
        if auto_dispatch["topk_2k_cases"]:
            lines.append(
                f"- Top-k 2048: {auto_dispatch['topk_2k_cases']} cases, "
                f"geomean **{auto_dispatch['topk_2k_geomean']:.4f}x**, "
                f"maximum **{auto_dispatch['topk_2k_max']:.4f}x**."
            )
        for row in auto_dispatch["misses"][:20]:
            lines.append(
                f"  - **{row['regret']:.3f}x regret** "
                f"({row['auto_ms']:.4f} ms vs {row['best_ms']:.4f} ms) — "
                f"{auto_case_label(row)}"
            )
    else:
        lines.append("- No cases contained both auto and forced-backend timings.")
    lines.append("")

    lines.append("## PyTorch SDPA context")
    if sdpa["rows"]:
        lines.append(
            f"- **{len(sdpa['good'])} materially faster**, **{len(sdpa['okay'])} within ±{parity*100:.0f}% parity**, "
            f"**{len(sdpa['bad'])} materially slower** across {len(sdpa['rows'])} masked-SDPA comparisons."
        )
        lines.append(
            f"- Aggregate geometric-mean speedup: **{sdpa['geomean']:.2f}x** versus "
            "`torch.nn.functional.scaled_dot_product_attention` with a dense broadcast boolean selected-set mask."
        )
        if sdpa["good"]:
            lines.append("- Best context points:")
            for row in sdpa["good"][:5]:
                lines.append(
                    f"  - **{fmt_speedup(row['speedup'])}** — {row['name']} [{row.get('dtype', 'unknown')}, {row['path']}]"
                )
        if sdpa["bad"]:
            lines.append("- Slower than masked SDPA:")
            for row in sdpa["bad"][:5]:
                lines.append(
                    f"  - **{fmt_speedup(row['speedup'])}** — {row['name']} [{row.get('dtype', 'unknown')}, {row['path']}]"
                )
    else:
        lines.append("- No PyTorch SDPA benchmark rows were recorded.")
    lines.append(
        "- This is a context baseline, not the primary dispatch gate: PyTorch may choose different "
        "SDPA kernels depending on shape, dtype, GQA support, and mask handling."
    )
    lines.append("")

    lines.append("## Prefill kernel experiments")
    grouped_rows = experiments["grouped_vs_scalar"]
    if grouped_rows:
        grouped_wins = sum(row["speedup"] > 1.0 for row in grouped_rows)
        lines.append(
            f"- Register-shared grouped warp versus one-head-per-warp: "
            f"**{experiments['grouped_geomean']:.2f}x** geometric-mean speedup "
            f"across {len(grouped_rows)} cases ({grouped_wins} wins)."
        )
        for row in sorted(grouped_rows, key=lambda item: item["speedup"], reverse=True)[:5]:
            lines.append(
                f"  - **{fmt_speedup(row['speedup'])}** — {row['name']} [{row['suite']}]"
            )
    else:
        lines.append("- No grouped-versus-scalar warp comparisons were recorded.")
    lines.append("")

    lines.append("### By selected backend")
    for path, stats in summary["by_path"].items():
        lines.append(
            f"- `{path}`: {stats['cases']} cases, geomean **{stats['geomean']:.2f}x**, "
            f"range {stats['min']:.2f}x–{stats['max']:.2f}x."
        )
    lines.append("")

    lines.append("### Works well")
    if summary["good"]:
        for row in summary["good"][:15]:
            lines.append(f"- **{fmt_speedup(row['speedup'])}** — {case_label(row)}")
    else:
        lines.append("- No material wins.")
    lines.append("")

    lines.append("### Okay / parity")
    if summary["okay"]:
        for row in summary["okay"][:15]:
            lines.append(f"- **{row['speedup']:.3f}x** — {case_label(row)}")
        if len(summary["okay"]) > 15:
            lines.append(f"- …and {len(summary['okay']) - 15} additional parity cases.")
    else:
        lines.append("- No parity cases.")
    lines.append("")

    lines.append("### Needs work")
    if summary["bad"]:
        for row in summary["bad"][:20]:
            lines.append(f"- **{fmt_speedup(row['speedup'])}** — {case_label(row)}")
    else:
        lines.append("- No material same-semantics regressions.")
    if summary["failed"]:
        lines.append("")
        lines.append("#### Failed benchmark cases")
        for failure in summary["failed"][:20]:
            lines.append(f"- `{failure['suite']}/{failure['name']}`: {failure['status']} — {failure['error']}")
    lines.append("")

    lines.append("### Suite overview")
    lines.append("| Suite | Cases | Geomean | Faster | Parity | Slower | Failed |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for item in summary["suite_summaries"]:
        gm = "—" if item["geomean"] is None else f"{item['geomean']:.2f}x"
        lines.append(
            f"| {item['suite']} | {item['cases']} | {gm} | {item['good']} | "
            f"{item['parity']} | {item['bad']} | {item['failed']} |"
        )
    lines.append("")

    lines.append("### Category overview")
    lines.append("| Category | Cases | Geomean | Range |")
    lines.append("|---|---:|---:|---:|")
    for category, stats in summary["by_category"].items():
        lines.append(
            f"| {category} | {stats['cases']} | {stats['geomean']:.2f}x | "
            f"{stats['min']:.2f}x–{stats['max']:.2f}x |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--junit", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--parity", type=float, default=0.02)
    parser.add_argument(
        "--auto-regret-threshold",
        type=float,
        default=1.03,
        help="report auto choices slower than the best forced backend by this ratio",
    )
    parser.add_argument(
        "--fail-auto-regret",
        type=float,
        default=None,
        help="exit nonzero if any comparable auto/best ratio exceeds this value",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = load_benchmarks(args.results_dir)
    summary = summarize(payloads, args.parity)
    auto_dispatch = summarize_auto_dispatch(payloads, args.auto_regret_threshold)
    sdpa = summarize_sdpa_context(payloads, args.parity)
    experiments = summarize_kernel_experiments(payloads)
    junit = load_junit(args.junit)
    text = render_markdown(summary, auto_dispatch, sdpa, experiments, junit, args.parity)
    output = args.output or args.results_dir / "SUMMARY.md"
    output.write_text(text + "\n")
    print(text)
    print(f"\nSummary: {output.resolve()}")

    correctness_bad = junit["present"] and (junit["failures"] > 0 or junit["errors"] > 0)
    auto_regret_bad = (
        args.fail_auto_regret is not None
        and auto_dispatch["max"] is not None
        and auto_dispatch["max"] > args.fail_auto_regret
    )
    if (args.strict and (correctness_bad or summary["bad"] or summary["failed"])) or auto_regret_bad:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
