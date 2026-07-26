#!/usr/bin/env python3
"""Print the auto-dispatch misses from one fast indexed-SM90 benchmark run."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_FORCED_BACKENDS = {
    "row": "indexed_warp_end_to_end_median_ms",
    "row_scalar": "indexed_warp_scalar_end_to_end_median_ms",
    "dense": "indexed_dense_end_to_end_median_ms",
    "union": "indexed_union_end_to_end_median_ms",
}


def _number(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def summarize_rows(
    rows: Iterable[dict[str, Any]],
    *,
    baseline_rows: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    baseline = {
        (row.get("name"), row.get("dtype")): _number(
            row.get("indexed_auto_end_to_end_median_ms")
        )
        for row in baseline_rows
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        auto_ms = _number(row.get("indexed_auto_end_to_end_median_ms"))
        if auto_ms is None:
            continue
        forced = {
            backend: value
            for backend, column in _FORCED_BACKENDS.items()
            if (value := _number(row.get(column))) is not None
        }
        best_backend, best_ms = (
            min(forced.items(), key=lambda item: item[1])
            if forced
            else ("n/a", auto_ms)
        )
        baseline_ms = baseline.get((row.get("name"), row.get("dtype")))
        raw_regret = auto_ms / best_ms if best_ms > 0 else math.inf
        result.append(
            {
                "name": row.get("name", "?"),
                "category": row.get("category", "?"),
                "topk": _number(row.get("topk")),
                "dtype": row.get("dtype", "?"),
                "auto_path": row.get("plan_path", "?"),
                "auto_ms": auto_ms,
                "best_backend": best_backend,
                "best_ms": best_ms,
                # Auto and a forced call to the same backend are timed
                # independently.  Noise can make auto appear faster than the
                # measured minimum, but policy regret is defined to be >= 1.
                "regret": max(1.0, raw_regret),
                "exact_speedup": _number(
                    row.get("indexed_e2e_speedup_vs_fa4_native_topk_exact_e2e")
                ),
                "baseline_speedup": (
                    baseline_ms / auto_ms
                    if baseline_ms is not None and auto_ms > 0
                    else None
                ),
            }
        )
    return sorted(result, key=lambda item: item["regret"], reverse=True)


def _geomean(values: Iterable[float]) -> float:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    return math.exp(sum(math.log(value) for value in positive) / len(positive)) if positive else math.nan


def _format_optional(value: float | None, suffix: str = "x") -> str:
    return "-" if value is None else f"{value:.3f}{suffix}"


def _stats(items: Iterable[dict[str, Any]], *, max_regret: float) -> dict[str, float | int]:
    rows = list(items)
    exact = [item["exact_speedup"] for item in rows if item["exact_speedup"] is not None]
    return {
        "cases": len(rows),
        "regret_geomean": _geomean(item["regret"] for item in rows),
        "regret_max": max((item["regret"] for item in rows), default=math.nan),
        "misses": sum(item["regret"] > max_regret for item in rows),
        "exact_geomean": _geomean(exact),
    }


def _print_group_summary(items: list[dict[str, Any]], *, max_regret: float) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["category"])].append(item)

    print()
    print(f"{'category':38} {'cases':>5} {'auto/best':>10} {'max':>8} {'miss':>5} {'vs exact':>10}")
    print("-" * 82)
    for category in sorted(grouped):
        stats = _stats(grouped[category], max_regret=max_regret)
        print(
            f"{category[:38]:38} {stats['cases']:5d} "
            f"{stats['regret_geomean']:10.4f} {stats['regret_max']:8.4f} "
            f"{stats['misses']:5d} {_format_optional(stats['exact_geomean']):>10}"
        )


def print_summary(items: list[dict[str, Any]], *, top: int, max_regret: float) -> None:
    header = (
        f"{'case':46} {'dt':4} {'auto path':18} {'auto ms':>9} "
        f"{'best':11} {'best ms':>9} {'regret':>8} {'exact':>8} {'vs base':>8}"
    )
    print(header)
    print("-" * len(header))
    for item in items[:top]:
        marker = "!" if item["regret"] > max_regret else " "
        print(
            f"{marker}{item['name'][:45]:45} {item['dtype'][:4]:4} "
            f"{item['auto_path'][:18]:18} {item['auto_ms']:9.4f} "
            f"{item['best_backend'][:11]:11} {item['best_ms']:9.4f} "
            f"{item['regret']:8.3f} {_format_optional(item['exact_speedup']):>8} "
            f"{_format_optional(item['baseline_speedup']):>8}"
        )

    overall = _stats(items, max_regret=max_regret)
    topk_2k = [item for item in items if item["topk"] == 2048]
    topk_stats = _stats(topk_2k, max_regret=max_regret)
    print()
    print(
        f"cases={overall['cases']}  auto/best geomean={overall['regret_geomean']:.4f}x  "
        f"max={overall['regret_max']:.4f}x  misses>{max_regret:.3f}x={overall['misses']}"
    )
    print(
        f"topk=2048 cases={topk_stats['cases']}  auto/best geomean="
        f"{topk_stats['regret_geomean']:.4f}x  max={topk_stats['regret_max']:.4f}x  "
        f"misses={topk_stats['misses']}  vs exact={_format_optional(topk_stats['exact_geomean'])}"
    )
    _print_group_summary(items, max_regret=max_regret)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path, help="iteration CSV result(s)")
    parser.add_argument("--baseline", action="append", type=Path, default=[])
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument(
        "--max-regret",
        type=float,
        default=1.05,
        help="mark auto choices slower than the best forced backend by this ratio",
    )
    parser.add_argument(
        "--fail-above-regret",
        type=float,
        default=None,
        help="exit nonzero when any auto/best ratio exceeds this value",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = summarize_rows(load_rows(args.csv), baseline_rows=load_rows(args.baseline))
    print_summary(items, top=args.top, max_regret=args.max_regret)
    if args.fail_above_regret is not None and any(
        item["regret"] > args.fail_above_regret for item in items
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
