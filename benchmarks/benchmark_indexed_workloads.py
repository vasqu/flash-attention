from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import torch

from flash_attn.cute.indexed_metrics import (
    analyze_index_tiles,
)
from flash_attn.cute.indexed_policy import (
    choose_indexed_plan,
)


def make_indices(
    *,
    pattern: str,
    batch: int,
    query_length: int,
    kv_length: int,
    topk: int,
    tail: int,
    device: str,
) -> torch.Tensor:
    if pattern == "identical":
        base = torch.randperm(
            kv_length,
            device=device,
        )[:topk]
        values = base[torch.randperm(topk, device=device)]
        return values.view(
            1, 1, -1
        ).expand(
            batch,
            query_length,
            -1,
        ).to(torch.int32).clone()

    if pattern == "random":
        values = torch.randint(
            0,
            kv_length,
            (
                batch,
                query_length,
                topk,
            ),
            device=device,
            dtype=torch.int32,
        )
        return values

    if pattern == "tail_random":
        tail = min(tail, topk, kv_length)
        random_width = topk - tail
        random_values = torch.randint(
            0,
            max(1, kv_length - tail),
            (
                batch,
                query_length,
                random_width,
            ),
            device=device,
            dtype=torch.int32,
        )
        tail_values = torch.arange(
            kv_length - 1,
            kv_length - tail - 1,
            -1,
            device=device,
            dtype=torch.int32,
        ).view(1, 1, -1).expand(
            batch,
            query_length,
            -1,
        )
        values = torch.cat(
            [random_values, tail_values],
            dim=-1,
        )
        order = torch.randperm(topk, device=device)
        return values[..., order]

    raise ValueError(f"unknown pattern: {pattern}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        choices=[
            "identical",
            "tail_random",
            "random",
        ],
        default="tail_random",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--query-length", type=int, default=512)
    parser.add_argument("--kv-length", type=int, default=32768)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--value-dim", type=int, default=256)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--tail", type=int, default=512)
    parser.add_argument("--sm-count", type=int, default=120)
    parser.add_argument("--backend", choices=("auto", "warp", "union", "dense"), default="auto")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    indices = make_indices(
        pattern=args.pattern,
        batch=args.batch_size,
        query_length=args.query_length,
        kv_length=args.kv_length,
        topk=args.topk,
        tail=args.tail,
        device=device,
    )

    plan = choose_indexed_plan(
        batch_size=args.batch_size,
        query_length=args.query_length,
        kv_length=args.kv_length,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        qk_head_dim=args.head_dim,
        value_head_dim=args.value_dim,
        topk=args.topk,
        sm_count=args.sm_count,
        backend=args.backend,
    )
    print("plan")
    print(json.dumps(asdict(plan), indent=2, default=str))

    union_plan = choose_indexed_plan(
        batch_size=args.batch_size,
        query_length=args.query_length,
        kv_length=args.kv_length,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        qk_head_dim=args.head_dim,
        value_head_dim=args.value_dim,
        topk=args.topk,
        sm_count=args.sm_count,
        backend="union",
    )
    if union_plan.tile_m:
        metrics = analyze_index_tiles(
            indices,
            kv_length=args.kv_length,
            tile_m=union_plan.tile_m,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            pack_gqa=union_plan.pack_gqa,
        )
        print("metrics")
        print(
            json.dumps(
                asdict(metrics),
                indent=2,
            )
        )

        estimated_dense_column_reduction = (
            args.kv_length
            / max(metrics.mean_union, 1.0)
        )
        print(
            "estimated dense-column reduction: "
            f"{estimated_dense_column_reduction:.2f}x"
        )


if __name__ == "__main__":
    main()
