import torch

from flash_attn.cute.indexed_metrics import (
    analyze_index_tiles,
)


def test_identical_lists_have_unit_compute_inflation():
    base = torch.tensor(
        [9, 8, 7, 6, -1, -1],
        dtype=torch.int32,
    )
    indices = base.view(1, 1, -1).expand(
        1,
        8,
        -1,
    ).clone()

    metrics = analyze_index_tiles(
        indices,
        kv_length=16,
        tile_m=8,
        query_heads=8,
        kv_heads=8,
    )

    assert metrics.mean_union == 4.0
    assert abs(
        metrics.mean_compute_inflation - 1.0
    ) < 1e-6
    assert metrics.mean_kv_reuse == 8.0


def test_disjoint_lists_report_inflation():
    rows = []
    for query in range(4):
        rows.append(
            torch.tensor(
                [
                    query * 4 + 3,
                    query * 4 + 2,
                    query * 4 + 1,
                    query * 4,
                ],
                dtype=torch.int32,
            )
        )
    indices = torch.stack(rows).unsqueeze(0)

    metrics = analyze_index_tiles(
        indices,
        kv_length=32,
        tile_m=4,
        query_heads=4,
        kv_heads=4,
    )

    assert metrics.mean_union == 16.0
    assert metrics.mean_compute_inflation == 4.0
    assert metrics.mean_kv_reuse == 1.0


def test_unpacked_nondivisible_gqa_metrics_do_not_raise():
    indices = torch.tensor(
        [[[7, 5, 3, 1], [6, 4, 2, 0]]],
        dtype=torch.int32,
    )
    metrics = analyze_index_tiles(
        indices,
        kv_length=8,
        tile_m=128,
        query_heads=28,
        kv_heads=4,
        pack_gqa=False,
    )
    assert metrics.qhead_per_kvhead == 1
    assert metrics.logical_queries_per_tile == 128
