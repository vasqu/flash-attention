import torch

from flash_attn.cute.indexed_block_sparse_lab import build_indexed_block_layout


def test_block_layout_tracks_union_per_query_tile_and_materializes_empty_full_list():
    indices = torch.tensor(
        [[
            [0, 1, 65, -1],
            [2, 66, 67, -1],
            [128, 129, -1, -1],
            [191, 192, -1, -1],
        ]],
        dtype=torch.int32,
    )
    mask_cnt, mask_idx, full_cnt, full_idx = build_indexed_block_layout(
        indices,
        seqlen_q=4,
        seqlen_k=256,
        tile_m=64,
        tile_n=64,
    )
    assert mask_cnt.shape == (1, 1, 1)
    assert mask_cnt.tolist() == [[[4]]]
    assert mask_idx[0, 0, 0, :4].tolist() == [0, 1, 2, 3]
    assert torch.count_nonzero(full_cnt) == 0
    assert full_idx.shape == mask_idx.shape


def test_block_layout_separates_query_tiles():
    indices = torch.full((1, 128, 2), -1, dtype=torch.int32)
    indices[:, :64, 0] = 1
    indices[:, 64:, 0] = 129
    mask_cnt, mask_idx, _, _ = build_indexed_block_layout(
        indices,
        seqlen_q=128,
        seqlen_k=256,
        tile_m=64,
        tile_n=64,
    )
    assert mask_cnt.tolist() == [[[1, 1]]]
    assert mask_idx[0, 0, 0, 0].item() == 0
    assert mask_idx[0, 0, 1, 0].item() == 2


def test_cute_metadata_source_is_native_and_persistent():
    from pathlib import Path

    root = Path(__file__).parents[2]
    lab = (root / "flash_attn" / "cute" / "indexed_block_sparse_lab.py").read_text()
    production = (
        root / "flash_attn" / "cute" / "indexed_block_sparse.py"
    ).read_text()
    kernel = (
        root / "flash_attn" / "cute" / "indexed_block_sparse_metadata_sm90.py"
    ).read_text()
    assert "triton" not in lab.lower()
    assert "triton" not in production.lower()
    assert "triton" not in kernel.lower()
    assert "@cute.kernel" in kernel
    assert "build_topk_bitmask" in production
    assert "IndexedBlockSparseCuteWorkspace" in production
    assert "mask_block_cnt=self.mask_cnt" in production
    assert "mask_block_idx=self.mask_idx" in production
