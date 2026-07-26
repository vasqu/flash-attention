import math

import pytest
import torch

from flash_attn.cute.interface import _flash_attn_fwd


def reference_indexed_attention(q, k, v, indices):
    batch, seqlen_q, num_heads, _ = q.shape
    num_kv_heads = k.shape[2]
    ratio = num_heads // num_kv_heads
    out = torch.zeros(
        (*q.shape[:-1], v.shape[-1]),
        dtype=torch.float32,
        device=q.device,
    )
    lse = torch.full(
        (batch, num_heads, seqlen_q),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    scale = 1.0 / math.sqrt(q.shape[-1])
    sentinel = -1 if indices.dtype == torch.int32 else 65_535
    for b in range(batch):
        for qi in range(seqlen_q):
            row = indices[b, qi].to(torch.int64)
            valid = row[(row != sentinel) & (row >= 0) & (row < k.shape[1])]
            valid = torch.unique(valid)
            if valid.numel() == 0:
                continue
            for h in range(num_heads):
                hk = h // ratio
                scores = (k[b, valid, hk].float() @ q[b, qi, h].float()) * scale
                probs = torch.softmax(scores, dim=0)
                out[b, qi, h] = probs @ v[b, valid, hk].float()
                lse[b, h, qi] = torch.logsumexp(scores, dim=0)
    return out, lse


def make_indices(batch, q_len, topk, k_len, device, dtype):
    sentinel = -1 if dtype != torch.uint16 else 65_535
    rows = torch.empty((batch, q_len, topk), device=device, dtype=torch.int64)
    for b in range(batch):
        for q in range(q_len):
            valid = torch.randperm(k_len, device=device)[: topk - 2]
            row = torch.cat(
                [valid, torch.tensor([sentinel, sentinel], device=device, dtype=torch.int64)]
            )
            rows[b, q] = row[torch.randperm(topk, device=device)]
    return rows.to(dtype)


@pytest.mark.parametrize("q_len,hq,hkv", [(1, 4, 4), (16, 8, 2)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.uint16])
def test_indexed_sm90_matches_reference(q_len, hq, hkv, index_dtype):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90 CUDA GPU")
    torch.manual_seed(0)
    device = "cuda"
    q = torch.randn((1, q_len, hq, 64), device=device, dtype=torch.float16)
    k = torch.randn((1, 128, hkv, 64), device=device, dtype=torch.float16)
    v = torch.randn((1, 128, hkv, 64), device=device, dtype=torch.float16)
    indices = make_indices(1, q_len, 16, 128, device, index_dtype)

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        gather_kv_indices=indices,
        return_lse=True,
    )
    ref_out, ref_lse = reference_indexed_attention(q, k, v, indices)

    torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=3e-2, atol=3e-2)
