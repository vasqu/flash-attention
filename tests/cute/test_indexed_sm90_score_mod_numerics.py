import math

import pytest
import torch
import cutlass.cute as cute
from cutlass import Float32, Int32

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.indexed_score_mod import (
    build_topk_bitmask,
    topk_bitmask_score_mod,
)


@cute.jit
def selected_token_bias(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    bias = aux_tensors[0]
    return scores + Float32(
        bias[Int32(batch_idx[0]), Int32(head_idx[0]), Int32(q_idx[0]), Int32(kv_idx[0])]
    )


selected_token_bias.__vec_size__ = 1


def reference(q, k, v, indices, bias=None):
    bsz, q_len, hq, d = q.shape
    hkv = k.shape[2]
    ratio = hq // hkv
    scale = 1.0 / math.sqrt(d)
    out = torch.zeros((bsz, q_len, hq, v.shape[-1]), device=q.device, dtype=torch.float32)
    lse = torch.full((bsz, hq, q_len), -float("inf"), device=q.device, dtype=torch.float32)
    for b in range(bsz):
        for qi in range(q_len):
            row = indices[b, qi].long()
            valid = row[(row >= 0) & (row < k.shape[1])]
            valid = torch.unique(valid)
            for h in range(hq):
                hk = h // ratio
                scores = (k[b, valid, hk].float() @ q[b, qi, h].float()) * scale
                if bias is not None:
                    scores = scores + bias[b, h, qi, valid]
                probs = torch.softmax(scores, dim=0)
                out[b, qi, h] = probs @ v[b, valid, hk].float()
                lse[b, h, qi] = torch.logsumexp(scores, dim=0)
    return out, lse


def make_indices(q_len, k_len, topk, device):
    rows = torch.empty((1, q_len, topk), device=device, dtype=torch.int32)
    for qi in range(q_len):
        valid = torch.randperm(k_len, device=device)[: topk - 1].to(torch.int32)
        row = torch.cat([valid, torch.tensor([-1], device=device, dtype=torch.int32)])
        rows[0, qi] = row[torch.randperm(topk, device=device)]
    return rows


@pytest.mark.parametrize("q_len", [1, 8])
def test_custom_indexed_score_mod_matches_reference(q_len):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires SM90")
    torch.manual_seed(0)
    q = torch.randn((1, q_len, 4, 64), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 4, 64), device="cuda", dtype=torch.float16)
    v = torch.randn((1, 128, 4, 64), device="cuda", dtype=torch.float16)
    indices = make_indices(q_len, 128, 16, q.device)
    bias = torch.randn((1, 4, q_len, 128), device="cuda", dtype=torch.float32) * 0.05

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        gather_kv_indices=indices,
        score_mod=selected_token_bias,
        aux_tensors=[bias],
        return_lse=True,
        _arch=90,
    )
    ref_out, ref_lse = reference(q, k, v, indices, bias)
    torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=3e-2, atol=3e-2)


def test_native_bitmask_score_mod_matches_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires SM90")
    torch.manual_seed(1)
    q = torch.randn((1, 8, 4, 64), device="cuda", dtype=torch.float16)
    k = torch.randn((1, 128, 4, 64), device="cuda", dtype=torch.float16)
    v = torch.randn((1, 128, 4, 64), device="cuda", dtype=torch.float16)
    indices = make_indices(8, 128, 16, q.device)
    bitmask = build_topk_bitmask(indices, 128)

    out, lse, _, _ = _flash_attn_fwd(
        q,
        k,
        v,
        score_mod=topk_bitmask_score_mod,
        aux_tensors=[bitmask],
        return_lse=True,
        _arch=90,
    )
    ref_out, ref_lse = reference(q, k, v, indices)
    torch.testing.assert_close(out.float(), ref_out, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(lse, ref_lse, rtol=3e-2, atol=3e-2)
