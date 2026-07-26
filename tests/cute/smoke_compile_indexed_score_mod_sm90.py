"""Fake-tensor CuTe compilation checks for indexed and native score_mod paths."""

import os
os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_90a")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
import cutlass.cute as cute
from cutlass import Float32, Int32

from flash_attn.cute.interface import _flash_attn_fwd
from flash_attn.cute.indexed_score_mod import topk_bitmask_score_mod


@cute.jit
def selected_token_bias(scores, batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    bias = aux_tensors[0]
    return scores + Float32(bias[Int32(batch_idx[0]), Int32(q_idx[0]), Int32(kv_idx[0])])


selected_token_bias.__vec_size__ = 1


def compile_indexed(q_len: int):
    with FakeTensorMode():
        q = torch.empty((1, q_len, 4, 64), device="cuda", dtype=torch.float16)
        k = torch.empty((1, 128, 4, 64), device="cuda", dtype=torch.float16)
        v = torch.empty((1, 128, 4, 64), device="cuda", dtype=torch.float16)
        # Public path accepts arbitrary order; preparation happens after policy selection.
        indices = torch.empty((1, q_len, 16), device="cuda", dtype=torch.int64)
        bias = torch.empty((1, q_len, 128), device="cuda", dtype=torch.float32)
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
        assert out.shape == q.shape
        assert lse.shape == (1, 4, q_len)


def compile_native():
    with FakeTensorMode():
        q = torch.empty((1, 8, 4, 64), device="cuda", dtype=torch.float16)
        k = torch.empty((1, 128, 4, 64), device="cuda", dtype=torch.float16)
        v = torch.empty((1, 128, 4, 64), device="cuda", dtype=torch.float16)
        bitmask = torch.empty((1, 8, 4), device="cuda", dtype=torch.int32)
        out, lse, _, _ = _flash_attn_fwd(
            q,
            k,
            v,
            score_mod=topk_bitmask_score_mod,
            aux_tensors=[bitmask],
            return_lse=True,
            _arch=90,
        )
        assert out.shape == q.shape
        assert lse.shape == (1, 4, 8)


if __name__ == "__main__":
    compile_indexed(8)
    print("compiled indexed union score_mod")
    compile_indexed(1)
    print("compiled indexed decode score_mod")
    compile_native()
    print("compiled native dense topk score_mod")
