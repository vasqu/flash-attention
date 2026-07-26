"""CPU-only CuTe smoke compilation for indexed FA4 SM90.

Run from the FA4 source root after applying this overlay:
    PYTHONPATH=. python benchmarks/fa4_only_exec.py tests/cute/smoke_compile_indexed_sm90.py
"""

import os

os.environ.setdefault("FLASH_ATTENTION_ARCH", "sm_90a")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_90a")

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn.cute.interface import _flash_attn_fwd


def compile_case(
    name,
    q_len,
    hq,
    hkv,
    dim,
    index_dtype=torch.int32,
    *,
    batch=1,
    k_len=256,
    topk=16,
    value_dim=None,
    backend=None,
):
    value_dim = dim if value_dim is None else value_dim
    with FakeTensorMode():
        q = torch.empty((batch, q_len, hq, dim), device="cuda", dtype=torch.float16)
        k = torch.empty((batch, k_len, hkv, dim), device="cuda", dtype=torch.float16)
        v = torch.empty((batch, k_len, hkv, value_dim), device="cuda", dtype=torch.float16)
        indices = torch.empty((batch, q_len, topk), device="cuda", dtype=index_dtype)
        out, lse, _, _ = _flash_attn_fwd(
            q,
            k,
            v,
            gather_kv_indices=indices,
            return_lse=True,
            indexed_backend=backend,
            _arch=90,
        )
        assert out.shape == (batch, q_len, hq, value_dim)
        assert lse.shape == (batch, hq, q_len)
    print(f"compiled: {name}")


def compile_split_decode():
    with FakeTensorMode():
        q = torch.empty((1, 1, 4, 64), device="cuda", dtype=torch.float16)
        k = torch.empty((1, 4096, 4, 64), device="cuda", dtype=torch.float16)
        v = torch.empty((1, 4096, 4, 64), device="cuda", dtype=torch.float16)
        indices = torch.empty((1, 1, 512), device="cuda", dtype=torch.int32)
        out, lse, _, _ = _flash_attn_fwd(
            q,
            k,
            v,
            gather_kv_indices=indices,
            return_lse=True,
            _arch=90,
        )
        assert out.shape == q.shape
        assert lse.shape == (1, 4, 1)
    print("compiled: split warp decode")


def compile_dense_regression():
    with FakeTensorMode():
        q = torch.empty((1, 64, 4, 64), device="cuda", dtype=torch.float16)
        k = torch.empty((1, 128, 4, 64), device="cuda", dtype=torch.float16)
        v = torch.empty((1, 128, 4, 64), device="cuda", dtype=torch.float16)
        out, lse, _, _ = _flash_attn_fwd(q, k, v, return_lse=True, _arch=90)
        assert out.shape == q.shape
        assert lse.shape == (1, 4, 64)
    print("compiled: dense regression")


if __name__ == "__main__":
    compile_case("union N128", 64, 4, 4, 64)
    compile_case("warp D96", 17, 8, 2, 96, k_len=2501, topk=257, backend="warp")
    compile_case(
        "grouped GQA warp Q64", 64, 32, 8, 128,
        k_len=4097, topk=1025, backend="warp_grouped"
    )
    compile_case(
        "scalar GQA warp Q64", 64, 32, 8, 128,
        k_len=4097, topk=1025, backend="warp_scalar"
    )
    compile_case("dense indexed D96", 127, 8, 2, 96, k_len=4097, topk=513, backend="dense")
    compile_case(
        "dense indexed launch-safe D192/DV256", 128, 6, 6, 192,
        k_len=192, topk=31, value_dim=256, backend="dense"
    )
    compile_case(
        "dense indexed launch-safe D96/DV192 ratio6", 128, 6, 1, 96,
        k_len=192, topk=31, value_dim=192, backend="dense"
    )
    compile_case(
        "dense indexed register-safe D64/DV256 ratio6", 64, 6, 1, 64,
        k_len=128, topk=17, value_dim=256, backend="dense"
    )
    compile_case("packed GQA", 64, 8, 1, 64)
    compile_case("non-divisible GQA fallback", 64, 10, 2, 64)
    compile_case("union N96", 64, 4, 4, 192)
    compile_case("union N64", 64, 4, 4, 256)
    compile_case("uint16", 64, 4, 4, 64, torch.uint16)
    compile_case("warp decode direct", 1, 4, 4, 64)
    compile_case(
        "odd warp Q3 K2500 T257",
        3,
        8,
        2,
        64,
        torch.int64,
        batch=2,
        k_len=2500,
        topk=257,
        backend="warp",
    )
    compile_case(
        "odd union Q65 K997 T73",
        65,
        8,
        2,
        64,
        k_len=997,
        topk=73,
        value_dim=128,
        backend="union",
    )
    compile_case(
        "auto direct MQA prefill Q129 K5003 T257",
        129,
        64,
        1,
        128,
        k_len=5003,
        topk=257,
        backend="auto",
    )
    compile_case(
        "odd dense indexed Q129 K509 T71",
        129,
        8,
        1,
        64,
        k_len=509,
        topk=71,
        backend="dense",
    )
    compile_split_decode()
    compile_dense_regression()
