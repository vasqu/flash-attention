"""Preflight used by the FA4-only validation runner."""

import sys
import flash_attn
from flash_attn.cute.interface import _flash_attn_fwd

assert getattr(flash_attn, "__fa4_only_namespace__", False)
assert "flash_attn_2_cuda" not in sys.modules
print(f"FA4-only preflight OK: {list(flash_attn.__path__)[0]}")
