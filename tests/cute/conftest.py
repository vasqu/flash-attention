"""Make FA4 CuTe tests runnable when only ``flash-attn-4`` is installed.

The legacy parent package may eagerly import ``flash_attn_2_cuda``.  FA4's
Python/CuTe modules do not need that extension, so expose the repository's
``flash_attn`` directory as a lightweight namespace before test collection.
"""

from __future__ import annotations

import importlib.machinery
from pathlib import Path
import sys
import types


def _install_fa4_namespace() -> None:
    package_dir = Path(__file__).resolve().parents[2] / "flash_attn"
    cute_interface = package_dir / "cute" / "interface.py"
    if not cute_interface.is_file():
        return

    existing = sys.modules.get("flash_attn")
    existing_paths = list(getattr(existing, "__path__", ())) if existing else []
    if str(package_dir) in existing_paths:
        return

    for name in tuple(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            del sys.modules[name]

    module = types.ModuleType("flash_attn")
    module.__file__ = str(package_dir / "__init__.py")
    module.__package__ = "flash_attn"
    module.__path__ = [str(package_dir)]
    spec = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_dir)]
    module.__spec__ = spec
    sys.modules["flash_attn"] = module


_install_fa4_namespace()
