"""Install an FA4-only ``flash_attn`` namespace for tests and scripts."""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import os
from pathlib import Path
import sys
import types
from typing import Any


def _normalize_package_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path if path.name == "flash_attn" else path / "flash_attn"


def discover_package_dirs(explicit_root: str | os.PathLike[str] | None = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit_root is not None:
        candidates.append(_normalize_package_dir(Path(explicit_root)))

    env_root = os.environ.get("FLASH_ATTN_FA4_ROOT")
    if env_root:
        candidates.append(_normalize_package_dir(Path(env_root)))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend((repo_root / "flash_attn", Path.cwd().resolve() / "flash_attn"))

    for distribution_name in ("flash-attn-4", "flash_attn_4"):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for entry in distribution.files or ():
            parts = tuple(entry.parts)
            if len(parts) >= 3 and parts[-3:] == ("flash_attn", "cute", "interface.py"):
                candidates.append(Path(distribution.locate_file(entry)).resolve().parents[1])
                break

    result: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            result.append(path)
            seen.add(key)
    return result


def install_fa4_only_namespace(
    explicit_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    package_dirs = [
        path
        for path in discover_package_dirs(explicit_root)
        if (path / "cute" / "interface.py").is_file()
    ]
    if not package_dirs:
        raise RuntimeError(
            "Could not locate FA4's flash_attn/cute/interface.py. "
            "Set FLASH_ATTN_FA4_ROOT to the FA4 repository or installation root."
        )

    for name in tuple(sys.modules):
        if name == "flash_attn" or name.startswith("flash_attn."):
            del sys.modules[name]

    primary = package_dirs[0]
    module = types.ModuleType("flash_attn")
    module.__file__ = str(primary / "__init__.py")
    module.__package__ = "flash_attn"
    module.__path__ = [str(path) for path in package_dirs]
    spec = importlib.machinery.ModuleSpec("flash_attn", loader=None, is_package=True)
    spec.submodule_search_locations = list(module.__path__)
    module.__spec__ = spec
    module.__fa4_only_namespace__ = True
    sys.modules["flash_attn"] = module

    return {
        "mode": "fa4-only-namespace",
        "package_dirs": list(module.__path__),
        "primary_package_dir": str(primary),
        "bypassed_parent_init": True,
        "flash_attn_2_cuda_required": False,
    }
