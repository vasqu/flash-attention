#!/usr/bin/env python3
"""Run a Python module or script after installing the FA4-only namespace.

Examples:
    python benchmarks/fa4_only_exec.py -m pytest -q tests/cute/test_indexed_sm90_sdpa.py
    python benchmarks/fa4_only_exec.py tests/cute/smoke_compile_indexed_sm90.py
"""

from __future__ import annotations

import runpy
from pathlib import Path
import sys

from fa4_only_bootstrap import install_fa4_only_namespace


def _usage() -> str:
    return "usage: fa4_only_exec.py [-m module | script.py] [arguments ...]"


def main() -> int:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(_usage())

    install_fa4_only_namespace()

    if args[0] == "-m":
        if len(args) < 2:
            raise SystemExit(_usage())
        module = args[1]
        module_args = args[2:]
        if module == "pytest":
            import pytest

            return int(pytest.main(module_args))
        sys.argv = [module, *module_args]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0

    script = Path(args[0])
    if not script.is_absolute():
        script = (Path.cwd() / script).resolve()
    if not script.is_file():
        raise SystemExit(f"script not found: {script}")
    sys.argv = [str(script), *args[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
