from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_SUMMARY = Path(__file__).parents[2] / "benchmarks" / "summarize_glm_moe_dsa_45_overnight.py"
_SPEC = spec_from_file_location("summarize_glm_moe_dsa_45_overnight", _SUMMARY)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_summary_promotes_safe_cute_sparse_candidate():
    payload = {
        "results": [
            {
                "status": "ok",
                "case": {
                    "name": f"prefill-{idx}",
                    "category": "glm45o-prefill",
                    "batch": 1,
                    "query_length": 8192,
                    "kv_length": 8192,
                    "topk": 2048,
                    "pattern": "causal-window",
                },
                "timings": {
                    "indexed_auto_end_to_end": {"median_ms": 10.0},
                    "indexed_dense_128x64_end_to_end": {"median_ms": 9.8},
                    "fa4_topk_128x64_end_to_end": {"median_ms": 9.7},
                    "fa4_topk_reuse_128x64_end_to_end": {"median_ms": 9.6},
                    "fa4_block_sparse_cute_128x64_end_to_end": {"median_ms": 8.0},
                    "fa4_block_sparse_cute_128x64_prepared_kernel": {"median_ms": 7.7},
                    "fa4_block_sparse_cute_128x64_metadata_only": {"median_ms": 0.2},
                    "fa4_native_topk_end_to_end": {"median_ms": 10.2},
                    "torch_sdpa_topk_end_to_end": {"median_ms": 15.0},
                },
                "diagnostics": {
                    "block_sparse_cute_128x64_active_fraction": 0.25,
                },
            }
            for idx in range(3)
        ]
    }
    report = _MODULE.render(payload)
    assert "PROMOTE" in report
    assert "block_sparse_cute_128x64" in report
    assert "CuTe exact block-sparse diagnostics" in report
    assert "Metadata budget" in report
