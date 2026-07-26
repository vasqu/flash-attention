from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_SUMMARY = Path(__file__).parents[2] / "benchmarks" / "summarize_glm_moe_dsa_45.py"
_SPEC = spec_from_file_location("summarize_glm_moe_dsa_45", _SUMMARY)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_summary_reports_model_stack_and_dispatch_regret():
    payload = {
        "suite": "glm-moe-dsa-45",
        "model_profile": {"num_hidden_layers": 45},
        "results": [
            {
                "status": "ok",
                "case": {
                    "name": "decode",
                    "category": "glm45-decode",
                    "batch": 1,
                    "query_length": 1,
                    "kv_length": 65536,
                    "topk": 2048,
                    "pattern": "random",
                },
                "plan": {"path": "row_sparse"},
                "timings": {
                    "indexed_auto_end_to_end": {"median_ms": 1.0},
                    "indexed_warp_end_to_end": {"median_ms": 0.8},
                    "indexed_dense_end_to_end": {"median_ms": 2.0},
                    "fa4_native_topk_end_to_end": {"median_ms": 4.0},
                    "torch_sdpa_topk_end_to_end": {"median_ms": 3.0},
                    "indexed_auto_preparation_only": {"median_ms": 0.1},
                    "indexed_auto_prepared_kernel": {"median_ms": 0.7},
                },
            }
        ],
    }
    rows = _MODULE._rows(payload)
    assert len(rows) == 1
    row = rows[0]
    assert row["stack_ms"] == 45.0
    assert row["vs_fa4"] == 4.0
    assert row["vs_sdpa"] == 3.0
    assert row["best_backend"] == "row"
    assert row["auto_regret"] == 1.25
    report = _MODULE.render(payload)
    assert "Dispatch misses above 3%" in report
    assert "45-layer" in report
