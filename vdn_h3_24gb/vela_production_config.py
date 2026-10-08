"""VELA v1.0.0 production configuration.

This module internalizes the validated RC5 runtime configuration so VELA can
be launched from a clean ComfyUI BAT with no VDN_H3_* environment variables.
v1.0.0 freezes the validated internal V3.28 production math and adds no V-Smooth.
"""
from __future__ import annotations
import os

# Validated production-critical settings internalized by VELA.
# Assignment (rather than setdefault) makes V0 deterministic even if a parent
# shell happens to contain stale VDN_H3_* variables.
PRODUCTION_ENV = {
    "VDN_H3_ABLATION": "hybrid",
    "VDN_H3_HYBRID_MEMORY_STRATEGY": "baseline",
    "VDN_H3_BRANCH_OVERLAP": "0",
    "VDN_H3_SELECTIVE_WEIGHT_LOAD": "1",
    "VDN_H3_STREAM_PREFETCH": "0",
    "VDN_H3_WINDOW_INNER_TRIM": "0",
    "VDN_H3_HYBRID_STRIDE": "1",
    "VDN_H3_HYBRID_BLOCKS": "0,2,4,5,9,12,13,15,17,21,24,27,30,36,42,48",
    "VDN_H3_WINDOW_BLOCKS": "",
    "VDN_H3_WINDOW_FALLBACK": "zero",
    "VDN_H3_OUTPROJ_CACHE_GIB": "0",
    "VDN_H3_OUTPROJ_CACHE_CLEAR_AFTER": "0",
    "VDN_H3_OUTPROJ_EXECUTION": "materialized",
    "VDN_H3_OUTPROJ_GEMM": "mm_bias",
    "VDN_H3_QKV_EXECUTION": "native",
    "VDN_H3_QKV_EXEC_LAB": "0",
    "VDN_H3_OUTPROJ_MAT_PROFILE": "0",
    "VDN_H3_OUTPROJ_GEOMETRY_LAB": "0",
    "VDN_H3_PROFILE": "0",
    "VDN_H3_WINDOW_DEEP_PROFILE": "0",
    "VDN_H3_QKV_DEEP_PROFILE": "0",
    "VDN_H3_OUTPROJ_BLOCK_PROFILE": "0",
    "VDN_H3_DELTA_DEEP_PROFILE": "0",
    "VDN_H3_DELTA_SOLVER_LAB": "0",
    "VDN_H3_VC_STAGE_C01_FORCE_BATCH": "3",
    "VDN_H3_WINDOW_GROUP_BATCH": "3",
    "VDN_H3_WINDOW_SDPA": "benchmark",
    "VDN_H3_SDPA_BENCH_WARMUP": "1",
    "VDN_H3_SDPA_BENCH_REPS": "3",
    "VDN_H3_VC_C40_FAST_SAGE": "1",
}

# Defensive shutdown of old research switches that could otherwise leak in
# from a shell/session. They are not part of VELA v1.0.0 production behaviour.
OFF_ENV = (
    "VDN_H3_VSMOOTH_DIAG", "VDN_H3_VSMOOTH_PHASE1A",
    "VDN_H3_VSMOOTH_PHASE1B", "VDN_H3_VSMOOTH_PHASE1B1", "VDN_H3_VSMOOTH_PHASE1B2",
    "VDN_H3_VSMOOTH_PHASE1C", "VDN_H3_VSMOOTH_PHASE1C1", "VDN_H3_VSMOOTH_PHASE1C2", "VDN_H3_VSMOOTH_PHASE1C3",
    "VDN_H3_VSMOOTH_PHASE1D", "VDN_H3_VSMOOTH_PHASE1D1", "VDN_H3_VSMOOTH_PHASE1D2",
    "VDN_H3_VSMOOTH_PHASE1E", "VDN_H3_VSMOOTH_PHASE1E1", "VDN_H3_VSMOOTH_PHASE1E2",
    "VDN_H3_VC_C15_SAGE_WINDOW", "VDN_H3_VC_C30_PROFILE", "VDN_H3_VC_C35_CLONE64",
    "VDN_H3_VC_STAGE_C0",
    # LongCache remains research-only in VELA 1.0.0. Production is core-patch-free.
    "VDN_H3_LONG_CACHE_ENABLE", "VDN_H3_LONG_CACHE_LAB",
)

def apply_vela_production_config() -> None:
    for key, value in PRODUCTION_ENV.items():
        os.environ[key] = value
    for key in OFF_ENV:
        os.environ[key] = "0"
    os.environ["VDN_H3_INTEGRATION_MODE"] = ""

apply_vela_production_config()
print("[VELA v1.0.0] production config active; external VDN_H3_* BAT flags are not required")
