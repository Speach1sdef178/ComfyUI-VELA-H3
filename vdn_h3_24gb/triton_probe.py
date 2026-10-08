"""v28: Triton exact-attention autotuning laboratory.

Diagnostic only: production attention always remains on the configured PyTorch/cuDNN
SDPA path.  v28 benchmarks several streaming online-softmax Triton launch shapes on
one *real* H3 window group, checks every candidate against the cuDNN reference, and
reports the fastest valid configuration.

Two mathematically equivalent softmax implementations are explored:
  * ``exp``  : natural exponent (v27 implementation)
  * ``exp2`` : base-2 exponent with log2(e) scale, a common Triton fast path

No Triton result is ever fed back into the model in v28.
"""
import math
import os
import logging
import itertools
import torch

_log = logging.getLogger(__name__)
_TRIED = set()


def _env_on(name, default=False):
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def enabled():
    # Keep the v27 variable as an alias so old BAT files still work.
    return _env_on("VDN_H3_TRITON_ATTN_AUTOTUNE", False) or _env_on("VDN_H3_TRITON_ATTN_PROBE", False)


def _int_env(name, default, lo, hi):
    try:
        v = int(os.environ.get(name, str(default)))
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _float_env(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def maybe_probe(q_bqhd, k_bkhd, v_bkhd, scale, reference_call):
    """Run the v28 laboratory once per real shape; never return a production output."""
    if not enabled() or q_bqhd.device.type != "cuda" or q_bqhd.shape[-1] != 128:
        return
    key = (tuple(q_bqhd.shape), tuple(k_bkhd.shape), str(q_bqhd.dtype), str(q_bqhd.device))
    if key in _TRIED:
        return
    _TRIED.add(key)

    try:
        import triton
        import triton.language as tl
    except Exception as e:
        _log.warning("[vdn-triton-v28] Triton unavailable; autotune skipped: %s", e)
        return

    @triton.jit
    def _attn_exp(Q, K, V, O, sm_scale: tl.constexpr,
                  QN: tl.constexpr, KN: tl.constexpr,
                  H: tl.constexpr, D: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        q_base = ((pid_b * QN + offs_m[:, None]) * H + pid_h) * D
        q = tl.load(Q + q_base + offs_d[None, :], mask=offs_m[:, None] < QN, other=0.0)
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        for start_n in range(0, KN, BLOCK_N):
            n = start_n + offs_n
            k_base = ((pid_b * KN + n[None, :]) * H + pid_h) * D
            k = tl.load(K + k_base + offs_d[:, None], mask=n[None, :] < KN, other=0.0)
            qk = tl.dot(q, k) * sm_scale
            qk = tl.where(n[None, :] < KN, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            v_base = ((pid_b * KN + n[:, None]) * H + pid_h) * D
            v = tl.load(V + v_base + offs_d[None, :], mask=n[:, None] < KN, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        acc = acc / l_i[:, None]
        o_base = ((pid_b * QN + offs_m[:, None]) * H + pid_h) * D
        tl.store(O + o_base + offs_d[None, :], acc, mask=offs_m[:, None] < QN)

    @triton.jit
    def _attn_exp2(Q, K, V, O, sm_scale_log2e: tl.constexpr,
                   QN: tl.constexpr, KN: tl.constexpr,
                   H: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        q_base = ((pid_b * QN + offs_m[:, None]) * H + pid_h) * D
        q = tl.load(Q + q_base + offs_d[None, :], mask=offs_m[:, None] < QN, other=0.0)
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, D), tl.float32)
        for start_n in range(0, KN, BLOCK_N):
            n = start_n + offs_n
            k_base = ((pid_b * KN + n[None, :]) * H + pid_h) * D
            k = tl.load(K + k_base + offs_d[:, None], mask=n[None, :] < KN, other=0.0)
            qk = tl.dot(q, k) * sm_scale_log2e
            qk = tl.where(n[None, :] < KN, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            v_base = ((pid_b * KN + n[:, None]) * H + pid_h) * D
            v = tl.load(V + v_base + offs_d[None, :], mask=n[:, None] < KN, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij
        acc = acc / l_i[:, None]
        o_base = ((pid_b * QN + offs_m[:, None]) * H + pid_h) * D
        tl.store(O + o_base + offs_d[None, :], acc, mask=offs_m[:, None] < QN)

    qb = q_bqhd[:1].contiguous()
    kb = k_bkhd[:1].contiguous()
    vb = v_bkhd[:1].contiguous()
    B, QN, H, D = qb.shape
    KN = kb.shape[1]
    tol = _float_env("VDN_H3_TRITON_ATTN_RTOL", 0.002)
    check_rows = min(QN, _int_env("VDN_H3_TRITON_ATTN_CHECK_ROWS", 128, 16, QN))
    reps = _int_env("VDN_H3_TRITON_TUNE_REPS", 2, 1, 10)
    final_reps = _int_env("VDN_H3_TRITON_FINAL_REPS", 4, 2, 20)
    topk = _int_env("VDN_H3_TRITON_TUNE_TOPK", 4, 1, 8)
    exhaustive = _env_on("VDN_H3_TRITON_TUNE_EXHAUSTIVE", False)

    ref = None
    try:
        # Warm and retain one reference tensor for correctness checks.
        reference_call(qb, kb, vb)
        torch.cuda.synchronize(qb.device)
        ref = reference_call(qb, kb, vb)
        torch.cuda.synchronize(qb.device)
    except Exception as e:
        _log.warning("[vdn-triton-v28] reference setup failed; autotune skipped: %s", e)
        return

    def time_ms(fn, nreps):
        torch.cuda.synchronize(qb.device)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(nreps):
            fn()
        e.record(); e.synchronize()
        return float(s.elapsed_time(e)) / nreps

    cudnn_ms = time_ms(lambda: reference_call(qb, kb, vb), final_reps)

    def run_candidate(strategy, bm, bn, warps, stages, out):
        grid = (triton.cdiv(QN, bm), H, B)
        if strategy == "exp2":
            _attn_exp2[grid](qb, kb, vb, out,
                sm_scale_log2e=float(scale) * 1.4426950408889634,
                QN=QN, KN=KN, H=H, D=D, BLOCK_M=bm, BLOCK_N=bn,
                num_warps=warps, num_stages=stages)
        else:
            _attn_exp[grid](qb, kb, vb, out,
                sm_scale=float(scale), QN=QN, KN=KN, H=H, D=D,
                BLOCK_M=bm, BLOCK_N=bn, num_warps=warps, num_stages=stages)
        return out

    def evaluate(cfg, nreps):
        strategy, bm, bn, warps, stages = cfg
        out = torch.empty_like(qb)
        fn = lambda: run_candidate(strategy, bm, bn, warps, stages, out)
        try:
            # compile + warm
            fn(); torch.cuda.synchronize(qb.device)
            a = out[:, :check_rows].float()
            b = ref[:, :check_rows].float()
            diff = a - b
            rel = float(torch.sqrt(torch.mean(diff * diff)) / (torch.sqrt(torch.mean(b*b)) + 1e-12))
            mx = float(diff.abs().max())
            ok = math.isfinite(rel) and rel <= tol
            if not ok:
                return {"cfg":cfg, "ok":False, "rel":rel, "max":mx, "ms":float("inf"), "err":"accuracy"}
            ms = time_ms(fn, nreps)
            return {"cfg":cfg, "ok":True, "rel":rel, "max":mx, "ms":ms, "err":""}
        except Exception as e:
            return {"cfg":cfg, "ok":False, "rel":float("inf"), "max":float("inf"), "ms":float("inf"), "err":str(e)}

    # Stage 1: broad shape/algorithm search with conservative stage count.
    if exhaustive:
        stage1 = list(itertools.product(("exp","exp2"), (32,64,128), (32,64,128), (4,8), (2,)))
    else:
        shape_seed = ((32,64),(64,32),(64,64),(64,128),(128,32),(128,64))
        stage1 = [(strategy,bm,bn,4,2) for strategy in ("exp","exp2") for bm,bn in shape_seed]

    _log.info("[vdn] v28 Triton exact-attention autotuning laboratory: on (diagnostic-only; %d stage-1 candidates, topk=%d)", len(stage1), topk)
    results = []
    seen = set()
    for cfg in stage1:
        seen.add(cfg)
        r = evaluate(cfg, reps)
        results.append(r)
        if r["ok"]:
            _log.info("[vdn-triton-v28-cand] phase=1 strat=%s BM=%d BN=%d warps=%d stages=%d | %.3fms speedup=%.3fx rel=%.6g max=%.6g ok=1",
                      *cfg, r["ms"], cudnn_ms/r["ms"], r["rel"], r["max"])
        else:
            _log.info("[vdn-triton-v28-cand] phase=1 strat=%s BM=%d BN=%d warps=%d stages=%d | rejected rel=%.6g err=%s",
                      *cfg, r["rel"], r["err"][:120])

    valid1 = sorted((r for r in results if r["ok"]), key=lambda x:x["ms"])
    finalists = valid1[:topk]

    # Stage 2: for the best stage-1 shapes/strategy, sweep occupancy/pipeline knobs.
    phase2 = []
    for r in finalists:
        strategy,bm,bn,_,_ = r["cfg"]
        for warps in (4,8):
            for stages in (2,3,4):
                cfg=(strategy,bm,bn,warps,stages)
                if cfg not in seen:
                    phase2.append(cfg); seen.add(cfg)
    for cfg in phase2:
        r = evaluate(cfg, reps)
        results.append(r)
        if r["ok"]:
            _log.info("[vdn-triton-v28-cand] phase=2 strat=%s BM=%d BN=%d warps=%d stages=%d | %.3fms speedup=%.3fx rel=%.6g max=%.6g ok=1",
                      *cfg, r["ms"], cudnn_ms/r["ms"], r["rel"], r["max"])
        else:
            _log.info("[vdn-triton-v28-cand] phase=2 strat=%s BM=%d BN=%d warps=%d stages=%d | rejected rel=%.6g err=%s",
                      *cfg, r["rel"], r["err"][:120])

    valid = sorted((r for r in results if r["ok"]), key=lambda x:x["ms"])
    if not valid:
        _log.warning("[vdn-triton-v28] Q%d K%d H%d D%d | no valid Triton candidate; cuDNN=%.3fms | diagnostic-only",
                     QN,KN,H,D,cudnn_ms)
        return

    # Rebenchmark the best few with more repetitions to reduce ranking noise.
    reranked = []
    for base in valid[:min(3,len(valid))]:
        rr = evaluate(base["cfg"], final_reps)
        if rr["ok"]:
            reranked.append(rr)
    best = min(reranked or valid, key=lambda x:x["ms"])
    strategy,bm,bn,warps,stages = best["cfg"]
    speed = cudnn_ms/best["ms"] if best["ms"] > 0 else 0.0
    top_text = ", ".join(
        "%s/%dx%d/w%d/s%d=%.3fms" % (r["cfg"][0],r["cfg"][1],r["cfg"][2],r["cfg"][3],r["cfg"][4],r["ms"])
        for r in valid[:5]
    )
    _log.info("[vdn-triton-v28] Q%d K%d H%d D%d bf=%s | cuDNN=%.3fms BEST=%.3fms speedup=%.3fx | strat=%s BM=%d BN=%d warps=%d stages=%d | rel_rms=%.6g max_abs=%.6g tol=%g | tested=%d valid=%d | top5=[%s] | diagnostic-only",
              QN,KN,H,D,str(qb.dtype).replace("torch.",""),cudnn_ms,best["ms"],speed,
              strategy,bm,bn,warps,stages,best["rel"],best["max"],tol,len(results),len(valid),top_text)
