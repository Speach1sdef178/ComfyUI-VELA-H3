"""VDN-H3 linear-attention branch (Video Delta Attention) for ComfyUI's MiniMax-H3.

Port of the official release's BidirectionalLinearBranch
(github.com/OpenVDN/vdn-minimax-h3, src/models/linear_attention/) with the checkpoint
held as plain tensors instead of a module tree, so ComfyUI's model patcher stays the
sole owner of the diffusion model's parameter tree.

The released 8-step checkpoint configuration: delta_rule="vdn_solve", bridge="alpha",
a_fp32=True, enable_text_state=True, short_conv on (k, v), linear_head_dim=128.
Everything here is eager PyTorch -- no Triton, no torch.compile, no CUDA kernels.
Numerics follow the reference inference bodies: A statistics in fp32 (TF32 GEMM), the
recurrence in fp32 via preallocated banks, bf16 features and readout.
"""
import collections
import contextlib
import logging
import math
import os
import warnings

import torch
import torch.nn.functional as F

from .profiler import section as prof_section

_log = logging.getLogger("comfy.vdn")


_V32_SOLVER_LAB_DONE = False

def _env_on(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}

@contextlib.contextmanager
def _v32_delta_prof(name, block_index):
    if _env_on("VDN_H3_DELTA_DEEP_PROFILE"):
        with prof_section(name, block_index):
            yield
    else:
        yield

def _v32_solver_lab(chol, rhs_t, reference, block_index=None):
    global _V32_SOLVER_LAB_DONE
    if _V32_SOLVER_LAB_DONE or not _env_on("VDN_H3_DELTA_SOLVER_LAB"):
        return
    wanted = int(os.environ.get("VDN_H3_DELTA_SOLVER_LAB_BLOCK", "0"))
    if block_index is not None and int(block_index) != wanted:
        return
    _V32_SOLVER_LAB_DONE = True
    reps = max(1, int(os.environ.get("VDN_H3_DELTA_SOLVER_LAB_REPS", "2")))
    rtol = float(os.environ.get("VDN_H3_DELTA_SOLVER_LAB_RTOL", "0.0005"))
    try:
        def bench(fn):
            fn()
            torch.cuda.synchronize()
            st = torch.cuda.Event(enable_timing=True)
            en = torch.cuda.Event(enable_timing=True)
            st.record()
            out = None
            for _ in range(reps):
                out = fn()
            en.record(); en.synchronize()
            return out, st.elapsed_time(en) / reps

        ref_out, t_cholsolve = bench(lambda: torch.cholesky_solve(rhs_t, chol))

        def tri2():
            y = torch.linalg.solve_triangular(chol, rhs_t, upper=False, left=True)
            return torch.linalg.solve_triangular(chol.transpose(-1, -2), y, upper=True, left=True)
        tri_out, t_tri2 = bench(tri2)

        def err(x):
            d = x.float() - reference.float()
            denom = reference.float().square().mean().sqrt().clamp_min(1e-12)
            rel = (d.square().mean().sqrt() / denom).item()
            mx = d.abs().max().item()
            return rel, mx
        rel_c, max_c = err(ref_out)
        rel_t, max_t = err(tri_out)
        ok_c = int(rel_c <= rtol)
        ok_t = int(rel_t <= rtol)
        best = "cholesky_solve" if t_cholsolve <= t_tri2 else "triangular2"
        speed = max(t_cholsolve, t_tri2) / max(1e-9, min(t_cholsolve, t_tri2))
        _log.info("[vdn-delta-v32-lab] block=%s reps=%d rhs=%s | cholesky_solve=%.3fms(rel=%.6g,max=%.6g,ok=%d) | triangular2=%.3fms(rel=%.6g,max=%.6g,ok=%d) | best=%s pair_speed_ratio=%.3fx | diagnostic-only", str(block_index), reps, tuple(rhs_t.shape), t_cholsolve, rel_c, max_c, ok_c, t_tri2, rel_t, max_t, ok_t, best, speed)
    except Exception as e:
        _log.warning("[vdn-delta-v32-lab] failed; production direct solver unchanged: %s", e)



# v33: real-path cholesky_solve execution laboratory. Diagnostic only.
_V33_SOLVE_LAB_DONE = set()

def _v33_cuda_ms(fn):
    st = torch.cuda.Event(enable_timing=True)
    en = torch.cuda.Event(enable_timing=True)
    st.record()
    out = fn()
    en.record(); en.synchronize()
    return out, st.elapsed_time(en)

def _v33_solve_realpath_lab(chol, rhs_t, reference, block_index=None):
    if not _env_on("VDN_H3_DELTA_REALPATH_LAB") or block_index is None:
        return
    wanted = {int(x) for x in os.environ.get("VDN_H3_DELTA_REALPATH_LAB_BLOCKS", "0,12,24,36,48").split(",") if x.strip()}
    bi = int(block_index)
    if bi not in wanted or bi in _V33_SOLVE_LAB_DONE:
        return
    _V33_SOLVE_LAB_DONE.add(bi)
    rtol = float(os.environ.get("VDN_H3_DELTA_REALPATH_LAB_RTOL", "0.0005"))
    try:
        def meta(x):
            return "shape=%s stride=%s contig=%d dtype=%s" % (tuple(x.shape), tuple(x.stride()), int(x.is_contiguous()), str(x.dtype).replace("torch.", ""))
        # Synchronize before each isolated measurement so CUDA-event attribution cannot
        # inherit queued work from the surrounding real path. Full outputs are deleted
        # immediately; none are fed back into production.
        torch.cuda.synchronize()
        out1, raw1 = _v33_cuda_ms(lambda: torch.cholesky_solve(rhs_t, chol))
        out2, raw2 = _v33_cuda_ms(lambda: torch.cholesky_solve(rhs_t, chol))

        rhs_c, rhs_clone = _v33_cuda_ms(lambda: rhs_t.contiguous())
        chol_c, chol_clone = _v33_cuda_ms(lambda: chol.contiguous())
        out_rc, rhs_contig_ms = _v33_cuda_ms(lambda: torch.cholesky_solve(rhs_c, chol))
        out_cc, both_contig_ms = _v33_cuda_ms(lambda: torch.cholesky_solve(rhs_c, chol_c))

        # Clone forces fresh physical storage while preserving layout; this detects
        # pointer/workspace/alias-sensitive behavior distinct from contiguous() no-ops.
        rhs_clone_t, rhs_fresh_ms = _v33_cuda_ms(lambda: rhs_t.clone())
        chol_clone_t, chol_fresh_ms = _v33_cuda_ms(lambda: chol.clone())
        out_fresh, fresh_solve_ms = _v33_cuda_ms(lambda: torch.cholesky_solve(rhs_clone_t, chol_clone_t))

        def err(x):
            d=(x.float()-reference.float())
            den=reference.float().square().mean().sqrt().clamp_min(1e-12)
            return (d.square().mean().sqrt()/den).item(), d.abs().max().item()
        rel, mx = err(out2)
        relc, mxc = err(out_cc)
        relf, mxf = err(out_fresh)
        ok=int(max(rel,relc,relf) <= rtol)
        _log.info("[vdn-delta-v33-real] block=%02d | rhs{%s} chol{%s} | repeat1=%.3fms repeat2=%.3fms | contiguous_copy rhs=%.3fms chol=%.3fms solve_rhsC=%.3fms solve_bothC=%.3fms | fresh_clone rhs=%.3fms chol=%.3fms fresh_solve=%.3fms | rel(raw2)=%.6g rel(contig)=%.6g rel(fresh)=%.6g max=%.6g ok=%d | diagnostic-only", bi, meta(rhs_t), meta(chol), raw1, raw2, rhs_clone, chol_clone, rhs_contig_ms, both_contig_ms, rhs_fresh_ms, chol_fresh_ms, fresh_solve_ms, rel, relc, relf, max(mx,mxc,mxf), ok)
        del out1,out2,out_rc,out_cc,out_fresh,rhs_c,chol_c,rhs_clone_t,chol_clone_t
    except Exception as e:
        _log.warning("[vdn-delta-v33-real] block=%s failed; production unchanged: %s", str(block_index), e)

# ---------------------------------------------------------------- delta rules --

class VdnDelta:
    """Exact VDN solve for S_out = (S_in Diag(alpha) + B)(I + A)^-1.

    v3 adds an Ampere experiment selected with VDN_H3_DELTA_SOLVE:
      reference  - original inverse-building path from the portable port.
      direct     - never materializes (I+A)^-1. It solves both required right-hand
                   products in one batched Cholesky solve. The algebra is exact:
                   R M^-1 = solve(M, R^T)^T for SPD M = I+A.

    The default remains ``reference`` so merely installing this profiled build does
    not alter released-model numerics. Set VDN_H3_DELTA_SOLVE=direct for the test.
    """

    def __init__(self, tokens_per_frame=None):
        mode = os.environ.get("VDN_H3_DELTA_SOLVE", "reference").strip().lower()
        aliases = {"ref": "reference", "inverse": "reference",
                   "chol": "direct", "cholesky": "direct",
                   "direct_cholesky_solve": "direct"}
        mode = aliases.get(mode, mode)
        if mode not in {"reference", "direct"}:
            _log.warning("[vdn] unknown VDN_H3_DELTA_SOLVE=%r; using reference", mode)
            mode = "reference"
        self.solve_mode = mode

    def _reference(self, alpha, a_raw, b_raw, retain):
        a32 = a_raw.float()
        eye = torch.eye(a32.shape[-1], device=a32.device,
                        dtype=torch.float32).expand_as(a32)
        if retain:
            scratch = _delta_scratch(a32.shape, a32.device)
        else:
            scratch = torch.empty(a32.shape, dtype=torch.float32,
                                  device=a32.device)
        torch.add(a32, eye, out=scratch)
        chol = torch.linalg.cholesky(scratch)
        linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True,
                                             out=scratch)
        del chol
        inv = linv.transpose(-1, -2) @ linv
        del linv
        transition = alpha.unsqueeze(-1) * inv
        injection = b_raw.float() @ inv
        del inv
        return transition.to(a_raw.dtype), injection.to(b_raw.dtype)

    def _direct(self, alpha, a_raw, b_raw, retain, block_index=None):
        # Exact same operator without explicitly constructing M^-1.  Let
        # M = I + A and R = [Diag(alpha); B].  Since M is symmetric SPD,
        # R M^-1 = solve(M, R^T)^T.  One batched cholesky_solve handles both
        # transition and injection RHS at once.
        with _v32_delta_prof("deep_delta_detail_prepare", block_index):
            a32 = a_raw.float()
            d = a32.shape[-1]
            eye = torch.eye(d, device=a32.device, dtype=torch.float32).expand_as(a32)
            if retain:
                scratch = _delta_scratch(a32.shape, a32.device)
            else:
                scratch = torch.empty(a32.shape, dtype=torch.float32, device=a32.device)
            torch.add(a32, eye, out=scratch)

        with _v32_delta_prof("deep_delta_detail_cholesky", block_index):
            chol = torch.linalg.cholesky(scratch)

        with _v32_delta_prof("deep_delta_detail_rhs_build", block_index):
            alpha_diag = torch.diag_embed(alpha.float())
            rhs_rows = torch.cat((alpha_diag, b_raw.float()), dim=-2)
            rhs_t = rhs_rows.transpose(-1, -2).contiguous()

        with _v32_delta_prof("deep_delta_detail_factored_solve", block_index):
            solved_t = torch.cholesky_solve(rhs_t, chol)
        solved_rows = solved_t.transpose(-1, -2)

        _v33_solve_realpath_lab(chol, rhs_t, solved_t, block_index=block_index)
        _v32_solver_lab(chol, rhs_t, solved_t, block_index=block_index)

        with _v32_delta_prof("deep_delta_detail_split_cast", block_index):
            transition = solved_rows[..., :d, :].to(a_raw.dtype)
            injection = solved_rows[..., d:, :].to(b_raw.dtype)
        return transition, injection

    def factor_apply(self, alpha, a_raw, b_raw, retain=True, block_index=None):
        if self.solve_mode == "direct":
            return self._direct(alpha, a_raw, b_raw, retain, block_index=block_index)
        return self._reference(alpha, a_raw, b_raw, retain)


class SanaDelta:
    """Scaled subtractive delta: S_out = (S_in Diag(D))(I - c^2 A) + c B."""

    def __init__(self, tokens_per_frame):
        self.inv_tokens = 1.0 / tokens_per_frame
        self.inv_sqrt_tokens = self.inv_tokens ** 0.5

    def factor_apply(self, alpha, a_raw, b_raw, retain=True, block_index=None):
        eye = torch.eye(a_raw.shape[-1], device=a_raw.device, dtype=a_raw.dtype)
        transition = alpha.unsqueeze(-1) * (eye - self.inv_tokens * a_raw)
        injection = self.inv_sqrt_tokens * b_raw
        return transition, injection


class VdnScaledDelta(VdnDelta):
    """Exact joint solve WITH SANA's key scaling:
    S_out = (S_in Diag(D) + cB)(I + c^2 A)^-1, c = 1/sqrt(S).

    A control arm, kept for interpretability, not to train with: once c^2 = 1/S
    forces trace(c^2 A) <= 1, the exact inverse and the first-order truncation
    (I - c^2 A) are very nearly the same operator. spec.py accepts the rule, so
    a checkpoint that names it must find it here."""

    def __init__(self, tokens_per_frame):
        super().__init__(tokens_per_frame)
        self.inv_tokens = 1.0 / tokens_per_frame              # c^2
        self.inv_sqrt_tokens = self.inv_tokens ** 0.5         # c

    def factor_apply(self, alpha, a_raw, b_raw, retain=True, block_index=None):
        a32 = a_raw.float() * self.inv_tokens
        eye = torch.eye(a32.shape[-1], device=a32.device,
                        dtype=torch.float32).expand_as(a32)
        chol = torch.linalg.cholesky(a32 + eye)
        inv = torch.cholesky_solve(eye.contiguous(), chol)
        transition = alpha.unsqueeze(-1) * inv
        injection = (b_raw.float() * self.inv_sqrt_tokens) @ inv
        return transition.to(a_raw.dtype), injection.to(b_raw.dtype)


DELTA_BACKENDS = {"vdn_solve": VdnDelta, "sana_scaled": SanaDelta,
                  "vdn_scaled": VdnScaledDelta}

TEXT_STATE_SCALE = 0.5

MAX_DELTA_SCRATCH = 4
_DELTA_SCRATCH = collections.OrderedDict()


def _delta_scratch(shape, device):
    """ONE reusable [F, H, d, d] fp32 scratch for the delta-rule Cholesky solve
    (I+A sum and the triangular solve rotate through it), instead of allocating
    the intermediates per block per step. fp32 always -- lifetime reuse only."""
    key = (tuple(shape), str(device))
    hit = _DELTA_SCRATCH.get(key)
    if hit is None:
        hit = torch.empty(shape, dtype=torch.float32, device=device)
        _DELTA_SCRATCH[key] = hit
        while len(_DELTA_SCRATCH) > MAX_DELTA_SCRATCH:
            _DELTA_SCRATCH.popitem(last=False)
    else:
        _DELTA_SCRATCH.move_to_end(key)
    return hit


# ------------------------------------------------------------- frame statistics --

def _tf32_matmul():
    class _Ctx:
        def __enter__(self):
            self.prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        def __exit__(self, *a):
            torch.backends.cuda.matmul.allow_tf32 = self.prev
    return _Ctx()


_STATISTICS_WORKSPACE_BYTES = 1 << 30


def frame_statistics(kf, vf, beta, a_fp32=True):
    """Prepare independent per-frame statistics in bounded batches.

    The original full-clip path temporarily materialized several FP32 copies of
    every frame.  Long clips can exceed the 24 GB budget even though the final A/B
    statistics fit.  Batching complete frames preserves the exact token reduction
    and dtypes while bounding those temporary allocations to roughly 1 GiB.
    """
    frames, heads, tokens, dim = kf.shape
    # K repack, FP32 K and weighted K, plus the weighted-V multiply/repack.
    per_frame = heads * tokens * (
        dim * (kf.element_size() + (8 if a_fp32 else kf.element_size()))
        + 2 * vf.shape[-1] * vf.element_size())
    batch = max(1, _STATISTICS_WORKSPACE_BYTES // max(1, per_frame))
    if frames <= batch:
        return _frame_statistics_chunk(kf, vf, beta, a_fp32)

    a = torch.empty((frames, heads, dim, dim), device=kf.device,
                    dtype=torch.float32)
    b = torch.empty((frames, heads, vf.shape[-1], dim), device=kf.device,
                    dtype=torch.float32)
    for start in range(0, frames, batch):
        stop = min(start + batch, frames)
        ac, bc = _frame_statistics_chunk(
            kf[start:stop], vf[start:stop], beta[start:stop], a_fp32)
        a[start:stop].copy_(ac)
        b[start:stop].copy_(bc)
        del ac, bc
    return a, b


def _frame_statistics_chunk(kf, vf, beta, a_fp32=True):
    """A[f,h,k,l] = sum_s k beta k,  B[f,h,v,k] = sum_s v beta k, over one chunk's
    rows. A in fp32 (bf16's 8 mantissa bits break the conditioning I+A needs), B left
    in bf16 for the tensor-core GEMM and promoted on the store. Operates with autocast
    off implicitly -- callers run under inference no_grad, no ambient autocast."""
    with torch.autocast(device_type=kf.device.type, enabled=False):
        kf16 = kf.contiguous()
        vb = (vf * beta.unsqueeze(-1).to(vf.dtype)).contiguous()
        if a_fp32:
            kf32 = kf16.float()
            scaled32 = (kf32 * beta.unsqueeze(-1).float()).contiguous()
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                a = torch.matmul(scaled32.transpose(-1, -2), kf32)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
        else:
            a = torch.matmul((kf * beta.unsqueeze(-1).to(kf.dtype)).contiguous()
                             .transpose(-1, -2), kf).float()
        a = 0.5 * (a + a.transpose(-1, -2))
        b = torch.matmul(vb.transpose(-1, -2), kf16).float()
        return a, b


# ------------------------------------------------------- compile small helpers --

_COMPILED_CACHE = {}
_COMPILED_BROKEN = set()


def _run_compiled(key, body, *args, _mode=None, **kwargs):
    """torch.compile(body, dynamic=False), built once per key, with a permanent
    eager fallback on failure -- the same policy linear_epilogue already uses:
    same math, one rounding at the store instead of one per op, just slower."""
    if key in _COMPILED_BROKEN:
        return body(*args, **kwargs)
    try:
        if key not in _COMPILED_CACHE:
            _COMPILED_CACHE[key] = torch.compile(body, dynamic=False, mode=_mode)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _COMPILED_CACHE[key](*args, **kwargs)
    except Exception as e:
        _COMPILED_BROKEN.add(key)
        _log.debug("[vdn] compile of %s failed (%s); using eager", key, e)
        return body(*args, **kwargs)


# ---------------------------------------------------------------------- scans --

MAX_SCAN_BANKS = 4
_SCAN_BANKS = collections.OrderedDict()


def _scan_banks(num_frames, state_shape, dtype, device, retain=True):
    """The prefix/suffix state banks. Retained mode: allocated ONCE per
    (F, H, dv, dk, device, dtype) and reused across blocks and steps (2xF
    baddbmm `out=` targets no longer churn the allocator). Transient mode (VRAM
    pressure): fresh per call -- the v1.3.1 pattern. LRU-capped; every call
    fully rewrites both banks."""
    if not retain:
        prefix = torch.empty((num_frames, *state_shape), dtype=dtype,
                             device=device)
        return prefix, torch.empty_like(prefix)
    key = (num_frames, tuple(state_shape), str(device), dtype)
    hit = _SCAN_BANKS.get(key)
    if hit is None:
        prefix = torch.empty((num_frames, *state_shape), dtype=dtype, device=device)
        hit = (prefix, torch.empty_like(prefix))
        _SCAN_BANKS[key] = hit
        while len(_SCAN_BANKS) > MAX_SCAN_BANKS:
            _SCAN_BANKS.popitem(last=False)
    else:
        _SCAN_BANKS.move_to_end(key)
    return hit


def clear_scan_banks():
    """Drop the reused scan banks and delta-solve scratch (and let compiled
    variants go) so a cancelled run's buffers don't pin VRAM into the next one."""
    _SCAN_BANKS.clear()
    _DELTA_SCRATCH.clear()
    for key in [k for k in _COMPILED_CACHE if isinstance(k, tuple) and k[:1] == ("scan",)]:
        _COMPILED_CACHE.pop(key, None)


def _scan_body(transitions, injections, start):
    """The two recurrence sweeps as one functional body (no out= aliases), so
    fast_kernels can hand the whole scan to torch.compile(reduce-overhead):
    2xF per-frame baddbmm kernel launches become one CUDA-graph replay. Same
    math as the eager loop; under cudagraphs the returned banks live in the
    graph pool and are valid until the next compiled call of the same shape --
    the gather consumes them before that can happen."""
    num_frames = transitions.shape[0]
    prefix = torch.empty((num_frames, *start.shape), dtype=injections.dtype,
                         device=injections.device)
    suffix = torch.empty_like(prefix)
    state = start
    for frame in range(num_frames):
        state = torch.baddbmm(injections[frame], state, transitions[frame])
        prefix[frame] = state
    state = start
    for frame in range(num_frames - 1, -1, -1):
        state = torch.baddbmm(injections[frame], state, transitions[frame])
        suffix[frame] = state
    return prefix, suffix


def run_scans(backend, alpha, a_raw, b_raw, text_state=None, fuse=False,
              retain=True, block_index=None):
    """Forward/reverse state banks; the plain linear recurrence
    state_t = state_{t-1} @ transition_t + injection_t, one baddbmm per frame.

    Eager writes into the reused banks (retained mode) or fresh per-call banks
    (transient, VRAM pressure); fuse=True (fast_kernels) runs the scan as one
    reduce-overhead compiled graph instead (latch-to-eager on failure)."""
    with torch.autocast(device_type=a_raw.device.type, enabled=False):
        with prof_section("deep_delta_solve", block_index):
            transitions, injections = backend.factor_apply(alpha, a_raw, b_raw,
                                                           retain=retain, block_index=block_index)
        num_frames = transitions.shape[0]
        start = (torch.zeros_like(injections[0]) if text_state is None
                 else text_state.to(injections.dtype))
        with prof_section("deep_scan_recurrence", block_index):
            if fuse:
                key = ("scan", num_frames, *start.shape, str(injections.device),
                       str(injections.dtype))
                return _run_compiled(key, _scan_body, transitions, injections, start,
                                     _mode="reduce-overhead")
            prefix, suffix = _scan_banks(num_frames, start.shape, injections.dtype,
                                         injections.device, retain=retain)
            state = start
            for frame in range(num_frames):
                torch.baddbmm(injections[frame], state, transitions[frame],
                              out=prefix[frame])
                state = prefix[frame]
            state = start
            for frame in range(num_frames - 1, -1, -1):
                torch.baddbmm(injections[frame], state, transitions[frame],
                              out=suffix[frame])
                state = suffix[frame]
            return prefix, suffix


MAX_CACHED_GATHERS = 64
_GATHER_INDEX_CACHE = collections.OrderedDict()


def gather_indices(bounds, num_frames, device):
    key = (tuple(bounds), num_frames, str(device))
    hit = _GATHER_INDEX_CACHE.get(key)
    if hit is not None:
        _GATHER_INDEX_CACHE.move_to_end(key)
        return hit
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    hit = dict(
        before_idx=last_before.clamp(min=0),
        after_idx=first_after.clamp(max=num_frames - 1),
        has_before=(last_before >= 0),
        has_after=(first_after < num_frames),
        bridge_before=(last_before + 1).clamp(min=0),
        bridge_after=first_after.clamp(max=num_frames),
        frames=torch.arange(num_frames, device=device),
    )
    _GATHER_INDEX_CACHE[key] = hit
    while len(_GATHER_INDEX_CACHE) > MAX_CACHED_GATHERS:
        _GATHER_INDEX_CACHE.popitem(last=False)
    return hit


def _gather_body(prefix_states, suffix_states, alpha, text_state, bridge_alpha,
                 out_dtype, before_idx, after_idx, has_before, has_after,
                 bridge_before, bridge_after, frames):
    """The arithmetic of gather_linear_state, with the index tensors already built.

    Split out so fast_kernels can hand the whole thing to one compiled kernel:
    eager it is two gathers, two wheres, two multiplies and a combine over the
    fp32 state bank -- seven passes for what is one read of each side and one
    store."""
    state_before = prefix_states[before_idx]
    state_after = suffix_states[after_idx]
    if text_state is not None:
        text_state = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before,
                                   text_state)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after,
                                  text_state)
    if bridge_alpha:
        log_alpha = torch.log(alpha.clamp_min(1e-12))
        log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
        alpha_from_before = torch.exp(
            log_prefix[frames + 1] - log_prefix[bridge_before])
        alpha_from_after = torch.exp(
            log_prefix[bridge_after] - log_prefix[frames])
        # alpha is per KEY channel: broadcast over d_v, not d_k
        state_before = state_before * alpha_from_before.unsqueeze(2)
        state_after = state_after * alpha_from_after.unsqueeze(2)
    if text_state is not None:
        out = state_before + state_after
    else:
        out = (state_before * has_before.view(-1, 1, 1, 1)
               + state_after * has_after.view(-1, 1, 1, 1))
    return out if out_dtype is None else out.to(out_dtype)


def gather_linear_state(prefix_states, suffix_states, alpha, bounds, bridge="alpha",
                        text_state=None, out_dtype=None, fuse=False):
    """The state of everything OUTSIDE the softmax window, in the query frame's frame
    of reference: prefix_states[lo-1] + suffix_states[hi+1], decayed in by the product
    of alpha over the window span (bridge="alpha"), with the scan start (the text
    state, when given) read by out-of-range sides.

    fuse=True (fast_kernels) runs the arithmetic as one compiled kernel, keyed on
    (bridge, text_state?, out_dtype); same math, rounded once at the store."""
    assert bridge in ("alpha", "none")
    num_frames = prefix_states.shape[0]
    idx = gather_indices(bounds, num_frames, prefix_states.device)
    if not fuse:
        return _gather_body(prefix_states, suffix_states, alpha, text_state,
                            bridge == "alpha", out_dtype, **idx)
    key = ("gather", bridge, text_state is not None, str(out_dtype))
    return _run_compiled(key, _gather_body, prefix_states, suffix_states, alpha,
                         text_state, bridge == "alpha", out_dtype, **idx)


# -------------------------------------------------------------------- features --

def _activate(tokens, l2norm):
    x = F.silu(tokens)
    if l2norm:
        return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype)
    return x


def _activate_fhsd_body(tokens, l2norm, num_frames, per_frame):
    """_activate storing q frame-major, [F, H, S, d] instead of [F*S, H, d].

    The readout below is a frame-major batched matmul; storing q this way (the
    official inference body) means the matmul consumes it without a permute-in
    copy. Only pays off compiled, where the strided store rides the activation
    kernel for free -- so callers gate it behind fast_kernels."""
    x = _activate(tokens, l2norm)
    heads, dim = x.shape[-2], x.shape[-1]
    return x.view(num_frames, per_frame, heads, dim).permute(0, 2, 1, 3).contiguous()


def _temporal_shift(x, w, kernel):
    """Depthwise k-tap conv over frames as shift-multiply-add. x [F, S, C]; w [C, k];
    zero-padded, symmetric."""
    pad = kernel // 2
    xp = F.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(kernel):
        part = xp[dt:dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    return out


def conv_features(tokens, sp_weight, tm_weight, num_frames, frame_size, l2norm):
    """Separable short conv: depthwise 5x5 spatial per frame (cudnn NHWC via a
    channels-last view), then the 5-tap temporal shift, then SiLU [+ L2Norm]."""
    heads, head_dim = tokens.shape[-2], tokens.shape[-1]
    grid_h, grid_w = frame_size
    channels = heads * head_dim
    volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
    volume = F.conv2d(volume, sp_weight, padding=2, groups=channels)
    x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
    tm = tm_weight.squeeze(1)                     # Conv1d [C, 1, K] -> [C, K]
    out = _temporal_shift(x, tm.to(x.dtype), tm.shape[-1])
    return _activate(out.reshape(-1, heads, head_dim), l2norm)


def alpha_gate(frame_mean, w_down, w_up, dt_bias, a_log, num_heads, head_dim):
    """alpha_t = exp(-exp(A_log) * softplus(delta + dt_bias)) per frame/head/channel,
    KDA's double-exponential gate in fla layout. fp32 throughout."""
    with torch.autocast(device_type=frame_mean.device.type, enabled=False):
        delta = F.linear(frame_mean.float(), w_down.float())
        delta = F.linear(delta, w_up.float())
        delta = delta + dt_bias.float()
        scale = torch.exp(a_log.float())[:, None]
        delta = delta.view(-1, num_heads, head_dim)
        return torch.exp(-scale * F.softplus(delta.float()))


def rms_norm(x, weight, eps):
    """Weighted RMSNorm with fp32 second-moment accumulation (vector_norm spelling)."""
    ms = torch.linalg.vector_norm(
        x, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / x.shape[-1]
    return x * torch.rsqrt(ms + eps).to(x.dtype) * weight.to(x.dtype)


def _linear_epilogue_body(readout_fhsd, norm_weight, gate, eps):
    """RMSNorm + output gate over a readout still in [F, H, S, d], with the transpose
    back to token order folded into the store."""
    ms = torch.linalg.vector_norm(
        readout_fhsd, dim=-1, keepdim=True, dtype=torch.float32).pow(2) \
        / readout_fhsd.shape[-1]
    normed = readout_fhsd * torch.rsqrt(ms + eps).to(readout_fhsd.dtype) \
        * norm_weight.to(readout_fhsd.dtype)
    frames, heads, per_frame, dim = normed.shape
    rows = frames * per_frame
    return (normed.permute(0, 2, 1, 3).reshape(rows, heads * dim)
            * gate.reshape(rows, heads * dim))


_EPILOGUE_FUSED = None
_EPILOGUE_FUSED_BROKEN = False


def linear_epilogue(readout_fhsd, norm_weight, gate, eps, fuse=False):
    """RMSNorm + output gate, optionally under torch.compile. Eager this walks the
    full readout several times (norm, rsqrt, two multiplies, the gated store); the
    fused variant is one inductor kernel. Compilation failure falls back to eager
    permanently (same math, just slower)."""
    global _EPILOGUE_FUSED, _EPILOGUE_FUSED_BROKEN
    if fuse and not _EPILOGUE_FUSED_BROKEN:
        try:
            if _EPILOGUE_FUSED is None:
                _EPILOGUE_FUSED = torch.compile(_linear_epilogue_body)
            return _EPILOGUE_FUSED(readout_fhsd, norm_weight, gate, eps)
        except Exception as e:
            _EPILOGUE_FUSED_BROKEN = True
            _log.warning("[vdn] fused epilogue compile failed (%s); using eager", e)
    return _linear_epilogue_body(readout_fhsd, norm_weight, gate, eps)


# ---------------------------------------------------------------- the branch --

class LinearBranch:
    """The checkpoint-backed linear-attention branch for ONE transformer block.

    Weights are plain CPU tensors under `w` (checkpoint keys minus the per-block
    prefix). Call `readout(...)` inside the block's attention forward; it consumes the
    raw (pre-QK-norm, pre-RoPE) q/k/v of the video rows and the hidden states, and
    returns the gated readout [video_rows, H*d_linear] pre-to_out_linear.
    """

    def __init__(self, w, num_heads, head_dim, delta_rule="vdn_solve", bridge="alpha",
                 a_fp32=True, short_conv=("k", "v"), enable_text_state=True,
                 retain_buffers=True):
        self.w = w
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bridge = bridge
        self.a_fp32 = a_fp32
        self.short_conv = tuple(short_conv) or None
        self.enable_text_state = enable_text_state
        self.delta_rule = delta_rule
        self.fuse_epilogue = False
        self.retain_buffers = retain_buffers
        self._backend = None
        self._backend_key = None

    def _features(self, w, q_raw, k_raw, v_raw, num_frames, frame_size, q_fhsd=False, block_index=None):
        """[ShortConv ->] SiLU [-> L2Norm for q/k]. NoPE: the branch consumes raw
        pre-RoPE features. q_fhsd (fast_kernels) stores q frame-major [F, H, S, d]
        straight out of the fused activation; n/a when q itself is convolved."""
        conv = self.short_conv
        with prof_section("deep_features_query", block_index):
            if q_fhsd and not (conv and "q" in conv):
                query = _run_compiled(("act_fhsd", True), _activate_fhsd_body, q_raw,
                                      True, num_frames, q_raw.shape[0] // num_frames)
            else:
                query = _activate(q_raw, l2norm=True)
        with prof_section("deep_short_conv_k", block_index):
            if conv and "k" in conv:
                key = conv_features(k_raw, w["short_conv.k_sp.weight"],
                                    w["short_conv.k_tm.weight"], num_frames, frame_size,
                                    l2norm=True)
            else:
                key = _activate(k_raw, l2norm=True)
        with prof_section("deep_short_conv_v", block_index):
            if conv and "v" in conv:
                value = conv_features(v_raw, w["short_conv.v_sp.weight"],
                                      w["short_conv.v_tm.weight"], num_frames, frame_size,
                                      l2norm=False)
            else:
                value = _activate(v_raw, l2norm=False)
        return query, key, value

    def _delta_backend(self, tokens_per_frame):
        key = (self.delta_rule, tokens_per_frame)
        if self._backend is None or self._backend_key != key:
            self._backend = DELTA_BACKENDS[self.delta_rule](tokens_per_frame)
            self._backend_key = key
        return self._backend

    def _text_state(self, w, text_x, text_k_raw, text_v_raw):
        """TEXT_STATE_SCALE * S_text: the whole prompt written into a zero state as ONE
        delta-rule chunk; both directional scans start from it."""
        if not self.enable_text_state or text_x is None:
            return None
        length = text_x.shape[0]
        n_heads, head_dim = self.num_heads, self.head_dim
        key = _activate(text_k_raw, l2norm=True)
        value = _activate(text_v_raw, l2norm=False)
        key = key.view(1, length, n_heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, n_heads, head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(F.linear(text_x, w["beta_proj.weight"]))
        beta = beta.view(1, length, n_heads).permute(0, 2, 1)
        a, b = frame_statistics(key, value, beta, a_fp32=self.a_fp32)
        backend = self._delta_backend(length)
        with torch.autocast(device_type=a.device.type, enabled=False):
            ones = torch.ones(1, n_heads, head_dim, device=a.device, dtype=a.dtype)
            _, injection = backend.factor_apply(ones, a, b,
                                            retain=self.retain_buffers)
        return TEXT_STATE_SCALE * injection[0]

    def readout(self, w, xv, q_raw, k_raw, v_raw, num_frames, tokens_per_frame,
                bounds, frame_size=None, text_x=None, text_k_raw=None,
                text_v_raw=None, skip_ends=False, block_index=None):
        """Everything the softmax window cannot see, summarised for every video row.

        w: the branch weights, already moved to the activations' device/dtype (see
        VDNState.weights_on). xv: [video_rows, hidden]; q/k/v_raw: [video_rows, H, d]
        raw features. bounds: per-frame inclusive window [lo, hi]. Returns
        [video_rows, H*d_linear] (gated + normalised; the caller adds
        to_out_linear(...) into the attention output's video rows).
        """
        n_heads, head_dim = self.num_heads, self.head_dim
        ref = xv

        if skip_ends:
            if num_frames <= 2:
                return ref.new_zeros(num_frames * tokens_per_frame, n_heads * head_dim)
            inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
            readout = self._readout(
                w, xv[inner] if xv is not None else None,
                tuple(t[inner] for t in (q_raw, k_raw, v_raw)),
                num_frames - 2, tokens_per_frame,
                [(lo - 1, hi - 1) for lo, hi in bounds[1:num_frames - 1]],
                frame_size, text_x, text_k_raw, text_v_raw, block_index)
            out = readout.new_empty(num_frames * tokens_per_frame, readout.shape[-1])
            out[:tokens_per_frame].zero_()
            out[(num_frames - 1) * tokens_per_frame:].zero_()
            out[inner] = readout
            return out
        return self._readout(w, xv, (q_raw, k_raw, v_raw), num_frames,
                             tokens_per_frame, bounds, frame_size, text_x,
                             text_k_raw, text_v_raw, block_index)

    def _readout(self, w, xv, qkv_raw, num_frames, tokens_per_frame, bounds,
                 frame_size, text_x, text_k_raw, text_v_raw, block_index):
        n_heads, head_dim = self.num_heads, self.head_dim
        num_tokens = num_frames * tokens_per_frame
        backend = self._delta_backend(tokens_per_frame)
        shape = (num_frames, tokens_per_frame, n_heads, head_dim)

        query, key, value = self._features(w, *qkv_raw, num_frames, frame_size,
                                           q_fhsd=self.fuse_epilogue,
                                           block_index=block_index)
        key_by_frame = key.view(shape).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape).permute(0, 2, 1, 3)
        with prof_section("deep_beta_proj", block_index):
            beta = torch.sigmoid(F.linear(xv, w["beta_proj.weight"]))
            beta = beta.view(num_frames, tokens_per_frame, n_heads).permute(0, 2, 1)

        with prof_section("deep_frame_statistics", block_index):
            a, b = frame_statistics(key_by_frame, value_by_frame, beta, a_fp32=self.a_fp32)

        with prof_section("deep_alpha_gate", block_index):
            frame_mean = xv.view(num_frames, tokens_per_frame, -1).mean(
                dim=1, dtype=torch.float32)
            alpha = alpha_gate(frame_mean, w["alpha.down.weight"], w["alpha.up.weight"],
                               w["alpha.dt_bias"], w["alpha.A_log"], n_heads, head_dim)

        with prof_section("deep_text_state", block_index):
            text_state = self._text_state(w, text_x, text_k_raw, text_v_raw)
        prefix_states, suffix_states = run_scans(backend, alpha, a, b,
                                                 text_state=text_state,
                                                 fuse=self.fuse_epilogue,
                                                 retain=self.retain_buffers,
                                                 block_index=block_index)
        with prof_section("deep_output_gate", block_index):
            gate = torch.sigmoid(F.linear(xv, w["output_gate.down.weight"])
                                 @ w["output_gate.up.weight"].T
                                 + w["output_gate.up.bias"])
        with prof_section("deep_state_gather", block_index):
            linear_state = gather_linear_state(
                prefix_states, suffix_states, alpha, bounds, bridge=self.bridge,
                text_state=text_state, out_dtype=gate.dtype, fuse=self.fuse_epilogue)
        del prefix_states, suffix_states

        if query.dim() == 4:
            query_fhsd = query
        else:
            query_fhsd = query.view(shape).permute(0, 2, 1, 3)
        with prof_section("deep_readout_matmul", block_index):
            readout = torch.matmul(query_fhsd, linear_state.transpose(-1, -2))
        with prof_section("deep_epilogue", block_index):
            return linear_epilogue(readout, w["norm.weight"], gate,
                                   w["norm.weight"].new_tensor(1e-6).item(),
                                   fuse=self.fuse_epilogue)
