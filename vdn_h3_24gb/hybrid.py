"""VDN-H3 hybrid attention for ComfyUI's MiniMax-H3: the integration layer.

Replaces each DiT block's Attention.forward on a model CLONE (object patch, the same
mechanism the MiniMax-H3-Turbo node uses) with the official hybrid:

    softmax_out = window_softmax(roped q, k, v)          # local frames + globals
    out         = out_proj(softmax_gate(x) * softmax_out)
    out[video] += to_out_linear(branch(video rows))       # everything the window can't see

The base QKV projection, QK-norm and RoPE are reused verbatim from
comfy/ldm/minimax/model.py (the checkpoint's own weights), and the linear branch
consumes the raw pre-norm pre-RoPE q/k/v exactly like the official HybridAttention.

Per-forward packed-sequence geometry (video span, frame grid, text span) is published
by a DIFFUSION_MODEL wrapper that reads the payload's PackedLayout -- the same object
the model itself consumes.
"""
import gc
import contextlib
import logging
import os
import time

import torch
import torch.nn.functional as F

import comfy.ldm.minimax.model as minimax_model
import comfy.cli_args
import comfy.model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention
from comfy.patcher_extension import WrappersMP

from vdn_h3_24gb.branch import LinearBranch
from vdn_h3_24gb.window import full_coverage, window_bounds
from vdn_h3_24gb.profiler import (
    PROFILER, section as prof_section, wall_section as prof_wall_section,
    memory_snapshot as prof_memory_snapshot,
    importance_sample as prof_importance_sample,
    window_importance_sample as prof_window_importance_sample,
    qkv_deep_profile_enabled, reuse_sample as prof_reuse_sample,
)

_log = logging.getLogger("comfy.vdn")
_seen = set()


def _once(key, message):
    if key not in _seen:
        _seen.add(key)
        _log.info(f"[vdn] {message}")


class _StreamPrefetcher:
    """Safe one-block lookahead for branch_weights="stream".

    v7.1 deliberately avoids a background Python thread.  ComfyUI's lazy /
    disk-backed QuantizedTensor resolve path is not thread-safe with CUDA on
    Windows and can trigger an asynchronous illegal-memory-access while the
    worker thread is materializing a tensor.

    The next block is therefore resolved on the *main Python thread* but inside
    a dedicated CUDA stream.  CPU-side page-cache / tensor resolution happens
    serially, while any asynchronous H2D work queued by PyTorch may still
    overlap the current block's later GPU work.  The consumer waits on an event
    before using the prefetched tensors.  Extra residency remains one block.
    """

    def __init__(self):
        self._done = {}
        self._stream = None

    def _record(self, t, stream):
        """Keep every prefetched storage alive through the consumer stream.

        cudaMallocAsync still requires cross-stream lifetime tracking.  Its
        warning concerns recording a tensor's original allocation stream, not
        this prefetch-to-consumer handoff.
        """
        inner = getattr(t, "_qdata", None)
        seen = [inner if inner is not None else t]
        params = getattr(t, "_params", None)
        for name in ("scale", "orig_weight", "bias"):
            sub = getattr(params, name, None)
            if isinstance(sub, torch.Tensor):
                seen.append(sub)
        for x in seen:
            x.record_stream(stream)

    def request(self, index, fetch):
        if index in self._done:
            return
        try:
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            # Important: resolve() stays on the main Python thread.  Only the
            # CUDA work it launches is assigned to the side stream.
            with torch.cuda.stream(self._stream):
                w = fetch()
                ev = torch.cuda.Event()
                ev.record(self._stream)
            self._done[index] = (w, ev)
        except Exception as e:
            _log.warning("[vdn] branch prefetch failed (%s); the consumer "
                         "will read synchronously", e)

    def take(self, index):
        hit = self._done.pop(index, None)
        if hit is None:
            return None
        w, ev = hit
        cur = torch.cuda.current_stream()
        cur.wait_event(ev)
        for t in w.values():
            self._record(t, cur)
        return w

    def reset(self):
        # If an interrupt happens while a side-stream copy is outstanding, wait
        # before dropping the last Python references to those tensors.
        if self._stream is not None:
            try:
                self._stream.synchronize()
            except Exception:
                pass
        self._done.clear()


class VDNLayout:
    """Published once per forward: the packed-sequence geometry the branches need."""

    __slots__ = ("video_start", "video_end", "num_frames", "tokens_per_frame",
             "frame_size", "text_start", "text_len", "bounds", "full_cover",
             "seq_len", "anchor_frames", "window_inner_trim", "segments")

    def __init__(self, video_start, video_end, num_frames, tokens_per_frame,
             frame_size, text_start, text_len, seq_len, radius, chunk,
             anchor_frames, segments=None):
        self.video_start = video_start
        self.video_end = video_end
        self.num_frames = num_frames
        self.tokens_per_frame = tokens_per_frame
        self.frame_size = frame_size
        self.text_start = text_start
        self.text_len = text_len
        self.seq_len = seq_len
        self.segments = tuple(segments or ())
        self.bounds = window_bounds(num_frames, radius, chunk)

        # v16: conservative exact-window shrink.  We keep exact SDPA in every
        # transformer block and preserve the trained chunk partition.  Only
        # fully-interior chunk windows are shortened; boundary windows (which
        # already have fewer visible frames after clamping) are left untouched.
        # This avoids the severe distribution shift of v14/v15 while reducing
        # K/V rows for the expensive interior window groups.
        raw_trim = os.environ.get("VDN_H3_WINDOW_INNER_TRIM", "0").strip()
        try:
            inner_trim = max(0, int(raw_trim))
        except Exception:
            inner_trim = 0
        self.window_inner_trim = inner_trim
        if inner_trim > 0 and not full_coverage(self.bounds, num_frames):
            left_trim = inner_trim // 2
            right_trim = inner_trim - left_trim
            shrunk = []
            for lo, hi in self.bounds:
                # A fully-interior trained window has both edges inside the
                # clip.  Edge windows retain their exact released geometry.
                if lo >= 0 and hi < num_frames:
                    nlo = lo + left_trim
                    nhi = hi - right_trim
                    if nlo <= nhi:
                        lo, hi = nlo, nhi
                shrunk.append((lo, hi))
            self.bounds = shrunk
        self.full_cover = full_coverage(self.bounds, num_frames)
        self.anchor_frames = anchor_frames


class VDNState:
    """Everything one Apply-VDN application owns: config, per-block branch weights,
    the per-forward layout, and the runtime weight-placement policy."""

    def __init__(self, name, cfg, branches, num_heads, head_dim):
        self.name = name
        self.cfg = cfg
        self.branches = branches              # [num_blocks] LinearBranch or None
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layout = None                    # published by the wrapper each forward
        self.cache_gpu = False
        self.retain_buffers = True            # auto-resolved at apply time
        self._gpu_cache = {}
        # v17: the softmax gate is needed in every transformer block, while the
        # heavy linear-branch tensors are only needed in blocks selected by the
        # sparse hybrid mask. Cache the tiny gate tensors independently so
        # inactive linear blocks never materialize an entire branch.
        self._gate_gpu_cache = {}
        # v20: optional exact GPU residency cache for the base H3 attention
        # out_proj weights.  This does not change attention math: on a cache
        # miss we ask ComfyUI's own CastBiasWeightContext for the exact weight
        # and bias it would use, keep those returned tensors alive on CUDA, and
        # subsequent NFEs call F.linear with the same tensors.  A small explicit
        # budget prevents the cache from pushing 24 GiB Ampere cards over the
        # memory cliff.
        self._outproj_gpu_cache = {}
        self._outproj_cache_bytes = 0
        self._outproj_cache_hits = 0
        self._outproj_cache_misses = 0
        try:
            self.outproj_cache_gib = max(0.0, float(os.environ.get(
                "VDN_H3_OUTPROJ_CACHE_GIB", "0")))
        except Exception:
            self.outproj_cache_gib = 0.0
        try:
            self.outproj_cache_clear_after = max(0, int(os.environ.get(
                "VDN_H3_OUTPROJ_CACHE_CLEAR_AFTER", "0")))
        except Exception:
            self.outproj_cache_clear_after = 0
        self._outproj_cache_log_once = False
        # v22: probe/benchmark ComfyUI's native quantized out_proj execution
        # against the exact materialized F.linear path used by v20/v21.  The
        # benchmark is guarded by a numerical check and falls back to the
        # materialized path if native execution is unavailable, slower, or
        # disagrees beyond tolerance.
        raw_exec = os.environ.get("VDN_H3_OUTPROJ_EXECUTION", "materialized").strip().lower()
        if raw_exec not in ("materialized", "native", "benchmark"):
            _log.warning("[vdn] invalid VDN_H3_OUTPROJ_EXECUTION=%r; using materialized", raw_exec)
            raw_exec = "materialized"
        self.outproj_execution = raw_exec
        self._outproj_exec_choice = None if raw_exec == "benchmark" else raw_exec
        self._outproj_exec_bench_done = False
        try:
            self.outproj_exec_bench_block = int(os.environ.get("VDN_H3_OUTPROJ_EXEC_BENCH_BLOCK", "0"))
        except Exception:
            self.outproj_exec_bench_block = 0
        try:
            self.outproj_exec_bench_reps = max(1, int(os.environ.get("VDN_H3_OUTPROJ_EXEC_BENCH_REPS", "2")))
        except Exception:
            self.outproj_exec_bench_reps = 2
        try:
            self.outproj_exec_rtol = max(0.0, float(os.environ.get("VDN_H3_OUTPROJ_EXEC_RTOL", "0.002")))
        except Exception:
            self.outproj_exec_rtol = 0.002
        self._outproj_exec_probe_logged = False
        # v21: optional explicit block eligibility for the exact out_proj cache.
        # Empty means the v20 first-fit policy; a comma-separated list keeps the
        # same VRAM budget but lets profiling choose which projections deserve
        # residency.
        raw_blocks = os.environ.get("VDN_H3_OUTPROJ_CACHE_BLOCKS", "").strip()
        self.outproj_cache_blocks = None
        if raw_blocks:
            try:
                self.outproj_cache_blocks = {int(x.strip()) for x in raw_blocks.split(",") if x.strip()}
            except Exception:
                self.outproj_cache_blocks = None
                _log.warning("[vdn] invalid VDN_H3_OUTPROJ_CACHE_BLOCKS=%r; using first-fit cache", raw_blocks)
        raw_fmt = os.environ.get("VDN_H3_OUTPROJ_CACHE_FORMAT_PROFILE", "1").strip().lower()
        self.outproj_cache_format_profile = raw_fmt not in ("0", "false", "no", "off")
        self._outproj_format_stats = {}
        # v23: diagnostic split of uncached out_proj cost into exact ComfyUI
        # materialization/preparation and the actual GEMM. This is opt-in
        # because synchronizing around materialization perturbs wall time.
        raw_mat_prof = os.environ.get("VDN_H3_OUTPROJ_MAT_PROFILE", "0").strip().lower()
        self.outproj_mat_profile = raw_mat_prof not in ("0", "false", "no", "off")
        self._outproj_mat_stats = {}
        self._outproj_hit_gemm_ms = 0.0
        self._outproj_hit_gemm_count = 0
        # v24: exact GEMM frontend autotuner for already-materialized out_proj
        # weights.  All candidates compute the same dense BF16/FP16 projection;
        # only the PyTorch/cuBLAS entry point changes.  Benchmark mode tests the
        # real H3 tensor shape once on NFE1 and keeps the fastest numerically
        # compatible implementation for the rest of the run.
        raw_gemm = os.environ.get("VDN_H3_OUTPROJ_GEMM", "linear").strip().lower()
        if raw_gemm not in ("linear", "addmm", "mm_bias", "benchmark"):
            _log.warning("[vdn] invalid VDN_H3_OUTPROJ_GEMM=%r; using linear", raw_gemm)
            raw_gemm = "linear"
        self.outproj_gemm = raw_gemm
        self._outproj_gemm_choice = None if raw_gemm == "benchmark" else raw_gemm
        self._outproj_gemm_bench_done = False
        try:
            self.outproj_gemm_bench_block = int(os.environ.get("VDN_H3_OUTPROJ_GEMM_BENCH_BLOCK", "0"))
        except Exception:
            self.outproj_gemm_bench_block = 0
        try:
            self.outproj_gemm_bench_reps = max(1, int(os.environ.get("VDN_H3_OUTPROJ_GEMM_BENCH_REPS", "3")))
        except Exception:
            self.outproj_gemm_bench_reps = 3
        try:
            self.outproj_gemm_rtol = max(0.0, float(os.environ.get("VDN_H3_OUTPROJ_GEMM_RTOL", "0.001")))
        except Exception:
            self.outproj_gemm_rtol = 0.001

        # v30: diagnostic-only QKV projection execution laboratory.  This probes
        # the real full H3 qkv_proj shape on NFE1, comparing the established
        # ComfyUI/MixedPrecisionOps module path against exact materialized dense
        # BF16/FP16 GEMM frontends.  Production execution is never changed by
        # this experiment; it only reports timings and numerical agreement.
        raw_qkv_lab = os.environ.get("VDN_H3_QKV_EXEC_LAB", "0").strip().lower()
        self.qkv_exec_lab = raw_qkv_lab not in ("0", "false", "no", "off")
        try:
            self.qkv_exec_lab_block = int(os.environ.get("VDN_H3_QKV_EXEC_LAB_BLOCK", "0"))
        except Exception:
            self.qkv_exec_lab_block = 0
        try:
            self.qkv_exec_lab_reps = max(1, int(os.environ.get("VDN_H3_QKV_EXEC_LAB_REPS", "1")))
        except Exception:
            self.qkv_exec_lab_reps = 1
        try:
            self.qkv_exec_lab_rtol = max(0.0, float(os.environ.get("VDN_H3_QKV_EXEC_LAB_RTOL", "0.002")))
        except Exception:
            self.qkv_exec_lab_rtol = 0.002
        self._qkv_exec_lab_done = False

        # v31: opt-in production QKV experiment. v30 showed that exact
        # transient materialization followed by F.linear reproduces the native
        # qkv_proj output for the real H3 shape.  v31 uses that path per block
        # without retaining QKV weights between blocks/NFEs.
        raw_qkv_exec = os.environ.get("VDN_H3_QKV_EXECUTION", "native").strip().lower()
        aliases = {
            "transient": "transient_linear",
            "linear": "transient_linear",
            "materialized": "transient_linear",
            "materialized_linear": "transient_linear",
        }
        raw_qkv_exec = aliases.get(raw_qkv_exec, raw_qkv_exec)
        if raw_qkv_exec not in ("native", "transient_linear"):
            _log.warning("[vdn] invalid VDN_H3_QKV_EXECUTION=%r; using native", raw_qkv_exec)
            raw_qkv_exec = "native"
        self.qkv_execution = raw_qkv_exec
        self._qkv_execution_failed = False
        self._qkv_execution_log_once = False
        self._qkv_transient_calls = 0

        # v34: one-shot diagnostic laboratory for the real H3 attention out_proj.
        # It benchmarks BLAS backend choice and row-chunk geometry on the exact
        # materialized production tensors, then restores all global settings.
        raw_v34 = os.environ.get("VDN_H3_OUTPROJ_GEOMETRY_LAB", "0").strip().lower()
        self.outproj_geometry_lab = raw_v34 not in ("0", "false", "no", "off")
        try:
            self.outproj_geometry_lab_block = int(os.environ.get("VDN_H3_OUTPROJ_GEOMETRY_LAB_BLOCK", "0"))
        except Exception:
            self.outproj_geometry_lab_block = 0
        try:
            self.outproj_geometry_lab_reps = max(1, int(os.environ.get("VDN_H3_OUTPROJ_GEOMETRY_LAB_REPS", "2")))
        except Exception:
            self.outproj_geometry_lab_reps = 2
        self._outproj_geometry_lab_done = False

        raw_sel = os.environ.get("VDN_H3_SELECTIVE_WEIGHT_LOAD", "1").strip().lower()
        self.selective_weight_load = raw_sel not in ("0", "false", "no", "off")
        self._selective_log_once = False
        self._act = None                      # per-geometry activation scratch
        self._act_key = None
        self._prefetcher = None
        self._window_stream = None           # v9: ordinary-tensor window branch only
        # v7: the one-block streaming prefetch can be enabled independently of
        # retained scratch buffers.  This matters on 24 GB Ampere: retained
        # scan/window scratch is too expensive near the VRAM cliff, while one
        # INT8 branch block is only ~43 MB and is cheap enough to overlap with
        # the current block's compute.
        raw_pf = os.environ.get("VDN_H3_STREAM_PREFETCH", "0").strip().lower()
        requested_pf = raw_pf in ("1", "true", "yes", "on")
        self.stream_prefetch = False
        if requested_pf:
            _log.warning("[vdn] VDN_H3_STREAM_PREFETCH is disabled in v8: "
                         "comfy-kitchen QuantizedTensor materialization is not "
                         "safe on auxiliary CUDA streams on Windows")
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self.forwards = 0
        # v6: optional temporary CUDA reservation.  The node allocates this
        # before ComfyUI loads the base H3 weights, forcing the loader to leave
        # deliberate VRAM headroom.  The wrapper releases it immediately before
        # the first diffusion forward, so that headroom becomes VDN workspace
        # rather than remaining wasted during sampling.
        self._headroom_chunks = []
        self._headroom_gib = 0.0
        self._headroom_released = False
        # v37 lifecycle fix: keep the headroom policy visible to ComfyUI's
        # model loader across *every* prompt.  v6 used temporary CUDA tensors
        # only before the first lazy H3 load; after VideoVAE decode Comfy could
        # reload H3 fully and the second prompt would hit the VRAM cliff.
        self._loader_headroom_installed = False
        self._loader_headroom_gib = 0.0

    @staticmethod
    def _driver_free_gib(device):
        try:
            free, total = torch.cuda.mem_get_info(device)
            gib = 1024.0 ** 3
            return free / gib, total / gib
        except Exception:
            return None, None

    def install_loader_headroom(self, gib, exact=False):
        """Persistently add VDN workspace headroom to ComfyUI model loads.

        Unlike the old v6 temporary CUDA reservation, this participates in
        ComfyUI's own ``minimum_inference_memory`` calculation, so it is applied
        again when H3 is reloaded after VideoVAE decode on later prompts.  The
        requested VDN headroom is added on top of ComfyUI's pre-existing
        reservation (Windows/--reserve-vram), preserving the user's baseline.

        The process-wide reservation is monotonic (max wins) so re-executing an
        ApplyVDN node cannot accidentally stack the same request repeatedly.
        """
        try:
            gib = max(0.0, float(gib))
        except Exception:
            gib = 0.0
        if gib <= 0:
            return 0.0
        mm = comfy.model_management
        try:
            baseline = getattr(mm, "_VDN_H3_BASE_RESERVED_VRAM")
        except Exception:
            baseline = int(getattr(mm, "EXTRA_RESERVED_VRAM", 0))
            setattr(mm, "_VDN_H3_BASE_RESERVED_VRAM", baseline)
        requested = baseline + int(gib * (1024 ** 3))
        previous = int(getattr(mm, "EXTRA_RESERVED_VRAM", baseline))
        # v38 AutoMemory may move both upward and downward as resolution changes.
        # Fixed v37 mode keeps the old monotonic max behavior.
        target = requested if exact else max(previous, requested)
        mm.EXTRA_RESERVED_VRAM = target
        self._loader_headroom_installed = True
        self._loader_headroom_gib = (target - baseline) / (1024.0 ** 3)
        _log.info(
            "[vdn-vram] v37/v38 persistent loader headroom: %.2f GiB VDN + %.2f GiB Comfy baseline = %.2f GiB reserved for every model load",
            self._loader_headroom_gib, baseline / (1024.0 ** 3),
            target / (1024.0 ** 3))
        return self._loader_headroom_gib

    def reserve_headroom(self, gib, device=None):
        """Temporarily occupy ``gib`` GiB before Comfy loads the base model.

        The reservation is deliberately made from ordinary live CUDA tensors so
        ComfyUI's model loader sees less free device memory and chooses partial
        residency/offload.  It is released by ``release_headroom`` immediately
        before the first diffusion forward.
        """
        try:
            gib = float(gib)
        except Exception:
            gib = 0.0
        if gib <= 0 or not torch.cuda.is_available() or self._headroom_chunks:
            return 0.0
        device = device or comfy.model_management.get_torch_device()
        if torch.device(device).type != "cuda":
            return 0.0

        before_free, total = self._driver_free_gib(device)
        # Use modest chunks instead of one giant allocation.  This is much less
        # fragile on 24 GiB cards and makes partial reservations possible if the
        # requested amount is close to the current free-memory limit.
        chunk_bytes = 256 * 1024 * 1024
        target_bytes = int(gib * (1024 ** 3))
        # Always leave 512 MiB of driver-visible free memory while constructing
        # the reservation so the node itself cannot strand Comfy during setup.
        safety_bytes = 512 * 1024 * 1024
        allocated = 0
        chunks = []
        try:
            while allocated < target_bytes:
                free_bytes, _ = torch.cuda.mem_get_info(device)
                room = int(free_bytes) - safety_bytes
                if room <= 0:
                    break
                n = min(chunk_bytes, target_bytes - allocated, room)
                if n < 16 * 1024 * 1024:
                    break
                t = torch.empty((n,), dtype=torch.uint8, device=device)
                chunks.append(t)
                allocated += n
        except torch.cuda.OutOfMemoryError:
            # Keep the successfully allocated prefix; it still provides a valid
            # smaller forced-headroom experiment.
            pass

        self._headroom_chunks = chunks
        self._headroom_gib = allocated / (1024.0 ** 3)
        self._headroom_released = False
        after_free, _ = self._driver_free_gib(device)
        _log.info(
            "[vdn-vram] forced headroom reserved %.2f GiB on %s before base "
            "model load (driver free %.2f -> %.2f GiB; requested %.2f GiB)",
            self._headroom_gib, str(device),
            before_free if before_free is not None else -1.0,
            after_free if after_free is not None else -1.0, gib)
        return self._headroom_gib

    def release_headroom(self, device=None):
        """Release the v6 pre-load reservation once, before first NFE."""
        if self._headroom_released:
            return
        self._headroom_released = True
        if not self._headroom_chunks:
            return
        device = device or comfy.model_management.get_torch_device()
        before_free, _ = self._driver_free_gib(device)
        released = self._headroom_gib
        self._headroom_chunks.clear()
        self._headroom_gib = 0.0
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        after_free, _ = self._driver_free_gib(device)
        _log.info(
            "[vdn-vram] released %.2f GiB forced headroom before first NFE "
            "(driver free %.2f -> %.2f GiB); base-model residency is left "
            "unchanged so this memory is now VDN workspace",
            released,
            before_free if before_free is not None else -1.0,
            after_free if after_free is not None else -1.0)

    def act_scratch(self, video_rows, text_rows, device, dtype):
        """The raw pre-RoPE q/k/v copies the linear branch reads. Retained mode:
        one buffer set lives across blocks and is dropped after each block's
        readout (the allocator re-serves it, so only one set is ever live).
        Transient mode (VRAM pressure): fresh per block -- the v1.3.1 pattern.
        The branch never writes to these."""
        key = (video_rows, text_rows, self.num_heads, self.head_dim,
               str(device), dtype)
        if not self.retain_buffers:
            shape = (video_rows, self.num_heads, self.head_dim)
            tshape = (text_rows, self.num_heads, self.head_dim)
            return {"q": torch.empty(shape, device=device, dtype=dtype),
                    "k": torch.empty(shape, device=device, dtype=dtype),
                    "v": torch.empty(shape, device=device, dtype=dtype),
                    "tk": torch.empty(tshape, device=device, dtype=dtype),
                    "tv": torch.empty(tshape, device=device, dtype=dtype)}
        if self._act is None or self._act_key != key:
            shape = (video_rows, self.num_heads, self.head_dim)
            tshape = (text_rows, self.num_heads, self.head_dim)
            self._act = {"q": torch.empty(shape, device=device, dtype=dtype),
                         "k": torch.empty(shape, device=device, dtype=dtype),
                         "v": torch.empty(shape, device=device, dtype=dtype),
                         "tk": torch.empty(tshape, device=device, dtype=dtype),
                         "tv": torch.empty(tshape, device=device, dtype=dtype)}
            self._act_key = key
        return self._act

    def _prefetch(self):
        if self._prefetcher is None:
            self._prefetcher = _StreamPrefetcher()
        return self._prefetcher

    def window_stream(self, device):
        """Dedicated v9 stream for the exact window branch only.

        Unlike the abandoned v7 prefetch experiment, no lazy / quantized
        comfy-kitchen tensor is ever materialized or executed on this stream.
        It receives only ordinary q/k/v tensors that have already been produced
        by the base H3 projection on the default stream.
        """
        if self._window_stream is None:
            self._window_stream = torch.cuda.Stream(device=device)
        return self._window_stream

    @staticmethod
    def _resident_bytes(obj):
        """Best-effort physical CUDA storage size for plain/quantized tensors."""
        if obj is None:
            return 0
        qd = getattr(obj, "_qdata", None)
        if qd is not None:
            total = VDNState._resident_bytes(qd)
            params = getattr(obj, "_params", None)
            if params is not None:
                for name in ("scale", "scale_weight", "scale_input", "bias",
                             "orig_weight"):
                    total += VDNState._resident_bytes(getattr(params, name, None))
            return total
        if isinstance(obj, torch.Tensor):
            try:
                return int(obj.numel()) * int(obj.element_size())
            except Exception:
                return 0
        if isinstance(obj, (tuple, list)):
            return sum(VDNState._resident_bytes(x) for x in obj)
        if isinstance(obj, dict):
            return sum(VDNState._resident_bytes(x) for x in obj.values())
        return 0

    @staticmethod
    def _cuda_time_ms(fn, reps=1):
        """Time a CUDA callable with explicit events and return (ms, last_output)."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        out = None
        start.record()
        for _ in range(max(1, int(reps))):
            out = fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / max(1, int(reps)), out

    @staticmethod
    def _outproj_probe_summary(module):
        w = getattr(module, "weight", None)
        if w is None:
            return "weight=None"
        parts = [f"type={type(w).__name__}", f"dtype={getattr(w, 'dtype', '?')}",
                 f"shape={tuple(getattr(w, 'shape', ())) }"]
        for attr in ("_layout", "layout", "_qdata", "_params"):
            obj = getattr(w, attr, None)
            if obj is not None:
                if attr == "_qdata":
                    parts.append(f"qdata={getattr(obj, 'dtype', '?')}/{tuple(getattr(obj, 'shape', ())) }")
                else:
                    parts.append(f"{attr}={type(obj).__name__}")
        return ", ".join(parts)

    def _benchmark_outproj_execution(self, index, module, inp):
        """v22 one-time exact native-vs-materialized execution benchmark.

        Uses the full real tensor shape, compares a cloned sample to avoid keeping
        two full outputs alive, and selects native only when it is numerically close
        and measurably faster.  Any failure leaves the established materialized
        path untouched.
        """
        if self._outproj_exec_bench_done:
            return self._outproj_exec_choice or "materialized", None
        if int(index) != int(self.outproj_exec_bench_block) or self.forwards != 1:
            return None, None
        self._outproj_exec_bench_done = True
        if not self._outproj_exec_probe_logged:
            self._outproj_exec_probe_logged = True
            _log.info("[vdn-outproj-v22-probe] block=%d %s", int(index), self._outproj_probe_summary(module))
        try:
            from comfy.ops import CastBiasWeightContext
            # Materialized exact reference.  Clone only a compact row sample for
            # the numerical check so the full output can be freed before native.
            with CastBiasWeightContext(module, inp, offloadable=True) as (weight, bias):
                mat_ms, mat_out = self._cuda_time_ms(lambda: F.linear(inp, weight, bias),
                                                     self.outproj_exec_bench_reps)
                nrows = min(64, int(mat_out.shape[0]))
                ref = mat_out[:nrows].detach().float().clone()
                del mat_out
            torch.cuda.synchronize(inp.device)
            native_ms, native_out = self._cuda_time_ms(lambda: module(inp),
                                                       self.outproj_exec_bench_reps)
            cand = native_out[:nrows].detach().float()
            diff = cand - ref
            ref_rms = float(ref.square().mean().sqrt().item())
            err_rms = float(diff.square().mean().sqrt().item())
            rel_rms = err_rms / max(ref_rms, 1e-8)
            max_abs = float(diff.abs().max().item())
            # Require a modest margin so noise does not flip the global path.
            native_ok = rel_rms <= self.outproj_exec_rtol
            choice = "native" if native_ok and native_ms < mat_ms * 0.97 else "materialized"
            self._outproj_exec_choice = choice
            _log.info("[vdn-outproj-v22-bench] block=%d materialized=%.3fms native=%.3fms speedup=%.3fx rel_rms=%.6g max_abs=%.6g tol=%.6g -> %s",
                      int(index), mat_ms, native_ms, mat_ms / max(native_ms, 1e-9),
                      rel_rms, max_abs, self.outproj_exec_rtol, choice)
            if choice == "native":
                return choice, native_out
            del native_out
            return choice, None
        except Exception as e:
            self._outproj_exec_choice = "materialized"
            _log.warning("[vdn-outproj-v22-bench] native probe failed (%s: %s); using materialized",
                         type(e).__name__, e)
            return "materialized", None

    @staticmethod
    def _v24_gemm(method, inp, weight, bias):
        """Execute the exact dense out_proj through a selected PyTorch frontend."""
        if method == "addmm":
            if bias is None:
                return torch.mm(inp, weight.t())
            return torch.addmm(bias, inp, weight.t())
        if method == "mm_bias":
            out = torch.mm(inp, weight.t())
            if bias is not None:
                out = out + bias
            return out
        return F.linear(inp, weight, bias)

    def _benchmark_outproj_gemm(self, index, inp, weight, bias):
        """v24 one-time exact GEMM frontend benchmark on the real H3 shape."""
        if self._outproj_gemm_bench_done:
            return self._outproj_gemm_choice or "linear"
        if int(index) != int(self.outproj_gemm_bench_block) or self.forwards != 1:
            return None
        self._outproj_gemm_bench_done = True
        methods = ("linear", "addmm", "mm_bias")
        try:
            # Reference sample from the established F.linear path.  Benchmark
            # calls are warmed once so lazy cuBLAS/cuBLASLt setup does not decide
            # the winner.  Only a compact row sample is retained for validation.
            warm = F.linear(inp, weight, bias)
            del warm
            torch.cuda.synchronize(inp.device)
            linear_ms, ref_out = self._cuda_time_ms(
                lambda: F.linear(inp, weight, bias), self.outproj_gemm_bench_reps)
            nrows = min(64, int(ref_out.shape[0]))
            ref = ref_out[:nrows].detach().float().clone()
            del ref_out
            results = {"linear": (linear_ms, 0.0, 0.0, True)}
            for method in methods[1:]:
                try:
                    warm = self._v24_gemm(method, inp, weight, bias)
                    del warm
                    torch.cuda.synchronize(inp.device)
                    ms, out = self._cuda_time_ms(
                        lambda m=method: self._v24_gemm(m, inp, weight, bias),
                        self.outproj_gemm_bench_reps)
                    cand = out[:nrows].detach().float()
                    diff = cand - ref
                    ref_rms = float(ref.square().mean().sqrt().item())
                    err_rms = float(diff.square().mean().sqrt().item())
                    rel_rms = err_rms / max(ref_rms, 1e-8)
                    max_abs = float(diff.abs().max().item())
                    ok = rel_rms <= self.outproj_gemm_rtol
                    results[method] = (ms, rel_rms, max_abs, ok)
                    del out
                except Exception as e:
                    results[method] = (float("inf"), float("inf"), float("inf"), False)
                    _log.warning("[vdn-outproj-v24-bench] %s candidate failed (%s: %s)",
                                 method, type(e).__name__, e)
            safe = [(m, v[0]) for m, v in results.items() if v[3]]
            best, best_ms = min(safe, key=lambda kv: kv[1]) if safe else ("linear", linear_ms)
            # Require a real margin over F.linear so timing noise cannot select a
            # different frontend for a negligible apparent win.
            choice = best if best != "linear" and best_ms < linear_ms * 0.97 else "linear"
            self._outproj_gemm_choice = choice
            def fmt(m):
                ms, rr, ma, ok = results[m]
                if not ok and ms == float("inf"):
                    return f"{m}=NA"
                return f"{m}={ms:.3f}ms(rel={rr:.3g},max={ma:.3g},ok={int(ok)})"
            _log.info("[vdn-outproj-v24-bench] block=%d reps=%d %s | %s | %s -> %s",
                      int(index), self.outproj_gemm_bench_reps,
                      fmt("linear"), fmt("addmm"), fmt("mm_bias"), choice)
            return choice
        except Exception as e:
            self._outproj_gemm_choice = "linear"
            _log.warning("[vdn-outproj-v24-bench] GEMM autotune failed (%s: %s); using F.linear",
                         type(e).__name__, e)
            return "linear"

    def _outproj_gemm_exact(self, index, inp, weight, bias):
        method = self._outproj_gemm_choice
        if self.outproj_gemm == "benchmark" and method is None:
            method = self._benchmark_outproj_gemm(index, inp, weight, bias)
        if method is None:
            # Blocks encountered before the designated NFE1 benchmark block use
            # the established path.
            method = "linear"
        return self._v24_gemm(method, inp, weight, bias)

    @staticmethod
    def _v30_qkv_gemm(method, inp, weight, bias):
        """Exact dense QKV projection through a selected PyTorch frontend."""
        if method == "addmm":
            if bias is None:
                return torch.mm(inp, weight.t())
            return torch.addmm(bias, inp, weight.t())
        if method == "mm_bias":
            out = torch.mm(inp, weight.t())
            if bias is not None:
                out = out + bias
            return out
        return F.linear(inp, weight, bias)

    def _v30_qkv_exec_probe(self, index, module, inp):
        """One-shot, diagnostic-only benchmark of the real full H3 qkv_proj.

        The probe intentionally runs before the production qkv_proj call so its
        very large output can be freed between candidates.  It never changes the
        production path.  ComfyUI's CastBiasWeightContext is used for the dense
        candidates so patches/quantization are resolved exactly as the module
        expects.
        """
        if (not self.qkv_exec_lab or self._qkv_exec_lab_done or
                self.forwards != 1 or int(index) != int(self.qkv_exec_lab_block)):
            return
        self._qkv_exec_lab_done = True
        if inp.device.type != "cuda":
            _log.info("[vdn-qkv-v30] skipped: non-CUDA input")
            return
        from comfy.ops import CastBiasWeightContext
        reps = self.qkv_exec_lab_reps
        try:
            # Established native/MixedPrecisionOps path. Warm once, then time.
            warm = module(inp)
            ref_rows = min(32, int(warm.shape[0]))
            ref = warm[:ref_rows].detach().float().clone()
            del warm
            torch.cuda.synchronize(inp.device)
            native_ms, native_out = self._cuda_time_ms(lambda: module(inp), reps)
            # Refresh the validation sample from the timed execution and free the
            # huge full output before materializing the dense weight.
            ref = native_out[:ref_rows].detach().float().clone()
            del native_out
            torch.cuda.synchronize(inp.device)

            # Materialize the exact weight/bias once, with both wall and CUDA
            # preparation timings.
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            ctx = CastBiasWeightContext(module, inp, offloadable=True)
            ev0.record()
            weight, bias = ctx.__enter__()
            ev1.record(); ev1.synchronize()
            prep_wall_ms = (time.perf_counter() - t0) * 1000.0
            prep_cuda_ms = float(ev0.elapsed_time(ev1))
            w_gib = (weight.numel() * weight.element_size() +
                     (0 if bias is None else bias.numel() * bias.element_size())) / (1024.0 ** 3)

            results = {}
            for method in ("linear", "addmm", "mm_bias"):
                try:
                    warm = self._v30_qkv_gemm(method, inp, weight, bias)
                    del warm
                    torch.cuda.synchronize(inp.device)
                    ms, out = self._cuda_time_ms(
                        lambda m=method: self._v30_qkv_gemm(m, inp, weight, bias), reps)
                    cand = out[:ref_rows].detach().float()
                    diff = cand - ref
                    ref_rms = float(ref.square().mean().sqrt().item())
                    err_rms = float(diff.square().mean().sqrt().item())
                    rel_rms = err_rms / max(ref_rms, 1e-8)
                    max_abs = float(diff.abs().max().item())
                    ok = rel_rms <= self.qkv_exec_lab_rtol
                    results[method] = (ms, rel_rms, max_abs, ok)
                    del out
                except Exception as e:
                    results[method] = (float("inf"), float("inf"), float("inf"), False)
                    _log.warning("[vdn-qkv-v30] %s candidate failed (%s: %s)",
                                 method, type(e).__name__, e)
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            del weight, bias
            torch.cuda.synchronize(inp.device)

            safe = [(m, v[0]) for m, v in results.items() if v[3]]
            best, best_ms = min(safe, key=lambda kv: kv[1]) if safe else ("none", float("inf"))
            def fmt(m):
                ms, rr, ma, ok = results[m]
                if ms == float("inf"):
                    return f"{m}=NA"
                return f"{m}={ms:.3f}ms(rel={rr:.3g},max={ma:.3g},ok={int(ok)})"
            speedup = native_ms / max(best_ms, 1e-9) if best != "none" else 0.0
            _log.info(
                "[vdn-qkv-v30] block=%d shape=%sx%s reps=%d native=%.3fms | "
                "%s | %s | %s | best=%s speedup=%.3fx | materialize=%.3fms-wall/%.3fms-cuda "
                "resident=%.3fGiB tol=%g | diagnostic-only",
                int(index), int(inp.shape[0]), int(inp.shape[-1]), reps, native_ms,
                fmt("linear"), fmt("addmm"), fmt("mm_bias"), best, speedup,
                prep_wall_ms, prep_cuda_ms, w_gib, self.qkv_exec_lab_rtol)
        except Exception as e:
            _log.warning("[vdn-qkv-v30] probe failed (%s: %s); production qkv path unchanged",
                         type(e).__name__, e)
            try:
                if 'ctx' in locals():
                    ctx.__exit__(None, None, None)
            except Exception:
                pass

    def _v31_qkv_project(self, index, module, inp):
        """Exact transient-materialized production QKV path with safe fallback."""
        if self.qkv_execution != "transient_linear" or self._qkv_execution_failed:
            return module(inp)
        if not self._qkv_execution_log_once:
            self._qkv_execution_log_once = True
            _log.info("[vdn] v31 QKV execution active: transient materialize -> F.linear -> release "
                      "(no persistent QKV cache)")
        try:
            from comfy.ops import CastBiasWeightContext
            ctx = CastBiasWeightContext(module, inp, offloadable=True)
            weight = bias = None
            try:
                weight, bias = ctx.__enter__()
                out = F.linear(inp, weight, bias)
            finally:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
                weight = None
                bias = None
            self._qkv_transient_calls += 1
            return out
        except Exception as e:
            self._qkv_execution_failed = True
            _log.warning("[vdn-qkv-v31] transient QKV failed at block %d (%s: %s); "
                         "falling back to native qkv_proj for the rest of the run",
                         int(index), type(e).__name__, e)
            return module(inp)

    def _v34_outproj_geometry_probe(self, index, inp, weight, bias):
        if (not self.outproj_geometry_lab or self._outproj_geometry_lab_done or
                int(index) != self.outproj_geometry_lab_block or inp.device.type != "cuda"):
            return
        self._outproj_geometry_lab_done = True
        reps = self.outproj_geometry_lab_reps
        try:
            ref = self._v24_gemm("mm_bias", inp, weight, bias)
            torch.cuda.synchronize(inp.device)
            ref_rows = min(32, int(ref.shape[0]))
            ref_small = ref[:ref_rows].detach().float()
            del ref
            results = []

            def bench(name, fn):
                ms, out = self._cuda_time_ms(fn, reps)
                cand = out[:ref_rows].detach().float()
                diff = cand - ref_small
                rr = float(diff.square().mean().sqrt().item()) / max(float(ref_small.square().mean().sqrt().item()), 1e-8)
                ma = float(diff.abs().max().item())
                results.append((name, ms, rr, ma))
                del out, cand, diff

            # Baseline full GEMM.
            bench("full_mm_bias", lambda: self._v24_gemm("mm_bias", inp, weight, bias))

            # Same math split only along independent output rows. This tests whether
            # the very tall M dimension is a poor cuBLAS geometry on Ampere.
            rows = int(inp.shape[0])
            for chunks in (2, 4, 8):
                def run_chunks(c=chunks):
                    step = (rows + c - 1) // c
                    parts = [self._v24_gemm("mm_bias", inp[a:min(a+step, rows)], weight, bias)
                             for a in range(0, rows, step)]
                    return torch.cat(parts, dim=0)
                bench(f"row_chunks{chunks}", run_chunks)

            # PyTorch exposes a preferred BLAS backend on builds that support it.
            pref = getattr(torch.backends.cuda, "preferred_blas_library", None)
            if pref is not None:
                old = None
                try:
                    old = pref()
                except Exception:
                    pass
                for lib in ("cublas", "cublaslt"):
                    try:
                        pref(lib)
                        bench(lib, lambda: self._v24_gemm("mm_bias", inp, weight, bias))
                    except Exception as e:
                        results.append((lib, float("inf"), float("inf"), float("inf")))
                        _log.info("[vdn-outproj-v34] backend %s unavailable: %s", lib, e)
                if old is not None:
                    try:
                        pref(old)
                    except Exception:
                        pass

            wshape = tuple(int(x) for x in weight.shape)
            ishape = tuple(int(x) for x in inp.shape)
            good = [r for r in results if r[1] != float("inf") and r[2] <= 0.001]
            best = min(good, key=lambda r: r[1]) if good else None
            txt = " | ".join((f"{n}={ms:.3f}ms(rel={rr:.3g},max={ma:.3g})" if ms != float("inf") else f"{n}=NA")
                             for n,ms,rr,ma in results)
            _log.info("[vdn-outproj-v34] block=%02d inp=%s weight=%s dtype=%s reps=%d | %s | best=%s | diagnostic-only",
                      int(index), ishape, wshape, str(inp.dtype).replace("torch.", ""), reps, txt,
                      (f"{best[0]}:{best[1]:.3f}ms" if best else "none"))
        except Exception as e:
            _log.warning("[vdn-outproj-v34] probe failed (%s: %s); production unchanged", type(e).__name__, e)

    def _v23_record_mat_stat(self, index, prep_wall_ms, prep_cuda_ms, gemm_cuda_ms):
        st = self._outproj_mat_stats.setdefault(int(index), {
            "count": 0, "prep_wall_ms": 0.0, "prep_cuda_ms": 0.0, "gemm_cuda_ms": 0.0
        })
        st["count"] += 1
        st["prep_wall_ms"] += float(prep_wall_ms)
        st["prep_cuda_ms"] += float(prep_cuda_ms)
        st["gemm_cuda_ms"] += float(gemm_cuda_ms)

    def _v23_profile_materialized_linear(self, module, inp):
        """Profile exact ComfyUI preparation separately from F.linear CUDA time."""
        from comfy.ops import CastBiasWeightContext
        torch.cuda.synchronize(inp.device)
        ctx = CastBiasWeightContext(module, inp, offloadable=True)
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        ev0.record()
        weight, bias = ctx.__enter__()
        ev1.record()
        ev1.synchronize()
        prep_wall_ms = (time.perf_counter() - t0) * 1000.0
        prep_cuda_ms = float(ev0.elapsed_time(ev1))
        gemm_cuda_ms, out = self._cuda_time_ms(lambda: self._outproj_gemm_exact(-1, inp, weight, bias), 1)
        return out, weight, bias, prep_wall_ms, prep_cuda_ms, gemm_cuda_ms, ctx

    def log_outproj_mat_profile(self):
        if not self.outproj_mat_profile:
            return
        vals = list(self._outproj_mat_stats.items())
        if not vals and not self._outproj_hit_gemm_count:
            return
        misses = sum(v["count"] for _, v in vals)
        prep_wall = sum(v["prep_wall_ms"] for _, v in vals)
        prep_cuda = sum(v["prep_cuda_ms"] for _, v in vals)
        gemm = sum(v["gemm_cuda_ms"] for _, v in vals)
        hit_gemm = float(self._outproj_hit_gemm_ms)
        hit_n = int(self._outproj_hit_gemm_count)
        avg_prep = prep_wall / max(misses, 1)
        avg_gemm = gemm / max(misses, 1)
        avg_hit = hit_gemm / max(hit_n, 1)
        denom = prep_wall + gemm
        prep_pct = 100.0 * prep_wall / max(denom, 1e-9)
        _log.info("[vdn-outproj-v23] NFE %d misses=%d hits=%d | materialize_wall=%.3fs (%.3fms/miss, %.1f%% of miss prep+gemm) materialize_cuda=%.3fs | miss_gemm_cuda=%.3fs (%.3fms/miss) cached_gemm_cuda=%.3fs (%.3fms/hit)",
                  self.forwards, misses, hit_n, prep_wall / 1000.0, avg_prep, prep_pct,
                  prep_cuda / 1000.0, gemm / 1000.0, avg_gemm, hit_gemm / 1000.0, avg_hit)
        if vals:
            ranked = sorted(vals, key=lambda kv: kv[1]["prep_wall_ms"], reverse=True)[:10]
            _log.info("[vdn-outproj-v23-top] NFE %d materialize=%s", self.forwards,
                      ", ".join(f"b{i:02d}:{v['prep_wall_ms']/max(v['count'],1):.3f}ms" for i, v in ranked))
        self._outproj_mat_stats.clear()
        self._outproj_hit_gemm_ms = 0.0
        self._outproj_hit_gemm_count = 0

    def outproj_exact(self, index, module, inp):
        """v20-v22 exact fast path for H3 attention out_proj.

        The first use of a block goes through ComfyUI's own casting context, so
        patches/quantization are resolved exactly as the module expects.  If the
        resulting CUDA tensors fit the configured cache budget we retain them
        across NFEs; later calls avoid the repeated offload/cast path.
        """
        # v22 execution policy.  Native means the ComfyUI module handles its
        # own QuantizedTensor/MixedPrecisionOps path directly.  Benchmark makes
        # one guarded measurement on NFE1 then sticks to one exact path.
        if inp.device.type == "cuda" and self.outproj_execution == "benchmark":
            choice, bench_out = self._benchmark_outproj_execution(index, module, inp)
            if choice == "native":
                if bench_out is not None:
                    return bench_out
                return module(inp)
            if choice is None:
                # Before the designated benchmark block, preserve the current
                # materialized path rather than guessing.
                pass
        elif self.outproj_execution == "native":
            return module(inp)
        if self._outproj_exec_choice == "native":
            return module(inp)

        budget = int(self.outproj_cache_gib * (1024 ** 3))
        if budget <= 0 or inp.device.type != "cuda":
            return module(inp)
        key = (index, str(inp.device), str(inp.dtype))
        hit = self._outproj_gpu_cache.get(key)
        if hit is not None:
            self._outproj_cache_hits += 1
            weight, bias = hit
            if self.outproj_mat_profile and inp.device.type == "cuda":
                ms, out = self._cuda_time_ms(lambda: self._outproj_gemm_exact(index, inp, weight, bias), 1)
                self._outproj_hit_gemm_ms += ms
                self._outproj_hit_gemm_count += 1
                return out
            return self._outproj_gemm_exact(index, inp, weight, bias)

        self._outproj_cache_misses += 1
        eligible = self.outproj_cache_blocks is None or int(index) in self.outproj_cache_blocks
        try:
            from comfy.ops import CastBiasWeightContext
            ctx = None
            if self.outproj_mat_profile and inp.device.type == "cuda":
                out, weight, bias, prep_wall_ms, prep_cuda_ms, gemm_cuda_ms, ctx = \
                    self._v23_profile_materialized_linear(module, inp)
                self._v23_record_mat_stat(index, prep_wall_ms, prep_cuda_ms, gemm_cuda_ms)
            else:
                ctx = CastBiasWeightContext(module, inp, offloadable=True)
                weight, bias = ctx.__enter__()
                self._v34_outproj_geometry_probe(index, inp, weight, bias)
                out = self._outproj_gemm_exact(index, inp, weight, bias)
            try:
                nbytes = self._resident_bytes(weight) + self._resident_bytes(bias)
                if self.outproj_cache_format_profile and int(index) not in self._outproj_format_stats:
                    raw_w = getattr(module, "weight", None)
                    raw_b = getattr(module, "bias", None)
                    raw_bytes = self._resident_bytes(raw_w) + self._resident_bytes(raw_b)
                    self._outproj_format_stats[int(index)] = {
                        "raw_bytes": int(raw_bytes),
                        "mat_bytes": int(nbytes),
                        "weight_type": type(raw_w).__name__ if raw_w is not None else "None",
                        "weight_dtype": str(getattr(raw_w, "dtype", "?")),
                    }
                free, _ = torch.cuda.mem_get_info(inp.device)
                safety = int(1.25 * (1024 ** 3))
                if (eligible and nbytes > 0 and
                        self._outproj_cache_bytes + nbytes <= budget and free - nbytes >= safety):
                    self._outproj_gpu_cache[key] = (weight, bias)
                    self._outproj_cache_bytes += nbytes
            finally:
                if ctx is not None:
                    ctx.__exit__(None, None, None)
            return out
        except Exception as e:
            _once(("outproj_cache_fallback", type(e).__name__),
                  f"v20 out_proj residency cache unavailable ({type(e).__name__}: {e}); using native module path")
            return module(inp)

    def log_outproj_format_summary(self):
        if not self.outproj_cache_format_profile or not self._outproj_format_stats:
            return
        vals = list(self._outproj_format_stats.values())
        raw = sum(v["raw_bytes"] for v in vals)
        mat = sum(v["mat_bytes"] for v in vals)
        types = {}
        for v in vals:
            key = f'{v["weight_type"]}/{v["weight_dtype"]}'
            types[key] = types.get(key, 0) + 1
        ratio = (mat / raw) if raw else 0.0
        _log.info("[vdn-outproj-format] blocks=%d raw=%.3f GiB materialized=%.3f GiB ratio=%.2fx types=%s",
                  len(vals), raw / (1024.0 ** 3), mat / (1024.0 ** 3), ratio, types)

    def clear_outproj_cache(self):
        if self._outproj_gpu_cache:
            self._outproj_gpu_cache.clear()
            self._outproj_cache_bytes = 0
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def weights_on(self, index, device, dtype, linear_active=True, need_gate=True):
        w = self.branches[index].w

        def fetch(t, copy=False):
            resolve = getattr(t, "resolve", None)
            if resolve is not None:
                return resolve(device, dtype)
            return comfy.model_management.cast_to(t, dtype=dtype, device=device,
                                                  copy=copy)

        # v17 selective residency. Window attention needs only the two tiny
        # softmax-gate tensors; the heavy branch is needed only by active linear
        # blocks. Cache the gate exactly on GPU and stream the rest on demand.
        gate_keys = ("softmax_gate.up.weight", "softmax_gate.up.bias")
        gate = {}
        if self.selective_weight_load and need_gate and torch.device(device).type == "cuda":
            gkey = (index, str(device), str(dtype))
            gate = self._gate_gpu_cache.get(gkey)
            if gate is None:
                gate = {k: fetch(w[k], copy=True) for k in gate_keys if k in w}
                self._gate_gpu_cache[gkey] = gate
            if not linear_active:
                return gate

        if self.cache_gpu:
            key = (index, str(device), str(dtype))
            hit = self._gpu_cache.get(key)
            if hit is None:
                hit = {k: fetch(t, copy=True) for k, t in w.items()}
                self._gpu_cache[key] = hit
            return hit
        if torch.device(device).type == "cuda":
            use_prefetch = self.retain_buffers or self.stream_prefetch
            if not use_prefetch:
                hit = {k: fetch(t) for k, t in w.items() if k not in gate}
                if gate:
                    hit.update(gate)
                return hit
            pf = self._prefetch()
            hit = pf.take(index)
            if hit is None:
                self._prefetch_misses += 1
                hit = {k: fetch(t) for k, t in w.items()}
            else:
                self._prefetch_hits += 1
            nxt = (index + 1) % len(self.branches)
            if self.branches[nxt] is not None:
                wn = self.branches[nxt].w
                pf.request(nxt, lambda: {k: fetch(t) for k, t in wn.items()})
            return hit
        return {k: fetch(t) for k, t in w.items()}


def layout_from_payload(payload, x, context, cfg):
    """Rebuild/adopt the PackedLayout the model itself uses, and derive the VDN
    geometry from it. Mirrors MiniMaxH3Model._forward's shape handling (including the
    patch-size padding of the video latent)."""
    payload = payload or {}
    layout = payload.get("layout")
    video_x = x[0]
    padded = comfy.ldm.common_dit.pad_to_patch_size(video_x, (1, 2, 2))
    latent_t, lat_h, lat_w = padded.shape[2], padded.shape[3], padded.shape[4]
    audio_t = x[1].shape[-1]
    text_len = context.shape[1]
    signature = (text_len, latent_t, lat_h, lat_w, audio_t)
    if layout is None or layout.signature != signature:
        layout = minimax_model.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                                            keyframes=payload.get("keyframes"),
                                            refs=payload.get("refs"))
    seg = next(s for s in layout.segments if s[2] == "video")
    text_seg = next(s for s in layout.segments if s[2] == "text")
    tokens_per_frame = (lat_h // 2) * (lat_w // 2)
    return VDNLayout(
    seg[0],
    seg[1],
    (seg[1] - seg[0]) // tokens_per_frame,
    tokens_per_frame,
    (lat_h // 2, lat_w // 2),
    text_seg[0],
    text_seg[1] - text_seg[0],
    layout.seq_len,
    cfg["radius"],
    cfg["chunk"],
    cfg["anchor_frames"],
    segments=layout.segments,
)


def make_layout_wrapper(state):
    """DIFFUSION_MODEL wrapper: publish the layout, run the model, clear it."""

    def wrap(executor, *args, **kwargs):
        # comfy builds with the model compiler crash on VDN forwards (the
        # malloc-graph planner cannot trace them); nodes.py flips the switch
        # off when the compiler stack exists, and we scope it to exactly this
        # forward so non-VDN workflows keep it.
        owns_switch = getattr(state, "owns_compiler_switch", False)
        if owns_switch:
            comfy.cli_args.args.disable_comfy_compiler = True
        # v37: loader headroom is now a persistent ComfyUI model-loading policy,
        # not live CUDA memory, so there is nothing to release between prompts.
        # Keep release_headroom for backwards compatibility with a state created
        # by an older build in the same process.
        if state.forwards == 0 and state._headroom_chunks:
            state.release_headroom()
        state.layout = layout_from_payload(kwargs.get("minimax_payload"),
                                           args[0], args[2], state.cfg)
        state.forwards += 1
        state._prefetch_hits = 0
        state._prefetch_misses = 0
        state._outproj_cache_hits = 0
        state._outproj_cache_misses = 0
        PROFILER.begin_forward(state.forwards)
        if state.forwards == 1:
            _log.info("[vdn] stream prefetch: %s (retain_buffers=%s, v7 decoupled mode)",
                      "on" if (state.retain_buffers or state.stream_prefetch) else "off",
                      state.retain_buffers)
            if not state._selective_log_once:
                state._selective_log_once = True
                _log.info("[vdn] v17 selective weight load: %s (softmax gate cached on GPU; full branch materialized only for active linear blocks)",
                          "on" if state.selective_weight_load else "off")
            if not state._outproj_cache_log_once:
                state._outproj_cache_log_once = True
                _log.info("[vdn] v20 exact out_proj residency cache: %.2f GiB budget, clear_after=%s NFE(s)",
                          state.outproj_cache_gib,
                          state.outproj_cache_clear_after if state.outproj_cache_clear_after else "disabled")
                _log.info("[vdn] v22 out_proj execution: %s (bench_block=%d reps=%d rtol=%g)",
                          state.outproj_execution, state.outproj_exec_bench_block,
                          state.outproj_exec_bench_reps, state.outproj_exec_rtol)
                if state.outproj_mat_profile:
                    _log.info("[vdn] v23 out_proj materialize/GEMM profiler: on (diagnostic synchronizations enabled)")
                if os.environ.get("VDN_H3_WINDOW_DEEP_PROFILE", "0").strip().lower() in ("1", "true", "yes", "on"):
                    _log.info("[vdn] v25 exact-window deep profiler: on (nested CUDA events; math unchanged)")
                _log.info("[vdn] v24 out_proj GEMM frontend: %s (bench_block=%d reps=%d rtol=%g)",
                          state.outproj_gemm, state.outproj_gemm_bench_block,
                          state.outproj_gemm_bench_reps, state.outproj_gemm_rtol)
                if state.qkv_exec_lab:
                    _log.info("[vdn] v30 QKV projection execution laboratory: on "
                              "(diagnostic-only; block=%d reps=%d rtol=%g)",
                              state.qkv_exec_lab_block, state.qkv_exec_lab_reps,
                              state.qkv_exec_lab_rtol)
                if state.qkv_execution == "transient_linear":
                    _log.info("[vdn] v31 QKV transient production path requested: materialize -> F.linear -> release")
                if os.environ.get("VDN_H3_DELTA_DEEP_PROFILE", "0").strip().lower() in {"1","true","yes","on"}:
                    _log.info("[vdn] v32 delta_solve deep profiler: on (nested CUDA events; math unchanged)")
                if os.environ.get("VDN_H3_DELTA_SOLVER_LAB", "0").strip().lower() in {"1","true","yes","on"}:
                    _log.info("[vdn] v32 exact solver laboratory: on (diagnostic-only; production remains direct cholesky_solve)")
        lay = state.layout
        _once(("layout", lay.seq_len, lay.num_frames, lay.tokens_per_frame,
               getattr(lay, "window_inner_trim", 0)),
              f"layout: seq {lay.seq_len} rows, video [{lay.video_start}, "
              f"{lay.video_end}), F={lay.num_frames}, S={lay.tokens_per_frame}, "
              f"frame {lay.frame_size}, text {lay.text_len} rows, "
              f"window {'dense (full cover)' if lay.full_cover else lay.bounds[0]}")
        if state.forwards == 1 and getattr(lay, "window_inner_trim", 0) > 0:
            widths = sorted(set(max(0, min(hi, lay.num_frames - 1) -
                                    max(lo, 0) + 1) for lo, hi in lay.bounds))
            _log.info("[vdn] v16 exact window inner trim: %d frame(s); "
                      "boundary groups unchanged, clamped temporal widths=%s",
                      lay.window_inner_trim, widths)
        try:
            return executor(*args, **kwargs)
        except comfy.model_management.InterruptProcessingException:
            # A cancelled mid-run leaves this node's GPU cache behind and the
            # CUDA allocator pool fragmented; drop everything the node owns so
            # the next run starts clean instead of OOM-ing on its first big
            # activation. (The base model's own residency is comfy's to manage.)
            state._gpu_cache.clear()
            state.clear_outproj_cache()
            # v37: an interrupted prompt must not leak its NFE index into the
            # next prompt.  The next diffusion call starts a fresh cycle.
            state.forwards = 0
            state._act = None
            state._act_key = None
            if state._prefetcher is not None:
                state._prefetcher.reset()
            from vdn_h3_24gb import branch as _b, window_advanced as _w
            _b.clear_scan_banks()
            _w.clear_window_state()
            torch.cuda.empty_cache()
            raise
        finally:
            if state._prefetch_hits or state._prefetch_misses:
                _log.info("[vdn-prefetch] NFE %d hits=%d misses=%d",
                          state.forwards, state._prefetch_hits, state._prefetch_misses)
            if state.outproj_cache_gib > 0:
                _log.info("[vdn-outproj-cache] NFE %d hits=%d misses=%d resident_blocks=%d resident=%.3f GiB",
                          state.forwards, state._outproj_cache_hits, state._outproj_cache_misses,
                          len(state._outproj_gpu_cache),
                          state._outproj_cache_bytes / (1024.0 ** 3))
                if state.forwards == 1:
                    if state.outproj_cache_blocks is None:
                        _log.info("[vdn] v21 out_proj cache selection: first-fit (no explicit block list)")
                    else:
                        _log.info("[vdn] v21 out_proj cache selection: explicit blocks=%s",
                                  ",".join(str(x) for x in sorted(state.outproj_cache_blocks)))
                    state.log_outproj_format_summary()
            state.log_outproj_mat_profile()
            PROFILER.finish_forward()
            if (state.outproj_cache_clear_after and
                    state.forwards == state.outproj_cache_clear_after):
                completed_nfe = state.forwards
                state.clear_outproj_cache()
                _log.info("[vdn-outproj-cache] cleared after NFE %d", completed_nfe)
                # v37 lifecycle fix: the counter is per sampling cycle, not per
                # lifetime of the patched MODEL.  This prevents the second
                # prompt from becoming NFE 9/10/... and re-clearing the cache
                # on every step.  Stage-DMD production uses 8 steps and the
                # cache clear boundary is deliberately set to 8 in the preset.
                state.forwards = 0
                _log.info("[vdn] v37 sampling-cycle reset after %d NFE(s); next prompt starts at NFE 1", completed_nfe)
            if owns_switch:
                comfy.cli_args.args.disable_comfy_compiler = False
            state.layout = None

    return wrap


def _base_attention(attn, x, rope_freqs, transformer_options):
    """comfy/ldm/minimax/model.py Attention.forward, verbatim (the dense teacher)."""
    s = x.shape[0]
    q, k, v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
    v = v.view(s, attn.heads, attn.head_dim)
    if rope_freqs is not None:
        q = q.view(1, s, attn.heads, attn.head_dim)
        k = k.view(1, s, attn.heads, attn.head_dim)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        rot = rope_freqs.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
        q = q[0]
        k = k[0]
    else:
        q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
        k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
    v = v.clone()
    q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
    k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
    v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
    out = optimized_attention(q, k, v, attn.heads, mask=None, skip_reshape=True,
                              transformer_options=transformer_options)
    return attn.out_proj(out.squeeze(0))


def make_vdn_forward(attn, state, block_index):
    """The object-patched Attention.forward for one DiT block."""
    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    qkv_proj, out_proj = attn.qkv_proj, attn.out_proj
    q_norm, k_norm = attn.q_norm, attn.k_norm
    branch = state.branches[block_index]
    cfg = state.cfg

    def vdn_forward(x, rope_freqs=None, transformer_options={}):
        lay = state.layout
        if lay is None or branch is None:
            return _base_attention(attn, x, rope_freqs, transformer_options)

        # VC-Attention Phase 1A: publish current NFE to exact-window helper.
        from vdn_h3_24gb.window import set_vc_forward
        set_vc_forward(state.forwards)

        s = x.shape[0]
        device, dtype = x.device, x.dtype
        ablation = os.environ.get("VDN_H3_ABLATION", "hybrid").strip().lower()
        if ablation not in ("hybrid", "window_only", "linear_only"):
            _once(("bad_ablation", ablation),
                  f"unknown VDN_H3_ABLATION={ablation!r}; using hybrid")
            ablation = "hybrid"
        _once(("ablation", ablation), f"ablation mode: {ablation}")
        memory_strategy = os.environ.get(
            "VDN_H3_HYBRID_MEMORY_STRATEGY", "baseline").strip().lower()
        if memory_strategy not in ("baseline", "recompute_qkv"):
            _once(("bad_memory_strategy", memory_strategy),
                  f"unknown VDN_H3_HYBRID_MEMORY_STRATEGY={memory_strategy!r}; using baseline")
            memory_strategy = "baseline"
        _once(("memory_strategy", memory_strategy),
              f"hybrid memory strategy: {memory_strategy}")
        raw_overlap = os.environ.get("VDN_H3_BRANCH_OVERLAP", "0").strip().lower()
        overlap_requested = raw_overlap in ("1", "true", "yes", "on")
        # v9 only overlaps the two released hybrid branches. Diagnostic
        # ablations and recompute_qkv keep the serial path. v10 stride and v12
        # explicit-mask runs are deliberately serial so A/B tests measure
        # compute removal rather than the small v9 stream-overlap effect.
        raw_stride_for_overlap = os.environ.get("VDN_H3_HYBRID_STRIDE", "1").strip()
        try:
            stride_for_overlap = max(1, int(raw_stride_for_overlap))
        except ValueError:
            stride_for_overlap = 1
        raw_block_mask_for_overlap = os.environ.get("VDN_H3_HYBRID_BLOCKS", "").strip()
        overlap_active = (overlap_requested and ablation == "hybrid" and
                          stride_for_overlap == 1 and
                          not raw_block_mask_for_overlap and
                          memory_strategy == "baseline" and
                          torch.device(device).type == "cuda")
        if overlap_requested and not overlap_active:
            _once(("overlap_inactive", ablation, memory_strategy),
                  "branch overlap requested but incompatible with the current "
                  "ablation/memory strategy; using serial execution")
        _once(("branch_overlap", overlap_active),
              f"hybrid branch overlap: {'on' if overlap_active else 'off'} "
              "(window-only auxiliary CUDA stream; quantized branch stays on default)")

        qkv_deep = qkv_deep_profile_enabled()
        if qkv_deep:
            _once(("qkv_deep_profiler",),
                  "v29 QKV data-path deep profiler: on (nested CUDA events; math unchanged)")
        qkv_ctx = (lambda name: prof_section(name, block_index)) if qkv_deep else \
                  (lambda name: contextlib.nullcontext())
        qkv_wall_ctx = (lambda name: prof_wall_section(name, block_index)) if qkv_deep else \
                       (lambda name: contextlib.nullcontext())

        with prof_section("qkv_and_raw_copy", block_index):
            # v30 diagnostic-only execution/GEMM laboratory. Run before the
            # production projection so each large benchmark output can be freed
            # before the normal qkv tensor is allocated.
            state._v30_qkv_exec_probe(block_index, qkv_proj, x)
            with qkv_ctx("qkv_deep_projection"):
                with qkv_wall_ctx("qkv_projection_wall"):
                    qkv_all = state._v31_qkv_project(block_index, qkv_proj, x)
            prof_reuse_sample("qkv", block_index, qkv_all)
            with qkv_ctx("qkv_deep_split_views"):
                q, k, v = qkv_all.split(inner, dim=-1)
                v = v.view(s, heads, head_dim)
                q_raw = q.view(s, heads, head_dim)
                k_raw = k.view(s, heads, head_dim)

            # Diagnostic ablations (v4) plus v5 memory strategy:
            #   hybrid      = released VDN path
            #   window_only = exact local/window softmax branch only
            #   linear_only = VDN linear branch only; softmax contribution is zero
            # v5 recompute_qkv avoids keeping ~video-sized raw q/k/v clones alive
            # during cuDNN window attention. It recomputes the deterministic QKV
            # projection once after the window branch instead.
            window_active = (not lay.full_cover) and ablation != "linear_only"
            linear_active = (not lay.full_cover) and cfg.get("linear_enabled", True) \
                and ablation != "window_only"

            # v14 sparse-window experiment.  VDN_H3_WINDOW_BLOCKS is an exact
            # causal mask for the local/window softmax branch in hybrid mode.
            # Blocks omitted from the mask receive a zero window contribution;
            # their linear branch is controlled independently by the v12 mask.
            # Empty/unset keeps the released behaviour (window in every block).
            raw_window_mask = os.environ.get("VDN_H3_WINDOW_BLOCKS", "").strip()
            window_block_mask = None
            if raw_window_mask and ablation == "hybrid" and not lay.full_cover:
                parsed = set()
                bad = []
                max_block = len(state.branches) - 1
                for tok in raw_window_mask.replace(";", ",").split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    try:
                        idx = int(tok)
                    except ValueError:
                        bad.append(tok)
                        continue
                    if 0 <= idx <= max_block:
                        parsed.add(idx)
                    else:
                        bad.append(tok)
                if parsed:
                    window_block_mask = frozenset(parsed)
                    window_active = window_active and (block_index in window_block_mask)
                    if bad:
                        _once(("bad_window_blocks", tuple(bad)),
                              f"ignored invalid VDN_H3_WINDOW_BLOCKS entries: {bad}")
                else:
                    _once(("empty_window_blocks", raw_window_mask),
                          "VDN_H3_WINDOW_BLOCKS contained no valid block indices; "
                          "using full window branch")

            raw_window_fallback = os.environ.get(
                "VDN_H3_WINDOW_FALLBACK", "zero").strip().lower()
            if raw_window_fallback not in ("zero", "temporal_proxy"):
                _once(("bad_window_fallback", raw_window_fallback),
                      f"unknown VDN_H3_WINDOW_FALLBACK={raw_window_fallback!r}; using zero")
                raw_window_fallback = "zero"

            if window_block_mask is not None:
                win_sorted = tuple(sorted(window_block_mask))
                _once(("window_blocks", win_sorted, raw_window_fallback),
                      f"window block mask: {len(win_sorted)}/{len(state.branches)} "
                      f"exact window blocks = {','.join(map(str, win_sorted))}; "
                      f"omitted fallback={raw_window_fallback}")
            else:
                _once(("window_blocks_full", ablation),
                      f"window block mask: full/default ({len(state.branches)}/{len(state.branches)})")

            # v10 sparse-hybrid experiment. Keep the exact window branch in
            # every transformer block, but evaluate the expensive VDN linear
            # correction only every Nth block.
            #
            # v12 adds an explicit block mask. If VDN_H3_HYBRID_BLOCKS is set,
            # it takes precedence over stride and selects the exact blocks that
            # receive the linear correction. Unlike v10 stride mode, no boundary
            # block is forced in mask mode: the profiler showed block 49 can be
            # among the weakest linear corrections. Example:
            #   VDN_H3_HYBRID_BLOCKS=0,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,19,24
            raw_stride = os.environ.get("VDN_H3_HYBRID_STRIDE", "1").strip()
            try:
                hybrid_stride = max(1, int(raw_stride))
            except ValueError:
                hybrid_stride = 1
                _once(("bad_hybrid_stride", raw_stride),
                      f"invalid VDN_H3_HYBRID_STRIDE={raw_stride!r}; using 1")

            raw_block_mask = os.environ.get("VDN_H3_HYBRID_BLOCKS", "").strip()
            block_mask = None
            if raw_block_mask:
                parsed = set()
                bad = []
                max_block = len(state.branches) - 1
                for tok in raw_block_mask.replace(";", ",").split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    try:
                        idx = int(tok)
                    except ValueError:
                        bad.append(tok)
                        continue
                    if 0 <= idx <= max_block:
                        parsed.add(idx)
                    else:
                        bad.append(tok)
                if parsed:
                    block_mask = frozenset(parsed)
                    if bad:
                        _once(("bad_hybrid_blocks", tuple(bad)),
                              f"ignored invalid VDN_H3_HYBRID_BLOCKS entries: {bad}")
                else:
                    _once(("empty_hybrid_blocks", raw_block_mask),
                          "VDN_H3_HYBRID_BLOCKS contained no valid block indices; "
                          "falling back to VDN_H3_HYBRID_STRIDE")

            if ablation == "hybrid" and linear_active:
                if block_mask is not None:
                    linear_active = block_index in block_mask
                elif hybrid_stride > 1:
                    last_block = len(state.branches) - 1
                    linear_active = (block_index == 0 or block_index == last_block or
                                     (block_index % hybrid_stride) == 0)

            if block_mask is not None:
                mask_sorted = tuple(sorted(block_mask))
                _once(("hybrid_blocks", mask_sorted),
                      f"hybrid block mask: {len(mask_sorted)}/{len(state.branches)} "
                      f"linear blocks = {','.join(map(str, mask_sorted))} "
                      "(explicit mask overrides stride; no forced boundary blocks)")
            else:
                _once(("hybrid_stride", hybrid_stride),
                      f"hybrid stride: {hybrid_stride} "
                      f"({'sparse linear branch' if hybrid_stride > 1 else 'full hybrid'}; "
                      "blocks 0 and last always hybrid)")

            recompute_raw = (ablation == "hybrid" and linear_active and
                             memory_strategy == "recompute_qkv")
            q_raw_video = k_raw_video = v_video = None
            text_x = text_k_raw = text_v_raw = None
            if linear_active and not recompute_raw:
                v_s, e_s = lay.video_start, lay.video_end
                with qkv_ctx("qkv_deep_scratch_acquire"):
                    with qkv_wall_ctx("qkv_scratch_acquire_wall"):
                        buf = state.act_scratch(
                            e_s - v_s,
                            lay.text_len if branch.enable_text_state else 0, device, dtype)
                with qkv_ctx("qkv_deep_video_q_copy"):
                    q_raw_video = buf["q"].copy_(q_raw[v_s:e_s])
                with qkv_ctx("qkv_deep_video_k_copy"):
                    k_raw_video = buf["k"].copy_(k_raw[v_s:e_s])
                with qkv_ctx("qkv_deep_video_v_copy"):
                    v_video = buf["v"].copy_(v[v_s:e_s])
                if branch.enable_text_state and lay.text_len:
                    t_a, t_b = lay.text_start, lay.text_start + lay.text_len
                    text_x = x[t_a:t_b]
                    with qkv_ctx("qkv_deep_text_k_copy"):
                        text_k_raw = buf["tk"].copy_(k_raw[t_a:t_b])
                    with qkv_ctx("qkv_deep_text_v_copy"):
                        text_v_raw = buf["tv"].copy_(v[t_a:t_b])
        prof_memory_snapshot("after_qkv", block_index)

        with prof_section("rope_norm", block_index):
            # The VDN linear branch consumes raw pre-norm/pre-RoPE q/k/v. In
            # linear_only mode RoPE/QK norm would be dead work, so skip it.
            if ablation != "linear_only":
                if rope_freqs is not None:
                    q4 = q.view(1, s, heads, head_dim)
                    k4 = k.view(1, s, heads, head_dim)
                    qw = comfy.model_management.cast_to(q_norm.weight, device=device)
                    kw = comfy.model_management.cast_to(k_norm.weight, device=device)
                    rot = rope_freqs.shape[-3] * 2
                    comfy.quant_ops.ck.rms_rope_split_half_(
                        q4, k4, rope_freqs, qw, kw, epsilon=q_norm.eps, rot_dim=rot)
                    q = q4[0]
                    k = k4[0]
                else:
                    q = q_norm(q_raw)
                    k = k_norm(k_raw)
        # v is NOT cloned: nothing downstream mutates it. The grouped window path
        # gathers k/v rows through index_select (which copies into contiguous
        # scratch), so the strided split view never reaches an SDPA kernel. The
        # two paths that feed v to an attention call directly get a contiguous
        # copy at the call site instead of one clone per block per step.

        def _compute_window():
            if ablation == "linear_only":
                return None
            if window_active:
                if getattr(state, "softmax_backend", "grouped") == "flex":
                    from vdn_h3_24gb.window import window_softmax_flex
                    try:
                        return window_softmax_flex(
                            q, k, v.contiguous(), lay.video_start, lay.video_end,
                            lay.num_frames, lay.tokens_per_frame, lay.bounds,
                            head_dim ** -0.5, anchor_frames=cfg["anchor_frames"])
                    except Exception as e:
                        state.softmax_backend = "grouped"
                        _log.warning("[vdn] flex attention failed (%s); falling back "
                                     "to grouped SDPA", e)
                if getattr(state, "softmax_backend", "grouped") != "flex":
                    from vdn_h3_24gb.window import window_softmax_grouped
                    return window_softmax_grouped(
                        q, k, v, lay.video_start, lay.video_end, lay.num_frames,
                        lay.tokens_per_frame, lay.bounds, head_dim ** -0.5,
                        anchor_frames=cfg["anchor_frames"],
                        retain_buffers=state.retain_buffers,
                        profile_block=block_index)
            q2 = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
            k2 = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
            v2 = AttentionTensorContainer(
                v.contiguous().transpose(0, 1).unsqueeze(0))
            return optimized_attention(
                q2, k2, v2, heads, mask=None, skip_reshape=True,
                transformer_options=transformer_options).squeeze(0)

        # ---------------------------------------------------------------- v9 --
        # Hybrid-only experiment: exact window attention runs on a side CUDA
        # stream while the default stream loads branch weights and evaluates the
        # linear VDN branch.  Crucially, *all* comfy-kitchen QuantizedTensor work
        # remains on the default stream; the side stream sees only ordinary
        # already-materialized q/k/v tensors.  This is intentionally different
        # from v7 prefetch, which proved unsafe on Windows.
        use_overlap = overlap_active and window_active and linear_active
        softmax_out = None
        window_done = None
        overlap_start = overlap_end = None

        if use_overlap:
            main_stream = torch.cuda.current_stream(device)
            side_stream = state.window_stream(device)
            # q/k/v and RoPE were produced on main_stream.  Make the window
            # stream depend on that work before touching them.
            side_stream.wait_stream(main_stream)
            if PROFILER.active:
                overlap_start = torch.cuda.Event(enable_timing=True)
                overlap_end = torch.cuda.Event(enable_timing=True)
                overlap_start.record(main_stream)
            with torch.cuda.stream(side_stream):
                with prof_section("window_softmax", block_index):
                    softmax_out = _compute_window()
                window_done = torch.cuda.Event()
                window_done.record(side_stream)
            # Keep source storages alive until the explicit wait below.  Under
            # cudaMallocAsync lifetime is stream-ordered; references are also
            # retained here so the Python allocator cannot recycle them early.
        else:
            if window_active or lay.full_cover:
                with prof_section("window_softmax", block_index):
                    softmax_out = _compute_window()
            elif (ablation == "hybrid" and window_block_mask is not None and
                  raw_window_fallback == "temporal_proxy"):
                # v15: omitted exact-window blocks still receive a cheap
                # attention-shaped temporal signal.  Unlike v14 zeroing, this
                # keeps same-position temporal mixing, compressed global
                # conditioning, and exact anchor/global query semantics.
                from vdn_h3_24gb.window import window_softmax_temporal_proxy
                with prof_section("window_proxy", block_index):
                    softmax_out = window_softmax_temporal_proxy(
                        q, k, v, lay.video_start, lay.video_end, lay.num_frames,
                        lay.tokens_per_frame, lay.bounds, head_dim ** -0.5,
                        anchor_frames=cfg["anchor_frames"],
                        transformer_options=None)
            else:
                # v14 diagnostic behavior: true zero branch ablation.
                softmax_out = None
            prof_memory_snapshot("after_window", block_index)

        # Branch weights are always resolved/materialized on the default stream.
        # This is the safety boundary of the v9 experiment.
        prof_memory_snapshot("before_branch_weights", block_index)
        with prof_wall_section("branch_weight_load_wall", block_index):
            with prof_section("branch_weight_load", block_index):
                w = state.weights_on(
                    block_index, device, dtype, linear_active=linear_active,
                    need_gate=(ablation != "linear_only" and softmax_out is not None))
        prof_memory_snapshot("after_branch_weights", block_index)

        # In overlap mode compute the complete linear contribution while cuDNN
        # window SDPA is in flight.  We postpone the softmax gate/out projection
        # until the window event is joined.
        linear_add = None
        if linear_active:
            if recompute_raw:
                with prof_section("qkv_recompute", block_index):
                    q2, k2, v2 = qkv_proj(x).split(inner, dim=-1)
                    q2 = q2.view(s, heads, head_dim)
                    k2 = k2.view(s, heads, head_dim)
                    v2 = v2.view(s, heads, head_dim)
                    v_s, e_s = lay.video_start, lay.video_end
                    q_raw_video = q2[v_s:e_s]
                    k_raw_video = k2[v_s:e_s]
                    v_video = v2[v_s:e_s]
                    if branch.enable_text_state and lay.text_len:
                        t_a, t_b = lay.text_start, lay.text_start + lay.text_len
                        text_x = x[t_a:t_b]
                        text_k_raw = k2[t_a:t_b]
                        text_v_raw = v2[t_a:t_b]
                prof_memory_snapshot("after_qkv_recompute", block_index)

            with prof_section("linear_branch", block_index):
                readout = branch.readout(
                    w, x[lay.video_start:lay.video_end], q_raw_video, k_raw_video,
                    v_video, lay.num_frames, lay.tokens_per_frame, lay.bounds,
                    frame_size=lay.frame_size, text_x=text_x, text_k_raw=text_k_raw,
                    text_v_raw=text_v_raw, skip_ends=(cfg["anchor_frames"] == "both"),
                    block_index=block_index)
            if recompute_raw:
                del q2, k2, v2
            state._act = None
            with prof_section("linear_outproj", block_index):
                linear_add = F.linear(readout.type_as(x), w["to_out_linear.weight"])
            del readout

        # Join the side-stream window branch only when its output is actually
        # needed.  The wait is GPU-side; Python does not synchronize the device.
        if use_overlap:
            torch.cuda.current_stream(device).wait_event(window_done)
            if PROFILER.active:
                overlap_end.record(torch.cuda.current_stream(device))
                PROFILER.events.append(("meta_overlap_span", block_index,
                                        overlap_start, overlap_end))
            prof_memory_snapshot("after_window", block_index)

        # The roped/raw full-sequence projections are dead after the join.  Raw
        # video clones used by the linear branch are separate scratch tensors.
        del q, k, v, q_raw, k_raw

        if softmax_out is not None:
            prof_reuse_sample("window", block_index, softmax_out)

        with prof_section("softmax_gate_outproj", block_index):
            if ablation == "linear_only" or softmax_out is None:
                with prof_section("gate_zero_alloc", block_index):
                    out = x.new_zeros((s, out_proj.weight.shape[0]))
            else:
                if cfg["enable_softmax_gate"]:
                    # v19: split the formerly opaque ~1 s/NFE gate/out-projection
                    # section into exact nested CUDA timings.  No math changes.
                    with prof_section("gate_linear", block_index):
                        gate_logits = F.linear(x, w["softmax_gate.up.weight"],
                                               w["softmax_gate.up.bias"])
                    with prof_section("gate_sigmoid", block_index):
                        gate = torch.sigmoid(gate_logits)
                    del gate_logits
                    with prof_section("gate_apply_reshape", block_index):
                        flat = (softmax_out * gate.view(s, heads, 1).to(softmax_out.dtype)) \
                            .reshape(s, -1)
                    del gate
                else:
                    with prof_section("gate_apply_reshape", block_index):
                        flat = softmax_out.reshape(s, -1)
                with prof_section("window_outproj", block_index):
                    out = state.outproj_exact(block_index, out_proj, flat.type_as(x))
                prof_reuse_sample("outproj", block_index, out)
                del flat
                del softmax_out

        # v13 diagnostic: profile the projected window-attention contribution
        # against the block's residual-stream input before any linear correction
        # is added.  This runs for every block, including blocks excluded by an
        # explicit sparse linear mask, and preserves the per-NFE trajectory.
        prof_window_importance_sample(
            block_index, x[lay.video_start:lay.video_end],
            out[lay.video_start:lay.video_end], linear_add)

        if linear_add is not None:
            # v11 diagnostic only: estimate how much this block's linear VDN
            # correction changes the final attention output.  This schedules
            # sampled GPU reductions and defers scalar extraction until the NFE
            # profiler's single synchronization point.
            prof_importance_sample(
                block_index, out[lay.video_start:lay.video_end], linear_add)
            out[lay.video_start:lay.video_end] += linear_add
            del linear_add
            prof_memory_snapshot("after_linear", block_index)
        return out

    vdn_forward._vdn_forward = True
    return vdn_forward


def apply_vdn(new_model, state):
    """Install the layout wrapper and one object patch per DiT block on a cloned
    ModelPatcher."""
    dm = new_model.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None or not hasattr(getattr(blocks[0], "attn", None), "qkv_proj"):
        raise RuntimeError(
            "ApplyVDNH3_24GB: the MODEL's diffusion model is not a ComfyUI MiniMax-H3 "
            "(expected blocks[].attn.qkv_proj). Load a MiniMax-H3 checkpoint first.")
    if len(blocks) != len(state.branches):
        raise RuntimeError(
            f"ApplyVDNH3_24GB: checkpoint has {len(state.branches)} blocks but the loaded "
            f"model has {len(blocks)}; the VDN checkpoint and the base model do not "
            "belong together.")
    for i, block in enumerate(blocks):
        new_model.add_object_patch(
            f"diffusion_model.blocks.{i}.attn.forward",
            make_vdn_forward(block.attn, state, i))
    new_model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "vdn_h3",
                                   make_layout_wrapper(state))
