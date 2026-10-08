"""Opt-in CUDA timing profiler for ComfyUI-VDN-H3.

Enable with VDN_H3_PROFILE=1 before starting ComfyUI. The profiler adds no
math changes. It records CUDA events around the major VDN stages and emits one
summary per MiniMax-H3 diffusion-model forward (normally one sampler NFE).
"""
from __future__ import annotations

import contextlib
import logging
import os
import time
from collections import defaultdict

import torch

_log = logging.getLogger("comfy.vdn")


def enabled() -> bool:
    return os.environ.get("VDN_H3_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

def memory_enabled() -> bool:
    return os.environ.get("VDN_H3_MEM_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

def importance_enabled() -> bool:
    return os.environ.get("VDN_H3_IMPORTANCE_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

def window_importance_enabled() -> bool:
    return os.environ.get("VDN_H3_WINDOW_IMPORTANCE_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

def outproj_block_profile_enabled() -> bool:
    return os.environ.get("VDN_H3_OUTPROJ_BLOCK_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

def window_deep_profile_enabled() -> bool:
    return os.environ.get("VDN_H3_WINDOW_DEEP_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }

def qkv_deep_profile_enabled() -> bool:
    return os.environ.get("VDN_H3_QKV_DEEP_PROFILE", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


class _Profiler:
    def __init__(self):
        self.active = False
        self.forward_index = 0
        self.events = []
        self.wall_events = []
        self.mem_events = []
        self.importance_events = []
        self.importance_history = defaultdict(list)
        self.window_importance_events = []
        self.window_importance_history = defaultdict(list)
        # v35: tiny persistent sampled fingerprints for NFE-to-NFE reuse diagnostics.
        self.reuse_prev = {}
        self.reuse_events = []

    def begin_forward(self, forward_index: int):
        if not enabled() or not torch.cuda.is_available():
            self.active = False
            self.events = []
            self.wall_events = []
            self.mem_events = []
            self.importance_events = []
            self.window_importance_events = []
            return
        self.active = True
        self.forward_index = int(forward_index)
        self.events = []
        self.wall_events = []
        self.mem_events = []
        self.importance_events = []
        self.window_importance_events = []
        self.reuse_events = []
        if int(forward_index) == 1:
            self.reuse_prev = {}

    def reuse_sample(self, stage: str, block: int, tensor: torch.Tensor):
        """v35 diagnostic-only NFE-to-NFE similarity on tiny deterministic samples."""
        if not self.active or os.environ.get("VDN_H3_REUSE_PROFILE", "0").strip().lower() not in {"1","true","yes","on"}:
            return
        raw = os.environ.get("VDN_H3_REUSE_BLOCKS", "0,4,12,24,36,48")
        try:
            blocks = {int(x.strip()) for x in raw.replace(";", ",").split(",") if x.strip()}
        except Exception:
            blocks = {0,4,12,24,36,48}
        if int(block) not in blocks:
            return
        try:
            rs = max(1, int(os.environ.get("VDN_H3_REUSE_ROW_STRIDE", "128")))
            cs = max(1, int(os.environ.get("VDN_H3_REUSE_CHANNEL_STRIDE", "64")))
            t = tensor.detach()
            if t.ndim == 3:
                t = t.reshape(t.shape[0], -1)
            elif t.ndim != 2:
                t = t.reshape(t.shape[0], -1)
            cur = t[::rs, ::cs].contiguous().clone()
            key = (str(stage), int(block))
            prev = self.reuse_prev.get(key)
            if prev is not None and prev.shape == cur.shape:
                a = prev.float(); b = cur.float()
                dot = torch.sum(a*b)
                aa = torch.sum(a*a); bb = torch.sum(b*b)
                diff2 = torch.mean((b-a)*(b-a))
                base2 = torch.mean(a*a)
                self.reuse_events.append((str(stage), int(block), dot, aa, bb, diff2, base2, tuple(cur.shape), rs, cs))
            self.reuse_prev[key] = cur
        except Exception as e:
            _log.warning("[vdn-reuse-v35] %s block %02d sample failed: %s", stage, int(block), e)

    def importance_sample(self, block: int, base_video: torch.Tensor, linear_add: torch.Tensor):
        """Schedule a cheap, deterministic estimate of linear-branch importance.

        Skipping the linear branch changes this attention block's video output by
        exactly ``linear_add``.  We therefore measure the sampled L2 magnitude of
        that delta relative to both the pre-linear (window) output and the final
        output.  Sampling avoids turning the profiler itself into a new bandwidth
        bottleneck on 15k x 5k activations.  No .item()/CPU sync occurs here.
        """
        if not self.active or not importance_enabled() or linear_add is None:
            return
        try:
            rs = max(1, int(os.environ.get("VDN_H3_IMPORTANCE_ROW_STRIDE", "16")))
            cs = max(1, int(os.environ.get("VDN_H3_IMPORTANCE_CHANNEL_STRIDE", "16")))
        except Exception:
            rs = cs = 16
        try:
            b = base_video[::rs, ::cs].float()
            d = linear_add[::rs, ::cs].float()
            f = b + d
            # mean-square is enough; ratios of RMS equal ratios of L2 norms for
            # equally-sized samples. Keep 0-D tensors on GPU until finish_forward.
            b2 = torch.mean(b * b)
            d2 = torch.mean(d * d)
            f2 = torch.mean(f * f)
            dot = torch.mean(b * d)
            self.importance_events.append((int(block), b2, d2, f2, dot, rs, cs))
        except Exception as e:
            _log.warning("[vdn-imp] block %02d sample failed: %s", int(block), e)


    def window_importance_sample(self, block: int, block_input: torch.Tensor,
                                 window_out: torch.Tensor, linear_add: torch.Tensor | None = None):
        """Sample the actual projected window contribution for one attention block.

        The primary metric is RMS(window_out) / RMS(block_input), which asks how
        large the window-attention update is relative to the residual-stream input.
        Unlike the linear-branch profiler, this is recorded for every block and
        every NFE so we can inspect whether window importance changes across the
        diffusion trajectory.  If a linear correction exists, we also retain the
        ratio to the final attention output for diagnostics.  All reductions stay
        on GPU until finish_forward()'s single synchronization point.
        """
        if not self.active or not window_importance_enabled() or window_out is None:
            return
        try:
            rs = max(1, int(os.environ.get("VDN_H3_WINDOW_IMPORTANCE_ROW_STRIDE",
                                            os.environ.get("VDN_H3_IMPORTANCE_ROW_STRIDE", "16"))))
            cs = max(1, int(os.environ.get("VDN_H3_WINDOW_IMPORTANCE_CHANNEL_STRIDE",
                                            os.environ.get("VDN_H3_IMPORTANCE_CHANNEL_STRIDE", "16"))))
        except Exception:
            rs = cs = 16
        try:
            x = block_input[::rs, ::cs].float()
            w = window_out[::rs, ::cs].float()
            if linear_add is not None:
                l = linear_add[::rs, ::cs].float()
                f = w + l
                f2 = torch.mean(f * f)
            else:
                f2 = torch.mean(w * w)
            x2 = torch.mean(x * x)
            w2 = torch.mean(w * w)
            dot = torch.mean(x * w)
            self.window_importance_events.append(
                (int(block), x2, w2, f2, dot, rs, cs))
        except Exception as e:
            _log.warning("[vdn-winimp] block %02d sample failed: %s", int(block), e)

    def memory_snapshot(self, name: str, block: int | None = None):
        if not self.active or not memory_enabled() or not torch.cuda.is_available():
            return
        try:
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            max_allocated = torch.cuda.max_memory_allocated(device)
            free, total = torch.cuda.mem_get_info(device)
            self.mem_events.append((name, block, allocated, reserved, max_allocated, free, total))
        except Exception:
            pass

    @contextlib.contextmanager
    def section(self, name: str, block: int | None = None):
        if not self.active:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((name, block, start, end))


    @contextlib.contextmanager
    def wall_section(self, name: str, block: int | None = None):
        """CPU wall-clock timing for blocking work such as disk/page-cache -> GPU
        staging. This complements CUDA events, which only see work queued on CUDA
        streams and can miss Python, filesystem, and synchronization overhead."""
        if not self.active:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.wall_events.append((name, block, time.perf_counter() - start))

    def finish_forward(self):
        if not self.active:
            return
        self.active = False
        if not self.events:
            return
        # One synchronization per NFE, after all timed work has already been queued.
        torch.cuda.synchronize()
        if self.reuse_events:
            import math
            rows = []
            for stage, block, dot, aa, bb, diff2, base2, shape, rs, cs in self.reuse_events:
                dv=float(dot.item()); av=float(aa.item()); bv=float(bb.item())
                cos=dv/max(1e-30, math.sqrt(max(av,0.0)*max(bv,0.0)))
                rel=math.sqrt(max(float(diff2.item()),0.0)/max(float(base2.item()),1e-30))
                rows.append((stage, block, cos, rel, shape, rs, cs))
            for stage in ("qkv", "window", "outproj"):
                rr=sorted((r for r in rows if r[0]==stage), key=lambda x:x[1])
                if rr:
                    _log.info("[vdn-reuse-v35] NFE %d stage=%s | %s", self.forward_index, stage,
                              " ".join(f"b{b:02d}:cos={cos:.6f},rel={rel:.6f}" for _,b,cos,rel,_,_,_ in rr))
            if rows:
                _log.info("[vdn-reuse-v35-meta] NFE %d sample_shape(first)=%s row_stride=%d channel_stride=%d diagnostic-only",
                          self.forward_index, rows[0][4], rows[0][5], rows[0][6])
        totals = defaultdict(float)
        per_block = defaultdict(lambda: defaultdict(float))
        for name, block, start, end in self.events:
            ms = start.elapsed_time(end)
            totals[name] += ms
            if block is not None:
                per_block[int(block)][name] += ms

        # Main sections are mutually exclusive. deep_* sections are nested inside
        # linear_branch and are therefore reported separately rather than added to
        # the NFE total (otherwise nested timings would double-count).
        # v19 gate_* / window_outproj sections are nested inside the parent
        # softmax_gate_outproj and must not be added to the main CUDA total.
        gate_nested_names = {"gate_linear", "gate_sigmoid", "gate_apply_reshape",
                             "window_outproj", "gate_zero_alloc"}
        window_deep_names = {k for k in totals if k.startswith("window_deep_")}
        qkv_deep_names = {k for k in totals if k.startswith("qkv_deep_")}
        delta_detail_names = {k for k in totals if k.startswith("deep_delta_detail_")}
        main_totals = {k: v for k, v in totals.items()
                       if not k.startswith("deep_") and not k.startswith("meta_")
                       and k not in gate_nested_names and k not in window_deep_names
                       and k not in qkv_deep_names}
        deep_totals = {k: v for k, v in totals.items() if k.startswith("deep_") and k not in delta_detail_names}
        delta_detail_totals = {k: v for k, v in totals.items() if k in delta_detail_names}
        meta_totals = {k: v for k, v in totals.items() if k.startswith("meta_")}
        gate_totals = {k: v for k, v in totals.items() if k in gate_nested_names}
        window_deep_totals = {k: v for k, v in totals.items() if k in window_deep_names}
        qkv_deep_totals = {k: v for k, v in totals.items() if k in qkv_deep_names}
        total_ms = sum(main_totals.values())
        order = (
            "qkv_and_raw_copy", "rope_norm", "window_softmax", "branch_weight_load",
            "softmax_gate_outproj", "qkv_recompute", "linear_branch", "linear_outproj"
        )
        parts = []
        for name in order:
            if name in main_totals:
                pct = 100.0 * main_totals[name] / total_ms if total_ms else 0.0
                parts.append(f"{name}={main_totals[name]/1000:.3f}s ({pct:.1f}%)")
        for name in sorted(k for k in main_totals if k not in order):
            pct = 100.0 * main_totals[name] / total_ms if total_ms else 0.0
            parts.append(f"{name}={main_totals[name]/1000:.3f}s ({pct:.1f}%)")
        _log.info("[vdn-prof] NFE %d CUDA total %.3fs | %s",
                  self.forward_index, total_ms / 1000.0, " | ".join(parts))

        if meta_totals:
            _log.info("[vdn-prof-overlap] NFE %d | %s", self.forward_index,
                      " | ".join(f"{k[5:]}={v/1000:.3f}s"
                                 for k, v in sorted(meta_totals.items())))

        if qkv_deep_totals:
            qkv_total = sum(qkv_deep_totals.values())
            qorder = (
                "qkv_deep_projection",
                "qkv_deep_split_views",
                "qkv_deep_scratch_acquire",
                "qkv_deep_video_q_copy",
                "qkv_deep_video_k_copy",
                "qkv_deep_video_v_copy",
                "qkv_deep_text_k_copy",
                "qkv_deep_text_v_copy",
            )
            qparts = []
            for name in qorder:
                if name in qkv_deep_totals:
                    pct = 100.0 * qkv_deep_totals[name] / qkv_total if qkv_total else 0.0
                    qparts.append(f"{name[9:]}={qkv_deep_totals[name]/1000:.3f}s ({pct:.1f}%)")
            for name in sorted(k for k in qkv_deep_totals if k not in qorder):
                pct = 100.0 * qkv_deep_totals[name] / qkv_total if qkv_total else 0.0
                qparts.append(f"{name[9:]}={qkv_deep_totals[name]/1000:.3f}s ({pct:.1f}%)")
            _log.info("[vdn-prof-qkv] NFE %d qkv/raw CUDA %.3fs (nested) | %s",
                      self.forward_index, qkv_total / 1000.0, " | ".join(qparts))

            rows = []
            for block, vals in per_block.items():
                subtotal = sum(v for k, v in vals.items() if k.startswith("qkv_deep_"))
                if subtotal > 0.0:
                    rows.append((int(block), subtotal, vals.get("qkv_deep_projection", 0.0),
                                 sum(v for k, v in vals.items() if k.startswith("qkv_deep_") and k.endswith("_copy"))))
            rows.sort(key=lambda x: x[1], reverse=True)
            if rows:
                topn = min(8, len(rows))
                _log.info("[vdn-prof-qkv-blocks] NFE %d top%d=%s",
                          self.forward_index, topn,
                          ", ".join(f"b{b:02d}:{tot:.3f}ms(proj={proj:.3f},copies={copies:.3f})"
                                    for b, tot, proj, copies in rows[:topn]))

        if delta_detail_totals:
            dd_total = sum(delta_detail_totals.values())
            dd_order = ("deep_delta_detail_prepare", "deep_delta_detail_cholesky", "deep_delta_detail_rhs_build", "deep_delta_detail_factored_solve", "deep_delta_detail_split_cast")
            ddparts = []
            for name in dd_order:
                if name in delta_detail_totals:
                    pct = 100.0 * delta_detail_totals[name] / dd_total if dd_total else 0.0
                    ddparts.append(f"{name[18:]}={delta_detail_totals[name]/1000:.3f}s ({pct:.1f}%)")
            _log.info("[vdn-prof-delta] NFE %d delta_solve CUDA %.3fs (nested) | %s", self.forward_index, dd_total / 1000.0, " | ".join(ddparts))
            rows = []
            for block_i, vals in per_block.items():
                subtotal = sum(v for k, v in vals.items() if k.startswith("deep_delta_detail_"))
                if subtotal > 0.0:
                    rows.append((int(block_i), subtotal, vals.get("deep_delta_detail_cholesky",0.0), vals.get("deep_delta_detail_factored_solve",0.0)))
            rows.sort(key=lambda x: x[1], reverse=True)
            if rows:
                topn=min(8,len(rows))
                _log.info("[vdn-prof-delta-blocks] NFE %d top%d=%s", self.forward_index, topn, ", ".join(f"b{b:02d}:{tot:.3f}ms(chol={chol:.3f},solve={solv:.3f})" for b,tot,chol,solv in rows[:topn]))

        if gate_totals:
            gate_total = sum(gate_totals.values())
            gorder = ("gate_linear", "gate_sigmoid", "gate_apply_reshape",
                      "window_outproj", "gate_zero_alloc")
            gparts = []
            for name in gorder:
                if name in gate_totals:
                    pct = 100.0 * gate_totals[name] / gate_total if gate_total else 0.0
                    gparts.append(f"{name}={gate_totals[name]/1000:.3f}s ({pct:.1f}%)")
            _log.info("[vdn-prof-gate] NFE %d gate/outproj CUDA %.3fs (nested) | %s",
                      self.forward_index, gate_total / 1000.0, " | ".join(gparts))

            # v21: expose block-level out_proj timing so a fixed VRAM budget can
            # be assigned to the blocks with the highest residency benefit.
            if outproj_block_profile_enabled():
                rows = []
                for block, vals in per_block.items():
                    ms = vals.get("window_outproj", 0.0)
                    if ms > 0.0:
                        rows.append((int(block), float(ms)))
                rows.sort(key=lambda x: x[1], reverse=True)
                if rows:
                    topn = min(12, len(rows))
                    _log.info("[vdn-outproj-prof] NFE %d top%d=%s",
                              self.forward_index, topn,
                              ", ".join(f"b{b:02d}:{ms:.3f}ms" for b, ms in rows[:topn]))
                    _log.info("[vdn-outproj-map] NFE %d %s", self.forward_index,
                              " ".join(f"b{b:02d}={ms:.3f}" for b, ms in sorted(rows)))

        if window_deep_totals:
            wd_total = sum(window_deep_totals.values())
            wd_order = (
                "window_deep_global_sdpa",
                "window_deep_global_scratch_copy",
                "window_deep_q_gather",
                "window_deep_kv_gather",
                "window_deep_group_sdpa",
                "window_deep_scatter",
                "window_deep_anchor_sdpa",
                "window_deep_batch_pack",
            )
            wd_parts = []
            for name in wd_order:
                if name in window_deep_totals:
                    pct = 100.0 * window_deep_totals[name] / wd_total if wd_total else 0.0
                    wd_parts.append(f"{name[12:]}={window_deep_totals[name]/1000:.3f}s ({pct:.1f}%)")
            for name in sorted(k for k in window_deep_totals if k not in wd_order):
                pct = 100.0 * window_deep_totals[name] / wd_total if wd_total else 0.0
                wd_parts.append(f"{name[12:]}={window_deep_totals[name]/1000:.3f}s ({pct:.1f}%)")
            _log.info("[vdn-prof-window] NFE %d exact-window CUDA %.3fs (nested) | %s",
                      self.forward_index, wd_total / 1000.0, " | ".join(wd_parts))

            rows = []
            for block, vals in per_block.items():
                subtotal = sum(v for k, v in vals.items() if k.startswith("window_deep_"))
                if subtotal > 0.0:
                    rows.append((int(block), subtotal))
            rows.sort(key=lambda x: x[1], reverse=True)
            if rows:
                topn = min(8, len(rows))
                _log.info("[vdn-prof-window-blocks] NFE %d top%d=%s",
                          self.forward_index, topn,
                          ", ".join(f"b{b:02d}:{ms:.3f}ms" for b, ms in rows[:topn]))

            # VELA V3.21 READ-ONLY: expose the already-recorded exact GLOBAL SDPA
            # CUDA events for the coherent critical island. No extra attention
            # call is made, so production math remains exactly V3.13.
            critical = []
            for block in range(40, 50):
                vals = per_block.get(block, {})
                ms = float(vals.get("window_deep_global_sdpa", 0.0))
                if ms > 0.0:
                    critical.append((block, ms))
            if critical:
                total_ms = sum(ms for _, ms in critical)
                _log.info("[vela-v3.21][EXACT-GLOBAL-CENSUS] NFE %d blocks40-49 total=%.3fms | %s",
                          self.forward_index, total_ms,
                          " ".join(f"b{b:02d}={ms:.3f}ms" for b, ms in critical))

        if deep_totals:
            deep_total = sum(deep_totals.values())
            deep_order = (
                "deep_features_query", "deep_short_conv_k", "deep_short_conv_v",
                "deep_beta_proj", "deep_frame_statistics", "deep_alpha_gate",
                "deep_text_state", "deep_delta_solve", "deep_scan_recurrence",
                "deep_output_gate", "deep_state_gather", "deep_readout_matmul",
                "deep_epilogue"
            )
            dparts = []
            for name in deep_order:
                if name in deep_totals:
                    pct = 100.0 * deep_totals[name] / deep_total if deep_total else 0.0
                    dparts.append(f"{name[5:]}={deep_totals[name]/1000:.3f}s ({pct:.1f}%)")
            for name in sorted(k for k in deep_totals if k not in deep_order):
                pct = 100.0 * deep_totals[name] / deep_total if deep_total else 0.0
                dparts.append(f"{name[5:]}={deep_totals[name]/1000:.3f}s ({pct:.1f}%)")
            _log.info("[vdn-prof-deep] NFE %d linear-branch CUDA %.3fs (nested) | %s",
                      self.forward_index, deep_total / 1000.0, " | ".join(dparts))

        if self.wall_events:
            wtot = defaultdict(float)
            for name, block, sec in self.wall_events:
                wtot[name] += sec
            _log.info("[vdn-prof-wall] NFE %d | %s", self.forward_index,
                      " | ".join(f"{k}={v:.3f}s" for k, v in sorted(wtot.items())))

        if self.mem_events:
            gib = 1024.0 ** 3
            by_name = defaultdict(list)
            for name, block, allocated, reserved, max_allocated, free, total in self.mem_events:
                by_name[name].append((allocated, reserved, max_allocated, free, total, block))
            parts = []
            for name in ("after_qkv", "after_window", "before_branch_weights",
                         "after_branch_weights", "after_qkv_recompute", "after_linear"):
                vals = by_name.get(name)
                if not vals:
                    continue
                max_alloc = max(v[0] for v in vals) / gib
                max_reserved = max(v[1] for v in vals) / gib
                min_free = min(v[3] for v in vals) / gib
                parts.append(f"{name}:alloc={max_alloc:.2f}GiB,res={max_reserved:.2f}GiB,minfree={min_free:.2f}GiB")
            if parts:
                _log.info("[vdn-prof-mem] NFE %d | %s", self.forward_index, " | ".join(parts))
            # Report the single lowest-free-memory sample to locate the pressure point.
            worst = min(self.mem_events, key=lambda e: e[5])
            _log.info("[vdn-prof-mem] NFE %d pressure-low: %s block=%s free=%.2fGiB alloc=%.2fGiB reserved=%.2fGiB",
                      self.forward_index, worst[0], str(worst[1]), worst[5]/gib,
                      worst[2]/gib, worst[3]/gib)

        if self.importance_events:
            # finish_forward already synchronized above, so scalar extraction is
            # cheap and does not add per-block synchronization.
            import math
            rows = []
            for block, b2_t, d2_t, f2_t, dot_t, rs, cs in self.importance_events:
                b2 = max(float(b2_t.item()), 0.0)
                d2 = max(float(d2_t.item()), 0.0)
                f2 = max(float(f2_t.item()), 0.0)
                dot = float(dot_t.item())
                eps = 1e-20
                ratio_base = math.sqrt(d2 / max(b2, eps))
                impact_final = math.sqrt(d2 / max(f2, eps))
                cosine = dot / math.sqrt(max(b2 * d2, eps)) if d2 > eps else 0.0
                rows.append((impact_final, ratio_base, cosine, block, rs, cs))
                self.importance_history[block].append((impact_final, ratio_base, cosine))

            rows.sort(reverse=True)
            top = rows[:10]
            bottom = sorted(rows, key=lambda x: x[0])[:10]
            fmt = lambda r: f"b{r[3]:02d}:{r[0]:.4f}"
            _log.info("[vdn-imp] NFE %d sampled impact=||linear_delta||/||final|| "
                      "(row_stride=%d channel_stride=%d) | top %s | bottom %s",
                      self.forward_index, rows[0][4], rows[0][5],
                      ", ".join(fmt(r) for r in top),
                      ", ".join(fmt(r) for r in bottom))

            running = []
            for block, vals in self.importance_history.items():
                n = len(vals)
                running.append((sum(v[0] for v in vals)/n,
                                sum(v[1] for v in vals)/n,
                                sum(v[2] for v in vals)/n, block, n))
            running.sort(reverse=True)
            _log.info("[vdn-imp-running] through NFE %d | top %s",
                      self.forward_index,
                      ", ".join(f"b{r[3]:02d}:{r[0]:.4f}" for r in running[:15]))
            # Full compact map makes it possible to design a deterministic block
            # mask after the run without rerunning the profiler.
            by_block = sorted(running, key=lambda r: r[3])
            _log.info("[vdn-imp-map] NFE %d | %s", self.forward_index,
                      " ".join(f"{r[3]:02d}={r[0]:.4f}" for r in by_block))

        if self.window_importance_events:
            import math
            win_rows = []
            for block, x2_t, w2_t, f2_t, dot_t, rs, cs in self.window_importance_events:
                x2 = max(float(x2_t.item()), 0.0)
                w2 = max(float(w2_t.item()), 0.0)
                f2 = max(float(f2_t.item()), 0.0)
                dot = float(dot_t.item())
                eps = 1e-20
                impact_input = math.sqrt(w2 / max(x2, eps))
                ratio_final = math.sqrt(w2 / max(f2, eps))
                cosine_input = dot / math.sqrt(max(x2 * w2, eps)) if w2 > eps else 0.0
                win_rows.append((impact_input, ratio_final, cosine_input, block, rs, cs))
                self.window_importance_history[block].append(
                    (self.forward_index, impact_input, ratio_final, cosine_input))

            win_rows.sort(reverse=True)
            vals = sorted(r[0] for r in win_rows)
            mean = sum(vals) / len(vals)
            median = vals[len(vals)//2] if len(vals) % 2 else 0.5 * (vals[len(vals)//2-1] + vals[len(vals)//2])
            top = win_rows[:10]
            bottom = sorted(win_rows, key=lambda x: x[0])[:10]
            fmt = lambda r: f"b{r[3]:02d}:{r[0]:.4f}"
            _log.info(
                "[vdn-winimp] NFE %d sampled impact=||window_out||/||block_input|| "
                "(row_stride=%d channel_stride=%d) mean=%.4f median=%.4f | top %s | bottom %s",
                self.forward_index, win_rows[0][4], win_rows[0][5], mean, median,
                ", ".join(fmt(r) for r in top), ", ".join(fmt(r) for r in bottom))
            by_block = sorted(win_rows, key=lambda r: r[3])
            _log.info("[vdn-winimp-map] NFE %d | %s", self.forward_index,
                      " ".join(f"{r[3]:02d}={r[0]:.4f}" for r in by_block))

            # Per-NFE mean is intentionally logged separately: unlike the old
            # running-average importance map, the temporal trend itself is the
            # signal we want for future step-sparse window experiments.
            _log.info("[vdn-winimp-step] NFE %d mean=%.4f median=%.4f max=%.4f min=%.4f",
                      self.forward_index, mean, median, vals[-1], vals[0])

        ranked = []
        for block, vals in per_block.items():
            main_vals = {k:v for k,v in vals.items()
                         if not k.startswith("deep_") and not k.startswith("meta_")
                         and not k.startswith("window_deep_")
                         and not k.startswith("qkv_deep_")
                         and k not in gate_nested_names}
            ranked.append((sum(main_vals.values()), block, main_vals))
        ranked.sort(reverse=True)
        for block_total, block, vals in ranked[:5]:
            compact = ", ".join(
                f"{k}={v/1000:.3f}s" for k, v in sorted(vals.items(), key=lambda kv: -kv[1])[:3]
            )
            _log.info("[vdn-prof] slow block %02d: %.3fs (%s)",
                      block, block_total / 1000.0, compact)
        self.events = []
        self.wall_events = []
        self.mem_events = []
        self.importance_events = []
        self.window_importance_events = []
        self.reuse_events = []


PROFILER = _Profiler()


def section(name: str, block: int | None = None):
    return PROFILER.section(name, block)

def wall_section(name: str, block: int | None = None):
    return PROFILER.wall_section(name, block)

def memory_snapshot(name: str, block: int | None = None):
    return PROFILER.memory_snapshot(name, block)


def importance_sample(block: int, base_video: torch.Tensor, linear_add: torch.Tensor):
    return PROFILER.importance_sample(block, base_video, linear_add)

def window_importance_sample(block: int, block_input: torch.Tensor,
                             window_out: torch.Tensor, linear_add: torch.Tensor | None = None):
    return PROFILER.window_importance_sample(block, block_input, window_out, linear_add)


def reuse_sample(stage: str, block: int, tensor: torch.Tensor):
    return PROFILER.reuse_sample(stage, block, tensor)
