"""Apply VDN-H3 (Video Delta Net hybrid attention) to a loaded MiniMax-H3 model."""

import logging
import os

import folder_paths

import comfy.model_management
import comfy.model_prefetch
import comfy.cli_args

from vdn_h3_24gb.apply import apply_adapters
from vdn_h3_24gb.hybrid import VDNState, apply_vdn
from vdn_h3_24gb.branch import LinearBranch
import vdn_h3_24gb.spec as spec

_log = logging.getLogger("comfy.vdn")


_COMPILER_DISABLED_BY_VDN = False
_COMPILER_WARNED = False


def _disable_comfy_compiler_on_broken_builds():
    """Comfy builds from 2026-09-04 ship a model compiler + aimdo malloc-graph
    that hard-fails on patched MiniMax-H3 forwards (graph breaks raise
    'aimdo memory compile error'; some paths abort the process mid-step). The
    node does not need that compiler, so on affected builds we switch it off --
    the same effect as launching with --disable-comfy-compiler, without asking
    anything of the user. hybrid.py scopes the switch to VDN forwards only
    (restored in a finally after every step), so non-VDN workflows keep it.
    No-op on builds without the compiler stack.

    Returns True when the switch is ours to manage."""
    global _COMPILER_DISABLED_BY_VDN, _COMPILER_WARNED
    try:
        args = comfy.cli_args.args
        # The compiler stack (comfy 2026-09-04, "Introduce Comfy Compiler") is
        # what breaks VDN: model_prefetch gains the aimdo malloc-graph planner
        # and cli_args gains the --disable-comfy-compiler switch. model_prefetch
        # binds the package (import comfy_aimdo.malloc_graph -> comfy_aimdo), so
        # probe the package for the submodule, not the module name.
        if not hasattr(args, "disable_comfy_compiler"):
            return False
        aimdo = getattr(comfy.model_prefetch, "comfy_aimdo", None)
        if aimdo is None or not hasattr(aimdo, "malloc_graph"):
            return False
        if getattr(args, "disable_comfy_compiler", False):
            return _COMPILER_DISABLED_BY_VDN
        args.disable_comfy_compiler = True
        _COMPILER_DISABLED_BY_VDN = True
        if not _COMPILER_WARNED:
            _COMPILER_WARNED = True
            _log.warning(
                "[vdn] this comfy build's model compiler crashes with VDN-H3 "
                "(aimdo malloc-graph); disabling it while VDN is sampling. "
                "Remove this once comfy fixes the compiler.")
        return True
    except Exception:
        return False


def _auto_memory_from_latent(latent):
    """v49 AutoLongCache policy with a VERY_LONG (T>=96) calibration profile.

    The H3 LATENT temporal dimension is already the temporal token/frame count
    consumed by MiniMax-H3 for the tested ComfyUI workflows (e.g. T=37 for the
    ~5 s clip and T=87 for the ~12 s clip).  Earlier v41-v44 builds attempted to
    re-expand small T values and could misclassify T=37 as a long clip.  v45
    therefore uses T directly and makes all duration decisions from work=T*S.

    Short clips retain the proven v39/v40 spatial memory curve.  Long clips
    switch to the lower-peak B=1 window path, disable out_proj residency cache,
    purge helper models before H3 load, and enable the validated tail-block
    LongCache.  Cache depth reaches the tested 0.60 profile at T>=85.  v47 adds
    a VERY_LONG class at T>=96. v49 lowers its loader headroom to 2.50 GiB so 15 s
    workloads can preserve some H3 GPU residency instead of falling to 0 MB.
    """
    try:
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None or samples.ndim < 5:
            return None
        t = max(1, int(samples.shape[-3]))
        h = int(samples.shape[-2]); w = int(samples.shape[-1])
        ph = h + (h & 1); pw = w + (w & 1)
        spatial_tokens = (ph // 2) * (pw // 2)
        work_tokens = int(spatial_tokens * t)

        # Proven v39/v40 short-video spatial curve.
        short_headroom = 4.0 + max(0, spatial_tokens - 405) * 0.0040
        short_headroom = min(6.5, max(4.0, short_headroom))
        if spatial_tokens > 798:
            upper_t = min(1.0, (spatial_tokens - 798) / float(1032 - 798))
            short_headroom -= 0.25 * upper_t

        if spatial_tokens <= 405:
            short_cache = 0.75
        elif spatial_tokens >= 798:
            short_cache = 0.0
        else:
            short_cache = 0.75 * (798 - spatial_tokens) / (798 - 405)

        # Duration classes use the REAL latent T.  T=37 (~5 s) stays short;
        # T=87 (~12 s) is the validated long profile.  Intermediate clips ramp
        # cache depth smoothly instead of jumping directly to 0.60.
        long_video = t > 48
        very_long = t >= 96
        group_batch = 3
        purge_before_h3 = False
        cache = short_cache
        headroom = short_headroom
        longcache_depth = 0.0

        if long_video:
            group_batch = 1
            purge_before_h3 = True
            cache = 0.0

            # Empirical 24 GiB long-video balance: at 0.8 MP / T87, 5.0 GiB
            # gives materially better H3 residency than larger reservations.
            # Move toward that value as duration grows, never increasing the
            # short-video reservation merely because T is larger.
            dur = max(0.0, min(1.0, (t - 48) / float(87 - 48)))
            target_long_hr = 5.0
            headroom = short_headroom + (target_long_hr - short_headroom) * dur
            if t >= 87:
                headroom = target_long_hr

            # v49 calibration for 15 s / T~107 on 24 GiB Ampere. v46 at
            # 5.00 GiB and v48 at 3.50 GiB both pushed H3 to 0 MB resident.
            # Try 2.50 GiB; this remains a calibration point, not an
            # extrapolated duration curve.
            if very_long:
                headroom = 2.50

            # AutoLongCache schedule: mild for medium clips, validated 0.60 at
            # T>=85.  The NFE pattern remains 3/5/7 with no consecutive CACHE.
            if t < 64:
                longcache_depth = 0.40
            elif t < 85:
                longcache_depth = 0.50
            else:
                longcache_depth = 0.60

        return {
            "spatial_tokens": spatial_tokens,
            "latent_t": t,
            "work_tokens": work_tokens,
            "headroom_gib": headroom,
            "cache_gib": cache,
            "latent_hw": (h, w),
            "group_batch": group_batch,
            "purge_before_h3": purge_before_h3,
            "long_video": long_video,
            "very_long": very_long,
            "longcache_depth": longcache_depth,
        }
    except Exception:
        return None


def _apply_vdn(model, vdn_checkpoint, strength, lora_mode, branch_weights,
               attention_backend, verbose, apply_turbo_adapter=True,
               cfg_overrides=None, fast_kernels=False, retain_buffers="auto",
               auto_memory_latent=None):
    """Shared core of ApplyVDNH3 and ApplyVDNH3Advanced. `strength` is a float or a
    {adapter_name: float} map; `cfg_overrides` deviates from the checkpoint's trained
    spec (ablation knobs); `fast_kernels` torch.compiles the branch's hot spots
    (epilogue, state gather, frame-major q store, bidirectional scan)."""
    _disable_comfy_compiler_on_broken_builds()
    path = spec.resolve_vdn_checkpoint(vdn_checkpoint)
    prefer_int8 = False
    retain = True
    if branch_weights == "auto" or retain_buffers == "auto":
        # free VRAM right now = after the base model's load in the same run
        free = comfy.model_management.get_free_memory(
            comfy.model_management.get_torch_device())
    else:
        free = None
    if branch_weights == "auto":
        branch_weights, prefer_int8 = spec.auto_branch_policy(path, free)
    cfg, branch_weights_by_block, adapters = spec.load_vdn_checkpoint(
        path, prefer_int8=prefer_int8)
    auto_memory_enabled = os.environ.get("VDN_H3_AUTO_MEMORY", "0").strip().lower() in ("1", "true", "yes", "on")
    if retain_buffers == "auto":
        if auto_memory_enabled:
            # v40: AutoMemory owns the VRAM policy.  A transiently high free-VRAM
            # reading between prompts must not enable retained scratch/banks,
            # because the subsequent loader-headroom reservation can then force
            # H3 into extreme offload and cause a performance cliff.
            retain = False
            _log.info("[vdn] v41 AutoMemory + retain_buffers=auto: forcing transient buffers (single VRAM policy owner)")
        else:
            retain = spec.auto_retain_policy(path, prefer_int8, free)
    else:
        retain = retain_buffers == "on"

    if cfg_overrides:
        changed = {k: (cfg.get(k), v) for k, v in cfg_overrides.items()
                   if k in cfg and cfg.get(k) != v}
        if changed:
            logging.warning("[vdn] deviating from the trained spec: %s", changed)
        cfg = dict(cfg, **cfg_overrides)

    dm = model.get_model_object("diffusion_model")
    blocks = getattr(dm, "blocks", None)
    if blocks is None or not hasattr(getattr(blocks[0], "attn", None), "qkv_proj"):
        raise RuntimeError(
            "ApplyVDNH3_24GB needs a MiniMax-H3 MODEL (blocks[].attn.qkv_proj). "
            "Connect a MiniMax-H3 diffusion model loader first.")
    if len(blocks) != len(branch_weights_by_block):
        raise RuntimeError(
            f"VDN checkpoint has {len(branch_weights_by_block)} blocks but the "
            f"loaded model has {len(blocks)}; this VDN checkpoint belongs to a "
            "different base.")

    for key in model.object_patches:
        if key.endswith(".attn.forward"):
            if getattr(model.object_patches[key], "_vdn_forward", False):
                raise RuntimeError("This MODEL already has VDN-H3 applied; chain "
                                   "it only once.")
            logging.warning("[vdn] replacing an existing attention forward patch "
                            "(%s); attention patches applied before VDN will no "
                            "longer run on the softmax path", key)

    attn0 = blocks[0].attn
    num_heads, head_dim = attn0.heads, attn0.head_dim
    hidden = dm.hidden_size
    lin_dim = cfg["linear_head_dim"]
    expected = {"to_out_linear.weight": (hidden, num_heads * lin_dim),
                "beta_proj.weight": (num_heads, hidden),
                "alpha.A_log": (num_heads,),
                "alpha.dt_bias": (num_heads * lin_dim,),
                "alpha.down.weight": (lin_dim, hidden),
                "alpha.up.weight": (num_heads * lin_dim, lin_dim),
                "output_gate.down.weight": (lin_dim, hidden),
                "output_gate.up.weight": (num_heads * lin_dim, lin_dim),
                "output_gate.up.bias": (num_heads * lin_dim,),
                "softmax_gate.up.weight": (num_heads, hidden),
                "softmax_gate.up.bias": (num_heads,),
                "norm.weight": (lin_dim,)}
    sample = branch_weights_by_block[0]
    for key, shape in expected.items():
        if key in sample and tuple(sample[key].shape) != shape:
            raise RuntimeError(
                f"VDN checkpoint/{path}: {key} has shape "
                f"{tuple(sample[key].shape)}, expected {shape} for this base "
                f"(heads={num_heads}, head_dim={head_dim}, hidden={hidden}).")

    branches = [LinearBranch(w, num_heads, head_dim,
                             delta_rule=cfg["delta_rule"], bridge=cfg["bridge"],
                             a_fp32=cfg["a_fp32"], short_conv=cfg["short_conv"],
                             enable_text_state=cfg["enable_text_state"],
                             retain_buffers=retain)
                for w in branch_weights_by_block]
    for b in branches:
        b.fuse_epilogue = fast_kernels
    if fast_kernels and "dmd" in os.path.basename(path).lower():
        _log.warning(
            "[vdn] fast_kernels on an 8-step DMD stage (%s): the compiled branch "
            "kernels are known to drift on 8-step DMD checkpoints (measurably "
            "visible output on torch 2.10) -- ablation use only, do not use for "
            "final renders", os.path.basename(path))
    state = VDNState(vdn_checkpoint, cfg, branches, num_heads, head_dim)
    state.owns_compiler_switch = _disable_comfy_compiler_on_broken_builds()
    state.retain_buffers = retain
    state.cache_gpu = branch_weights == "cache_gpu"
    state.softmax_backend = attention_backend

    new_model = model.clone()
    apply_vdn(new_model, state)

    wanted = {"default"}
    if apply_turbo_adapter:
        wanted.add("turbo")
    missing = wanted - set(adapters)
    if "default" in missing:
        raise RuntimeError(
            f"{vdn_checkpoint}: the 'default' (Stage-B) adapter is missing; this "
            "checkpoint cannot reproduce the released model. Re-download the "
            "stage directory.")
    converted = {}
    for name in sorted(wanted & set(adapters)):
        loader, adapter_cfg = adapters[name]
        from vdn_h3_24gb.adapters import convert_adapter
        converted[name] = convert_adapter(loader(), adapter_cfg)
        if verbose:
            _log.info("[vdn] adapter %s: %d modules (%s)", name,
                      len(converted[name]),
                      ", ".join(sorted(converted[name])[:3]) + ", ...)")
    report = apply_adapters(new_model, converted, strength, lora_mode)

    # v37 lifecycle fix: reserve VDN workspace through ComfyUI's model-loader
    # accounting instead of a one-shot temporary CUDA allocation.  This policy
    # survives VideoVAE decode and is therefore applied again when H3 is loaded
    # for a second/subsequent prompt.
    auto_enabled = auto_memory_enabled
    auto = _auto_memory_from_latent(auto_memory_latent) if auto_enabled else None
    if auto is not None:
        spatial_tokens = auto["spatial_tokens"]
        forced_headroom_gib = auto["headroom_gib"]
        auto_cache_gib = auto["cache_gib"]
        latent_hw = auto["latent_hw"]

        # Calibration escape hatches.  VERY_LONG has its own override so 15 s
        # tuning does not disturb the validated T87 / 12 s LONG profile.
        if auto["long_video"]:
            env_name = "VDN_H3_VERY_LONG_HEADROOM_GIB" if auto.get("very_long") else "VDN_H3_LONG_HEADROOM_GIB"
            raw_long_hr = os.environ.get(env_name, "").strip()
            if raw_long_hr:
                try:
                    forced_headroom_gib = max(0.0, float(raw_long_hr))
                    _log.info("[vdn-vram] v49 %s headroom manual override: %.2f GiB",
                              "VERY_LONG" if auto.get("very_long") else "LONG", forced_headroom_gib)
                except ValueError:
                    _log.warning("[vdn-vram] invalid %s=%r; using AutoLongCache value %.2f GiB",
                                 env_name, raw_long_hr, forced_headroom_gib)
        state.outproj_cache_gib = auto_cache_gib

        # v46 transition lifecycle fix.  AutoMemory can change the process-wide
        # loader reservation between prompts, but ComfyUI may keep the previous
        # H3 residency alive.  In particular LONG -> SHORT could therefore keep
        # a heavily offloaded H3 even after the SHORT headroom was restored.
        # Track the last memory class process-wide and, on a class transition,
        # unload stale model residency before the next H3 lazy load.
        mm = comfy.model_management
        memory_class = "LONG" if auto["long_video"] else "SHORT"
        previous_class = getattr(mm, "_VDN_H3_LAST_MEMORY_CLASS", None)
        transitioned = previous_class is not None and previous_class != memory_class
        if transitioned:
            try:
                # Install the new reservation first so any subsequent lazy load
                # observes the destination class policy, then evict stale H3/
                # helper residency and clear allocator slack.
                state.install_loader_headroom(forced_headroom_gib, exact=True)
                mm.unload_all_models()
                mm.soft_empty_cache()
                _log.info(
                    "[vdn-vram] v49 transition reset: %s -> %s; unloaded stale model residency before H3 reload",
                    previous_class, memory_class)
            except Exception as e:
                _log.warning(
                    "[vdn-vram] v49 transition reset failed for %s -> %s: %s",
                    previous_class, memory_class, e)
        setattr(mm, "_VDN_H3_LAST_MEMORY_CLASS", memory_class)

        # v41 long-video hygiene: prompt encoding is already complete when this
        # node executes, so unload currently resident TE/VAE weights before H3
        # residency is chosen.  Their model objects remain valid and ComfyUI
        # reloads them later for decode.  This prevents a 10-15 s prompt from
        # leaving almost the entire H3 on CPU simply because TE/VAE stayed hot.
        if auto["purge_before_h3"] and not transitioned:
            try:
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                _log.info("[vdn-vram] v41 long-video pre-H3 purge: unloaded resident helper models before H3 load")
            except Exception as e:
                _log.warning("[vdn-vram] v41 long-video pre-H3 purge failed: %s", e)

        if not transitioned:
            state.install_loader_headroom(forced_headroom_gib, exact=True)
        os.environ["VDN_H3_WINDOW_GROUP_BATCH"] = str(auto["group_batch"])
        _log.info(
            "[vdn-vram] v49 AutoMemory: latent=T%d %sx%s S=%d work=%d -> "
            "headroom=%.2f GiB, out_proj_cache=%.2f GiB, window_group_batch=%d%s%s",
            auto["latent_t"], latent_hw[0], latent_hw[1], spatial_tokens,
            auto["work_tokens"], forced_headroom_gib, auto_cache_gib,
            auto["group_batch"], (" VERY_LONG" if auto.get("very_long") else " LONG") if auto["long_video"] else "",
            (" depth=%.2f" % auto["longcache_depth"]) if auto["long_video"] else "")
    else:
        raw_headroom = os.environ.get("VDN_H3_FORCED_HEADROOM_GIB", "0").strip()
        try:
            forced_headroom_gib = max(0.0, float(raw_headroom or "0"))
        except ValueError:
            forced_headroom_gib = 0.0
            _log.warning("[vdn-vram] invalid VDN_H3_FORCED_HEADROOM_GIB=%r; disabled",
                         raw_headroom)
        if forced_headroom_gib > 0:
            state.install_loader_headroom(forced_headroom_gib)
        if auto_enabled:
            _log.warning("[vdn-vram] v41 AutoMemory requested but no LATENT is connected; using fixed BAT values")

    _log.info("[vdn] %s applied on %d blocks (%s): %s", vdn_checkpoint,
              len(branches),
              f"r={cfg['radius']} c={cfg['chunk']} anchors={cfg['anchor_frames']} "
              f"rule={cfg['delta_rule']}"
              + (" fused" if fast_kernels else ""), report)
    return (new_model,)


class ApplyVDNH3:
    @classmethod
    def INPUT_TYPES(cls):
        names = spec.list_vdn_checkpoints()
        return {"required": {
            "model": ("MODEL", {
                "tooltip": "The MiniMax-H3 diffusion model to patch. Chain once, "
                           "between the model loader and the sampler."}),
            "vdn_checkpoint": (names or ["<place a VDN stage-... directory under models/vdn>"], {
                "tooltip": "The VDN stage directory (models/vdn) holding the "
                           "linear-branch weights and spec. Must match the loaded "
                           "base (stage-dmd-* = 8-step distilled model)."}),
            "apply_turbo_adapter": ("BOOLEAN", {
                "default": True,
                "tooltip": "Apply the 'turbo' adapter when the checkpoint carries one "
                           "(stage-dmd = the 8-step VDN-H3 model). OFF gives the "
                           "50-step model the checkpoint was distilled from. Use 8 "
                           "sampler steps with it ON, 50 with it OFF."}),
            "strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Adapter strength. 1.0 is the released model."}),
            "lora_mode": (["bypass", "merge"], {
                "default": "merge",
                "tooltip": "merge: adapters folded into the weights -- reproduces "
                           "the validated model exactly. REQUIRED for 8-step DMD "
                           "checkpoints (stage-dmd-*): bypass's activation-space "
                           "rounding noise is amplified by the deep blocks and "
                           "visibly degrades output."}),            "branch_weights": (["stream", "auto", "cache_gpu"], {
                "default": "stream",
                "tooltip": "stream (24GB tested default): branch weights are moved to the GPU per block per step. auto: cache_gpu when the free VRAM after the "
                           "base load exceeds 1.5x the stage size + 4 GiB headroom, "
                           "else stream (prefers the int8_convrot stage file under "
                           "memory pressure). stream: the ~4.3 GB of linear-branch "
                           "weights are moved to the GPU per block per step, with a "
                           "one-block lookahead prefetch (safe on small cards). "
                           "cache_gpu: resident on the GPU after the first step "
                           "(faster; keep ~4.3 GB VRAM free)."}),
            "retain_buffers": (["auto", "on", "off"], {
                "default": "auto",
                "tooltip": "Retained branch scratch/banks (scan banks, delta "
                           "solve, window gather, q/k/v copies + prefetch) trade "
                           "~0.5-1 GiB VRAM for churn-free steps. auto: retain "
                           "when free VRAM >= stage + 10 GiB headroom, else "
                           "transient (v1.3.1 allocation pattern, peak VRAM "
                           "priority on small cards). on/off override."}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Log the applied "
                        "adapters and the per-forward layout to the console."}),
            "attention_backend": (["grouped", "flex"], {
                "default": "grouped",
                "tooltip": "How the windowed softmax runs. grouped: one dense SDPA "
                           "per window group (portable, exact). flex: the whole "
                           "pattern as one compiled FlexAttention kernel over the "
                           "full sequence (faster on long clips; first run compiles, "
                           "falls back to grouped if compile fails)."}),
        }, "optional": {
            "auto_memory_latent": ("LATENT", {
                "tooltip": "v49 AutoLongCache: connect the SAME H3 LATENT that goes to the sampler. "
                           "The node derives spatial workload before H3 lazy-load and automatically "
                           "chooses 24 GiB Ampere headroom/cache for ~0.2-1.0 MP and up to ~15 s."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "VDN-H3: hybrid attention for MiniMax-H3. Nearby frames keep exact softmax "
        "attention; distant context goes through the checkpoint's linear Video Delta "
        "Attention branch. Requires a VDN checkpoint (models/vdn, from "
        "OpenVDN/vdn-minimax-h3) and a MiniMax-H3 base model.")

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, strength, lora_mode,
              branch_weights, attention_backend, verbose, retain_buffers="auto",
              auto_memory_latent=None):
        return _apply_vdn(model, vdn_checkpoint, strength, lora_mode, branch_weights,
                          attention_backend, verbose,
                          apply_turbo_adapter=apply_turbo_adapter,
                          retain_buffers=retain_buffers,
                          auto_memory_latent=auto_memory_latent)


class ApplyVDNH3Advanced:
    """Everything the base node does, plus per-adapter strengths, ablation knobs that
    deviate from the released spec (window radius/chunk, anchor frames, text state,
    linear branch), and compile-fused branch kernels (fast_kernels)."""

    @classmethod
    def INPUT_TYPES(cls):
        names = spec.list_vdn_checkpoints()
        return {"required": {
            "model": ("MODEL", {
                "tooltip": "The MiniMax-H3 diffusion model to patch. Chain once, "
                           "between the model loader and the sampler."}),
            "vdn_checkpoint": (names or ["<place a VDN stage-... directory under models/vdn>"], {
                "tooltip": "The VDN stage directory (models/vdn) holding the "
                           "linear-branch weights and spec. Must match the loaded "
                           "base (stage-dmd-* = 8-step distilled model)."}),
            "apply_turbo_adapter": ("BOOLEAN", {
                "default": True,
                "tooltip": "Apply the 'turbo' adapter when the checkpoint carries "
                           "one (8-step model). See the base node's tooltip."}),
            "stage_b_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Strength of the 'default' (Stage-B) adapter."}),
            "turbo_strength": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "Strength of the 'turbo' (8-step DMD) adapter."}),
            "lora_mode": (["bypass", "merge"], {
                "default": "merge",
                "tooltip": "merge required for 8-step DMD checkpoints; see the "
                           "base node's tooltip."}),
            "branch_weights": (["stream", "auto", "cache_gpu"], {
                "default": "stream",
                "tooltip": "stream (24GB tested default): branch weights are moved to the GPU per block per step. auto: cache_gpu when the free VRAM after the "
                           "base load exceeds 1.5x the stage size + 4 GiB headroom, "
                           "else stream (prefers the int8_convrot stage file under "
                           "memory pressure). stream: branch weights move to the "
                           "GPU per block per step with a one-block lookahead "
                           "prefetch (safe on small cards). cache_gpu: resident on "
                           "the GPU after the first step (faster; keep ~4.3 GB "
                           "VRAM free)."}),
            "retain_buffers": (["auto", "on", "off"], {
                "default": "auto",
                "tooltip": "Retained branch scratch/banks (scan banks, delta "
                           "solve, window gather, q/k/v copies + prefetch) trade "
                           "~0.5-1 GiB VRAM for churn-free steps. auto: retain "
                           "when free VRAM >= stage + 10 GiB headroom, else "
                           "transient (v1.3.1 allocation pattern, peak VRAM "
                           "priority on small cards). on/off override."}),
            "verbose": ("BOOLEAN", {"default": False, "tooltip": "Log the applied "
                        "adapters and the per-forward layout to the console."}),
            "attention_backend": (["grouped", "flex"], {
                "default": "grouped",
                "tooltip": "How the windowed softmax runs. grouped: one dense SDPA "
                           "per window group (portable, exact). flex: the whole "
                           "pattern as one compiled FlexAttention kernel over the "
                           "full sequence (faster on long clips; first run compiles, "
                           "falls back to grouped if compile fails)."}),
        }, "optional": {
            "window_radius": ("INT", {
                "default": 1, "min": 0, "max": 8,
                "tooltip": "Softmax window radius in chunks. Trained: 1."}),
            "window_chunk": ("INT", {
                "default": 5, "min": 0, "max": 64,
                "tooltip": "Chunk size for the aligned window; 0 = per-frame "
                           "centered window. Trained: 5."}),
            "anchor_frames": (["both", "columns", "rows", "none"], {
                "default": "both",
                "tooltip": "Boundary-frame anchors. Trained: both."}),
            "text_state": ("BOOLEAN", {
                "default": True,
                "tooltip": "Write the prompt into the linear branch's states at "
                           "init. Trained: on."}),
            "linear_branch": ("BOOLEAN", {
                "default": True,
                "tooltip": "Off = window-only ablation (debug; output then lacks "
                           "all long-range context on clips longer than the "
                           "window)."}),
            "fast_kernels": ("BOOLEAN", {
                "default": False,
                "tooltip": "torch.compile the branch's hot spots (RMSNorm+gate "
                           "epilogue, state gather, frame-major q store, and the "
                           "bidirectional scan as one CUDA-graph replay). Same "
                           "math; falls back to eager if compile fails. First run "
                           "compiles. Known to drift on 8-step DMD stages "
                           "(stage-dmd-*) on torch 2.10 -- ablation use only, "
                           "keep off for final renders (a warning is logged)."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model_patch/video"
    DESCRIPTION = (
        "VDN-H3 advanced: per-adapter strengths, window/anchor/text/branch ablations, "
        "and compile-fused branch kernels. Defaults reproduce the released "
        "model exactly.")

    def apply(self, model, vdn_checkpoint, apply_turbo_adapter, stage_b_strength,
              turbo_strength, lora_mode, branch_weights, attention_backend, verbose,
              retain_buffers="auto",
              window_radius=1, window_chunk=5, anchor_frames="both", text_state=True,
              linear_branch=True, fast_kernels=False):
        strength = {"default": stage_b_strength, "turbo": turbo_strength}
        cfg_overrides = {"radius": window_radius, "chunk": window_chunk,
                         "anchor_frames": anchor_frames,
                         "enable_text_state": text_state,
                         "linear_enabled": linear_branch}
        return _apply_vdn(model, vdn_checkpoint, strength, lora_mode, branch_weights,
                          attention_backend, verbose,
                          apply_turbo_adapter=apply_turbo_adapter,
                          cfg_overrides=cfg_overrides, fast_kernels=fast_kernels,
                          retain_buffers=retain_buffers)


NODE_CLASS_MAPPINGS = {"ApplyVDNH3_24GB": ApplyVDNH3}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyVDNH3_24GB": "Apply VDN + VELA H3 [1.0.0]"}
