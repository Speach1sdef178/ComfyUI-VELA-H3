# ComfyUI VELA H3

**Version 1.0.0** · developed and maintained by **Speach1sdef178**

VELA is a trajectory-aware production execution optimization layer for **VDN-H3 on MiniMax H3 in ComfyUI**. It is the separate continuation of the optimization work started in [ComfyUI-VDN-H3-24GB](https://github.com/Speach1sdef178/ComfyUI-VDN-H3-24GB). VELA keeps the VDN checkpoint format and model behavior while selectively changing how expensive attention paths are executed.

VELA is not a distilled MiniMax model and does not ship model weights. The VDN checkpoint is required separately.

## Why VELA exists

The project began with a simple question: which parts of the VDN + MiniMax H3 runtime can be made faster on a 24 GB Ampere GPU **without treating every transformer block, attention branch or diffusion step as equally tolerant to approximation?**

The resulting production policy is deliberately heterogeneous. Faster approximate backends are used where validation showed they were acceptable, while trajectory-sensitive regions are returned to exact execution. The final release is therefore the result of profiling, numerical A/B testing and block/timestep sensitivity mapping rather than a global backend replacement.

## VELA 1.0.0 production policy

- **QKV projection:** native comfy-kitchen INT8 / ConvRot path.
- **GROUP attention:** Stock Sage path (C40).
- **GLOBAL attention:** Sage normally.
- **Critical GLOBAL island:** gate label 2, transformer blocks **40–49** use exact direct-cuDNN SDPA.
- **ANCHOR attention:** exact.
- **Anchor-column optimization:** extra anchor columns are removed only in blocks **0 and 36**.
- Research/deep profilers are disabled in production.
- The validated VELA configuration is internalized; no external `VDN_H3_*` BAT flags are required.
- Experimental Triton QKV is **not** enabled in VELA 1.0.0.

The internal research checkpoint that became this public release was VELA V3.28. The public version starts at **1.0.0** and follows normal semantic versioning.

## Benchmarks

Matched end-to-end measurements on the project test system:

| Test | VDN 1.1.0 | VELA 1.0.0 | Change |
|---|---:|---:|---:|
| 0.4 MP / 5 sec | 1:14 | **1:01** | **17.6% faster** |
| 0.8 MP / 5 sec | 2:53 | **2:13** | **23.1% faster** |
| 0.8 MP / 7 sec | 3:10 | **3:02** | **4.2% faster** |
| 0.8 MP / 8 sec | 3:31 | **3:23** | **3.8% faster** |
| 0.8 MP / 9 sec | 4:07 | **3:57** | **4.0% faster** |
| 0.8 MP / 10 sec | **4:12** | 4:17 | **2.0% slower** |

Additional VELA scaling validation: **1.0 MP / 5 sec / 8 steps = 2:46 total sampler time, 20.80 s/it**. There is no matched VDN 1.1.0 1.0 MP measurement, so no speedup claim is made for that row.

These results are intentionally reported as a full table rather than a single “up to” number. VELA's benefit is workload-dependent; the relative cost of the optimized attention paths changes with spatial and temporal sequence geometry.

See [BENCHMARKS.md](BENCHMARKS.md) for methodology and [RESEARCH.md](RESEARCH.md) for the complete research history.

## Tested stack

- NVIDIA RTX 3090 Ti 24 GB
- Windows
- Python 3.13.15
- PyTorch 2.13.0 + CUDA 13.0
- ComfyUI 0.33.0 (`v0.33.0-36-g76135e55`)
- comfy-kitchen 0.2.31
- comfy-aimdo 0.4.13
- MiniMax H3 INT8 ConvRot + VDN

This is the validated stack, not a guarantee that every future ComfyUI/PyTorch/comfy-kitchen combination is numerically or performance-equivalent.

## Installation

### ComfyUI Manager / Registry

After the package is published to the Comfy Registry, install **ComfyUI VELA H3** through ComfyUI Manager and restart ComfyUI.

### Manual

Clone or extract the repository to:

```text
<ComfyUI>/custom_nodes/ComfyUI-VELA-H3/
```

The directory should directly contain `__init__.py`, `vdn_h3_24gb/`, `tools/`, `pyproject.toml` and this README.

## VDN checkpoint

VELA uses the same separately distributed VDN stage format as the preceding VDN-H3 project. Place the complete stage at:

```text
<ComfyUI>/models/vdn/stage-dmd-step-250-int8_convrot_comfyui/
├─ model_spec.json
├─ linear_branch/
│  └─ model_int8_convrot_comfyui.safetensors
└─ adapters/
   ├─ default/
   │  ├─ adapter_config.json
   │  └─ adapter_model.safetensors
   └─ turbo/
      ├─ adapter_config.json
      └─ adapter_model.safetensors
```

Checkpoint source: [speach1sdef178/VDN-H3-INT8-ConvRot-ComfyUI](https://huggingface.co/speach1sdef178/VDN-H3-INT8-ConvRot-ComfyUI). Model weights are not included in this repository.

## Core-patch-free integration

VELA v1.0.0 does **not** modify `comfy/ldm/minimax/model.py` or any other ComfyUI core source file. The older VDN research line included an experimental block-loop hook for LongCache; that hook is deliberately not part of the VELA 1.0.0 production path. LongCache remains research-only and disabled in this release. This avoids a fragile on-disk dependency on MiniMax-H3 core implementation details and improves compatibility with upstream ComfyUI features that use the standard model patch system.

## Expected release markers

At startup:

```text
[VELA v1.0.0] production config active; external VDN_H3_* BAT flags are not required
```

During generation:

```text
[VELA v1.0.0] ACTIVE production path | GLOBAL gate2 blocks=40-49 exact direct-cuDNN | other GLOBAL Sage | ANCHOR exact | GROUP Stock Sage | anchor-column removal blocks=0,36
```

These markers are useful when validating that the release package, rather than an older experimental VELA folder, is active.

## Project lineage

VELA is published as a separate project because its purpose is broader than the original 24 GB VDN integration, but it is explicitly a continuation of that work:

**OpenVDN / VideoDeltaNet → ComfyUI VDN-H3 port → Speach1sdef178/ComfyUI-VDN-H3-24GB → ComfyUI VELA H3**

See [NOTICE](NOTICE) and [LICENSE](LICENSE) for attribution.

## Research documentation

- [RESEARCH.md](RESEARCH.md) — full technical research article and experimental history.
- [BENCHMARKS.md](BENCHMARKS.md) — publication benchmark table and methodology.
- [CHANGELOG.md](CHANGELOG.md) — public release history.

## License

Apache License 2.0. Existing upstream notices are preserved. VELA modifications are Copyright 2026 **Speach1sdef178**. MiniMax-H3 model weights and separately distributed checkpoints may be governed by their own licenses; review those terms before redistribution or commercial use.
