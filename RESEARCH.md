# VELA: Trajectory-Aware Execution Optimization for VDN-H3 / MiniMax H3

**Release:** VELA 1.0.0  
**Author / maintainer:** Speach1sdef178  
**Reference hardware:** NVIDIA RTX 3090 Ti 24 GB

## Abstract

VELA is an execution-optimization study and production runtime for the VDN-H3 / MiniMax H3 pipeline in ComfyUI. The work began from a 24 GB-oriented VDN integration and investigated where runtime could be reduced without assuming that all attention branches, transformer depths, diffusion evaluations, or low-level kernels have equal numerical sensitivity. The study combined end-to-end profiling, attention-backend comparison, selective anchor sparsity, diffusion-step and depth isolation, direct cuDNN execution, QKV kernel decomposition, Triton backend experiments, single-block trajectory mapping, and multi-block interaction tests.

The principal result is a heterogeneous production policy: retain native INT8/ConvRot QKV, use Stock Sage for GROUP attention, use Sage for ordinary GLOBAL attention, restore exact direct-cuDNN GLOBAL attention for a critical gate/depth island (gate label 2, blocks 40–49), keep ANCHOR exact, and remove extra anchor columns only in validated blocks 0 and 36. Experimental Triton QKV kernels were measurably faster but were rejected from the default release because their small end-to-end gain at the primary 0.8 MP geometry was accompanied by additional trajectory drift.

## 1. Motivation

VDN makes MiniMax H3 practical on constrained consumer hardware, but the resulting runtime is a mixture of dense and sparse attention, model patching, INT8 projections, long-sequence memory management, and sampler-dependent execution. A global rule such as “replace every attention call with the fastest backend” is attractive but unsafe: diffusion models can amplify small numerical changes, and the sensitivity can depend on timestep and transformer depth.

VELA therefore treated speed and trajectory preservation as a joint optimization problem. Every promising microbenchmark was required to survive an end-to-end generation A/B before being considered for production.

## 2. Baseline and methodology

The development stack was frozen around an RTX 3090 Ti 24 GB, Windows, Python 3.13.15, PyTorch 2.13.0+cu130, ComfyUI 0.33.0, comfy-kitchen 0.2.31 and comfy-aimdo 0.4.13. The standard diagnostic run used approximately 0.8028 MP (896×896), 5 seconds and 8 steps with the same model, seed and workflow.

Candidate changes were evaluated at three levels:

1. **Kernel/operation timing** to locate expensive execution paths.
2. **Sampler/end-to-end timing** to determine whether a local speedup survived integration overhead.
3. **Latent trajectory comparison**, primarily relative RMSE of saved video/audio latents against a frozen reference, to detect changes that could be visually subtle in a single sample but structurally significant.

This separation became essential: several kernels that looked faster in isolation were neutral or harmful end-to-end, and several small numerical changes produced unexpectedly large diffusion-trajectory changes.

## 3. Selective anchor sparsity

Early VELA work tested whether anchor columns could be removed from window attention. The key finding was that this optimization was not uniformly safe across the transformer. Blocks **0 and 36** formed a useful production subset: removing the extra anchor columns there reduced work while preserving the validated behavior. Expanding the same idea to additional blocks produced either quality drift or insufficient performance benefit.

This became the first durable VELA optimization and remains enabled in 1.0.0. Natural local edge-frame information is preserved; the optimization removes only the additional anchor columns in the selected blocks.

## 4. Why GROUP, GLOBAL and ANCHOR were separated

Profiling showed that the three attention paths should not share one backend policy. GROUP attention has different tensor geometry and batching behavior from GLOBAL and ANCHOR. Backend census experiments compared exact cuDNN/memory-efficient SDPA with Sage on real production tensors rather than synthetic shapes.

For dominant GROUP geometry, Stock Sage was substantially faster than exact cuDNN and was already the correct production choice. Attempts to improve GROUP execution further through alternative batching/streaming did not produce a meaningful end-to-end gain. This branch was therefore closed rather than repeatedly re-optimized.

ANCHOR, in contrast, remained exact. GLOBAL attention became the main target for trajectory-aware backend routing.

## 5. Diffusion-step sensitivity

A global Sage substitution produced substantial trajectory drift. Instead of abandoning the backend, VELA isolated individual diffusion evaluations. The tests showed that most of the observed error was concentrated in a small part of the sampler trajectory.

The second critical gate/evaluation (internally logged as **gate label 2**) was particularly important. Restoring exact GLOBAL attention only at that gate recovered a large fraction of the lost trajectory fidelity while allowing Sage to remain active elsewhere. Later gates contributed much less to the measured drift.

This established a core VELA principle: **numerical sensitivity is timestep-dependent**.

## 6. Transformer-depth sensitivity and the critical island

The critical gate was then mapped across transformer depth. Coarse depth bands showed that late blocks were disproportionately important. Finer isolation established that **blocks 40–49 act as an atomic critical island** for the production policy. Smaller subsets such as 45–49, 40–44 or 42–49 did not recover the same trajectory.

A tempting hypothesis was that blocks with the largest local exact-vs-Sage tensor error would be the blocks that needed exact execution. That hypothesis failed. A selective set based on local error magnitude recovered little of the end-to-end trajectory. Local numerical error was therefore not a reliable proxy for diffusion importance.

VELA 1.0.0 consequently uses exact GLOBAL attention for **gate label 2, blocks 40–49**, while ordinary GLOBAL calls remain on Sage.

## 7. Direct cuDNN execution and the block-40 anomaly

Once the critical exact island was identified, profiling exposed an apparent execution anomaly: block 40 could take roughly an order of magnitude longer than neighboring blocks despite similar geometry. The cause was not the required exact math itself but execution/shape-selection overhead in the path used to reach it.

VELA introduced a direct cuDNN SDPA path for the validated critical island. This preserved the exact attention computation while bypassing the expensive path responsible for the block-40 spike. The critical-island execution time dropped dramatically, and the resulting latent trajectory matched the validated exact-island reference.

This was one of the most important production results: a speedup obtained by changing **how exact math is dispatched**, not by approximating the math.

## 8. Production bottleneck census

After attention routing stabilized, a full NFE census showed that window softmax/attention remained the largest component, but QKV projection was also substantial. QKV was dominated by the native comfy-kitchen INT8 GEMM path; quantization, ConvRot handling and dequantization were comparatively smaller components.

This redirected the next phase from generic PyTorch linear replacements toward the actual INT8 backend.

## 9. Dense QKV replacement: a useful failure

A dense `mm+bias` implementation looked promising in a small diagnostic but became slower in transient production execution and caused large trajectory drift when activated in selected blocks. The experiment demonstrated again that a local linear benchmark did not predict the behavior of the full patched INT8 pipeline. Dense QKV replacement was closed.

## 10. Triton QKV backend study

A fixed Triton INT8 QKV candidate was then compared with the native CUDA/comfy-kitchen path. On a representative block it reduced local QKV latency by roughly 17–21%, establishing that a faster kernel was technically possible. The output was close but not bit-exact.

The next question was whether that numerical difference was safe for diffusion. Five 10-block bands were tested across the transformer. Every band changed the trajectory beyond the production target. Importantly, the sensitivity was not monotonic with depth.

## 11. Single-block trajectory map

Blocks 10–29 were then tested individually with Triton QKV. The resulting video relative-RMSE map was highly jagged. Examples include:

- block 10: **13.580203%** — best individual candidate;
- block 23: **14.500488%**;
- block 17: **15.047031%**;
- block 26: **15.218851%**;
- block 16: **27.361589%** — the most destructive single block in the map.

The production reference for this phase was **12.925689% video / 4.421429% audio**.

Adjacent blocks could have radically different sensitivity, and audio/video drift did not necessarily move in the same direction. The map ruled out simple policies based on even/odd blocks, depth, or a fixed periodic selection.

## 12. Non-linear interaction and compensation

The best single blocks were combined to test whether drift was additive. It was not. Some pairs exhibited **positive compensation**: two approximated blocks together could produce less video drift than expected from either component.

The strongest mapped pair was blocks **17+26**, reaching **14.385004% video / 4.742346% audio**. Other pairs such as 10+23 or 10+26 were strongly destructive. A triple 17+23+26 was also worse than the good pairs, proving that pairwise compatibility did not compose transitively.

This is an important empirical result for diffusion-runtime optimization: approximation error can interact through the trajectory in a non-linear, sign-changing way. A block sensitivity map alone is insufficient; interactions matter.

## 13. Why Triton is not in VELA 1.0.0

The best pair, 17+26, was promoted to a clean performance candidate and tested after startup/Registry activity had completed. At 640×640 it produced a meaningful gain, but at the primary 896×896 geometry the clean sampler time improved only from roughly **2:14 / 16.77 s/it** for the matched V3.28 control to **2:10 / 16.27 s/it**. That is about a 3% timing improvement while video relative RMSE increased from **12.925689%** to **14.385004%**.

The project therefore chose the more conservative production point. Triton remains a successful research result and possible future fast mode, but **native INT8/ConvRot QKV is the VELA 1.0.0 default**.

## 14. Final VELA 1.0.0 architecture

The public release freezes the internal V3.28 production policy:

```text
QKV projection          native INT8 / ConvRot
GROUP attention         Stock Sage / C40
GLOBAL attention        Sage normally
critical GLOBAL island  gate label 2, blocks 40-49 -> exact direct-cuDNN
ANCHOR attention        exact
anchor-column removal   blocks 0 and 36 only
research profilers      disabled
experimental Triton     disabled
```

No external VELA/VDN research BAT flags are required; the validated production settings are internalized by the node.

## 15. End-to-end benchmark results

The publication comparison against VDN 1.1.0 shows that the gain is workload-dependent rather than constant:

| Test | VDN 1.1.0 | VELA 1.0.0 | Result |
|---|---:|---:|---:|
| 0.4 MP / 5 sec | 1:14 | **1:01** | 17.6% faster |
| 0.8 MP / 5 sec | 2:53 | **2:13** | 23.1% faster |
| 0.8 MP / 7 sec | 3:10 | **3:02** | 4.2% faster |
| 0.8 MP / 8 sec | 3:31 | **3:23** | 3.8% faster |
| 0.8 MP / 9 sec | 4:07 | **3:57** | 4.0% faster |
| 0.8 MP / 10 sec | **4:12** | 4:17 | 2.0% slower |

An additional VELA-only 1.0 MP / 5 sec / 8-step run completed in **2:46 / 20.80 s/it**. No matched VDN 1.1.0 1.0 MP result is available, so no comparative claim is made for that point.

The duration dependence is expected: spatial and temporal sequence geometry change the fraction of total runtime spent in the branches VELA optimizes. The data therefore support a workload-specific claim, not a universal constant speedup.

## 16. Negative results that shaped the release

VELA intentionally records failures because they narrowed the design space:

- aggressive head grouping failed reconstruction;
- INT4 value paths were numerically unsafe;
- active head splitting did not improve end-to-end runtime;
- global low-rank residual compression was unstable across blocks/timesteps;
- alternate GROUP batching/streaming produced no useful gain;
- globally replacing exact attention with Sage caused excessive drift;
- selecting exact blocks by largest local error failed to predict trajectory importance;
- dense QKV replacement was slower/unsafe in production;
- Triton QKV was locally faster but trajectory-sensitive;
- good Triton pairs did not necessarily form good triples.

These negative results are part of the engineering conclusion: the final policy is selective because the model itself is selective in its sensitivity.

## 17. Limitations

VELA 1.0.0 is validated on a specific Ampere 24 GB stack. Its implementation relies on details of ComfyUI's MiniMax H3 integration, comfy-kitchen INT8/ConvRot dispatch, SageAttention availability and PyTorch/cuDNN SDPA behavior. Future versions of these components may change performance or numerical behavior.

The benchmark table is not a guarantee for every duration or resolution. The 10-second 0.8 MP result is slightly slower than VDN 1.1.0, demonstrating that VELA is not universally faster for every sequence geometry.

The relative-RMSE metrics used during development compare latent trajectories under fixed inputs. They are diagnostic metrics, not direct perceptual quality scores.

## 18. Conclusion

VELA's main contribution is not a single kernel. It is an empirical strategy for optimizing a diffusion/video transformer runtime according to **where approximation is tolerated and where exact execution matters**. The final runtime combines sparse structural changes, backend specialization, timestep/depth-sensitive exact islands, and dispatch-level optimization.

The project also shows why microbenchmarks alone are insufficient for diffusion systems: faster kernels can lose end-to-end value through overhead, and numerically small differences can interact non-linearly across the sampler trajectory. VELA 1.0.0 therefore freezes the conservative production point rather than the fastest experimental point.

## Attribution

VELA is a continuation of `Speach1sdef178/ComfyUI-VDN-H3-24GB`, itself derived from the released VideoDeltaNet/OpenVDN work and existing ComfyUI VDN-H3 integration. Existing Apache-2.0 notices are preserved. See `NOTICE` and `LICENSE`. MiniMax-H3 weights are separately licensed and are not included.


## Release integration decision: no ComfyUI core patch

The earlier VDN research line included an experimental `block_loop` replacement used by LongCache and installed by editing `comfy/ldm/minimax/model.py`. During VELA release engineering this dependency was reviewed against ComfyUI's standard model-patch composition mechanism and the compatibility risk of modifying an upstream core file. The validated VELA v1.0.0 production configuration does not enable LongCache, so the core modification is not required for the published production path.

VELA v1.0.0 therefore removes the on-disk MiniMax block-loop installer entirely and explicitly disables the LongCache research switches. The released node uses runtime model patches only and leaves ComfyUI core files unchanged. A patch-free LongCache implementation is retained as a separate research direction and is not part of v1.0.0.
