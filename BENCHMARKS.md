# VELA 1.0.0 Benchmarks

## Publication benchmark

The following measurements compare the preceding **VDN 1.1.0** runtime with the final VELA 1.0.0 production policy. Values are wall/sampler times recorded on the same RTX 3090 Ti development system using the same 8-step generation setup.

| Resolution / duration | VDN 1.1.0 | VELA 1.0.0 | Seconds saved | Time change |
|---|---:|---:|---:|---:|
| 0.4 MP / 5 sec | 1:14 (74 s) | **1:01 (61 s)** | **13 s** | **17.57% faster** |
| 0.8 MP / 5 sec | 2:53 (173 s) | **2:13 (133 s)** | **40 s** | **23.12% faster** |
| 0.8 MP / 7 sec | 3:10 (190 s) | **3:02 (182 s)** | **8 s** | **4.21% faster** |
| 0.8 MP / 8 sec | 3:31 (211 s) | **3:23 (203 s)** | **8 s** | **3.79% faster** |
| 0.8 MP / 9 sec | 4:07 (247 s) | **3:57 (237 s)** | **10 s** | **4.05% faster** |
| 0.8 MP / 10 sec | **4:12 (252 s)** | 4:17 (257 s) | −5 s | **1.98% slower** |

## Additional VELA-only scaling point

- **1.0 MP / 5 sec / 8 steps:** `2:46`, `20.80 s/it`.

A matched VDN 1.1.0 measurement was not recorded at 1.0 MP, so this point is reported only as VELA scaling validation and is not used for a speedup claim.

## Standard validation conditions

The VELA development A/B protocol froze the model, seed, workflow and 8-step sampler while changing only the tested execution policy. The standard high-resolution diagnostic geometry was approximately **0.8028 MP (896×896), 5 seconds, 8 steps**.

The public duration table above comes from end-to-end user runs. It should not be interpreted as a hardware-independent throughput guarantee. Performance depends on sequence geometry, GPU, ComfyUI version, PyTorch, comfy-kitchen, attention backend and background startup activity.

For clean timing after a ComfyUI restart, wait until ComfyUI Manager reports that all startup tasks have completed before starting the measured prompt.

## Numerical validation used during development

The final internal V3.28 production checkpoint preserved the validated reference trajectory used by the VELA study: relative RMSE against the frozen V1.8 reference was **12.925689% for video** and **4.421429% for audio**. This metric was used to compare candidate execution changes under identical inputs; it is not a perceptual quality score.

The best experimental selective Triton pair (blocks 17+26) reached a clean 896×896 sampler time of **2:10 / 16.27 s/it**, but increased video relative RMSE to **14.385004%**. The approximately 3% timing gain at the standard 0.8 MP geometry did not justify the additional trajectory drift, so Triton QKV was excluded from VELA 1.0.0.
