# Changelog

## 1.0.0

- Final release packaging is core-patch-free: no modification of `comfy/ldm/minimax/model.py`; LongCache research switches are disabled and the old block-loop installer is not shipped. — 2026-10-08

First public VELA release.

- Freezes the validated internal V3.28 production execution policy.
- Uses native INT8/ConvRot QKV.
- Uses Stock Sage for GROUP attention and Sage for ordinary GLOBAL attention.
- Uses exact direct-cuDNN GLOBAL attention at gate label 2 for blocks 40–49.
- Keeps ANCHOR attention exact.
- Keeps validated anchor-column removal in blocks 0 and 36.
- Internalizes production configuration; external research BAT flags are not required.
- Experimental Triton QKV remains disabled after trajectory/performance validation.
- Adds public benchmark, research, attribution and Comfy Registry metadata.
