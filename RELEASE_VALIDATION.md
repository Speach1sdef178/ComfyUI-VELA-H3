# VELA 1.0.0 Release Sanity Test

Run this once from the **exact final release folder** before publishing.

## Test

- Resolution: **896×896 (~0.8028 MP)**
- Duration: **5 seconds**
- Steps: **8**
- Same MiniMax H3 INT8 ConvRot model, VDN checkpoint, seed and workflow used for the publication benchmark.
- Restart ComfyUI after replacing the node folder.
- Ensure no older VELA experimental folder is present in `custom_nodes`.
- For a clean timing observation, wait until ComfyUI Manager reports that all startup tasks have completed before queuing the prompt.

## Required startup marker

```text
[VELA v1.0.0] production config active; external VDN_H3_* BAT flags are not required
```

## Required execution marker

```text
[VELA v1.0.0] ACTIVE production path | GLOBAL gate2 blocks=40-49 exact direct-cuDNN | other GLOBAL Sage | ANCHOR exact | GROUP Stock Sage | anchor-column removal blocks=0,36
```

## Pass criteria

- `ComfyUI VELA H3` imports without error.
- The node appears as `Apply VDN + VELA H3 [1.0.0]`.
- Both required VELA 1.0.0 markers appear.
- No V3.3x Triton-QKV activation marker appears.
- The 8-step sampler completes and saves the expected video/audio latent shapes for the frozen workflow.
- Output is visually sane.

Timing is recorded as a release-validation datapoint, not used to redefine the already collected publication table unless the test conditions are an exact matched benchmark run.
