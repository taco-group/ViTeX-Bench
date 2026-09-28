# Bundled DiffSynth-Studio

This directory is a trimmed copy of [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
v2.0.7 (commit `166e6d2`, 2026-03-24), Apache-2.0 (see [`LICENSE`](./LICENSE)).
Only the modules that the Wan video pipeline and trainer import are kept.

Changes for ViTeX-Edit-14B:

| file | change |
|---|---|
| `models/wan_video_vace.py` | `GlyphEncoder`, `ConditionCrossAttention` (and the unused `TargetTextEncoder`); VACE blocks return `(hint, state)` instead of stacking all hints; CPU-offload autograd helpers; `load_state_dict` initializes the new modules only when their weights are missing from the checkpoint |
| `pipelines/wan_video.py` | `glyph_video` / `target_text` inputs to the VACE unit (the glyph video is VAE-encoded separately); activation and hint CPU offload in `model_fn_wan_video` |
| `configs/model_configs.py` | Wan entries only; the Wan2.1-VACE-14B entry sets `glyph_channels=16` |
| `configs/vram_management_module_maps.py` | glyph / text encoders are offloaded as whole modules |
| `core/vram/layers.py` | disk offload: when VRAM is above `vram_limit`, a DiT block's temporary copy is shallow instead of a `deepcopy` that failed on the open safetensors handles of its wrapped children |
| `core/data/operators.py` | `LoadVideo` pads short clips with their last frame |
| `core/gradient/gradient_checkpoint.py`, `core/loader/model.py`, `diffusion/runner.py`, `diffusion/logger.py`, `diffusion/parsers.py` | DeepSpeed ZeRO-3 training (VAE kept out of ZeRO-3, sharded checkpoint loading, `--resume_step`, loss log) |
| `utils/state_dict_converters/wan_video_vace.py` | helper to widen the VACE patch embedding (unused by the released model) |
