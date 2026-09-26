# Baselines

ViTeX-Bench is anchored by an eight-baseline cross-family reference grid spanning the four practical strategies for video scene text editing, plus the **identity sanity baseline** and the **ViTeX-Edit-14B reference model**. Each baseline was selected to expose the characteristic failure mode of its family rather than implementation-specific differences between methods that share a design principle.

## Family A — per-frame image scene-text editing

The most direct strategy applies a state-of-the-art image scene-text editor to each frame independently and concatenates the results. Family A isolates how a strong static editor behaves without temporal coupling and exposes inter-frame inconsistency — flicker, character jitter, and glyph-identity drift — as the structural failure mode.

| baseline | reference | inference |
|---|---|---|
| **AnyText2** (ft.) | Tuo et al., 2024 | Multi-language diffusion editor with a glyph-conditioned auxiliary loss |
| **TextCtrl** (ft.) | Wang et al., 2024 | Structure-and-style disentangled diffusion model |
| **FLUX-Text** (ft.) | Wang et al., 2025 | Regional-attention rewrite for the FLUX backbone |
| **RS-STE** (ft.) | Liu et al., 2025 | Recognition-supervised editor trained on synthetic and real-scene pairs |

Each Family A model is **fine-tuned on first-frame pairs** from the 230-clip training split using the hyperparameters recommended in its original paper, then applied independently to all 120 frames. The four representatives span recent design choices, so the family-level failure mode is not attributable to a single editor-specific artifact. Per-frame inference runs on a single H100 80 GB.

## Family B — first-frame edit and image-to-video propagation

A straightforward strategy for converting a per-frame editor into a video editor is to edit only frame 1 and propagate through an image-to-video model. The first frame is typically faithful, but propagated frames progressively drift from the source motion — the temporal-consistency failure mode of single-keyframe propagation.

| baseline | reference | inference |
|---|---|---|
| **TextCtrl + AnyV2V** | Ku et al., 2024 (TMLR) | TextCtrl edits frame 1; AnyV2V injects temporal features from the edited frame into a frozen Wan2.2-I2V backbone with default tuning-free hyperparameters |

Family B inference runs on $2 \times \text{H100}$.

## Family C — mask-conditioned video inpainting

A third strategy uses a mask-conditioned video editor by supplying the dilated text-region mask and the target string as prompt and allowing the model to re-render the masked region in place. Family C exposes the **instruction-following limitation of generic video inpainters**: under a dilated text mask both editors frequently return output visually close to the source, leaving the masked region either unchanged or filled with glyph-like artifacts that do not spell $s_{\mathrm{tgt}}$.

| baseline | reference | inference |
|---|---|---|
| **Wan2.1-VACE-14B** (zero-shot) | Wan Team, 2025 | Unified Video Condition Unit DiT — same backbone ViTeX-Edit-14B is fine-tuned on. Queried zero-shot, no glyph conditioning |
| **VideoPainter** (zero-shot) | Bian et al., 2025 | Dual-stream context-controlled video inpainter; lightweight context encoder added to a frozen video DiT |

Both Family C baselines run zero-shot. Each receives the dilated text-region mask $M$ and the same fixed prompt template as Family D. We use default hyperparameters from each official release; no per-clip prompt tuning. Family C inference runs on $2 \times \text{H100}$.

## Family D — instruction-guided video-to-video editing

The final strategy provides a free-form natural-language edit instruction to a generic instruction-conditioned video-to-video model. As with Family C, the dominant failure mode is **limited responsiveness to the character-level intent of the instruction**: motion and surface reconstruction are typically strong, but the rendered text either remains unchanged or substitutes plausible-looking but incorrect characters.

| baseline | reference | inference |
|---|---|---|
| **Kling Video 3.0 Omni** | Kuaishou Kling Team, 2025 | Closed-source commercial V2V system; queried through the public API with a fixed instruction template |

Each of the 157 test clips is submitted, the returned video is retrieved, and the output is re-encoded to $1280 \times 720$ / 24 fps / 120 frames to match the evaluation grid. The API version and query date are reported alongside the final results; per-clip cost is reported in the paper supplementary.

We chose **one** representative closed-source commercial system rather than three to keep the baseline grid focused — the failure-mode signal at the family level is comparable across Runway, Luma, and Kling. Google Veo is not included because its public API exposes text-to-video, image-to-video, and scene extension but no in-place V2V mode that accepts a user-supplied source video as input.

## Identity sanity baseline

We additionally release an **identity** baseline (built by `benchmark/make_identity_baseline.py`) that copies the source video verbatim as the prediction; it is reported as the **Source video** row in the paper and on the leaderboard and is never ranked. Identity anchors text correctness from below (SeqAcc should be near zero unless the source string already matches the target by chance; TTS stays high, about 0.76, because an unedited string is trivially stable, which is why TTS must be read together with SeqAcc and CharAcc) and edit locality from above for edit-locality (PSNR_loc = 100, SSIM_loc = 1, LPIPS_loc = 0, modulo the 100 dB cap on PSNR). It also calibrates Flicker against codec re-encoding noise — Flicker_full of the source itself.

## Excluded methods

- **STRIVE** (Subramanian et al., 2024) — code and weights unavailable.
- **GlyphMastero** (Li et al., 2025) — image-only, no video equivalent released.
- **SwapText** (Yang et al., 2020) — code and weights unavailable.
- Earlier GAN-era image methods (SRNet, STEFANN, MOSTEL) — superseded in published benchmarks; included by their successor diffusion editors instead.

---

## Reference: ViTeX-Edit-14B (ours)

ViTeX-Edit-14B is the reference model anchored under ViTeX-Bench. It is fine-tuned from Wan2.1 with a glyph-video conditioning pathway (rendered target-glyph video tracked along the mask, encoded to fixed-length tokens, attended via per-block ConditionCrossAttention with zero-initialized residuals). Training and inference details are in the paper §3.3 and supplementary §E. The model is released alongside the dataset on the public mirror after deanonymization.

`results/summary.tsv` contains the headline numbers from the paper. The full per-clip JSON for every baseline and ViTeX-Edit-14B is also published under `results/` so independent analyses (per-language stratification, target-text length sweeps, motion-intensity sweeps, …) can be run without re-executing OCR or the GPU pipeline.
