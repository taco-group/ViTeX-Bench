---
license: apache-2.0
language:
- en
- zh
- ja
- ru
tags:
- video-editing
- scene-text-editing
- benchmark
- ocr
- neurips-2026
---

# ViTeX-Bench (evaluation code)

> **ViTeX-Bench: Benchmarking High Fidelity Video Scene Text Editing.**
> Submitted to NeurIPS 2026 Datasets and Benchmarks Track (under double-blind review).

This repository contains the **evaluation code** for ViTeX-Bench — a **13-metric, three-axis** scoring pipeline plus the aggregate **ViTeX-Score**. Companion repos:

| component | repo |
|---|---|
| **ViTeX-Dataset** (387 paired clips, 230 train / 157 frozen test) | https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset |
| **ViTeX-14B** (open reference model) | https://huggingface.co/ViTeX-Bench/ViTeX-14B |
| **ViTeX-Bench** (this repo) | https://huggingface.co/ViTeX-Bench/ViTeX-Bench |

The Datasheet, Croissant 1.0 metadata, and dataset license live on the ViTeX-Dataset repo above.

---

## Quickstart

```bash
# 1. Clone + install
git clone https://huggingface.co/ViTeX-Bench/ViTeX-Bench && cd ViTeX-Bench

conda create -n paddleocr   python=3.12 -y && conda activate paddleocr   && pip install paddleocr opencv-python && conda deactivate
conda create -n vitex-bench python=3.12 -y && conda activate vitex-bench && pip install -r requirements.txt        && conda deactivate

# 2. Download dataset (387 clips with masks, ~12 GB)
huggingface-cli download ViTeX-Bench/ViTeX-Dataset --repo-type dataset --local-dir data

# 3. Drop your method's predictions in baseline_output_videos/<your_method>/<id>.mp4
#    (1280x720, 24 fps, 120 frames)

# 4. Evaluate (one command — runs OCR on CPU then 13 metrics + ViTeX-Score on GPU)
bash scripts/run_benchmark.sh <your_method>
```

`outputs/<your_method>/eval.json` will contain the per-clip metric vector, the 13 test-split aggregates with 95% bootstrap CIs, the per-axis aggregates, and the ViTeX-Score. `outputs/summary.tsv` accumulates a one-row-per-baseline TSV across runs.

The evaluator separates OCR (PaddleOCR / CPU) from the GPU metric pipeline because PaddleOCR's deps don't co-install cleanly with PyTorch + pyiqa; the runner handles both stages and conda environment switching automatically. For all paper baselines at once, `bash scripts/run_benchmark.sh` (no argument) runs the full grid with CPU↔GPU pipelining.

---

## Headline numbers (157-clip frozen test split)

ViTeX-Score (geometric mean across the three per-axis aggregates):

| Method | Family | ViTeX-Score | Correctness | Visual | Locality |
|---|---|---:|---:|---:|---:|
| **ViTeX-14B (Corp)** | — | **0.689** | 0.541 | 0.635 | **0.954** |
| TextCtrl | A | 0.686 | **0.562** | 0.613 | 0.939 |
| **ViTeX-14B** | — | 0.650 | 0.534 | **0.647** | 0.794 |
| RS-STE | A | 0.641 | 0.491 | 0.594 | 0.901 |
| VideoPainter$^{\dagger}$ | C | 0.623 | 0.515 | 0.608 | 0.773 |
| AnyText2 | A | 0.549 | 0.407 | 0.580 | 0.699 |
| FLUX-Text | A | 0.392 | 0.502 | 0.140 | 0.851 |
| TextCtrl + AnyV2V | B | 0.333 | 0.165 | 0.437 | 0.512 |
| Wan2.1-VACE-14B | C | 0.000 | 0.000 | 0.631 | 0.889 |
| Kling Video 3.0 Omni | D | 0.000 | 0.000 | 0.569 | 0.583 |
| identity (sanity) | — | 0.000 | 0.000 | 0.641 | 1.000 |

$^{\dagger}$VideoPainter Flicker$_{f/c}$ and Warp$_{f/c}$ are excluded from per-metric ranking because its CogVideoX-1.0 backbone runs on a sub-protocol grid that requires temporal interpolation. The full 13-metric breakdown for every baseline (with bootstrap CIs and per-clip OCR strings) is under [`results/`](./results).

---

## Evaluation protocol

**Axis 1 — Text correctness.** PP-OCRv5 reads the dilated-mask crop with per-clip language routing (Latin / Chinese / Japanese / Cyrillic). Source-detectability gating restricts text scoring to frames whose source text the recognizer can read at $\ge 0.5$ similarity. Comparisons use a substring (semi-global) edit distance so a correct target embedded in a longer OCR string is not penalized for surrounding characters. Three metrics: **SeqAcc**, **CharAcc**, **TTS** (temporal text stability).

**Axis 2 — Visual quality.** Six primitives at two scopes $S \in \{\text{full}, \text{crop}\}$:
- **Flicker$_S$** — adjacent-frame MAE; rewards smoothness, hackable by interpolation.
- **Warp$_S$** — adjacent-frame MAE after backward-warping along RAFT optical flow on the source; only scores low when the prediction follows the same per-pixel motion as the source.
- **MUSIQ$_S$** — KonIQ-pretrained no-reference perceptual quality.

The crop scope uses one static union-mask bbox per clip so adjacent crops are spatially aligned.

**Axis 3 — Edit locality.** Construct $\hat f_t^{\text{loc}} = (1 - m_t) \cdot \hat f_t + m_t \cdot f_t$ (predicted pixels outside the mask, source pixels inside) and score four metrics against the source: **PSNR**, **SSIM**, **LPIPS** (pixel-level) and **DreamSim** (learned perceptual identity, robust to VAE codec noise that pixel-level metrics over-penalize).

**ViTeX-Score.** All 13 metrics are normalized to $[0, 1]$ via a frozen v1 endpoint table; within-axis aggregation is a weighted geometric mean (visual axis: text-crop scope weighted $2\times$ full-frame); cross-axis is an unweighted geometric mean. The geometric mean enforces non-substitutivity: any axis at zero collapses the score, so visual quality cannot rescue a missing edit. Aggregates carry a 95% percentile bootstrap CI from 1000 clip-level resamples.

Full math, normalization endpoints, and edge-case rules are in [`docs/PROTOCOL.md`](./docs/PROTOCOL.md). Baseline implementations and protocol-fit notes are in [`docs/BASELINES.md`](./docs/BASELINES.md).

---

## Submitting a new baseline

1. Run your method on every clip in `parsed_records.json`. Outputs at `baseline_output_videos/<your_method>/<id>.mp4`, $1280 \times 720$, 24 fps, 120 frames.
2. Run `bash scripts/run_benchmark.sh <your_method>`.
3. Submit `eval.json` and the prediction videos to the leaderboard (URL added after deanonymization). The leaderboard server replays the metric script against the videos to validate the submission.

---

## License

Code is released under the **Apache License 2.0** (see [`LICENSE`](./LICENSE)).
The dataset itself (ViTeX-Dataset) is released under **CC-BY-NC 4.0** for non-commercial research; see the dataset repo for the full license and Datasheet.

---

## Citation

```bibtex
@misc{vitexbench2026,
  title  = {ViTeX-Bench: Benchmarking High Fidelity Video Scene Text Editing},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted to NeurIPS 2026 Datasets and Benchmarks Track. Author list and DOI updated after deanonymization.},
}
```
