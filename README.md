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
> Submitted to NeurIPS 2026 Track on Evaluations and Datasets (under double-blind review).

🌐 **Project page: [vitex-bench.github.io](https://vitex-bench.github.io/)**

This repository contains the **evaluation code** for ViTeX-Bench — a **13-metric, three-axis** scoring pipeline. The thirteen-metric vector is the unit of report; for ranking purposes the public leaderboard sorts on **TextScore** (geometric mean of the three text-correctness primitives), but the full vector is always visible alongside it. Companion repos:

| component | repo |
|---|---|
| **Project page** | https://vitex-bench.github.io/ |
| **Dataset** (387 paired clips, 230 train / 157 frozen test) | https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset |
| **Benchmark code** (this repo) | https://huggingface.co/ViTeX-Bench/ViTeX-Bench |
| **Model & Inference code** (ViTeX-14B reference model) | https://huggingface.co/ViTeX-Bench/ViTeX-14B |
| **Leaderboard** (public submission Space) | https://huggingface.co/spaces/ViTeX-Bench/ViTeX-Bench-Leaderboard |

The Datasheet, Croissant 1.0 metadata, and dataset license live on the ViTeX-Dataset repo above.

---

## Quickstart

```bash
# 1. Clone + install (two conda envs because PaddleOCR conflicts with PyTorch/pyiqa)
git clone https://huggingface.co/ViTeX-Bench/ViTeX-Bench && cd ViTeX-Bench

conda create -n paddleocr   python=3.12 -y && conda activate paddleocr   && pip install paddleocr opencv-python && conda deactivate
conda create -n vitex-bench python=3.12 -y && conda activate vitex-bench && pip install -r requirements.txt        && conda deactivate

# 2. Drop your method's predictions in baseline_output_videos/<your_method>/<id>.mp4
#    (1280x720, 24 fps, 120 frames; one .mp4 per clip id from parsed_records.json)

# 3. Evaluate. The runner downloads the 157-clip eval split of ViTeX-Dataset on
#    first run, then runs OCR (CPU) + 13 metrics (GPU) in one shot.
bash scripts/run_benchmark.sh <your_method>
```

`outputs/<your_method>/eval.json` contains per-clip metric values, the 13 test-split aggregates with 95% bootstrap CIs, and the TextScore leaderboard sort key. `outputs/summary.tsv` accumulates a one-row-per-baseline TSV across runs.

For the full paper baseline grid, run `bash scripts/run_benchmark.sh` with no argument; it pipelines CPU OCR and GPU metrics across baselines.

### What the runner needs

| | source | size | when |
|---|---|---|---|
| 157-clip evaluation split (videos + masks + records) | [ViTeX-Dataset/eval](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset/tree/main/eval) | ~6 GB | auto-downloaded by `run_benchmark.sh` on first run, cached at `data/eval/` |
| PP-OCRv5 weights (4 scripts) | PaddlePaddle | ~200 MB | auto-downloaded by `paddleocr` to `~/.paddleocr/` on first OCR call |
| RAFT-large optical flow | torchvision hub | ~80 MB | auto-downloaded to `~/.cache/torch/hub/` on first Warp call |
| LPIPS (AlexNet) | `lpips` package | ~10 MB | auto-downloaded on first call |
| MUSIQ-KonIQ | `pyiqa` | ~50 MB | auto-downloaded on first call |
| DreamSim ensemble | HuggingFace hub | ~600 MB | auto-downloaded on first call |
| Your prediction videos | you | n/a | one `.mp4` per clip id, dropped in `baseline_output_videos/<method>/` |

That is the full external dependency list. **TextScore is self-contained** — it is the geometric mean of three primitives (SeqAcc, CharAcc, TTS) that each live natively in $[0, 1]$, so your sort key does not depend on any other baseline's numbers. The reference baseline rows under [`results/`](./results) and the table below are for context, not for score computation.

---

## Headline numbers (157-clip frozen test split)

TextScore (geomean of SeqAcc, CharAcc, TTS) is the leaderboard sort key. The full thirteen-metric vector is reported alongside it under [`results/eval_all.json`](./results/eval_all.json) and [`results/summary.tsv`](./results/summary.tsv); only TextScore + the three text primitives are reproduced inline below.

| Rank | Method | Family | TextScore | SeqAcc | CharAcc | TTS |
|---:|---|---|---:|---:|---:|---:|
| 1 | TextCtrl | A | **0.5624** | **0.475** | **0.733** | 0.511 |
| 2 | **ViTeX-14B (Composite)** | Reference | 0.5410 | 0.345 | 0.689 | **0.666** |
| 3 | **ViTeX-14B** | Reference | 0.5338 | 0.341 | 0.688 | 0.648 |
| 4 | VideoPainter$^{\dagger}$ | C | 0.5151 | 0.365 | 0.619 | 0.606 |
| 5 | FLUX-Text | A | 0.5023 | 0.528 | **0.737** | 0.326 |
| 6 | RS-STE | A | 0.4908 | 0.354 | 0.626 | 0.534 |
| 7 | AnyText2 | A | 0.4074 | 0.280 | 0.633 | 0.382 |
| 8 | TextCtrl + AnyV2V | B | 0.1649 | 0.057 | 0.308 | 0.257 |
| – | Identity (sanity) | — | 0.0000 | 0.000 | 0.317 | 0.760 |
| – | Wan2.1-VACE-14B | C | 0.0000 | 0.000 | 0.298 | 0.690 |
| – | Kling Video 3.0 Omni | D | 0.0000 | 0.000 | 0.207 | 0.641 |

The three rows at the bottom tie at TextScore = 0 because $\mathrm{SeqAcc} = 0$ collapses the geometric mean — these methods never produce the requested target string. They are still reported because the rest of their thirteen-metric vector is informative (e.g., Wan2.1-VACE-14B's high MUSIQ_crop and PSNR_loc; see `results/`).

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

**TextScore.** $\mathrm{TextScore} = \sqrt[3]{\mathrm{SeqAcc} \cdot \mathrm{CharAcc} \cdot \mathrm{TTS}}$ over the per-clip aggregated text means. Used as the public leaderboard sort key only — the full thirteen-metric vector remains the unit of report, because no axis substitutes for another. Aggregates carry a 95% percentile bootstrap CI from 1000 clip-level resamples.

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
