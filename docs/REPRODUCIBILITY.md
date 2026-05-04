# Reproducibility

This document specifies the exact environment, hardware, and runtime in which the headline numbers in `results/summary.tsv` were produced.

## Environments

ViTeX-Bench uses **two** conda environments because PaddleOCR and PyTorch + pyiqa do not co-install cleanly.

### `paddleocr` (Stage 1 — OCR, CPU)

```bash
conda create -n paddleocr python=3.12 -y
conda activate paddleocr
pip install paddleocr opencv-python
```

PaddleOCR pulls PaddlePaddle 3.x. The pipeline forces CPU execution (`CUDA_VISIBLE_DEVICES=""`) because PaddlePaddle 3.2 does not yet support Blackwell GPUs (RTX 5090, compute capability 12.0). On older NVIDIA generations (Ampere / Hopper) you can switch to GPU OCR by passing `--device gpu` to `benchmark/ocr_extract.py`; for that case set `--workers 1`.

### `vitex-bench` (Stage 2 — GPU metrics)

```bash
conda create -n vitex-bench python=3.12 -y
conda activate vitex-bench
pip install -r requirements.txt
```

A CUDA-enabled PyTorch wheel matching your local CUDA toolkit is required. `lpips` ships its own AlexNet checkpoint on first call; `pyiqa` downloads MUSIQ KonIQ weights into `~/.cache/torch/hub/pyiqa/`.

## Hardware

The headline numbers were produced on:

| component | spec |
|---|---|
| CPU | 32-core x86_64 (Linux) |
| GPU | NVIDIA RTX 5090, 24 GB |
| RAM | 32 GB DDR5 |
| swap | 32 GB (incidentally present from earlier workloads, not needed by this pipeline) |

The pipeline does not require an RTX 5090 specifically — any CUDA-12-capable GPU with $\ge 16$ GB of VRAM is sufficient (LPIPS + MUSIQ + locality compose run with $\le 6$ GB peak on the largest 1280×720 × 120 batch). On a single H100 80 GB the GPU stage is bottlenecked on CPU-side video decoding rather than GPU compute, so a smaller card sees similar wall time.

The CPU stage scales roughly linearly with the number of `--workers`; we use 8 workers on a 32-core box. Each worker holds one PaddleOCR instance per language family in its task list; peak per-worker resident memory is $\sim 1.5$ GB.

## Runtime

End-to-end wall clock for the **full nine-baseline grid plus identity sanity** (10 baselines, 157 clips, the source-video OCR cached and reused across baselines):

| stage | time |
|---|---:|
| identity OCR + source cache build (159 clips × $V$ + $\hat V$) | ~170 min |
| identity GPU evaluate | ~28 min |
| each subsequent baseline OCR ($\hat V$ only, source cached) | ~85 min |
| each subsequent baseline GPU evaluate | ~28 min |
| total (with CPU↔GPU pipelining) | **~13 hours** |

The CPU stage and the GPU stage overlap in the master script (`scripts/run_benchmark.sh`): while baseline N is on the GPU, baseline N+1 is on the CPU. The 13-hour total is therefore $\approx 170 + 9 \times 85 + 28$ min — the source-OCR cost amortized once, plus the per-baseline CPU OCR of every prediction, plus the final GPU evaluate.

## Determinism

PP-OCRv5 inference is deterministic given a fixed input image and language code. MUSIQ, LPIPS, SSIM, PSNR are deterministic GPU operations under default PyTorch settings.

The bootstrap CI uses a Python `random.Random(0)` seeded RNG, so re-running `benchmark/evaluate.py` against the same `eval.json` gives identical CIs.

The video-decoder backend (`cv2.VideoCapture`) decodes H.264 deterministically across runs on the same machine; we have not verified bit-level reproducibility across libavcodec versions.

## Step-by-step

```bash
# 1. Layout
data/inference/parsed_records.json
data/inference/original_videos/<id>.mp4
data/inference/text_masks/<id>_<ts>_<hash>.mp4
baseline_output_videos/<method>/<id>.mp4

# 2. Identity sanity (also bootstraps the source-OCR cache)
python benchmark/make_identity_baseline.py \
    --records  data/inference/parsed_records.json \
    --src_dir  data/inference/original_videos \
    --out_dir  baseline_output_videos/identity

# 3. Run the full grid (10 baselines, ~13 hours)
bash scripts/run_benchmark.sh

# 4. Inspect aggregates
column -t -s $'\t' outputs/summary.tsv

# 5. Per-clip JSON for the analysis you want
python -c "
import json
d = json.load(open('outputs/ViTeX-14B/eval.json'))
for k, v in sorted(d['per_clip'].items())[:5]:
    print(k, v['SeqAcc'], v['CharAcc'], v['TTS'])
"
```

## What if my numbers don't match?

Differences of a few percentage points are expected because:
- Mask file timestamps in `parsed_records.json` change between dataset re-builds; the binary mask itself is stable.
- Some Family A baselines (especially those distributed without exact training-data hashes) may produce slightly different fine-tuned outputs across hardware.
- PP-OCRv5 model weights are versioned; the `_PADDLEX_VERSION` in your install determines the exact recognizer.

If your numbers diverge by more than ±1 % on aggregate metrics, the most common causes are (i) prediction videos at the wrong spatial resolution (the pipeline auto-resamples but Lanczos vs bilinear can differ slightly) and (ii) mask binarization threshold mismatches when masks are stored as PNG sequences instead of MP4 — convert to MP4 before invoking the pipeline.

For substantive deviations, please open an issue with your `outputs/<baseline>/log.txt` and a short `diff` of `summary.tsv`.
