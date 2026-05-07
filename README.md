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

# ViTeX-Bench (Benchmark code)

🌐 [Project page](https://vitex-bench.github.io/) &nbsp;·&nbsp;
📊 [Dataset](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset) &nbsp;·&nbsp;
🧪 Benchmark code &nbsp;·&nbsp;
🤖 [Model & Inference code](https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B) &nbsp;·&nbsp;
🏆 [Leaderboard](https://huggingface.co/spaces/ViTeX-Bench/ViTeX-Bench-Leaderboard)

Evaluation pipeline for **video scene text editing**. A 13-metric, three-axis protocol (text correctness, visual quality, edit locality) on the frozen 157-clip evaluation split of [ViTeX-Dataset](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset). The full thirteen-metric vector is the unit of report; the public [Leaderboard](https://huggingface.co/spaces/ViTeX-Bench/ViTeX-Bench-Leaderboard) sorts on **TextScore** = ∛(SeqAcc · CharAcc · TTS).

> Anonymous release under double-blind review at NeurIPS 2026 Datasets and Benchmarks Track. Author list and DOI updated after deanonymization.

## Quickstart

```bash
git clone https://huggingface.co/ViTeX-Bench/ViTeX-Bench && cd ViTeX-Bench

# Two envs because PaddleOCR conflicts with PyTorch / pyiqa.
conda create -n paddleocr   python=3.12 -y && conda activate paddleocr   && pip install paddleocr opencv-python && conda deactivate
conda create -n vitex-bench python=3.12 -y && conda activate vitex-bench && pip install -r requirements.txt        && conda deactivate

# Drop your method's predictions in baseline_output_videos/<your_method>/<id>.mp4
# (1280×720, 24 fps, 120 frames; one .mp4 per clip id from parsed_records.json)
bash scripts/run_benchmark.sh <your_method>
```

The runner auto-downloads the ViTeX-Dataset eval split on first run. Output:

* `outputs/<your_method>/eval.json` — per-clip metrics + 13 aggregates with 95 % bootstrap CIs + TextScore.
* `outputs/summary.tsv` — one-row-per-baseline TSV across runs.

## Submitting

Upload the `eval.json` to the [Leaderboard](https://huggingface.co/spaces/ViTeX-Bench/ViTeX-Bench-Leaderboard) **Submit** tab; entries are reviewed before they appear on the public ranking. Pre-computed paper baselines and TSV summary live in [`results/`](./results); metric definitions and normalization rules in [`docs/PROTOCOL.md`](./docs/PROTOCOL.md); reference baselines and reproducibility notes in [`docs/BASELINES.md`](./docs/BASELINES.md) and [`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md).

## License

Apache-2.0 (this code; see [`LICENSE`](./LICENSE)). The dataset itself is CC-BY-4.0; see the [Dataset](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset) repo.

## Citation

```bibtex
@misc{vitex2026,
  title  = {ViTeX-Bench: Benchmarking High Fidelity Video Scene Text Editing},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted to NeurIPS 2026 Datasets and Benchmarks Track. Author list and DOI updated after deanonymization.},
}
```
