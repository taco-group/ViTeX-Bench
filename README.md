# ViTeX-Bench

📄 [Paper](https://arxiv.org/abs/2609.40356) &nbsp;·&nbsp;
🌐 [Project page](https://vitex-bench.github.io/) &nbsp;·&nbsp;
📊 [Dataset](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset) &nbsp;·&nbsp;
🧪 Code &nbsp;·&nbsp;
🤖 [Model weights](https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B) &nbsp;·&nbsp;
🏆 [Leaderboard](https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/)

This repository holds the benchmark's evaluation code and the code of the reference model, [ViTeX-Edit-14B](./vitex_edit): glyph rendering, inference, Composite post-processing and training.

Evaluation pipeline for **video scene text editing**. A 13-metric, three-axis protocol (text correctness, visual quality, edit locality) on the frozen 157-clip evaluation split of [ViTeX-Dataset](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset). The full thirteen-metric vector is the unit of report. One metric per axis is primary — **SeqAcc** (text correctness), **Warp_crop** (temporal quality), **DreamSim_loc** (edit locality) — and the public [Leaderboard](https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/) marks the Pareto set on these three instead of computing an aggregate score.

> Accompanies [*ViTeX-Bench: Benchmarking High-Fidelity Video Scene Text Editing*](https://arxiv.org/abs/2609.40356) (NeurIPS 2026 Track on Evaluations and Datasets).

## Quickstart

```bash
git clone https://github.com/taco-group/ViTeX-Bench.git && cd ViTeX-Bench

# Two envs because PaddleOCR conflicts with PyTorch / pyiqa.
conda create -n paddleocr   python=3.12 -y && conda activate paddleocr   && pip install paddleocr opencv-python && conda deactivate
conda create -n vitex-bench python=3.12 -y && conda activate vitex-bench && pip install -r requirements.txt        && conda deactivate

# Drop your method's predictions in baseline_output_videos/<your_method>/<id>.mp4
# (1280×720, 24 fps, 120 frames; one .mp4 per clip id from parsed_records.json)
bash scripts/run_benchmark.sh <your_method>
```

The runner auto-downloads the ViTeX-Dataset eval split on first run. Output:

* `outputs/<your_method>/eval.json` — per-clip metrics + 13 aggregates with 95 % bootstrap CIs.
* `outputs/summary.tsv` — one-row-per-baseline TSV across runs.

## Reference model: ViTeX-Edit-14B

[`vitex_edit/`](./vitex_edit) contains everything needed to run the reference editor on your own clips or to reproduce it on the evaluation split: typeface selection, glyph-video rendering, inference with multi-GPU sharding and low-memory modes, the Composite wrapper, and the two-stage training recipe. Weights are on [Hugging Face](https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B), and the method is described in Section 4 of the [paper](https://arxiv.org/abs/2609.40356). See [`vitex_edit/README.md`](./vitex_edit/README.md).

## Submitting

[Open a submission issue](https://github.com/ViTeX-Bench/ViTeX-Bench-Leaderboard/issues/new?template=submission.yml) on the leaderboard repository and attach the `eval.json`; entries are reviewed before they appear on the public [Leaderboard](https://vitex-bench.github.io/ViTeX-Bench-Leaderboard/). Pre-computed paper baselines and TSV summary live in [`results/`](./results); metric definitions and normalization rules in [`docs/PROTOCOL.md`](./docs/PROTOCOL.md); reference baselines and reproducibility notes in [`docs/BASELINES.md`](./docs/BASELINES.md) and [`docs/REPRODUCIBILITY.md`](./docs/REPRODUCIBILITY.md).

## License

Apache-2.0 (this code; see [`LICENSE`](./LICENSE)). `vitex_edit/diffsynth/` is adapted from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (Apache-2.0). The dataset is CC-BY-NC 4.0 (non-commercial research only); see the [Dataset](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset) card.

## Citation

```bibtex
@inproceedings{chen2026vitexbench,
  title         = {{ViTeX-Bench}: Benchmarking High-Fidelity Video Scene Text Editing},
  author        = {Chen, Xinghao and Gao, Xiangbo and Yu, Jiongze and Wu, Yuheng and Tu, Zhengzhong},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year          = {2026},
  eprint        = {2609.40356},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.40356}
}
```
