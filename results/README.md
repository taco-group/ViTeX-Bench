# Results

`summary.tsv` is the per-baseline aggregate written by `scripts/run_benchmark.sh` after every baseline's `evaluate.py` pass. Columns:

| column | meaning |
|---|---|
| `baseline` | baseline identifier (also the directory name under `baseline_output_videos/`) |
| `n_clips` | number of evaluation clips that produced a valid metric value (`None` per-clip values, e.g. when the support set $\mathcal D$ is empty, are skipped) |
| `SeqAcc` | Axis 1 — exact-match rate over detectable frames |
| `CharAcc` | Axis 1 — fitting-similarity over detectable frames |
| `TTS` | Axis 1 — adjacent-detectable-pair string stability |
| `Flicker_full` | Axis 2 — full-frame mean absolute pixel difference between adjacent frames |
| `Flicker_crop` | Axis 2 — same on the per-clip union mask bbox |
| `MUSIQ_full` | Axis 2 — KonIQ-pretrained MUSIQ on full frames |
| `MUSIQ_crop` | Axis 2 — same on detectable-frame text-region crops |
| `PSNR_loc` | Axis 3 — PSNR on the locality-only prediction (capped at 100 dB) |
| `SSIM_loc` | Axis 3 — SSIM on the locality-only prediction |
| `LPIPS_loc` | Axis 3 — LPIPS-Alex on the locality-only prediction |

Means only — full 95 % bootstrap CIs over 1000 clip-level resamples are written into each baseline's `eval.json`:

```python
import json
agg = json.load(open('outputs/ViTeX-14B/eval.json'))['aggregate']
print(agg['SeqAcc'])
# {'mean': 0.341, 'ci_lo': 0.265, 'ci_hi': 0.418, 'n': 157}
```

The headline numbers in this directory were produced on the test split distributed with this repository's accompanying paper. See `docs/REPRODUCIBILITY.md` for the exact environment and the expected runtime.
