# Results

This directory holds the public release artifacts produced by `scripts/build_results.py` from a paper-grade run of the benchmark over all baselines.

| file | contents |
|---|---|
| `eval_all.json` | per-method per-clip records + aggregates for all eleven baselines on the 157-clip frozen test split. Each method carries its label, family tag, and the 13 primitives + TextScore. The full per-clip OCR strings are kept so failure cases can be re-analyzed without re-running the pipeline. |
| `summary.tsv` | one row per baseline, sorted by TextScore descending. Columns: `baseline`, `family`, `n_clips`, `TextScore`, then the 13 primitives. |
| `leaderboard.jsonl` | one record per baseline in the format consumed by the [ViTeX-Bench-Leaderboard](https://huggingface.co/spaces/ViTeX-Bench/ViTeX-Bench-Leaderboard) Space; pre-populates the public leaderboard. |

## summary.tsv schema

| column | meaning |
|---|---|
| `baseline` | public method label (paper-aligned) |
| `family` | A — per-frame image editor / B — first-frame + I2V propagation / C — mask-conditioned video inpainting / D — instruction-guided V2V / Reference / `—` (sanity) |
| `n_clips` | number of clips with at least one valid metric value (per-clip `None`s are skipped during the mean) |
| `TextScore` | $\sqrt[3]{\mathrm{SeqAcc} \cdot \mathrm{CharAcc} \cdot \mathrm{TTS}}$ — leaderboard sort key |
| `SeqAcc` / `CharAcc` / `TTS` | Axis 1 — exact match / fitting similarity / adjacent-pair stability over detectable frames |
| `Flicker_full` / `Flicker_crop` | Axis 2 — adjacent-frame MAE on full frames / on the per-clip union mask bbox |
| `Warp_full` / `Warp_crop` | Axis 2 — adjacent-frame MAE after backward-warping along RAFT optical flow on the source |
| `MUSIQ_full` / `MUSIQ_crop` | Axis 2 — KonIQ-pretrained MUSIQ on full frames / detectable-frame text-region crops |
| `PSNR_loc` / `SSIM_loc` / `LPIPS_loc` / `DreamSim_loc` | Axis 3 — pixel-level and learned similarity on the locality-only prediction |

Means only — full 95 % bootstrap CIs over 1000 clip-level resamples are written into each method's per-method `outputs/<method>/eval.json` after a re-run:

```python
import json
agg = json.load(open('outputs/ViTeX-14B/eval.json'))['aggregate']
print(agg['SeqAcc'])
# {'mean': 0.341, 'ci_lo': 0.265, 'ci_hi': 0.418, 'n': 157}
```

The numbers in this directory were produced on the test split distributed with this repository's accompanying paper. See `docs/REPRODUCIBILITY.md` for the environment and expected runtime.
