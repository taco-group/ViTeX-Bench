# Examples

The simplest end-to-end check of an installation is the identity sanity baseline, which copies each source clip as its own prediction:

```bash
hf download ViTeX-Bench/ViTeX-Dataset --repo-type dataset --include "eval/*" --local-dir data
python benchmark/make_identity_baseline.py \
    --records  data/eval/parsed_records.json \
    --src_dir  data/eval/original_videos \
    --out_dir  baseline_output_videos/identity
bash scripts/run_benchmark.sh identity
```

Identity should give `PSNR_loc = 100`, `SSIM_loc = 1` and `LPIPS_loc = 0` on every clip, and its aggregates should match the *Source video* row of [`results/summary.tsv`](../results/summary.tsv). If they do not, the pipeline is not installed correctly.

To produce predictions with the reference model, see [`vitex_edit/README.md`](../vitex_edit/README.md).
