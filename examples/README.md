# Examples

A single-clip walkthrough lives here once the public dataset mirror goes live (URL added at deanonymization). The example will include:

- A source clip with mask (`example_clip.mp4`, `example_mask.mp4`)
- The expected `parsed_records.json` entry
- A pre-computed identity-baseline `eval.json` so you can verify your local install end-to-end against known-good numbers

For now, the simplest end-to-end smoke test is the identity sanity baseline:

```bash
python benchmark/make_identity_baseline.py \
    --records  data/inference/parsed_records.json \
    --src_dir  data/inference/original_videos \
    --out_dir  baseline_output_videos/identity
```

then run Stage 1 + Stage 2 from the top-level Quickstart. Identity should produce `PSNR_loc = 100`, `SSIM_loc = 1`, `LPIPS_loc = 0` on every clip — if your numbers deviate, the pipeline isn't installed correctly.
