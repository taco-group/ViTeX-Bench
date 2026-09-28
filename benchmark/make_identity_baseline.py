"""Build the 'identity' sanity baseline by copying source videos verbatim.

Identity is an upper bound for edit-locality (PSNR_loc / SSIM_loc / LPIPS_loc
all reach their best possible values because the unedited region is exactly
preserved) and a lower bound for text-correctness (SeqAcc / CharAcc / TTS
should all be near zero unless the source text already matches the target).
It also calibrates Flicker against codec re-encoding noise.

Usage
-----
    python benchmark/make_identity_baseline.py \\
        --records  data/eval/parsed_records.json \\
        --src_dir  data/eval/original_videos \\
        --out_dir  baseline_output_videos/identity
"""

import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.records) as f:
        records = json.load(f)

    n_ok, n_skip = 0, 0
    for r in records:
        vid = r["id"]
        src = os.path.join(args.src_dir, vid + ".mp4")
        dst = os.path.join(args.out_dir, vid + ".mp4")
        if not os.path.exists(src):
            print(f"  SKIP {vid}: source missing at {src}")
            n_skip += 1
            continue
        shutil.copyfile(src, dst)
        n_ok += 1
    print(f"Identity baseline: {n_ok} clips copied, {n_skip} skipped → {args.out_dir}")


if __name__ == "__main__":
    main()
