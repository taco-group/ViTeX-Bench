"""Write the training metadata.csv for the ViTeX-Dataset train split.

Columns: video (edited target), vace_video (source), vace_video_mask,
glyph_video, prompt (the target string only, as used for ViTeX-Edit-14B).

  python vitex_edit/train/make_metadata.py --data_root data/train \
      --glyph_dir data/train/glyph_videos --output data/train/metadata.csv
"""

import argparse
import csv
import json
import os


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="Train split root (holds parsed_records.json)")
    p.add_argument("--glyph_dir", required=True, help="Glyph videos from render_glyph.py")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    with open(os.path.join(args.data_root, "parsed_records.json")) as f:
        records = json.load(f)
    rows, missing = [], []
    for r in records:
        glyph = os.path.join(args.glyph_dir, r["id"] + ".mp4")
        if not os.path.exists(glyph):
            missing.append(r["id"])
            continue
        rows.append({
            "video": r["edited_video"],
            "vace_video": r["original_video"],
            "vace_video_mask": r["mask_video"],
            "glyph_video": os.path.relpath(glyph, args.data_root),
            "prompt": r["target_text"],
        })
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "vace_video", "vace_video_mask", "glyph_video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {args.output}")
    if missing:
        print(f"WARNING: {len(missing)} clips have no glyph video and were left out: {missing[:5]}...")


if __name__ == "__main__":
    main()
