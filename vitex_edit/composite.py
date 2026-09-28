"""Composite post-processing: ViTeX-Edit-14B -> ViTeX-Edit-14B (Composite).

A deterministic, training-free wrapper applied to raw predictions (the paper
applies the same parameters to every baseline). Per frame:
  1. Reinhard color transfer in LAB: the prediction is re-normalized so a
     20-px band just outside the mask matches the source's mean and std.
  2. Signed-distance feathered alpha (4 px, centered on the mask edge)
     composites the corrected prediction onto the source.
Outside the feather the output is the source video; inside the mask it is
the color-matched prediction.

  python vitex_edit/composite.py --records data/eval/parsed_records.json \
      --data_root data/eval \
      --pred_dir baseline_output_videos/ViTeX-Edit-14B \
      --out_dir  baseline_output_videos/ViTeX-Edit-14B-Composite
"""

import argparse
import json
import os
import subprocess
from multiprocessing import Pool

import cv2
import numpy as np


def _read_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    return out


def _read_mask_video(path, target_h, target_w, max_frames=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if (gray.shape[0], gray.shape[1]) != (target_h, target_w):
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        out.append((gray > 127).astype(np.uint8))
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    return out


def _color_correct_lab(src_rgb, pred_rgb, mask_bin, band_width=20):
    """Reinhard-style LAB transfer using a band around the mask as reference."""
    band = cv2.dilate(mask_bin, np.ones((band_width * 2 + 1, band_width * 2 + 1),
                                        dtype=np.uint8)) - mask_bin
    band_idx = band > 0
    if band_idx.sum() < 100:
        return pred_rgb  # not enough reference, leave as-is

    src_lab = cv2.cvtColor(src_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    pred_lab = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    mean_src = src_lab[band_idx].mean(axis=0)
    std_src = src_lab[band_idx].std(axis=0) + 1e-6
    mean_pred = pred_lab[band_idx].mean(axis=0)
    std_pred = pred_lab[band_idx].std(axis=0) + 1e-6

    pred_corrected = (pred_lab - mean_pred) / std_pred * std_src + mean_src
    pred_corrected = np.clip(pred_corrected, 0, 255).astype(np.uint8)
    return cv2.cvtColor(pred_corrected, cv2.COLOR_LAB2RGB)


def _feathered_alpha(mask_bin, feather=4):
    """Smooth alpha centered on the mask boundary."""
    sdf_in = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
    sdf_out = cv2.distanceTransform(1 - mask_bin, cv2.DIST_L2, 5)
    sdf = sdf_in - sdf_out
    return np.clip((sdf + feather / 2.0) / feather, 0.0, 1.0).astype(np.float32)


def _process_frame(src_rgb, pred_rgb, mask_bin, band_width, feather):
    pred_cc = _color_correct_lab(src_rgb, pred_rgb, mask_bin, band_width=band_width)
    alpha = _feathered_alpha(mask_bin, feather=feather)[..., None]
    out = src_rgb.astype(np.float32) * (1 - alpha) + pred_cc.astype(np.float32) * alpha
    return out.astype(np.uint8)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _encode_video(frames, out_path, fps=24):
    if not frames:
        raise RuntimeError("no frames to encode")
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen([
        _ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        out_path,
    ], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def _process_clip(args):
    rec, data_root, pred_dir, out_dir, target_frames, band_width, feather = args
    vid = rec["id"]
    out_path = os.path.join(out_dir, vid + ".mp4")
    if os.path.exists(out_path):
        return vid, "skip"

    src_path = os.path.join(data_root, rec["original_video"])
    mask_path = os.path.join(data_root, rec["mask_video"])
    pred_path = os.path.join(pred_dir, vid + ".mp4")
    if not (os.path.exists(src_path) and os.path.exists(mask_path) and os.path.exists(pred_path)):
        return vid, "missing"

    src_frames = _read_video(src_path, max_frames=target_frames)
    pred_frames = _read_video(pred_path, max_frames=target_frames)
    if not src_frames or not pred_frames:
        return vid, "empty"
    h, w = src_frames[0].shape[:2]
    # Pred may be smaller (e.g., other res); resample to source grid.
    pred_frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4)
                   if (f.shape[0], f.shape[1]) != (h, w) else f
                   for f in pred_frames]
    mask_frames = _read_mask_video(mask_path, target_h=h, target_w=w, max_frames=target_frames)

    n = min(len(src_frames), len(pred_frames), len(mask_frames), target_frames)
    out_frames = []
    for t in range(n):
        out_frames.append(_process_frame(
            src_frames[t], pred_frames[t], mask_frames[t], band_width, feather,
        ))
    _encode_video(out_frames, out_path, fps=24)
    return vid, f"ok ({n}f)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, help="parsed_records.json")
    ap.add_argument("--data_root", required=True, help="Root that record paths are relative to")
    ap.add_argument("--pred_dir", required=True, help="Raw predictions, <id>.mp4")
    ap.add_argument("--out_dir", required=True, help="Where composited <id>.mp4 are written")
    ap.add_argument("--target_frames", type=int, default=120)
    ap.add_argument("--band_width", type=int, default=20,
                    help="Width in px of the reference band around the mask")
    ap.add_argument("--feather", type=int, default=4,
                    help="Feather width in px centered on the mask edge")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.records) as f:
        records = json.load(f)

    tasks = [(r, args.data_root, args.pred_dir, args.out_dir,
              args.target_frames, args.band_width, args.feather)
             for r in records]

    n_ok, n_skip, n_miss, n_err = 0, 0, 0, 0
    with Pool(args.workers) as p:
        for i, (vid, status) in enumerate(p.imap_unordered(_process_clip, tasks), 1):
            if status.startswith("ok"):
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            elif status == "missing":
                n_miss += 1
            else:
                n_err += 1
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {vid}: {status}", flush=True)
    print(f"\nDone: ok={n_ok} skipped={n_skip} missing={n_miss} errors={n_err}")


if __name__ == "__main__":
    main()
