"""ViTeX-Bench evaluation entry point.

Reports the 13-metric protocol over three axes:
    Axis 1 — text correctness:  SeqAcc, CharAcc, TTS
    Axis 2 — visual quality:    Flicker_{full,crop}, Warp_{full,crop}, MUSIQ_{full,crop}
    Axis 3 — edit locality:     PSNR_loc, SSIM_loc, LPIPS_loc, DreamSim_loc

The thirteen-metric vector is the unit of report. No cross-axis
aggregate is computed; the leaderboard sorts by TextScore (geometric
mean of the three text-correctness primitives) but every metric is
still emitted alongside it so the full vector is always visible.
Each test-split aggregate is reported with a 95% bootstrap confidence
interval (1000 resamples over clips). Per-clip records carry the raw
source / prediction OCR strings together with all metric values so
failure cases can be re-analyzed without re-running OCR or the GPU
metric pass.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import (
    align_resolution,
    load_mask_frames,
    load_video_frames,
    truncate_to_common_length,
)
import text_metrics
import visual_metrics
import locality_metrics
from text_score import text_score


METRIC_KEYS = [
    "SeqAcc", "CharAcc", "TTS",
    "Flicker_full", "Flicker_crop", "Warp_full", "Warp_crop",
    "MUSIQ_full", "MUSIQ_crop",
    "PSNR_loc", "SSIM_loc", "LPIPS_loc", "DreamSim_loc",
]


def _bootstrap_ci(values, n_resamples=1000, seed=0, alpha=0.05):
    """95% percentile bootstrap CI over a list of per-clip metric values.
    `values` is already filtered for None entries."""
    if len(values) < 2:
        return (None, None)
    rng = random.Random(seed)
    N = len(values)
    means = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(N)] for _ in range(N)]
        means.append(sum(sample) / N)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return (float(lo), float(hi))


def _aggregate_metrics(per_clip):
    """Per-metric aggregate (mean and bootstrap CI over clips)."""
    out = {}
    for k in METRIC_KEYS:
        vals = [c[k] for c in per_clip.values() if c.get(k) is not None]
        if not vals:
            out[k] = {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
            continue
        lo, hi = _bootstrap_ci(vals)
        out[k] = {
            "mean": float(np.mean(vals)),
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(vals),
        }
    return out


def _aggregate_text_score(per_clip, n_resamples=1000, seed=0, alpha=0.05):
    """TextScore on aggregated text metrics, with bootstrap CI from
    clip-level resamples (the score is recomputed on each resample
    rather than being propagated from per-metric CIs)."""
    clip_list = list(per_clip.values())
    n = len(clip_list)

    def _agg_text_means(clips):
        out = {}
        for k in ("SeqAcc", "CharAcc", "TTS"):
            vals = [c[k] for c in clips if c.get(k) is not None]
            if vals:
                out[k] = sum(vals) / len(vals)
        return out

    main = text_score(_agg_text_means(clip_list))
    if n < 2:
        return {"TextScore": {"mean": main, "ci_lo": None, "ci_hi": None, "n": n}}

    rng = random.Random(seed)
    samples = []
    for _ in range(n_resamples):
        sample = [clip_list[rng.randrange(n)] for _ in range(n)]
        s = text_score(_agg_text_means(sample))
        if s is not None:
            samples.append(s)
    if not samples:
        return {"TextScore": {"mean": main, "ci_lo": None, "ci_hi": None, "n": n}}
    samples.sort()
    lo = samples[int(alpha / 2 * len(samples))]
    hi = samples[int((1 - alpha / 2) * len(samples)) - 1]
    return {"TextScore": {"mean": main, "ci_lo": float(lo), "ci_hi": float(hi), "n": n}}


def _print_summary(agg, score_agg, n_clips):
    def fmt(d):
        if d["mean"] is None:
            return "N/A"
        if d["ci_lo"] is None:
            return f"{d['mean']:.4f} (n={d['n']})"
        return f"{d['mean']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] (n={d['n']})"
    print()
    print("=" * 70)
    print(f"ViTeX-Bench results ({n_clips} clips, 95% bootstrap CI)")
    print("=" * 70)
    print(f"  TextScore (leaderboard sort key): {fmt(score_agg['TextScore'])}")
    print()
    print("Axis 1 — Text correctness")
    print(f"  SeqAcc        : {fmt(agg['SeqAcc'])}")
    print(f"  CharAcc       : {fmt(agg['CharAcc'])}")
    print(f"  TTS           : {fmt(agg['TTS'])}")
    print("Axis 2 — Visual quality  (Flicker/Warp lower; MUSIQ higher)")
    print(f"  Flicker_full  : {fmt(agg['Flicker_full'])}")
    print(f"  Flicker_crop  : {fmt(agg['Flicker_crop'])}")
    print(f"  Warp_full     : {fmt(agg['Warp_full'])}")
    print(f"  Warp_crop     : {fmt(agg['Warp_crop'])}")
    print(f"  MUSIQ_full    : {fmt(agg['MUSIQ_full'])}")
    print(f"  MUSIQ_crop    : {fmt(agg['MUSIQ_crop'])}")
    print("Axis 3 — Edit locality   (PSNR/SSIM higher, LPIPS/DreamSim lower)")
    print(f"  PSNR_loc      : {fmt(agg['PSNR_loc'])}")
    print(f"  SSIM_loc      : {fmt(agg['SSIM_loc'])}")
    print(f"  LPIPS_loc     : {fmt(agg['LPIPS_loc'])}")
    print(f"  DreamSim_loc  : {fmt(agg['DreamSim_loc'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--ocr_results", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=float, default=0.5,
                    help="Detectability threshold (paper: 0.5)")
    args = ap.parse_args()

    output = args.output or os.path.join(args.pred_dir, "eval_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)

    with open(args.records) as f:
        records = json.load(f)
    with open(args.ocr_results) as f:
        ocr_data = json.load(f)

    per_clip = {}
    t0 = time.time()
    for i, rec in enumerate(records):
        vid = rec["id"]
        if vid not in ocr_data:
            print(f"  [{i+1}/{len(records)}] {vid}: SKIP (no OCR)")
            continue
        src_path = os.path.join(args.data_root, rec["original_video"])
        mask_path = os.path.join(args.data_root, rec["mask_video"])
        pred_path = os.path.join(args.pred_dir, vid + ".mp4")
        if not (os.path.exists(src_path) and os.path.exists(mask_path)
                and os.path.exists(pred_path)):
            print(f"  [{i+1}/{len(records)}] {vid}: SKIP (missing)")
            continue

        pred_frames = load_video_frames(pred_path)
        src_frames = load_video_frames(src_path)
        if not pred_frames or not src_frames:
            continue
        # Source resolution (1280x720 in this dataset) is the reference grid.
        # Predictions emitted at a different resolution -- e.g. text_ctrl+anyv2v
        # at 512x512 -- are upsampled here so MUSIQ / PSNR / LPIPS are computed
        # in a common pixel space rather than against a downsampled source.
        h, w = src_frames[0].shape[:2]
        pred_frames = align_resolution(pred_frames, h, w)
        mask_frames = load_mask_frames(mask_path, target_h=h, target_w=w)
        pred_frames, src_frames, mask_frames = truncate_to_common_length(
            pred_frames, src_frames, mask_frames
        )
        if len(pred_frames) < 30:
            print(f"  WARN: {vid} usable frames={len(pred_frames)} after truncation")

        oc = ocr_data[vid]
        src_ocr = oc["source_ocr"][: len(pred_frames)]
        pred_ocr = oc["pred_ocr"][: len(pred_frames)]

        text_scores = text_metrics.evaluate_clip(
            src_ocr, pred_ocr, oc["source_text"], oc["target_text"], tau=args.tau,
        )
        detectable = text_metrics.detectable_frames(src_ocr, oc["source_text"], args.tau)
        visual_scores = visual_metrics.evaluate_clip(
            src_frames, pred_frames, mask_frames,
            detectable_idx=detectable, device=args.device,
        )
        locality_scores = locality_metrics.evaluate_clip(
            src_frames, pred_frames, mask_frames, device=args.device,
        )

        clip_metrics = {
            **text_scores,
            **visual_scores,
            **locality_scores,
        }
        per_clip[vid] = {
            **clip_metrics,
            "TextScore": text_score(clip_metrics),
            "lang": oc.get("lang", "en"),
            "source_text": oc["source_text"],
            "target_text": oc["target_text"],
            "source_ocr": src_ocr,
            "pred_ocr": pred_ocr,
        }
        d = per_clip[vid]
        seq = "N/A" if d["SeqAcc"] is None else f"{d['SeqAcc']:.2f}"
        cha = "N/A" if d["CharAcc"] is None else f"{d['CharAcc']:.2f}"
        tts_ = "N/A" if d["TTS"] is None else f"{d['TTS']:.2f}"
        elapsed = time.time() - t0
        rate = (i + 1) / max(elapsed, 1e-6)
        eta = (len(records) - i - 1) / max(rate, 1e-6)
        print(f"  [{i+1}/{len(records)}] {vid}: "
              f"|D|={text_scores['num_detectable']} |P|={text_scores['num_pairs']} "
              f"Seq={seq} Char={cha} TTS={tts_} "
              f"PSNR={d['PSNR_loc']:.1f} LPIPS={d['LPIPS_loc']:.3f} "
              f"eta={eta/60:.1f}min", flush=True)

    aggregate = _aggregate_metrics(per_clip)
    score_aggregate = _aggregate_text_score(per_clip)
    aggregate.update(score_aggregate)
    with open(output, "w") as f:
        json.dump({"per_clip": per_clip, "aggregate": aggregate}, f,
                  indent=2, ensure_ascii=False)
    _print_summary(aggregate, score_aggregate, len(per_clip))
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()
