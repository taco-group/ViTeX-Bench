"""Build the public release artifacts under results/ from a local
outputs/eval_all.json (the paper run combining all baselines).

Outputs:
  results/eval_all.json   — per-method per-clip + aggregate, with TextScore.
  results/summary.tsv     — one row per baseline, 13 metrics + TextScore.
  results/leaderboard.jsonl — one record per baseline, ready for the
                              leaderboard Space (sorted by TextScore desc).

The legacy `vitex_score` / `axis_scores` fields are stripped — the public
ranking key is TextScore alone (see benchmark/text_score.py).

Usage:
    python scripts/build_results.py path/to/outputs/eval_all.json
"""

import argparse
import json
import math
import os
import sys
from typing import Dict


METRIC_KEYS = [
    "SeqAcc", "CharAcc", "TTS",
    "Flicker_full", "Flicker_crop", "Warp_full", "Warp_crop",
    "MUSIQ_full", "MUSIQ_crop",
    "PSNR_loc", "SSIM_loc", "LPIPS_loc", "DreamSim_loc",
]
TEXT_KEYS = ["SeqAcc", "CharAcc", "TTS"]

# Map raw output directory names to public method labels (paper-aligned).
METHOD_LABEL = {
    "identity":         ("Identity (sanity)",      "—"),
    "anytext2":         ("AnyText2",               "A — per-frame image editor"),
    "text_ctrl":        ("TextCtrl",               "A — per-frame image editor"),
    "fluxtext":         ("FLUX-Text",              "A — per-frame image editor"),
    "re-ste":           ("RS-STE",                 "A — per-frame image editor"),
    "text_ctrl+anyv2v": ("TextCtrl + AnyV2V",      "B — first-frame + I2V propagation"),
    "wan2.2vace14b":    ("Wan2.1-VACE-14B",        "C — mask-conditioned video inpainting"),
    "videopainter":     ("VideoPainter",           "C — mask-conditioned video inpainting"),
    "kling":            ("Kling Video 3.0 Omni",   "D — instruction-guided V2V"),
    "ViTeX-14B":        ("ViTeX-Edit-14B",              "Reference"),
    "ViTeX-14B_Corp":   ("ViTeX-Edit-14B (Composite)",  "Reference"),
}

PAPER_NOTE = "Anonymous (NeurIPS 2026 D&B submission)"
EXTERNAL_REF = {
    "AnyText2":              ("Tuo et al., 2024",  "https://arxiv.org/abs/2411.15245",
                              "https://github.com/tyxsspa/AnyText2"),
    "TextCtrl":              ("Zeng et al., 2024", "https://arxiv.org/abs/2410.10133",
                              "https://github.com/weichaozeng/TextCtrl"),
    "FLUX-Text":             ("Chen et al., 2025", "https://arxiv.org/abs/2505.03329",
                              "https://github.com/AMAP-ML/FluxText"),
    "RS-STE":                ("Zhao et al., 2025", "https://arxiv.org/abs/2503.17774",
                              "https://github.com/honglei-zhao/RS-STE"),
    "TextCtrl + AnyV2V":     ("Composite of Zeng 2024 + Ku 2024", "", ""),
    "Wan2.1-VACE-14B":       ("Wan-AI, 2025",      "https://arxiv.org/abs/2503.07598",
                              "https://huggingface.co/Wan-AI/Wan2.1-VACE-14B"),
    "VideoPainter":          ("Bian et al., 2025", "https://arxiv.org/abs/2503.05639",
                              "https://github.com/TencentARC/VideoPainter"),
    "Kling Video 3.0 Omni":  ("Kuaishou (closed)", "", ""),
    "ViTeX-Edit-14B":             (PAPER_NOTE, "", "https://huggingface.co/ViTeX-Bench/ViTeX-14B"),
    "ViTeX-Edit-14B (Composite)": (PAPER_NOTE, "", "https://huggingface.co/ViTeX-Bench/ViTeX-14B"),
    "Identity (sanity)":     ("—", "", ""),
}


def text_score(seq, char, tts):
    if seq is None or char is None or tts is None:
        return None
    if seq <= 0.0 or char <= 0.0 or tts <= 0.0:
        return 0.0
    return math.exp((math.log(seq) + math.log(char) + math.log(tts)) / 3.0)


def clean_per_clip(per_clip: Dict, keep_ocr_strings: bool = False) -> Dict:
    """Keep all 13 metrics + TextScore + the source/target string pair plus
    detectable-frame counts. Drop the per-frame OCR string arrays by default
    (they balloon eval_all.json above the HF 10 MiB push limit and are
    re-derivable from the OCR cache); pass keep_ocr_strings=True to retain."""
    drop_unless_kept = ("source_ocr", "pred_ocr")
    out = {}
    for vid, c in per_clip.items():
        rec = {k: c.get(k) for k in METRIC_KEYS}
        rec["TextScore"] = text_score(c.get("SeqAcc"), c.get("CharAcc"), c.get("TTS"))
        for k in ("lang", "source_text", "target_text", "source_ocr", "pred_ocr",
                  "num_detectable", "num_pairs", "bbox_local"):
            if k in c and (keep_ocr_strings or k not in drop_unless_kept):
                rec[k] = c[k]
        out[vid] = rec
    return out


def aggregate_with_textscore(agg: Dict, n_clips: int) -> Dict:
    """Strip vitex_score/axis_* fields, add TextScore from aggregate text means."""
    out = {k: agg.get(k) for k in METRIC_KEYS if k in agg}
    means = {k: (agg[k]["mean"] if isinstance(agg[k], dict) else agg[k]) for k in TEXT_KEYS}
    ts = text_score(*[means[k] for k in TEXT_KEYS])
    out["TextScore"] = {"mean": ts, "ci_lo": None, "ci_hi": None, "n": n_clips}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_all", help="Path to outputs/eval_all.json")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.eval_all) as f:
        data = json.load(f)

    cleaned_baselines = {}
    rows = []
    leaderboard = []

    for raw_name, payload in data["baselines"].items():
        if raw_name not in METHOD_LABEL:
            print(f"[skip] unknown method: {raw_name}", file=sys.stderr)
            continue
        label, family = METHOD_LABEL[raw_name]
        n = payload.get("n_clips") or len(payload.get("per_clip", {}))
        agg_clean = aggregate_with_textscore(payload["aggregate"], n)
        clip_clean = clean_per_clip(payload.get("per_clip", {}))
        cleaned_baselines[raw_name] = {
            "label": label,
            "family": family,
            "n_clips": n,
            "aggregate": agg_clean,
            "per_clip": clip_clean,
        }

        means = {k: (agg_clean[k]["mean"] if isinstance(agg_clean[k], dict) else agg_clean[k])
                 for k in METRIC_KEYS if k in agg_clean}
        ts = agg_clean["TextScore"]["mean"]
        rows.append({"raw": raw_name, "label": label, "family": family,
                     "n": n, "TextScore": ts, **means})

        org, paper_url, code_url = EXTERNAL_REF.get(label, ("—", "", ""))
        entry = {
            "method": label,
            "family": family,
            "organization": org,
            "paper_url": paper_url,
            "code_url": code_url,
            "submitter": "admin",
            "submitted_at": "2026-05-04 00:00:00 UTC",
            "approved_at": "2026-05-04 00:00:00 UTC",
            "status": "approved",
            "TextScore": ts,
            **{k: means.get(k) for k in METRIC_KEYS},
            "n_clips": n,
        }
        leaderboard.append(entry)

    # results/eval_all.json: cleaned full dump
    with open(os.path.join(out_dir, "eval_all.json"), "w") as f:
        json.dump({
            "metric_keys": METRIC_KEYS,
            "ranking_key": "TextScore",
            "ranking_definition": "geomean(SeqAcc, CharAcc, TTS); each in [0, 1]",
            "baselines": cleaned_baselines,
        }, f, indent=2, ensure_ascii=False)

    # results/summary.tsv
    header = ["baseline", "family", "n_clips", "TextScore", *METRIC_KEYS]
    rows.sort(key=lambda r: -(r["TextScore"] if r["TextScore"] is not None else -1))
    with open(os.path.join(out_dir, "summary.tsv"), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            cells = [r["label"], r["family"], str(r["n"]),
                     f"{r['TextScore']:.4f}" if r["TextScore"] is not None else "N/A"]
            for k in METRIC_KEYS:
                v = r.get(k)
                cells.append(f"{v:.4f}" if v is not None else "N/A")
            f.write("\t".join(cells) + "\n")

    # results/leaderboard.jsonl: pre-populated for the Space (sorted by TextScore)
    leaderboard.sort(key=lambda e: -(e["TextScore"] if e["TextScore"] is not None else -1))
    with open(os.path.join(out_dir, "leaderboard.jsonl"), "w") as f:
        for e in leaderboard:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {out_dir}/eval_all.json ({sum(len(b['per_clip']) for b in cleaned_baselines.values())} clip records, {len(cleaned_baselines)} baselines)")
    print(f"Wrote {out_dir}/summary.tsv")
    print(f"Wrote {out_dir}/leaderboard.jsonl")


if __name__ == "__main__":
    main()
