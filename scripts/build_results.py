"""Build the public release artifacts under results/ from a local
outputs/eval_all.json (the paper run combining all baselines).

Outputs:
  results/eval_all.json     — per-method per-clip + aggregate (13 metrics).
  results/summary.tsv       — one row per method, 13 metrics + Pareto flag.
  results/leaderboard.jsonl — one record per method in the format of
                              https://github.com/ViTeX-Bench/ViTeX-Bench-Leaderboard
                              (data/submissions.jsonl).

Methods are compared on the three primary metrics (SeqAcc, Warp_crop,
DreamSim_loc) through the Pareto set; no aggregate score is computed.
Legacy `vitex_score` / `axis_scores` / `TextScore` fields are stripped.

Usage:
    python scripts/build_results.py path/to/outputs/eval_all.json
"""

import argparse
import json
import os
import sys
from typing import Dict


METRIC_KEYS = [
    "SeqAcc", "CharAcc", "TTS",
    "Flicker_full", "Flicker_crop", "Warp_full", "Warp_crop",
    "MUSIQ_full", "MUSIQ_crop",
    "PSNR_loc", "SSIM_loc", "LPIPS_loc", "DreamSim_loc",
]
PRIMARY_KEYS = ["SeqAcc", "Warp_crop", "DreamSim_loc"]   # higher, lower, lower

# Map raw output directory names to public method labels (paper-aligned).
METHOD_LABEL = {
    "identity":         ("Source video",           ""),
    "anytext2":         ("AnyText2",               "A — per-frame image editing"),
    "text_ctrl":        ("TextCtrl",               "A — per-frame image editing"),
    "fluxtext":         ("FLUX-Text",              "A — per-frame image editing"),
    "re-ste":           ("RS-STE",                 "A — per-frame image editing"),
    "text_ctrl+anyv2v": ("TextCtrl + AnyV2V",      "B — first-frame editing + propagation"),
    "wan2.2vace14b":    ("Wan2.1-VACE-14B",        "C — mask-conditioned video inpainting"),
    "videopainter":     ("VideoPainter",           "C — mask-conditioned video inpainting"),
    "kling":            ("Kling Video 3.0 Omni",   "D — instruction-guided video editing"),
    "ViTeX-14B":        ("ViTeX-Edit-14B",              "Reference editor"),
    "ViTeX-14B_Corp":   ("ViTeX-Edit-14B (Composite)",  "Reference editor"),
}

PAPER_NOTE = "Chen et al., 2026"
PAPER_URL = "https://arxiv.org/abs/2609.40356"
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
    "ViTeX-Edit-14B":             (PAPER_NOTE, PAPER_URL, "https://github.com/taco-group/ViTeX-Bench/tree/main/vitex_edit"),
    "ViTeX-Edit-14B (Composite)": (PAPER_NOTE, PAPER_URL, "https://github.com/taco-group/ViTeX-Bench/tree/main/vitex_edit"),
    "Source video":          ("", "", ""),
}


# Rows that are not independent editors and are never ranked.
KIND = {"Source video": "reference", "ViTeX-Edit-14B (Composite)": "postprocessed"}
# Temporal metrics come from linear-blend frame interpolation (paper App. F).
TEMPORAL_NOT_COMPARABLE = {"VideoPainter"}


def pareto_front(rows):
    """Mark r['pareto'] for ranked editors on (SeqAcc up, Warp_crop down,
    DreamSim_loc down); None for rows outside the comparison."""
    pool = [r for r in rows if r["kind"] == "editor" and r["temporal_comparable"]]

    def dominates(o, r):
        ge = (o["SeqAcc"] >= r["SeqAcc"] and o["Warp_crop"] <= r["Warp_crop"]
              and o["DreamSim_loc"] <= r["DreamSim_loc"])
        gt = (o["SeqAcc"] > r["SeqAcc"] or o["Warp_crop"] < r["Warp_crop"]
              or o["DreamSim_loc"] < r["DreamSim_loc"])
        return ge and gt

    for r in rows:
        r["pareto"] = (not any(dominates(o, r) for o in pool if o is not r)) if r in pool else None


def clean_per_clip(per_clip: Dict, keep_ocr_strings: bool = False) -> Dict:
    """Keep all 13 metrics + the source/target string pair plus
    detectable-frame counts. Drop the per-frame OCR string arrays by default
    (they balloon eval_all.json above the HF 10 MiB push limit and are
    re-derivable from the OCR cache); pass keep_ocr_strings=True to retain."""
    drop_unless_kept = ("source_ocr", "pred_ocr")
    out = {}
    for vid, c in per_clip.items():
        rec = {k: c.get(k) for k in METRIC_KEYS}
        for k in ("lang", "source_text", "target_text", "source_ocr", "pred_ocr",
                  "num_detectable", "num_pairs", "bbox_local"):
            if k in c and (keep_ocr_strings or k not in drop_unless_kept):
                rec[k] = c[k]
        out[vid] = rec
    return out


def clean_aggregate(agg: Dict) -> Dict:
    """Keep only the 13 metric aggregates."""
    return {k: agg.get(k) for k in METRIC_KEYS if k in agg}


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
        agg_clean = clean_aggregate(payload["aggregate"])
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
        kind = KIND.get(label, "editor")
        temporal_ok = label not in TEMPORAL_NOT_COMPARABLE
        rows.append({"raw": raw_name, "label": label, "family": family, "n": n,
                     "kind": kind, "temporal_comparable": temporal_ok, **means})

        org, paper_url, code_url = EXTERNAL_REF.get(label, ("—", "", ""))
        entry = {
            "method": label,
            "kind": kind,
            "family": family,
            "organization": org,
            "paper_url": paper_url,
            "code_url": code_url,
            "temporal_comparable": temporal_ok,
            "submitter": "admin",
            "added": "2026-05-04",
            **{k: means.get(k) for k in METRIC_KEYS},
            "n_clips": n,
        }
        if kind == "reference":
            entry["PSNR_loc"] = None  # identical to the source: PSNR is infinite
        leaderboard.append(entry)

    # results/eval_all.json: cleaned full dump
    with open(os.path.join(out_dir, "eval_all.json"), "w") as f:
        json.dump({
            "metric_keys": METRIC_KEYS,
            "primary_metrics": PRIMARY_KEYS,
            "comparison": "Pareto set on the primary metrics (SeqAcc higher, "
                          "Warp_crop lower, DreamSim_loc lower); no aggregate score",
            "baselines": cleaned_baselines,
        }, f, indent=2, ensure_ascii=False)

    # results/summary.tsv
    pareto_front(rows)
    header = ["method", "family", "n_clips", "pareto", *METRIC_KEYS]
    # Ranked editors first, then unranked rows; each group by SeqAcc desc.
    rows.sort(key=lambda r: (r["kind"] != "editor", -(r["SeqAcc"] or 0)))
    with open(os.path.join(out_dir, "summary.tsv"), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            pareto = {True: "yes", False: "no", None: "-"}[r["pareto"]]
            cells = [r["label"], r["family"] or "-", str(r["n"]), pareto]
            for k in METRIC_KEYS:
                v = r.get(k)
                cells.append(f"{v:.4f}" if v is not None else "N/A")
            f.write("\t".join(cells) + "\n")

    # results/leaderboard.jsonl: seed data for the GitHub leaderboard
    leaderboard.sort(key=lambda e: (e["kind"] != "editor", -(e["SeqAcc"] or 0)))
    with open(os.path.join(out_dir, "leaderboard.jsonl"), "w") as f:
        for e in leaderboard:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"Wrote {out_dir}/eval_all.json ({sum(len(b['per_clip']) for b in cleaned_baselines.values())} clip records, {len(cleaned_baselines)} baselines)")
    print(f"Wrote {out_dir}/summary.tsv")
    print(f"Wrote {out_dir}/leaderboard.jsonl")


if __name__ == "__main__":
    main()
