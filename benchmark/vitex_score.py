"""ViTeX-Score: a single-number summary of the 13-metric protocol.

Design philosophy
-----------------
The score is structured as a two-level geometric aggregate:

    axis_score = geomean(normalized metrics in axis)
    ViTeX-Score = geomean(axis scores)

with all 13 metrics normalized to [0, 1] using **frozen** endpoints (a
benchmark contract that does not adapt to new methods, so historical
numbers stay stable).

The geometric mean is used at both levels because the three axes are
non-substitutive: a method that fails the task on any axis -- e.g., a
Family-D commercial model that re-renders every pixel beautifully but
emits SeqAcc = 0 -- must score near zero overall. Within an axis the same
intuition holds: SeqAcc = 0 means the model never produces the requested
target string, and no amount of partial CharAcc credit on near-misses
should hide that. The geomean is robust to small partial-credit
fluctuations -- two metrics at 0.6 and 0.4 give 0.49, close to their 0.5
arithmetic mean -- but a hard zero on any single metric annihilates the
axis (and the overall score), as intended.

Endpoint selection (frozen v1)
------------------------------
| Metric            | Range        | Justification                       |
|-------------------|--------------|-------------------------------------|
| SeqAcc/CharAcc/TTS| [0, 1]       | native bounds                       |
| SSIM_loc          | [0, 1]       | native                              |
| LPIPS_loc         | [0, 1]       | native (inverted)                   |
| PSNR_loc          | [0, 50] dB   | clip then scale (cap matches our    |
|                   |              | identity 100 dB cap; methods rarely |
|                   |              | exceed 45 dB on real video)         |
| MUSIQ_f, MUSIQ_c  | [0, 100]     | model design range                  |
| Flicker_f         | [0, 6]       | observed worst-case 5.11 with head- |
|                   |              | room                                |
| Flicker_c         | [0, 15]      | observed worst-case 14.81 (FLUX-    |
|                   |              | Text) with headroom                 |
| Warp_f            | [0, 5]       | observed worst 4.11 with headroom   |
| Warp_c            | [0, 15]      | observed worst 13.01 with headroom  |
| DreamSim_loc      | [0, 0.1]     | observed worst 0.073 with headroom  |
"""

import math


# FROZEN v1.0 -- DO NOT MODIFY
# Endpoints define the [0, 1] normalization range. (min, max, direction).
ENDPOINTS = {
    "SeqAcc":       (0.0, 1.0,   "up"),    # native
    "CharAcc":      (0.0, 1.0,   "up"),    # native
    "TTS":          (0.0, 1.0,   "up"),    # native
    "Flicker_full": (0.0, 6.0,   "down"),  # observed worst with headroom
    "Flicker_crop": (0.0, 15.0,  "down"),  # observed worst with headroom
    "Warp_full":    (0.0, 5.0,   "down"),  # observed worst with headroom
    "Warp_crop":    (0.0, 15.0,  "down"),  # observed worst with headroom
    "MUSIQ_full":   (0.0, 100.0, "up"),    # model design range
    "MUSIQ_crop":   (0.0, 100.0, "up"),    # model design range
    "PSNR_loc":     (0.0, 50.0,  "up"),    # dB, clip-and-scale
    "SSIM_loc":     (0.0, 1.0,   "up"),    # native
    "LPIPS_loc":    (0.0, 1.0,   "down"),  # native (invert)
    "DreamSim_loc": (0.0, 0.1,   "down"),  # observed worst with headroom
}

AXIS_METRICS = {
    "correctness": ["SeqAcc", "CharAcc", "TTS"],
    "visual":      ["Flicker_full", "Flicker_crop", "Warp_full", "Warp_crop",
                    "MUSIQ_full", "MUSIQ_crop"],
    "locality":    ["PSNR_loc", "SSIM_loc", "LPIPS_loc", "DreamSim_loc"],
}

# Within-axis weights. Visual weights crop scope twice as much as full because
# the text region is the task focus; correctness and locality use uniform
# weights across their constituents.
AXIS_WEIGHTS = {
    "correctness": [1.0, 1.0, 1.0],
    "visual":      [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    "locality":    [1.0, 1.0, 1.0, 1.0],
}


def normalize(name, value):
    """Map one metric value to [0, 1] using its frozen endpoint, with
    direction (higher-is-better vs lower-is-better) handled in-place."""
    if value is None:
        return None
    if name not in ENDPOINTS:
        raise KeyError(f"unknown metric: {name}")
    lo, hi, direction = ENDPOINTS[name]
    span = hi - lo
    if span <= 0:
        raise ValueError(f"degenerate range for {name}: {(lo, hi)}")
    x = (value - lo) / span if direction == "up" else (hi - value) / span
    return max(0.0, min(1.0, x))


def _geomean(values, weights=None):
    """Weighted geometric mean over a finite list of values in [0, 1]. Any
    value exactly 0 collapses the result to 0 (deliberate; that is the
    structural-failure semantics the score is built around). If `weights`
    is None all values get unit weight."""
    if weights is None:
        finite = [(v, 1.0) for v in values if v is not None]
    else:
        finite = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not finite:
        return None
    if any(v <= 0.0 for v, _ in finite):
        return 0.0
    total_w = sum(w for _, w in finite)
    return math.exp(sum(w * math.log(v) for v, w in finite) / total_w)


def axis_scores(metrics):
    """Return dict {correctness, visual, locality} of axis scores in [0, 1]
    using the within-axis AXIS_WEIGHTS. An axis is None if all of its
    metrics are None for the input clip."""
    out = {}
    for axis, names in AXIS_METRICS.items():
        normed = [normalize(n, metrics.get(n)) for n in names]
        out[axis] = _geomean(normed, AXIS_WEIGHTS[axis])
    return out


def vitex_score(metrics, exclude_axes=()):
    """Compute ViTeX-Score for one metric dict. Returns None if no axis
    is evaluable; otherwise the geometric mean over the included axes.

    `exclude_axes` is a tuple of axis names ("correctness", "visual", or
    "locality") to drop before the cross-axis aggregation. This is used
    to mark methods whose architecture trivializes one of the axes -- in
    particular, per-frame ``bbox-local'' editors (Family A) get their
    locality score by passing the source pixels through unchanged, not
    by learning background reconstruction, and including their locality
    would inflate their aggregate against full-frame regenerative methods.
    """
    axes = axis_scores(metrics)
    kept = {k: v for k, v in axes.items()
            if v is not None and k not in exclude_axes}
    if not kept:
        return None
    return _geomean(list(kept.values()))
