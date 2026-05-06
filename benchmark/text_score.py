"""TextScore: a single-axis ranking key for the public leaderboard.

ViTeX-Bench reports each method as a full thirteen-metric vector across
three axes (text correctness, visual quality, edit locality). The vector
is the unit of report; no cross-axis aggregate is published, because no
axis substitutes for another (see paper, Sec. "evaluation protocol").

The leaderboard, however, needs a single column to sort on. We use
TextScore, the unweighted geometric mean of the three text-correctness
primitives -- the axis whose failure is least recoverable, since a clip
with the wrong characters cannot be fixed by any amount of visual
quality or locality fidelity:

    TextScore = (SeqAcc * CharAcc * TTS) ** (1 / 3)

All three primitives are natively in [0, 1], so no normalization is
applied. SeqAcc = 0 collapses TextScore to 0 -- the intended semantics
for methods that never produce the requested target string. The
remaining ten primitives still appear next to TextScore on the
leaderboard so reviewers can read the full vector.
"""

import math


TEXT_METRICS = ["SeqAcc", "CharAcc", "TTS"]


def text_score(metrics):
    """Geometric mean of SeqAcc, CharAcc, TTS. Returns None if any of
    the three is missing (the score is not defined on partial inputs).
    Returns 0.0 if any of the three is exactly 0."""
    vals = [metrics.get(k) for k in TEXT_METRICS]
    if any(v is None for v in vals):
        return None
    if any(v <= 0.0 for v in vals):
        return 0.0
    return math.exp(sum(math.log(v) for v in vals) / len(vals))
