"""Axis 1: text-correctness metrics with source-detectability gating.

Implements the protocol from ViTeX-Bench Section 3.2 (Text correctness):

    fit(pattern, text)
        = min edits to align `pattern` to a contiguous region of `text`,
          treating any prefix/suffix of `text` outside the match as free
          (semi-global / "fitting" Levenshtein alignment).

    Sim(target, pred)
        = 1 - fit(target, pred) / max(|target|, 1)

    D = {t : Sim(s_src, s_t^V)   >= tau}             (detectable frames)
    P = {(t, t+1) : t in D, t+1 in D}                (adjacent pairs)

    SeqAcc  = mean over D of 1[fit(s_tgt, s_hat_t) == 0]
    CharAcc = mean over D of Sim(s_tgt, s_hat_t)
    TTS     = mean over P of 1[norm(s_hat_t) == norm(s_hat_{t+1})]

Why fitting alignment? The dilated mask used by every baseline pads beyond
the original glyph boundary, so a successful edit (target='BIG') often
appears inside a longer OCR string (e.g., 'ABIGA') because the pad picks
up neighbouring scene pixels. Standard Levenshtein would penalize 'A...A'
as two substitutions; fitting alignment treats them as free since 'BIG'
matches as a substring of 'ABIGA'. The same logic stabilizes the source-
detectability gate against OCR over-segmentation.
"""

import unicodedata


def _normalize(s):
    """Canonicalize for character-level edit-distance comparison.

    NFKC (fold full-width / compatibility forms) -> upper-case -> keep only
    letters (any script, including CJK / Cyrillic / Hangul / Kana) and
    digits. Whitespace and all punctuation are dropped so that
    target='35,000' / OCR='35 000' and target="MIKE'S" / OCR='MIKES' are
    not penalized for tokenizer-level noise.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s).upper()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in ("L", "N"))


def fitting_distance(pattern, text):
    """Semi-global Levenshtein. `pattern`'s match can start/end anywhere in
    `text`; the unmatched prefix/suffix of `text` is free.

    When len(pattern) > len(text) this degenerates to the regular
    Levenshtein distance (no free region available).
    """
    m, n = len(pattern), len(text)
    if m == 0:
        return 0
    if n == 0:
        return m
    # dp[i][j] = min ops to align pattern[:i] to a substring of text[:j]
    # ending at position j. Free first row: empty pattern aligns anywhere.
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if pattern[i - 1] == text[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return min(prev[1:])


def sim(target, pred):
    """Fitting similarity in [0, 1]. Higher is better."""
    t = _normalize(target)
    p = _normalize(pred)
    fit = fitting_distance(t, p)
    return 1.0 - fit / max(len(t), 1)


def detectable_frames(src_ocr, s_src, tau=0.5):
    """Return frame indices where the source string is reliably visible."""
    return [i for i, s in enumerate(src_ocr) if sim(s_src, s) >= tau]


def adjacent_detectable_pairs(detectable):
    s = set(detectable)
    return [(t, t + 1) for t in detectable if (t + 1) in s]


def seq_acc(pred_ocr, s_tgt, detectable):
    if not detectable:
        return None
    t_norm = _normalize(s_tgt)
    hits = sum(
        1 for t in detectable
        if fitting_distance(t_norm, _normalize(pred_ocr[t])) == 0
    )
    return hits / len(detectable)


def char_acc(pred_ocr, s_tgt, detectable):
    if not detectable:
        return None
    return sum(sim(s_tgt, pred_ocr[t]) for t in detectable) / len(detectable)


def tts(pred_ocr, pairs):
    if not pairs:
        return None
    hits = sum(
        1 for t, tn in pairs
        if _normalize(pred_ocr[t]) == _normalize(pred_ocr[tn])
    )
    return hits / len(pairs)


def evaluate_clip(src_ocr, pred_ocr, s_src, s_tgt, tau=0.5):
    """Compute the three text-correctness metrics for one clip.

    Returns dict with SeqAcc, CharAcc, TTS plus bookkeeping fields. Metrics
    are None when their support set is empty (excluded from the test-split
    average per the paper).
    """
    n = min(len(src_ocr), len(pred_ocr))
    src_ocr = src_ocr[:n]
    pred_ocr = pred_ocr[:n]
    D = detectable_frames(src_ocr, s_src, tau)
    P = adjacent_detectable_pairs(D)
    return {
        "SeqAcc": seq_acc(pred_ocr, s_tgt, D),
        "CharAcc": char_acc(pred_ocr, s_tgt, D),
        "TTS": tts(pred_ocr, P),
        "num_detectable": len(D),
        "num_pairs": len(P),
    }
