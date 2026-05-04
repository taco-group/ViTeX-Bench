"""Per-record language detection by Unicode block.

Maps source/target text strings to one of four ViTeX-Bench OCR families:
en (Latin / digits / symbols), zh (CJK Unified), ja (Hiragana / Katakana),
ru (Cyrillic). PP-OCRv5 has a dedicated weight set for each. Korean is
absent from the current 158-clip test split; if it appears later, add a
'ko' branch and the matching PP-OCRv5 lang code.
"""

from collections import Counter


_RANGES = [
    ("zh", (0x4E00, 0x9FFF)),    # CJK Unified Ideographs
    ("zh", (0x3400, 0x4DBF)),    # CJK Extension A
    ("ja", (0x3040, 0x309F)),    # Hiragana
    ("ja", (0x30A0, 0x30FF)),    # Katakana
    ("ko", (0xAC00, 0xD7AF)),    # Hangul Syllables
    ("ko", (0x1100, 0x11FF)),    # Hangul Jamo
    ("ru", (0x0400, 0x04FF)),    # Cyrillic
    ("en", (0x0041, 0x007A)),    # ASCII letters (a-zA-Z, plus a few punct)
    ("en", (0x00C0, 0x024F)),    # Latin Extended-A/B
]


def detect_lang(text):
    """Return one of {'en', 'zh', 'ja', 'ko', 'ru'}.

    Pure-digit / pure-symbol strings ('35,000', '$2.33 x 430', '-20°') are
    classified as 'en' since the English PP-OCRv5 model recognizes Latin
    digits and ASCII punctuation correctly.
    """
    counts = Counter()
    for ch in text:
        cp = ord(ch)
        for lang, (lo, hi) in _RANGES:
            if lo <= cp <= hi:
                counts[lang] += 1
                break
    if not counts:
        return "en"
    return counts.most_common(1)[0][0]


_PADDLE_LANG = {
    "en": "en",
    "zh": "ch",          # PP-OCRv5 'ch' = Chinese + English bilingual
    "ja": "japan",
    "ko": "korean",
    "ru": "ru",          # PP-OCRv5 routes 'ru' to the Cyrillic recognizer
}


def paddle_lang_code(lang):
    return _PADDLE_LANG.get(lang, "en")


def annotate_records_with_lang(records):
    """Return a copy of records with a 'lang' field. The longer of source /
    target text drives detection (more characters => more reliable signal).
    """
    out = []
    for r in records:
        src = r.get("source_text", "") or ""
        tgt = r.get("target_text", "") or ""
        text = src if len(src) >= len(tgt) else tgt
        lang = detect_lang(text)
        out.append({**r, "lang": lang})
    return out
