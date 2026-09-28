"""Font library used to render ViTeX-Edit-14B glyph videos.

Fonts come from standard Debian/Ubuntu packages; on other systems copy the
same files into a directory and point VITEX_FONT_DIR at it:

  sudo apt install fonts-liberation fonts-liberation-sans-narrow \
      fonts-urw-base35 fonts-noto-core fonts-noto-cjk fonts-freefont-ttf fonts-dejavu-core

The table maps the typeface names Qwen3-VL answers with (select_font.py) to
metric-compatible free fonts. Non-Latin target strings always use a
script-specific font instead (see `font_for_text`).
"""

import glob
import os

# VLM font name -> font file name.
# Liberation fonts are metric-compatible replacements: Arial -> Liberation Sans,
# Times New Roman -> Liberation Serif, Courier New -> Liberation Mono.
FONT_MAP = {
    # Sans-serif
    "arial": "LiberationSans-Regular.ttf",
    "arial bold": "LiberationSans-Bold.ttf",
    "helvetica": "NimbusSans-Regular.otf",
    "helvetica bold": "NimbusSans-Bold.otf",
    "verdana": "NotoSans-Regular.ttf",
    "calibri": "LiberationSans-Regular.ttf",
    "trebuchet ms": "NotoSans-Regular.ttf",
    "sans-serif": "LiberationSans-Regular.ttf",
    "sans-serif bold": "LiberationSans-Bold.ttf",
    # Condensed / narrow sans
    "arial narrow": "LiberationSansNarrow-Regular.ttf",
    "condensed": "NimbusSansNarrow-Regular.otf",
    "condensed bold": "NimbusSansNarrow-Bold.otf",
    # Serif
    "times new roman": "NimbusRoman-Regular.otf",
    "times new roman bold": "NimbusRoman-Bold.otf",
    "times": "NimbusRoman-Regular.otf",
    "georgia": "P052-Roman.otf",
    "palatino": "P052-Roman.otf",
    "book antiqua": "P052-Roman.otf",
    "bookman": "URWBookman-Light.otf",
    "bookman bold": "URWBookman-Demi.otf",
    "century schoolbook": "C059-Roman.otf",
    "serif": "LiberationSerif-Regular.ttf",
    "serif bold": "LiberationSerif-Bold.ttf",
    # Monospace
    "courier": "NimbusMonoPS-Regular.otf",
    "courier new": "NimbusMonoPS-Regular.otf",
    "courier new bold": "NimbusMonoPS-Bold.otf",
    "monospace": "LiberationMono-Regular.ttf",
    # Display / heavy
    "impact": "LiberationSansNarrow-Bold.ttf",
    "impact bold": "LiberationSansNarrow-Bold.ttf",
    "franklin gothic": "NimbusSans-Bold.otf",
    # Script / handwritten
    "script": "Z003-MediumItalic.otf",
    "script mt bold": "Z003-MediumItalic.otf",
    "cursive": "Z003-MediumItalic.otf",
    "handwritten": "Z003-MediumItalic.otf",
    "calligraphy": "Z003-MediumItalic.otf",
    "zapf chancery": "Z003-MediumItalic.otf",
    "comic sans ms": "Z003-MediumItalic.otf",
    "comic sans": "Z003-MediumItalic.otf",
    "brush script": "Z003-MediumItalic.otf",
    # Gothic / geometric
    "avant garde": "URWGothic-Book.otf",
    "avant garde bold": "URWGothic-Demi.otf",
    "century gothic": "URWGothic-Book.otf",
    "futura": "URWGothic-Book.otf",
    # Italic
    "italic": "LiberationSans-Italic.ttf",
    "sans-serif italic": "LiberationSans-Italic.ttf",
    "serif italic": "LiberationSerif-Italic.ttf",
    # Symbols
    "symbol": "StandardSymbolsPS.otf",
    "dingbats": "D050000L.otf",
    # CJK
    "simhei": "NotoSansCJK-Bold.ttc",
    "microsoft yahei": "NotoSansCJK-Bold.ttc",
    "malgun gothic": "NotoSansCJK-Bold.ttc",
    "noto sans cjk": "NotoSansCJK-Bold.ttc",
    "songti": "NotoSerifCJK-Regular.ttc",
    "simsun": "NotoSerifCJK-Regular.ttc",
}

DEFAULT_FONT = "LiberationSans-Bold.ttf"

# Target strings in these scripts ignore the VLM choice and use a font that covers them.
SCRIPT_FONTS = {
    "cjk": "NotoSansCJK-Bold.ttc",
    "cyrillic": "FreeSansBold.ttf",
    "symbol": "DejaVuSans.ttf",
}

FONT_DIRS = [
    os.environ.get("VITEX_FONT_DIR", ""),
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.local/share/fonts"),
    os.path.expanduser("~/.fonts"),
    "/Library/Fonts",
    os.path.expanduser("~/Library/Fonts"),
]


def detect_script(text):
    """Dominant script of `text`: cjk, cyrillic, symbol or latin."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:  # CJK ideographs
            return "cjk"
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:  # Hangul
            return "cjk"
        if 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:  # Hiragana / Katakana
            return "cjk"
        if 0x0400 <= cp <= 0x04FF:  # Cyrillic
            return "cyrillic"
        if 0x2200 <= cp <= 0x22FF or 0x2190 <= cp <= 0x21FF:  # math operators, arrows
            return "symbol"
        if 0x2160 <= cp <= 0x217F:  # Roman numerals
            return "symbol"
        if 0x2460 <= cp <= 0x24FF:  # enclosed alphanumerics
            return "symbol"
        if 0x2700 <= cp <= 0x27BF:  # dingbats
            return "symbol"
        if 0x2600 <= cp <= 0x26FF:  # misc symbols
            return "symbol"
        if 0x2070 <= cp <= 0x209F:  # super- / subscripts
            return "symbol"
    return "latin"


def font_file_for_name(font_name):
    """Map a VLM-returned font name to a font file name from FONT_MAP."""
    key = font_name.strip().lower()
    if key in FONT_MAP:
        return FONT_MAP[key]
    for k, v in FONT_MAP.items():
        if k in key or key in k:
            return v
    if any(w in key for w in ("script", "handwrit", "cursive", "brush", "calligraph")):
        return "Z003-MediumItalic.otf"
    if "gothic" in key or "futura" in key or "avant" in key:
        return "URWGothic-Book.otf"
    if "condensed" in key or "narrow" in key:
        return "LiberationSansNarrow-Bold.ttf"
    if "italic" in key:
        return "LiberationSans-Italic.ttf"
    if "bold" in key and "serif" in key and "sans" not in key:
        return "NimbusRoman-Bold.otf"
    if "bold" in key:
        return "LiberationSans-Bold.ttf"
    if "serif" in key and "sans" not in key:
        return "NimbusRoman-Regular.otf"
    if "mono" in key:
        return "NimbusMonoPS-Regular.otf"
    return DEFAULT_FONT


def font_for_text(text, font_file):
    """Script-specific font for non-Latin `text`, otherwise `font_file`."""
    return SCRIPT_FONTS.get(detect_script(text), font_file)


_found = {}


def find_font(font_file):
    """Absolute path of `font_file` (a file name, or a path that already exists)."""
    if os.path.isfile(font_file):
        return font_file
    if font_file not in _found:
        _found[font_file] = None
        for d in FONT_DIRS:
            if d and os.path.isdir(d):
                hits = glob.glob(os.path.join(d, "**", font_file), recursive=True)
                if hits:
                    _found[font_file] = sorted(hits)[0]
                    break
    if _found[font_file] is None:
        raise FileNotFoundError(
            f"font {font_file!r} not found in {[d for d in FONT_DIRS if d]}; install the packages "
            "listed in vitex_edit/fonts.py or set VITEX_FONT_DIR"
        )
    return _found[font_file]
