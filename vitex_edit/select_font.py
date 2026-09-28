"""Pick a typeface for each clip's glyph video with Qwen3-VL.

Qwen3-VL (served by Ollama, model `qwen3-vl:8b-instruct`) looks at the source
text region in the first frame and names the closest typeface; `fonts.py`
maps the name to a free font file. Non-Latin target strings later override
this choice with a script-specific font (see render_glyph.py).

  ollama pull qwen3-vl:8b-instruct     # once; `ollama serve` must be running
  python vitex_edit/select_font.py --records data/eval/parsed_records.json \
      --data_root data/eval --output data/eval/fonts.json

Single clip:
  python vitex_edit/select_font.py --video source.mp4 --mask mask.mp4 --source_text "OPEN"

Without Ollama, `--no_vlm` assigns the default font (Liberation Sans Bold) to every clip.
"""

import argparse
import base64
import json
import os
import sys

import cv2
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fonts import DEFAULT_FONT, font_file_for_name

PROMPT = (
    'This image shows the text "{source_text}" from a video frame. '
    'Identify the font style. Consider these categories:\n'
    '- Sans-serif regular (clean, no serifs): Arial, Helvetica, Verdana, Calibri\n'
    '- Sans-serif bold (thick, heavy): Arial Bold, Helvetica Bold, Impact, Franklin Gothic\n'
    '- Sans-serif condensed (tall and narrow): Arial Narrow, Condensed Bold\n'
    '- Serif regular (with serifs): Times New Roman, Georgia, Palatino, Bookman\n'
    '- Serif bold: Times New Roman Bold, Georgia Bold\n'
    '- Monospace (fixed-width): Courier New, Courier New Bold\n'
    '- Script/Handwritten (cursive, flowing): Script, Brush Script, Cursive, Handwritten\n'
    '- Gothic/Geometric (round, modern): Century Gothic, Futura, Avant Garde\n'
    '- Italic (slanted): Italic, Sans-serif Italic\n'
    'Reply with ONLY the font name from the examples above. No explanation.'
)
FALLBACK_NAME = "Arial Bold"


def text_crop_png(video_path, mask_path, pad=20):
    """First-frame crop around the mask's bounding box, as PNG bytes."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    cap = cv2.VideoCapture(mask_path)
    ok_m, mask = cap.read()
    cap.release()
    if not (ok and ok_m):
        return None
    _, binary = cv2.threshold(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    crop = frame[max(0, y - pad):min(frame.shape[0], y + h + pad),
                 max(0, x - pad):min(frame.shape[1], x + w + pad)]
    return cv2.imencode(".png", crop)[1].tobytes()


def query_vlm(png, source_text, model, url):
    """Ask the VLM for a font name. Returns None on failure."""
    session = requests.Session()
    session.trust_env = False  # never route the local Ollama server through a proxy
    try:
        resp = session.post(
            f"{url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": PROMPT.format(source_text=source_text),
                              "images": [base64.b64encode(png).decode()]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 30},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json().get("message", {}).get("content", "").strip()
    except Exception as e:
        print(f"    VLM error: {e}")
        return None
    if "<think>" in content:  # qwen3 may wrap its answer in a think block
        content = content.split("</think>")[-1].strip()
    return content.split("\n")[0].strip() or None


def choose(video, mask, source_text, args):
    if args.no_vlm:
        return None, DEFAULT_FONT
    png = text_crop_png(video, mask)
    name = query_vlm(png, source_text, args.model, args.ollama_url) if png else None
    if name is None:
        print(f"    falling back to {FALLBACK_NAME!r}")
        name = FALLBACK_NAME
    return name, font_file_for_name(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", help="parsed_records.json")
    p.add_argument("--data_root", help="Root that record paths are relative to")
    p.add_argument("--output", help="JSON written as {id: {vlm_font_name, font_file}}; existing ids are kept")
    p.add_argument("--video", help="Single clip: source video")
    p.add_argument("--mask", help="Single clip: mask video")
    p.add_argument("--source_text", default="", help="Single clip: the text currently in the video")
    p.add_argument("--model", default="qwen3-vl:8b-instruct")
    p.add_argument("--ollama_url", default="http://localhost:11434")
    p.add_argument("--no_vlm", action="store_true", help="Use the default font for every clip")
    args = p.parse_args()

    if args.records is None:
        if not (args.video and args.mask):
            p.error("pass --records/--data_root/--output, or --video/--mask")
        name, font_file = choose(args.video, args.mask, args.source_text, args)
        print(json.dumps({"vlm_font_name": name, "font_file": font_file}))
        return
    if not (args.data_root and args.output):
        p.error("--records needs --data_root and --output")

    with open(args.records) as f:
        records = json.load(f)
    fonts = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            fonts = json.load(f)
    todo = [r for r in records if r["id"] not in fonts]
    print(f"{len(records)} clips, {len(records) - len(todo)} already done")
    for i, rec in enumerate(todo, 1):
        name, font_file = choose(os.path.join(args.data_root, rec["original_video"]),
                                 os.path.join(args.data_root, rec["mask_video"]),
                                 rec["source_text"], args)
        fonts[rec["id"]] = {"vlm_font_name": name, "font_file": font_file}
        print(f"  [{i}/{len(todo)}] {rec['id']}: {name!r} -> {font_file}")
        with open(args.output, "w") as f:  # rewrite each time so an interrupted run resumes
            json.dump(fonts, f, indent=2, ensure_ascii=False)

    counts = {}
    for v in fonts.values():
        counts[v["font_file"]] = counts.get(v["font_file"], 0) + 1
    print("font files:", json.dumps(dict(sorted(counts.items(), key=lambda kv: -kv[1])), indent=2))


if __name__ == "__main__":
    main()
