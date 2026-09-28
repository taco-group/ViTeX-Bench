"""Render the motion-aligned glyph video that conditions ViTeX-Edit-14B.

  1. EasyOCR finds the source-text quadrilateral inside the mask in one of the
     first 10 frames (full frame first, then an enlarged crop of the mask).
  2. CoTracker3 tracks its 4 corners through the clip.
  3. The target string is rendered white-on-black at 2x the box size in the
     chosen typeface and perspective-warped onto the tracked corners.

If no text is detected, the clip falls back to rendering the string centered
in each frame's mask bounding box (method "mask_bbox" in the manifest).

Eval split (fonts.json from select_font.py; without it every clip uses the default font):
  python vitex_edit/render_glyph.py --records data/eval/parsed_records.json \
      --data_root data/eval --fonts data/eval/fonts.json --output_dir data/eval/glyph_videos

Single clip:
  python vitex_edit/render_glyph.py --video source.mp4 --mask mask.mp4 \
      --target_text "HILTON" --source_text "HOTEL" --font "Arial Bold" --output glyph.mp4
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fonts import DEFAULT_FONT, find_font, font_file_for_name, font_for_text

COTRACKER_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"


# ---------------------------------------------------------------------------
# Step 1: EasyOCR detection of the source-text quadrilateral
# ---------------------------------------------------------------------------

def init_ocr(device):
    import easyocr
    return easyocr.Reader(["en", "ch_sim"], gpu=device.startswith("cuda"), verbose=False)


def _best_box_in_mask(results, mask_bin, source_text):
    """Highest-scoring detection whose center lies inside the mask.
    Score = OCR confidence + 0.5 * character overlap with the source text."""
    best, best_score = None, -1
    for box, text, conf in results:
        pts = np.array(box, dtype=np.float32)
        ix, iy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        if not (0 <= iy < mask_bin.shape[0] and 0 <= ix < mask_bin.shape[1]) or mask_bin[iy, ix] == 0:
            continue
        score = conf
        if source_text:
            s1, s2 = text.lower().replace(" ", ""), source_text.lower().replace(" ", "")
            if s1 and s2:
                score += 0.5 * sum(1 for c in s1 if c in s2) / max(len(s1), len(s2))
        if score > best_score:
            best, best_score = pts, score
    return best


def detect_full_frame(reader, frame_rgb, mask_bin, source_text):
    results = reader.readtext(frame_rgb)
    return _best_box_in_mask(results, mask_bin, source_text) if results else None


def detect_in_crop(reader, frame_rgb, mask_bin, source_text, pad_ratio=0.3):
    """OCR on the mask region enlarged to >= 256 px (catches small text)."""
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return None
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = frame_rgb.shape[:2]
    pad_h, pad_w = int((y2 - y1) * pad_ratio), int((x2 - x1) * pad_ratio)
    y1p, y2p = max(0, y1 - pad_h), min(h, y2 + pad_h)
    x1p, x2p = max(0, x1 - pad_w), min(w, x2 + pad_w)
    crop, crop_mask = frame_rgb[y1p:y2p, x1p:x2p], mask_bin[y1p:y2p, x1p:x2p]
    scale = 1.0
    if min(crop.shape[:2]) < 256:
        scale = 256.0 / min(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop_mask = cv2.resize(crop_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    results = reader.readtext(crop)
    best = _best_box_in_mask(results, crop_mask, source_text) if results else None
    if best is None:
        return None
    best = best / scale
    best[:, 0] += x1p
    best[:, 1] += y1p
    return best


def detect_text_box(reader, frames_rgb, masks_bin, source_text, max_tries=10):
    """(box, frame_index) from the first frames: full-frame pass, then crop pass."""
    n = min(max_tries, len(frames_rgb))
    for detect in (detect_full_frame, detect_in_crop):
        for i in range(n):
            box = detect(reader, frames_rgb[i], masks_bin[i], source_text)
            if box is not None:
                return box, i
    return None, -1


# ---------------------------------------------------------------------------
# Step 2: CoTracker3 corner tracking
# ---------------------------------------------------------------------------

_cotracker = None


def get_cotracker(device):
    global _cotracker
    if _cotracker is None:
        from cotracker.predictor import CoTrackerPredictor
        ckpt = os.path.join(os.path.expanduser("~"), ".cache", "cotracker", "scaled_offline.pth")
        if not os.path.exists(ckpt):
            os.makedirs(os.path.dirname(ckpt), exist_ok=True)
            print("Downloading CoTracker3 checkpoint...")
            torch.save(torch.hub.load_state_dict_from_url(COTRACKER_URL, map_location="cpu"), ckpt)
        _cotracker = CoTrackerPredictor(checkpoint=ckpt).to(device)
    return _cotracker


def track_corners(frames_rgb, corners, start_frame, device, max_side=480):
    """Track 4 corners through the clip. Returns (T, 4, 2) points and (T, 4) visibility."""
    video = np.stack(frames_rgb)
    T, H, W, _ = video.shape
    scale = min(1.0, max_side / max(H, W))
    if scale != 1.0:
        video = np.stack([cv2.resize(f, (int(W * scale), int(H * scale))) for f in video])
    video = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
    queries = torch.zeros(1, 4, 3)
    queries[0, :, 0] = float(start_frame)
    queries[0, :, 1:] = torch.from_numpy(corners * scale)
    with torch.no_grad():
        tracks, visible = get_cotracker(device)(video, queries=queries.to(device))
    return tracks[0].cpu().numpy() / scale, visible[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Step 3: rendering
# ---------------------------------------------------------------------------

def _fit_font(draw, text, font_path, width, height):
    """Font at 80% of the height, shrunk to fit the width and then the height."""
    size = max(8, int(height * 0.8))
    font = ImageFont.truetype(font_path, size)
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] > width * 0.95 and bbox[2] - bbox[0] > 0:
        size = max(8, int(size * width * 0.9 / (bbox[2] - bbox[0])))
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[3] - bbox[1] > height * 0.95 and bbox[3] - bbox[1] > 0:
        size = max(8, int(size * height * 0.85 / (bbox[3] - bbox[1])))
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
    return font, bbox


def render_text_canvas(text, width, height, font_path):
    """White text centered on a black (height, width) canvas. Drawn on a padded
    canvas first so ascenders and descenders are not clipped."""
    pad = max(height // 2, 20)
    img = Image.new("L", (width + 2 * pad, height + 2 * pad), 0)
    draw = ImageDraw.Draw(img)
    font, bbox = _fit_font(draw, text, font_path, width, height)
    tx = (img.width - (bbox[2] - bbox[0])) // 2 - bbox[0]
    ty = (img.height - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((tx, ty), text, fill=255, font=font)
    return np.array(img)[pad:pad + height, pad:pad + width]


def render_warped(text, corners, frame_h, frame_w, font_path):
    """Render `text` at 2x the quadrilateral size and warp it onto `corners`
    (top-left, top-right, bottom-right, bottom-left)."""
    pts = np.asarray(corners, dtype=np.float32)
    w, h = int(np.linalg.norm(pts[1] - pts[0])), int(np.linalg.norm(pts[3] - pts[0]))
    if w < 5 or h < 5:
        return np.zeros((frame_h, frame_w), dtype=np.uint8)
    cw, ch = max(w * 2, 20), max(h * 2, 20)
    canvas = render_text_canvas(text, cw, ch, font_path)
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    return cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, pts), (frame_w, frame_h))


def render_in_bbox(text, mask_bin, font_path):
    """Fallback: text centered in the mask's axis-aligned bounding box."""
    img = Image.new("L", (mask_bin.shape[1], mask_bin.shape[0]), 0)
    coords = cv2.findNonZero(mask_bin)
    if coords is None:
        return np.array(img)
    x, y, w, h = cv2.boundingRect(coords)
    draw = ImageDraw.Draw(img)
    size = max(8, h - 4)
    font = ImageFont.truetype(font_path, size)
    bbox = draw.textbbox((0, 0), text, font=font)
    if bbox[2] - bbox[0] > w > 0:
        size = max(8, int(size * w / (bbox[2] - bbox[0]) * 0.95))
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((x + (w - (bbox[2] - bbox[0])) // 2, y + (h - (bbox[3] - bbox[1])) // 2), text, fill=255, font=font)
    return np.array(img)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return frames, fps


def read_masks(path, n):
    cap = cv2.VideoCapture(path)
    masks = []
    while len(masks) < n:
        ok, frame = cap.read()
        if not ok:
            break
        masks.append(cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1])
    cap.release()
    while masks and len(masks) < n:
        masks.append(masks[-1])
    return masks


def render_glyph_video(reader, video_path, mask_path, target_text, source_text, font_path, output_path,
                       device="cuda", fallback=True):
    """Write the glyph video. Returns "tracked", "mask_bbox", or None on failure."""
    frames, fps = read_video(video_path)
    if not frames:
        raise ValueError(f"cannot read {video_path}")
    masks = read_masks(mask_path, len(frames))
    frame_h, frame_w = frames[0].shape[:2]

    box, start = detect_text_box(reader, frames, masks, source_text)
    if box is not None:
        tracks, visible = track_corners(frames, box, start, device)
        method = "tracked"
    elif fallback:
        method = "mask_bbox"
    else:
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))
    last = None
    for t in range(len(frames)):
        if method == "tracked":
            if visible[t].sum() >= 2:  # enough corners visible: use them
                pts = tracks[t].copy()
                pts[:, 0] = np.clip(pts[:, 0], 0, frame_w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, frame_h - 1)
                last = pts
            else:  # otherwise keep the last good position
                pts = last
            glyph = (render_warped(target_text, pts, frame_h, frame_w, font_path) if pts is not None
                     else np.zeros((frame_h, frame_w), dtype=np.uint8))
        else:
            glyph = render_in_bbox(target_text, masks[t], font_path)
        writer.write(cv2.merge([glyph, glyph, glyph]))
    writer.release()
    return method


def resolve_font(target_text, font_arg):
    """Font path for `target_text` given a VLM name, a font file name, or None."""
    if font_arg is None:
        font_file = DEFAULT_FONT
    elif os.path.splitext(font_arg)[1].lower() in (".ttf", ".otf", ".ttc"):
        font_file = font_arg
    else:
        font_file = font_file_for_name(font_arg)
    return find_font(font_for_text(target_text, font_file))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--records", help="parsed_records.json")
    p.add_argument("--data_root", help="Root that record paths are relative to")
    p.add_argument("--fonts", help="fonts.json from select_font.py (default font for clips not in it)")
    p.add_argument("--output_dir", help="Where <id>.mp4 glyph videos are written")
    p.add_argument("--ids", nargs="*", help="Only render these clip ids")
    p.add_argument("--video", help="Single clip: source video")
    p.add_argument("--mask", help="Single clip: mask video")
    p.add_argument("--target_text", help="Single clip: string to render")
    p.add_argument("--source_text", default="", help="Single clip: current text (helps pick the OCR box)")
    p.add_argument("--font", help="Single clip: VLM font name (e.g. 'Arial Bold') or font file")
    p.add_argument("--output", default="glyph.mp4", help="Single clip: output path")
    p.add_argument("--no_fallback", action="store_true", help="Skip clips where OCR finds no text")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    reader = None
    if args.records is None:
        if not (args.video and args.mask and args.target_text):
            p.error("pass --records/--data_root/--output_dir, or --video/--mask/--target_text")
        reader = init_ocr(args.device)
        method = render_glyph_video(reader, args.video, args.mask, args.target_text, args.source_text,
                                    resolve_font(args.target_text, args.font), args.output,
                                    args.device, fallback=not args.no_fallback)
        print(f"{method or 'no text detected, nothing written'}: {args.output}")
        return
    if not (args.data_root and args.output_dir):
        p.error("--records needs --data_root and --output_dir")

    with open(args.records) as f:
        records = json.load(f)
    if args.ids:
        records = [r for r in records if r["id"] in set(args.ids)]
    fonts = {}
    if args.fonts:
        with open(args.fonts) as f:
            fonts = json.load(f)
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    os.makedirs(args.output_dir, exist_ok=True)

    for i, rec in enumerate(records, 1):
        out = os.path.join(args.output_dir, rec["id"] + ".mp4")
        if os.path.exists(out):
            continue
        reader = reader or init_ocr(args.device)
        font_path = resolve_font(rec["target_text"], fonts.get(rec["id"], {}).get("font_file"))
        print(f"  [{i}/{len(records)}] {rec['id']}: {rec['source_text']!r} -> {rec['target_text']!r} "
              f"({os.path.basename(font_path)})")
        try:
            method = render_glyph_video(
                reader, os.path.join(args.data_root, rec["original_video"]),
                os.path.join(args.data_root, rec["mask_video"]),
                rec["target_text"], rec["source_text"], font_path, out,
                args.device, fallback=not args.no_fallback)
        except Exception as e:  # keep going; one bad clip should not stop the split
            print(f"    ERROR: {e}")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            method = None
        if method != "tracked":
            print(f"    {method or 'skipped'}")
        manifest[rec["id"]] = {"method": method, "font": os.path.basename(font_path)}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    methods = [v["method"] for v in manifest.values()]
    print(f"done: {methods.count('tracked')} tracked, {methods.count('mask_bbox')} mask_bbox fallback, "
          f"{methods.count(None)} failed -> {args.output_dir}")


if __name__ == "__main__":
    main()
