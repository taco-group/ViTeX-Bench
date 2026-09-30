"""Render the motion-aligned glyph video that conditions ViTeX-Edit-14B.

  1. EasyOCR finds the source text inside the mask in one of the first 10
     frames (on the full frame and on an enlarged crop of the mask). A detected
     line that contains the source text is cut to the matching characters.
     Only detections centred in the mask and lying >= 80% inside it are used
     (the masks are loose blobs around the text, so a neighbouring word or a
     whole line would otherwise be picked up). The ones that read like the
     source text (else the closest or the largest one) and the other
     detections on their text line (words the OCR misread) are merged into
     one quadrilateral.
  2. CoTracker3 tracks its 4 corners through the clip.
  3. Each frame's tracked quadrilateral is checked against that frame's mask:
     convex, centred in it, >= 70% inside it, and within 0.5-2x the size
     relative to the mask that it had when detected. Where the check fails
     (tracker drift, zoom, occlusion), the quadrilateral is placed from the
     mask instead, keeping the nearest valid frame's position relative to the
     mask's rotated rectangle. If fewer than half of the masked frames pass, every
     frame is placed that way from the detection; if no text is detected,
     the text fills the mask's rotated rectangle (method "mask_rect").
  4. The target string is rendered white-on-black at 2x the quadrilateral
     size in the chosen typeface and perspective-warped onto it. Frames whose
     mask is empty stay black.

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
from difflib import SequenceMatcher

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fonts import DEFAULT_FONT, find_font, font_file_for_name, font_for_text

COTRACKER_URL = "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth"


# ---------------------------------------------------------------------------
# Quadrilaterals and the mask
# ---------------------------------------------------------------------------

def _region(mask_bin):
    """The mask, dilated by 3 px against boundary effects."""
    return cv2.dilate((mask_bin > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


def _frac_inside(quad, region):
    """Share of the polygon's pixels that lie inside the boolean `region`."""
    poly = np.zeros(region.shape, np.uint8)
    cv2.fillPoly(poly, [np.round(quad).astype(np.int32)], 1)
    n = int(poly.sum())
    return float((poly.astype(bool) & region).sum()) / n if n else 0.0


def _centred_in(quad, region):
    cx, cy = np.asarray(quad, np.float32).mean(0)
    x, y = int(round(float(cx))), int(round(float(cy)))
    return 0 <= y < region.shape[0] and 0 <= x < region.shape[1] and bool(region[y, x])


def order_quad(pts):
    """Corners as top-left, top-right, bottom-right, bottom-left, reading along
    the side closest to horizontal."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    edges = [pts[(i + 1) % 4] - pts[i] for i in range(4)]

    def tilt(e):
        a = abs(np.arctan2(e[1], e[0]))
        return min(a, np.pi - a)

    d = min(edges, key=tilt)
    d = d / (np.linalg.norm(d) + 1e-9)
    if d[0] < 0:
        d = -d
    n = np.array([-d[1], d[0]], np.float32)  # points down the image when d points right
    rel = pts - pts.mean(0)
    u, v = rel @ d, rel @ n
    idx = [int(np.argmin(u + v)), int(np.argmax(u - v)), int(np.argmax(u + v)), int(np.argmin(u - v))]
    return pts[idx] if len(set(idx)) == 4 else pts


def mask_rect(mask_bin):
    """The mask's minimum-area rotated rectangle as ordered corners, or None. A
    roundish mask (sides within 1.3x) has no clear text direction; its upright
    bounding box is used instead."""
    pts = cv2.findNonZero((mask_bin > 0).astype(np.uint8))
    if pts is None or len(pts) < 16:
        return None
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    if max(w, h) < 1.3 * min(w, h):
        x, y, bw, bh = cv2.boundingRect(pts)
        return np.array([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], np.float32)
    return order_quad(cv2.boxPoints(((cx, cy), (w, h), angle)))


def _convex(q):
    cross = []
    for i in range(4):
        a, b = q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]
        cross.append(a[0] * b[1] - a[1] * b[0])
    return all(c > 0 for c in cross) or all(c < 0 for c in cross)


def size_ratio(quad, rect):
    """Area of `quad` relative to the mask's rotated rectangle."""
    return cv2.contourArea(np.ascontiguousarray(quad, np.float32)) / max(cv2.contourArea(rect), 1.0)


def quad_ok(quad, mask_bin, rect, ref_ratio=None):
    """Is `quad` a plausible place for the text in this frame? Convex, centred in the
    mask and inside it: >= 80% for a detection, >= 70% for a tracked frame, whose
    size relative to the mask must also stay within 0.5-2x `ref_ratio` (the
    detection's)."""
    q = np.ascontiguousarray(quad, np.float32)
    if rect is None or not _convex(q) or cv2.contourArea(q) < 16:
        return False
    region = _region(mask_bin)
    if not _centred_in(q, region):
        return False
    ratio = size_ratio(q, rect)
    if ref_ratio is None:
        return _frac_inside(q, region) >= 0.8 and 0.02 <= ratio <= 2.5
    return _frac_inside(q, region) >= 0.7 and 0.5 <= ratio / ref_ratio <= 2.0


def to_rect_coords(quad, rect):
    """Corners of `quad` in the frame of the ordered rectangle `rect` ((0,0) = its
    top-left, (1,1) = its bottom-right)."""
    origin, ex, ey = rect[0], rect[1] - rect[0], rect[3] - rect[0]
    return np.linalg.solve(np.stack([ex, ey], 1), (np.asarray(quad, np.float32) - origin).T).T


def from_rect_coords(uv, rect):
    origin, ex, ey = rect[0], rect[1] - rect[0], rect[3] - rect[0]
    return (origin + uv[:, :1] * ex + uv[:, 1:] * ey).astype(np.float32)


# Where text placed from the mask alone goes inside the mask's rotated rectangle:
# the median margins of the detected source text on the 157-clip eval split
# (the masks are loose blobs around the text): 10% left/right, 20% top/bottom.
MASK_UV = np.array([[0.1, 0.2], [0.9, 0.2], [0.9, 0.8], [0.1, 0.8]], np.float32)


# ---------------------------------------------------------------------------
# Step 1: EasyOCR detection of the source-text quadrilateral
# ---------------------------------------------------------------------------

def init_ocr(device):
    import easyocr
    return easyocr.Reader(["en", "ch_sim"], gpu=device.startswith("cuda"), verbose=False)


def _ocr_boxes(reader, frame_rgb, mask_bin, crop, pad_ratio=0.3):
    """EasyOCR (box 4x2 in frame coordinates, text) on the full frame, or on the
    mask region enlarged to >= 256 px (catches small text)."""
    if not crop:
        return [(np.array(box, np.float32), text) for box, text, _conf in reader.readtext(frame_rgb)]
    ys, xs = np.where(mask_bin > 0)
    if len(ys) == 0:
        return []
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = frame_rgb.shape[:2]
    pad_h, pad_w = int((y2 - y1) * pad_ratio), int((x2 - x1) * pad_ratio)
    y1p, y2p = max(0, y1 - pad_h), min(h, y2 + pad_h)
    x1p, x2p = max(0, x1 - pad_w), min(w, x2 + pad_w)
    crop_img = frame_rgb[y1p:y2p, x1p:x2p]
    scale = 1.0
    if min(crop_img.shape[:2]) < 256:
        scale = 256.0 / min(crop_img.shape[:2])
        crop_img = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return [(np.array(box, np.float32) / scale + np.array([x1p, y1p], np.float32), text)
            for box, text, _conf in reader.readtext(crop_img)]


def _match(ocr, source):
    """How well an OCR string reads like the source text (1 = same), or like one
    of its words for a multi-word source."""
    a, b = "".join(ocr.lower().split()), "".join(source.lower().split())
    if not a or not b:
        return 0.0
    if a in source.lower().split():  # one whole word of the source, however short
        return 1.0
    best = SequenceMatcher(None, a, b).ratio()
    if max(3, len(b) // 2) <= len(a) < len(b):
        for i in range(len(b) - len(a) + 1):
            best = max(best, SequenceMatcher(None, a, b[i:i + len(a)]).ratio())
    return best


def _cut_to_source(box, ocr, source):
    """If the detected text (e.g. a whole line) contains the source text, the part
    of `box` covering those characters (widths taken as proportional to counts)."""
    a, b = ocr.lower(), source.lower().strip()
    if not b or len(a) <= len(b) + 1:
        return box, ocr
    best, start, length = 0.0, 0, len(b)
    for n in range(max(1, len(b) - 1), len(b) + 2):
        for i in range(len(a) - n + 1):
            r = SequenceMatcher(None, a[i:i + n], b).ratio()
            if r > best:
                best, start, length = r, i, n
    if best < 0.75:
        return box, ocr
    tl, tr, br, bl = box
    f0, f1 = start / len(a), (start + length) / len(a)
    cut = np.array([tl + f0 * (tr - tl), tl + f1 * (tr - tl), bl + f1 * (br - bl), bl + f0 * (br - bl)], np.float32)
    return cut, ocr[start:start + length]


def _merge(boxes):
    if len(boxes) == 1:
        return order_quad(boxes[0])
    return order_quad(cv2.boxPoints(cv2.minAreaRect(np.concatenate(boxes))))


def _same_line(box, line):
    """Is `box` on the text line `line` (centre within the line's height, similar
    height)?"""
    tl, tr, _br, bl = line
    d = (tr - tl) / (np.linalg.norm(tr - tl) + 1e-9)
    n = np.array([-d[1], d[0]], np.float32)
    height = float((bl - tl) @ n)
    b = order_quad(box)
    v = float((b.mean(0) - tl) @ n)
    h = float((b[3] - b[0]) @ n)
    return height > 0 and 0 <= v <= height and 0.5 <= h / height <= 2.0


def text_quad(detections, mask_bin, source_text=""):
    """One quadrilateral over the OCR detections centred in the mask and >= 80%
    inside it: those that read like the source text (else the closest one if it
    is somewhat alike, else the largest) plus the other detections on their line
    (words of a multi-word source are merged, one text line only), or None."""
    region = _region(mask_bin)
    cands = []
    for box, text in detections:
        box, text = _cut_to_source(order_quad(box), text, source_text)
        if _centred_in(box, region) and _frac_inside(box, region) >= 0.8:
            cands.append((box, text))
    if not cands:
        return None
    score = {id(c): _match(c[1], source_text) if source_text else 0.0 for c in cands}
    anchor = [c for c in cands if score[id(c)] >= 0.6]
    if not anchor:
        best = max(cands, key=lambda c: score[id(c)])
        if score[id(best)] < 0.4:  # nothing reads like the source text: take the largest
            best = max(cands, key=lambda c: cv2.contourArea(np.ascontiguousarray(c[0])))
        anchor = [best]
    line = _merge([box for box, _text in anchor])
    same_line = [c for c in cands if not any(c is a for a in anchor) and _same_line(c[0], line)]
    return _merge([box for box, _text in anchor + same_line])


def detect_text_quad(reader, frames_rgb, masks_bin, source_text="", max_tries=10):
    """(quad, frame_index) of the source text in the first of the first frames where
    OCR on the full frame and on an enlarged crop of the mask finds it. A detection
    counts only if it is plausible for that frame's mask (see quad_ok)."""
    for i in range(min(max_tries, len(frames_rgb))):
        rect = mask_rect(masks_bin[i])
        if rect is None:
            continue
        detections = (_ocr_boxes(reader, frames_rgb[i], masks_bin[i], crop=False)
                      + _ocr_boxes(reader, frames_rgb[i], masks_bin[i], crop=True))
        quad = text_quad(detections, masks_bin[i], source_text)
        if quad is not None and quad_ok(quad, masks_bin[i], rect):
            return quad, i
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
    return np.ascontiguousarray(tracks[0].cpu().numpy() / scale, dtype=np.float32), visible[0].cpu().numpy()


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
    pts = np.ascontiguousarray(corners, dtype=np.float32)
    w, h = int(np.linalg.norm(pts[1] - pts[0])), int(np.linalg.norm(pts[3] - pts[0]))
    if w < 5 or h < 5:
        return np.zeros((frame_h, frame_w), dtype=np.uint8)
    cw, ch = max(w * 2, 20), max(h * 2, 20)
    canvas = render_text_canvas(text, cw, ch, font_path)
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
    return cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, pts), (frame_w, frame_h))


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
    """Write the glyph video. Returns (method, frame counts): method "tracked" or
    "mask_rect", or (None, {}) if no text is detected and `fallback` is off.
    `source_text` helps pick the right detection inside the mask."""
    frames, fps = read_video(video_path)
    if not frames:
        raise ValueError(f"cannot read {video_path}")
    masks = read_masks(mask_path, len(frames))
    n = len(frames)
    frame_h, frame_w = frames[0].shape[:2]
    rects = [mask_rect(m) for m in masks]

    tracks, uv, template = None, [None] * n, MASK_UV
    quad, start = detect_text_quad(reader, frames, masks, source_text)
    if quad is not None:
        template = to_rect_coords(quad, rects[start])  # the text's place within the mask
        ref_ratio = size_ratio(quad, rects[start])
        tracks, _visible = track_corners(frames, quad, start, device)
        for t in range(n):
            if quad_ok(tracks[t], masks[t], rects[t], ref_ratio):
                uv[t] = to_rect_coords(tracks[t], rects[t])
        masked = sum(r is not None for r in rects)
        if sum(u is not None for u in uv) < 0.5 * masked:  # tracking unreliable for this clip
            uv = [None] * n
    elif not fallback:
        return None, {}
    good = [t for t in range(n) if uv[t] is not None]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))
    counts = {"tracked": 0, "mask_rect": 0, "empty": 0}
    for t in range(n):
        if rects[t] is None:  # no text to replace in this frame
            glyph, kind = np.zeros((frame_h, frame_w), dtype=np.uint8), "empty"
        elif uv[t] is not None:
            glyph, kind = render_warped(target_text, tracks[t], frame_h, frame_w, font_path), "tracked"
        else:  # nearest valid frame's placement relative to the mask, else the detection's
            ref = min(good, key=lambda g: (abs(g - t), g > t)) if good else None
            q = from_rect_coords(uv[ref] if ref is not None else template, rects[t])
            glyph, kind = render_warped(target_text, q, frame_h, frame_w, font_path), "mask_rect"
        counts[kind] += 1
        writer.write(cv2.merge([glyph, glyph, glyph]))
    writer.release()
    return ("tracked" if good else "mask_rect"), counts


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
    p.add_argument("--no_fallback", action="store_true", help="Skip clips where OCR finds no text in the mask")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    reader = None
    if args.records is None:
        if not (args.video and args.mask and args.target_text):
            p.error("pass --records/--data_root/--output_dir, or --video/--mask/--target_text")
        reader = init_ocr(args.device)
        method, counts = render_glyph_video(reader, args.video, args.mask, args.target_text, args.source_text,
                                            resolve_font(args.target_text, args.font), args.output,
                                            args.device, fallback=not args.no_fallback)
        print(f"{method or 'no text detected, nothing written'} {counts}: {args.output}")
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
            method, counts = render_glyph_video(
                reader, os.path.join(args.data_root, rec["original_video"]),
                os.path.join(args.data_root, rec["mask_video"]),
                rec["target_text"], rec["source_text"], font_path, out,
                args.device, fallback=not args.no_fallback)
        except Exception as e:  # keep going; one bad clip should not stop the split
            print(f"    ERROR: {e}")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            method, counts = None, {}
        if method != "tracked" or counts.get("mask_rect"):
            print(f"    {method or 'skipped'} {counts}")
        manifest[rec["id"]] = {"method": method, "font": os.path.basename(font_path), "frames": counts}
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    methods = [v["method"] for v in manifest.values()]
    patched = sum(1 for v in manifest.values() if v["method"] == "tracked" and v.get("frames", {}).get("mask_rect"))
    print(f"done: {methods.count('tracked')} tracked ({patched} with some frames placed from the mask), "
          f"{methods.count('mask_rect')} placed from the mask, {methods.count(None)} failed -> {args.output_dir}")


if __name__ == "__main__":
    main()
