"""Shared I/O and geometry utilities for ViTeX-Bench."""

import cv2
import numpy as np


def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_mask_frames(path, target_h=None, target_w=None):
    cap = cv2.VideoCapture(path)
    masks = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if target_h is not None and (gray.shape[0] != target_h or gray.shape[1] != target_w):
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        _, m = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
        masks.append(m.astype(np.uint8))
    cap.release()
    return masks


def align_resolution(frames, target_h, target_w):
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    if (h, w) == (target_h, target_w):
        return frames
    return [cv2.resize(f, (target_w, target_h)) for f in frames]


def truncate_to_common_length(*sequences):
    n = min(len(s) for s in sequences)
    return tuple(s[:n] for s in sequences)


def mask_bbox(mask, pad_ratio=0.1):
    """Axis-aligned bbox of the mask region with margin. Returns (x1,y1,x2,y2) or None."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    h, w = mask.shape[:2]
    pad_h = int((y2 - y1) * pad_ratio)
    pad_w = int((x2 - x1) * pad_ratio)
    y1 = max(0, y1 - pad_h)
    y2 = min(h, y2 + pad_h)
    x1 = max(0, x1 - pad_w)
    x2 = min(w, x2 + pad_w)
    return x1, y1, x2, y2


def union_mask_bbox(masks, pad_ratio=0.1):
    """Bbox covering the union of all per-frame masks, padded by pad_ratio."""
    h, w = masks[0].shape[:2]
    union = None
    for m in masks:
        ys, xs = np.where(m > 0)
        if ys.size == 0:
            continue
        b = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        union = b if union is None else (
            min(union[0], b[0]), min(union[1], b[1]),
            max(union[2], b[2]), max(union[3], b[3]),
        )
    if union is None:
        return None
    x1, y1, x2, y2 = union
    pad_h = int((y2 - y1) * pad_ratio)
    pad_w = int((x2 - x1) * pad_ratio)
    return (max(0, x1 - pad_w), max(0, y1 - pad_h),
            min(w, x2 + pad_w), min(h, y2 + pad_h))


def crop_with_bbox(frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]


def locality_prediction(pred_frames, src_frames, mask_frames):
    """Compose f_hat^loc = (1 - m) * f_hat + m * f, per the ViTeX-Bench locality axis.

    The masked text region is replaced with the source pixels so the metric
    measures preservation of the unedited scene only.
    """
    out = []
    for i, pred in enumerate(pred_frames):
        m = mask_frames[min(i, len(mask_frames) - 1)]
        m3 = m[..., None]  # (H, W, 1) broadcasts over channels
        composed = (1 - m3) * pred + m3 * src_frames[i]
        out.append(composed.astype(pred.dtype))
    return out
