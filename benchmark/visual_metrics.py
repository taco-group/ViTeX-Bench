"""Axis 2: visual-quality metrics.

Six metrics, two scopes (full-frame `f` and text-crop `c`):
    Flicker_S = mean adjacent-frame MAE         (lower is better)
    Warp_S    = mean RAFT-warping error in MAE  (lower is better)
    MUSIQ_S   = mean per-frame MUSIQ-KonIQ      (higher is better)

Why Warp on top of Flicker. Flicker (raw adjacent-frame MAE) can be hacked
by over-smoothing: a method that emits frames with no high-frequency
structure between them gets a low Flicker even though it does not actually
follow the source motion. Warp uses RAFT optical flow on the source video
to predict pred_t from pred_{t+1}, then compares to the actual pred_t.
A method must follow the same per-pixel motion as the source to score low,
which is exactly what "temporally consistent with the underlying motion"
means in the protocol description.

Crop scope shares one union-mask bbox across the clip so adjacent crops
are spatially aligned (Flicker_crop and Warp_crop are otherwise dominated
by bbox jitter).
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from bench_utils import union_mask_bbox

_MUSIQ = None
_RAFT = None


def _musiq(device):
    global _MUSIQ
    if _MUSIQ is None:
        import pyiqa
        _MUSIQ = pyiqa.create_metric("musiq", device=device)
    return _MUSIQ


def _raft(device):
    """RAFT-large on `device`. Loaded lazily and cached."""
    global _RAFT
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).to(device)
        model.eval()
        _RAFT = model
    return _RAFT


def _flicker(frames):
    """Mean absolute RGB difference between adjacent frames."""
    if len(frames) < 2:
        return None
    arr = np.stack(frames).astype(np.int16)
    diff = np.abs(arr[1:] - arr[:-1]).mean(axis=(1, 2, 3))
    return float(diff.mean())


def _backward_warp(image, flow):
    """Backward-warp `image` (B, C, H, W) using the forward flow (B, 2, H, W).

    Output pixel (x, y) is sampled from `image` at (x + flow_x, y + flow_y).
    Returns the warped image plus a binary mask (B, H, W) marking pixels
    whose source location lies inside the input frame.
    """
    B, C, H, W = image.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=torch.float32),
        torch.arange(W, device=image.device, dtype=torch.float32),
        indexing="ij",
    )
    grid_x = xx.unsqueeze(0).expand(B, -1, -1) + flow[:, 0]
    grid_y = yy.unsqueeze(0).expand(B, -1, -1) + flow[:, 1]
    valid = (
        (grid_x >= 0) & (grid_x <= W - 1)
        & (grid_y >= 0) & (grid_y <= H - 1)
    )
    grid_x_norm = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y_norm = 2.0 * grid_y / max(H - 1, 1) - 1.0
    grid = torch.stack([grid_x_norm, grid_y_norm], dim=-1)
    warped = F.grid_sample(image, grid, mode="bilinear",
                           padding_mode="zeros", align_corners=True)
    return warped, valid


def _warping_errors(src_frames, pred_frames, mask_bbox, device, batch=4):
    """RAFT-flow warping error per frame pair.

    Returns (err_full_list, err_crop_list) where each entry is the mean L1
    error per pair in [0, 255] (full frame and union-mask-crop, respectively).
    Crop list is empty when `mask_bbox` is None.
    """
    if len(pred_frames) < 2 or len(src_frames) < 2:
        return [], []
    n = min(len(pred_frames), len(src_frames))
    src_arr = np.stack(src_frames[:n])
    pred_arr = np.stack(pred_frames[:n])
    H, W = src_arr.shape[1:3]

    src_t = torch.from_numpy(src_arr).permute(0, 3, 1, 2).float().to(device) / 255.0
    pred_t = torch.from_numpy(pred_arr).permute(0, 3, 1, 2).float().to(device) / 255.0
    # RAFT-large expects inputs in [-1, 1].
    src_for_raft = src_t * 2.0 - 1.0
    flow_model = _raft(device)

    err_full, err_crop = [], []
    if mask_bbox is not None:
        x1, y1, x2, y2 = mask_bbox
    with torch.no_grad():
        for i in range(0, n - 1, batch):
            j = min(i + batch, n - 1)
            flow = flow_model(src_for_raft[i:j], src_for_raft[i + 1:j + 1])[-1]
            warped, valid = _backward_warp(pred_t[i + 1:j + 1], flow)
            err = (pred_t[i:j] - warped).abs().mean(dim=1)  # (b, H, W) per-pixel L1
            valid_f = valid.float()
            err_m = err * valid_f
            denom = valid_f.sum(dim=(1, 2)).clamp(min=1.0)
            per_pair_full = (err_m.sum(dim=(1, 2)) / denom) * 255.0
            err_full.extend(per_pair_full.cpu().numpy().tolist())
            if mask_bbox is not None:
                err_c = err_m[:, y1:y2, x1:x2]
                den_c = valid_f[:, y1:y2, x1:x2].sum(dim=(1, 2)).clamp(min=1.0)
                per_pair_crop = (err_c.sum(dim=(1, 2)) / den_c) * 255.0
                err_crop.extend(per_pair_crop.cpu().numpy().tolist())
    return err_full, err_crop


def _musiq_batch(frames, device, max_dim=512, batch_size=32):
    if not frames:
        return None
    model = _musiq(device)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            f = cv2.resize(f, (int(w * s), int(h * s)))
        resized.append(f)
    by_shape = {}
    for i, f in enumerate(resized):
        by_shape.setdefault(f.shape, []).append((i, f))
    scores = [0.0] * len(resized)
    with torch.no_grad():
        for shape, items in by_shape.items():
            stk = np.stack([f for _, f in items])
            t = torch.from_numpy(stk).permute(0, 3, 1, 2).float().to(device) / 255.0
            for j in range(0, len(t), batch_size):
                out = model(t[j:j + batch_size]).flatten().cpu().numpy()
                for k, val in enumerate(out):
                    scores[items[j + k][0]] = float(val)
    return float(np.mean(scores))


def _crop_frames(frames, masks, pad_ratio=0.1):
    bbox = union_mask_bbox(masks, pad_ratio=pad_ratio)
    if bbox is None:
        return [], None
    x1, y1, x2, y2 = bbox
    crops = [f[y1:y2, x1:x2] for f in frames if f[y1:y2, x1:x2].size > 0]
    return crops, bbox


def evaluate_clip(src_frames, pred_frames, mask_frames,
                  detectable_idx=None, device="cuda"):
    """Returns Flicker / Warp / MUSIQ at full and crop scopes for one clip.

    The crop scope uses the union-mask bbox shared across the clip. MUSIQ_crop
    is restricted to detectable frames when an index is provided.
    """
    crops, bbox = _crop_frames(pred_frames, mask_frames)
    if detectable_idx is not None:
        det = set(detectable_idx)
        crops_for_musiq = [c for i, c in enumerate(crops) if i in det]
    else:
        crops_for_musiq = crops

    warp_full_list, warp_crop_list = _warping_errors(
        src_frames, pred_frames, bbox, device,
    )
    warp_full = float(np.mean(warp_full_list)) if warp_full_list else None
    warp_crop = float(np.mean(warp_crop_list)) if warp_crop_list else None
    return {
        "Flicker_full": _flicker(pred_frames),
        "Flicker_crop": _flicker(crops),
        "Warp_full":    warp_full,
        "Warp_crop":    warp_crop,
        "MUSIQ_full":   _musiq_batch(pred_frames, device=device),
        "MUSIQ_crop":   _musiq_batch(crops_for_musiq, device=device),
    }
