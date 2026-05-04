"""Axis 3: edit-locality metrics.

Four metrics on the locality-only prediction f_hat^loc = (1-m)*f_hat + m*f:
    PSNR_loc       (higher is better, capped at 100 dB)
    SSIM_loc       (higher is better, in [0, 1])
    LPIPS_loc      (lower is better, AlexNet backbone)
    DreamSim_loc   (lower is better, perceptual identity)

DreamSim (Fu et al., NeurIPS 2023) is a DINO+CLIP+OpenCLIP ensemble trained
on human "are these images the same?" judgments. It is robust to the kind
of low-level pixel noise introduced by VAE round-trip in video diffusion
backbones, so it isolates real semantic deviation in the unedited region
rather than the codec/encoding gap that PSNR/SSIM/LPIPS conflate with
locality.
"""

import numpy as np
import torch
import torch.nn.functional as F

_LPIPS = None
_SSIM = None
_DREAMSIM = None


def _lpips(device):
    global _LPIPS
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="alex").to(device)
        _LPIPS.eval()
    return _LPIPS


def _ssim_metric(device):
    global _SSIM
    if _SSIM is None:
        import pyiqa
        _SSIM = pyiqa.create_metric("ssim", device=device)
    return _SSIM


def _dreamsim(device):
    global _DREAMSIM
    if _DREAMSIM is None:
        from dreamsim import dreamsim as ds_factory
        model, _ = ds_factory(pretrained=True, device=device, dreamsim_type="ensemble")
        model.eval()
        _DREAMSIM = model
    return _DREAMSIM


def _to_tensor(frames, device):
    arr = np.stack(frames)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().float().to(device)
    return t  # range [0, 255]


def _compose_locality_gpu(pred_t, src_t, mask_t):
    return (1.0 - mask_t) * pred_t + mask_t * src_t


def _psnr_batch(src_t, loc_t):
    diff = (src_t - loc_t).pow(2).mean(dim=(1, 2, 3))
    psnr = 10.0 * torch.log10((255.0 ** 2) / diff.clamp(min=1e-12))
    return psnr.clamp(max=100.0).cpu().numpy().tolist()


def _ssim_batch(src_t, loc_t, device, batch_size=16):
    metric = _ssim_metric(device)
    s = src_t / 255.0
    l = loc_t / 255.0
    out = []
    with torch.no_grad():
        for i in range(0, len(s), batch_size):
            v = metric(s[i:i + batch_size], l[i:i + batch_size]).flatten().cpu().numpy()
            out.extend(float(x) for x in v)
    return out


def _lpips_batch(src_t, loc_t, device, batch_size=16):
    model = _lpips(device)
    s = src_t / 127.5 - 1.0
    l = loc_t / 127.5 - 1.0
    out = []
    with torch.no_grad():
        for i in range(0, len(s), batch_size):
            v = model(s[i:i + batch_size], l[i:i + batch_size]).flatten().cpu().numpy()
            out.extend(float(x) for x in v)
    return out


def _dreamsim_batch(src_t, loc_t, device, batch_size=16):
    """Per-frame DreamSim (ensemble). Inputs at [0, 255], (T, 3, H, W)."""
    model = _dreamsim(device)
    # DreamSim's own preprocess resizes to 224 and ImageNet-normalizes; we
    # apply the same on GPU to avoid PIL round-trip.
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    s = F.interpolate(src_t / 255.0, size=(224, 224),
                      mode="bilinear", align_corners=False)
    l = F.interpolate(loc_t / 255.0, size=(224, 224),
                      mode="bilinear", align_corners=False)
    s = (s - mean) / std
    l = (l - mean) / std
    out = []
    with torch.no_grad():
        for i in range(0, len(s), batch_size):
            v = model(s[i:i + batch_size], l[i:i + batch_size]).flatten().cpu().numpy()
            out.extend(float(x) for x in v)
    return out


def evaluate_clip(src_frames, pred_frames, mask_frames, device="cuda"):
    src_t = _to_tensor(src_frames, device)
    pred_t = _to_tensor(pred_frames, device)
    m = np.stack([m_.astype(np.float32) for m_ in mask_frames])
    mask_t = torch.from_numpy(m).unsqueeze(1).to(device)

    loc_t = _compose_locality_gpu(pred_t, src_t, mask_t)

    psnr_vals = _psnr_batch(src_t, loc_t)
    ssim_vals = _ssim_batch(src_t, loc_t, device)
    lpips_vals = _lpips_batch(src_t, loc_t, device)
    dsim_vals = _dreamsim_batch(src_t, loc_t, device)

    del src_t, pred_t, mask_t, loc_t
    torch.cuda.empty_cache()

    return {
        "PSNR_loc":     float(np.mean(psnr_vals)) if psnr_vals else None,
        "SSIM_loc":     float(np.mean(ssim_vals)) if ssim_vals else None,
        "LPIPS_loc":    float(np.mean(lpips_vals)) if lpips_vals else None,
        "DreamSim_loc": float(np.mean(dsim_vals)) if dsim_vals else None,
    }
