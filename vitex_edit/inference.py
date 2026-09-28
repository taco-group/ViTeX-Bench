"""ViTeX-Edit-14B inference.

Edits the scene text inside a mask, conditioned on the target string and a
glyph video (see `render_glyph.py`). Runs on one clip, or on every clip of a
`parsed_records.json` (e.g. the ViTeX-Dataset eval split) with optional
multi-GPU sharding.

Single clip:
  python vitex_edit/inference.py --model_dir models/ViTeX-Edit-14B \
      --video source.mp4 --mask mask.mp4 --glyph glyph.mp4 \
      --target_text "HILTON" --output out.mp4

Eval split (writes <output_dir>/<id>.mp4, ready for scripts/run_benchmark.sh):
  python vitex_edit/inference.py --model_dir models/ViTeX-Edit-14B \
      --records data/eval/parsed_records.json --data_root data/eval \
      --glyph_dir data/eval/glyph_videos \
      --output_dir baseline_output_videos/ViTeX-Edit-14B

Weights: either a local copy of the Hugging Face repo ViTeX-Bench/ViTeX-Edit-14B
(`--model_dir`, which holds `base_model/` and `vitex_14b.safetensors`), or the
official Wan2.1-VACE-14B checkpoint (`--base_dir`, byte-identical to
`base_model/`) plus the ViTeX adapter (`--adapter`).

Memory: `--offload none` (default) keeps every model on the GPU and needs an
80 GB card at 720p x 121 frames. `--offload cpu` keeps the weights in CPU RAM
(~47 GB) and streams layers to the GPU. `--offload disk` streams the DiT and VACE
from the safetensors files and keeps only T5 in RAM, at the cost of speed. These three compute the
same result. `--fp8` additionally stores the DiT/VACE weights in FP8, which
halves their memory but changes outputs slightly.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # use the bundled `diffsynth/`

import numpy as np
import torch
from PIL import Image

from diffsynth.configs.model_configs import MODEL_CONFIGS
from diffsynth.core.loader import load_model
from diffsynth.models.model_loader import ModelPool
from diffsynth.models.wan_video_vace import VaceWanModel
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

HEIGHT = 720
WIDTH = 1280
NUM_FRAMES = 121  # Wan requires 4k+1 frames; 120-frame clips are padded with their last frame
NUM_INFERENCE_STEPS = 50
CFG_SCALE = 5.0
SEED = 42

# Wan2.1-VACE-14B entry of MODEL_CONFIGS; its extra_kwargs include glyph_channels=16.
VACE_14B_HASH = "7a513e1f257a861512b1afd387a8ecd9"


def load_video_frames(path, target_frames=NUM_FRAMES, resize=(HEIGHT, WIDTH)):
    """Read a video as PIL images, resized to `(H, W)` and sub-sampled or
    last-frame padded to `target_frames`. Returns (frames, source_frame_count)."""
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if resize:
            img = img.resize((resize[1], resize[0]), Image.LANCZOS)  # (W, H)
        frames.append(img)
    cap.release()

    if not frames:
        raise ValueError(f"empty or unreadable video: {path}")
    n_src = len(frames)
    if target_frames and n_src > target_frames:
        idx = np.linspace(0, n_src - 1, target_frames, dtype=int)
        frames = [frames[i] for i in idx]
    elif target_frames and n_src < target_frames:
        frames.extend([frames[-1]] * (target_frames - n_src))
    return frames, n_src


def video_fps(path, default=24.0):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps and fps > 0 else default


def save_video(frames, path, fps=24):
    """Write PIL images to an H.264 MP4 (CRF 18, yuv420p)."""
    import imageio_ffmpeg
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    w, h = frames[0].size
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames:
        proc.stdin.write(np.asarray(fr).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed writing {path}")


def resolve_weights(args):
    base_dir = args.base_dir or (os.path.join(args.model_dir, "base_model") if args.model_dir else None)
    adapter = args.adapter or (os.path.join(args.model_dir, "vitex_14b.safetensors") if args.model_dir else None)
    if base_dir is None or adapter is None:
        sys.exit("error: pass --model_dir, or both --base_dir and --adapter (see vitex_edit/README.md)")
    shards = sorted(glob.glob(os.path.join(base_dir, "diffusion_pytorch_model*.safetensors")))
    required = {
        "Wan2.1-VACE-14B DiT shards": shards,
        "T5 encoder": os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        "Wan VAE": os.path.join(base_dir, "Wan2.1_VAE.pth"),
        "tokenizer": os.path.join(base_dir, "google", "umt5-xxl"),
        "ViTeX adapter": adapter,
    }
    for name, path in required.items():
        if not path or (isinstance(path, str) and not os.path.exists(path)):
            sys.exit(f"error: missing {name}: {path or base_dir}")
    return base_dir, shards, adapter


def make_vram_config(offload, device, fp8=False):
    """Per-model VRAM config (see diffsynth docs, "VRAM management"). With
    `fp8`, weights are stored in float8 and cast to bf16 for each layer's compute."""
    bf16 = torch.bfloat16
    store = torch.float8_e4m3fn if fp8 else bf16
    if offload == "none":
        if not fp8:
            return {}
        return dict(offload_dtype=store, offload_device=device, onload_dtype=store, onload_device=device,
                    preparing_dtype=store, preparing_device=device, computation_dtype=bf16, computation_device=device)
    if offload == "cpu":
        return dict(offload_dtype=store, offload_device="cpu", onload_dtype=store, onload_device="cpu",
                    preparing_dtype=store, preparing_device=device, computation_dtype=bf16, computation_device=device)
    if offload == "disk":
        return dict(offload_dtype="disk", offload_device="disk", onload_dtype="disk", onload_device="disk",
                    preparing_dtype=store, preparing_device=device, computation_dtype=bf16, computation_device=device)
    raise ValueError(offload)


def load_vitex_vace(adapter, device, vram_config, vram_limit, model_kwargs=None):
    """Build the ViTeX-Edit-14B VACE branch (VACE blocks + glyph encoder +
    condition cross-attention) entirely from the adapter file."""
    # The DiT shards hash to both a DiT and a VACE entry; take the VACE one.
    config = next(c for c in MODEL_CONFIGS
                  if c["model_hash"] == VACE_14B_HASH and c["model_name"] == "wan_video_vace")
    kwargs = model_kwargs or config["extra_kwargs"]

    # The adapter must cover every parameter of the model; fail loudly otherwise.
    from safetensors import safe_open
    with safe_open(adapter, framework="pt") as f:
        file_keys = set(f.keys())
    with torch.device("meta"):
        model_keys = set(VaceWanModel(**kwargs).state_dict().keys())
    if file_keys != model_keys:
        raise RuntimeError(
            f"{adapter} does not match the ViTeX-Edit-14B VACE layout: "
            f"{len(model_keys - file_keys)} missing, {len(file_keys - model_keys)} unexpected tensors"
        )

    pool = ModelPool()
    full_config = {**pool.default_vram_config(), **vram_config}
    full_config["computation_dtype"] = full_config["computation_dtype"] or torch.bfloat16
    full_config["computation_device"] = device
    module_map = pool.fetch_module_map(config["model_class"], full_config)
    return load_model(
        VaceWanModel, adapter, kwargs, torch.bfloat16, device,
        state_dict_converter=None, use_disk_map=True,
        module_map=module_map, vram_config=full_config, vram_limit=vram_limit,
    )


def build_pipeline(args):
    base_dir, shards, adapter = resolve_weights(args)
    device = args.device
    vram = make_vram_config(args.offload, device, fp8=args.fp8)  # DiT and VACE
    # T5 and VAE stay bf16. They are CPU-offloaded whenever the DiT is managed
    # (their .pth files cannot be disk-mapped, and FP8 mode frees the GPU for the DiT).
    text_vae_vram = vram if args.offload == "none" and not args.fp8 else make_vram_config("cpu", device)
    vram_limit = None
    if args.offload != "none":
        total = torch.cuda.get_device_properties(torch.device(device)).total_memory / 1024 ** 3
        vram_limit = args.vram_limit if args.vram_limit is not None else max(total - 4.0, 1.0)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=shards, **vram),
            ModelConfig(path=os.path.join(base_dir, "models_t5_umt5-xxl-enc-bf16.pth"), **text_vae_vram),
            ModelConfig(path=os.path.join(base_dir, "Wan2.1_VAE.pth"), **text_vae_vram),
        ],
        tokenizer_config=ModelConfig(path=os.path.join(base_dir, "google", "umt5-xxl")),
        redirect_common_files=False,
        vram_limit=vram_limit,
    )

    # The DiT shards also contain the original Wan2.1-VACE weights, which
    # from_pretrained loads as `pipe.vace`. Replace them with ViTeX-Edit-14B.
    pipe.vace = None
    torch.cuda.empty_cache()
    print(f"Loading ViTeX-Edit-14B VACE branch from {adapter}")
    pipe.vace = load_vitex_vace(adapter, device, vram, vram_limit)
    pipe.vram_management_enabled = pipe.check_vram_management_state()
    return pipe


def edit_clip(pipe, video_path, mask_path, glyph_path, prompt, output_path, args):
    size = (args.height, args.width)
    vace_video, n_src = load_video_frames(video_path, args.num_frames, size)
    vace_mask, _ = load_video_frames(mask_path, args.num_frames, size)
    glyph = None if args.no_glyph else load_video_frames(glyph_path, args.num_frames, size)[0]
    frames = pipe(
        prompt=prompt,
        negative_prompt="",
        vace_video=vace_video,
        vace_video_mask=vace_mask,
        glyph_video=glyph,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        tiled=True,
    )
    # Drop the frames that only exist because of 4k+1 padding; a sub-sampled
    # clip keeps its duration.
    fps = video_fps(video_path)
    if n_src > len(frames):
        fps = fps * len(frames) / n_src
    save_video(frames[:min(n_src, len(frames))], output_path, fps=fps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    w = p.add_argument_group("weights")
    w.add_argument("--model_dir", help="Local copy of ViTeX-Bench/ViTeX-Edit-14B (base_model/ + vitex_14b.safetensors)")
    w.add_argument("--base_dir", help="Wan2.1-VACE-14B directory (overrides <model_dir>/base_model)")
    w.add_argument("--adapter", help="vitex_14b.safetensors (overrides <model_dir>/vitex_14b.safetensors)")

    s = p.add_argument_group("single clip")
    s.add_argument("--video", help="Source video")
    s.add_argument("--mask", help="Text-region mask video (white = replace)")
    s.add_argument("--glyph", help="Glyph video from render_glyph.py")
    s.add_argument("--target_text", help="Target string (used as the prompt, as in training)")
    s.add_argument("--output", default="output.mp4")

    b = p.add_argument_group("batch over parsed_records.json")
    b.add_argument("--records", help="parsed_records.json (e.g. data/eval/parsed_records.json)")
    b.add_argument("--data_root", help="Root that record paths are relative to")
    b.add_argument("--glyph_dir", help="Directory of <id>.mp4 glyph videos")
    b.add_argument("--output_dir", help="Where <id>.mp4 predictions are written")
    b.add_argument("--ids", nargs="*", help="Only process these clip ids")
    b.add_argument("--worker_rank", type=int, default=0, help="Shard index when running one process per GPU")
    b.add_argument("--num_workers", type=int, default=1, help="Total number of shards")

    g = p.add_argument_group("generation")
    g.add_argument("--no_glyph", action="store_true",
                   help="Do not condition on the glyph video (ablation, or when no glyph video is available).")
    g.add_argument("--height", type=int, default=HEIGHT)
    g.add_argument("--width", type=int, default=WIDTH)
    g.add_argument("--num_frames", type=int, default=NUM_FRAMES)
    g.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS)
    g.add_argument("--cfg_scale", type=float, default=CFG_SCALE)
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--offload", choices=["none", "cpu", "disk"], default="none",
                   help="none: all on GPU (80 GB); cpu: layer streaming from RAM; disk: from safetensors files")
    g.add_argument("--vram_limit", type=float, default=None, help="GB of VRAM to target when offloading")
    g.add_argument("--fp8", action="store_true",
                   help="Store DiT/VACE weights in FP8 (compute stays bf16). With --offload none this fits "
                        "a 32 GB GPU at reduced resolution/length; outputs differ slightly from bf16.")
    args = p.parse_args()

    batch = args.records is not None
    if batch:
        missing = [n for n in ("data_root", "output_dir") if getattr(args, n) is None]
        if not args.no_glyph and args.glyph_dir is None:
            missing.append("glyph_dir")
        if missing:
            p.error("batch mode needs --" + ", --".join(missing))
        if not (0 <= args.worker_rank < args.num_workers):
            p.error("need 0 <= --worker_rank < --num_workers")
    else:
        missing = [n for n in ("video", "mask", "target_text") if getattr(args, n) is None]
        if not args.no_glyph and args.glyph is None:
            missing.append("glyph")
        if missing:
            p.error("single-clip mode needs --" + ", --".join(missing))

    pipe = build_pipeline(args)
    print(f"Glyph conditioning: {'OFF (--no_glyph)' if args.no_glyph else 'ON'}")

    if not batch:
        edit_clip(pipe, args.video, args.mask, args.glyph, args.target_text, args.output, args)
        print(f"saved: {args.output}")
        return

    with open(args.records) as f:
        records = json.load(f)
    if args.ids:
        wanted = set(args.ids)
        records = [r for r in records if r["id"] in wanted]
    records = [r for i, r in enumerate(records) if i % args.num_workers == args.worker_rank]
    print(f"[rank {args.worker_rank}/{args.num_workers}] {len(records)} clips")

    for i, rec in enumerate(records, 1):
        out = os.path.join(args.output_dir, rec["id"] + ".mp4")
        if os.path.exists(out):
            print(f"  [{i}/{len(records)}] {rec['id']}: exists, skip")
            continue
        glyph = None if args.no_glyph else os.path.join(args.glyph_dir, rec["id"] + ".mp4")
        if glyph is not None and not os.path.exists(glyph):
            print(f"  [{i}/{len(records)}] {rec['id']}: no glyph video at {glyph}, skip")
            continue
        print(f"  [{i}/{len(records)}] {rec['id']}: {rec['source_text']!r} -> {rec['target_text']!r}")
        edit_clip(
            pipe,
            os.path.join(args.data_root, rec["original_video"]),
            os.path.join(args.data_root, rec["mask_video"]),
            glyph, rec["target_text"], out, args,
        )


if __name__ == "__main__":
    main()
