# ViTeX-Edit-14B

Reference editor for video scene text editing, released with ViTeX-Bench. It fine-tunes the VACE branch of [Wan2.1-VACE-14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) on the 230-clip training split of [ViTeX-Dataset](https://huggingface.co/datasets/ViTeX-Bench/ViTeX-Dataset) and adds a glyph-video stream. The glyph stream renders the target string in a typeface matched to the source text and warps it along the tracked source-text quadrilateral. A glyph encoder pools the result into 64 tokens, and every VACE block attends to those tokens through a zero-initialized condition cross-attention layer. Weights: [ViTeX-Bench/ViTeX-Edit-14B](https://huggingface.co/ViTeX-Bench/ViTeX-Edit-14B).

| file | purpose |
|---|---|
| `select_font.py` | Qwen3-VL (via Ollama) picks the closest typeface for the source text |
| `render_glyph.py` | EasyOCR + CoTracker3 + projective warp → glyph video |
| `inference.py` | ViTeX-Edit-14B on one clip or on a whole split, multi-GPU sharding, low-memory modes |
| `composite.py` | training-free Composite post-processing → ViTeX-Edit-14B (Composite) |
| `train/` | two-stage fine-tuning recipe used for the released checkpoint |
| `fonts.py` | font table and script-specific fonts |
| `diffsynth/` | trimmed DiffSynth-Studio with the ViTeX changes ([NOTICE](diffsynth/NOTICE.md)) |

## Setup

```bash
conda create -n vitex-edit python=3.12 -y && conda activate vitex-edit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r vitex_edit/requirements.txt

# Fonts for the glyph renderer (Debian/Ubuntu; elsewhere copy the files listed in fonts.py and set VITEX_FONT_DIR)
sudo apt install fonts-liberation fonts-liberation-sans-narrow fonts-urw-base35 \
    fonts-noto-core fonts-noto-cjk fonts-freefont-ttf fonts-dejavu-core

# Optional, for select_font.py: https://ollama.com, then
ollama pull qwen3-vl:8b-instruct
```

Weights. The Hugging Face repository bundles the adapter together with an unmodified copy of Wan2.1-VACE-14B, so one download gives you everything (83 GB):

```bash
hf download ViTeX-Bench/ViTeX-Edit-14B --local-dir models/ViTeX-Edit-14B      # → --model_dir models/ViTeX-Edit-14B
```

If you already have Wan2.1-VACE-14B, download only the 8 GB adapter. The files under `base_model/` are byte-identical to the official release.

```bash
hf download ViTeX-Bench/ViTeX-Edit-14B vitex_14b.safetensors --local-dir models/ViTeX-Edit-14B
hf download Wan-AI/Wan2.1-VACE-14B --local-dir models/Wan2.1-VACE-14B          # or modelscope download --model Wan-AI/Wan2.1-VACE-14B
# → --base_dir models/Wan2.1-VACE-14B --adapter models/ViTeX-Edit-14B/vitex_14b.safetensors
```

## Edit one clip

Inputs: a source video, a mask video (white where the text should be replaced), the current string and the target string. 720p, 24 fps, up to 121 frames. Longer clips are sub-sampled to 121 frames.

```bash
# 1. typeface (skip to use the default Liberation Sans Bold)
python vitex_edit/select_font.py --video source.mp4 --mask mask.mp4 --source_text "HOTEL"
#    → {"vlm_font_name": "Arial Bold", "font_file": "LiberationSans-Bold.ttf"}

# 2. glyph video
python vitex_edit/render_glyph.py --video source.mp4 --mask mask.mp4 \
    --source_text "HOTEL" --target_text "HILTON" --font "Arial Bold" --output glyph.mp4

# 3. edit
python vitex_edit/inference.py --model_dir models/ViTeX-Edit-14B \
    --video source.mp4 --mask mask.mp4 --glyph glyph.mp4 --target_text "HILTON" --output edited.mp4
```

## Run it on the ViTeX-Bench evaluation split

From the repository root:

```bash
hf download ViTeX-Bench/ViTeX-Dataset --repo-type dataset --include "eval/*" --local-dir data
R=data/eval/parsed_records.json
python vitex_edit/select_font.py  --records $R --data_root data/eval --output data/eval/fonts.json
python vitex_edit/render_glyph.py --records $R --data_root data/eval --fonts data/eval/fonts.json \
    --output_dir data/eval/glyph_videos

# one process per GPU
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i python vitex_edit/inference.py --model_dir models/ViTeX-Edit-14B \
      --records $R --data_root data/eval --glyph_dir data/eval/glyph_videos \
      --output_dir baseline_output_videos/ViTeX-Edit-14B --worker_rank $i --num_workers 8 &
done; wait

python vitex_edit/composite.py --records $R --data_root data/eval \
    --pred_dir baseline_output_videos/ViTeX-Edit-14B --out_dir baseline_output_videos/ViTeX-Edit-14B-Composite

bash scripts/run_benchmark.sh ViTeX-Edit-14B
bash scripts/run_benchmark.sh ViTeX-Edit-14B-Composite
```

Predictions are saved at the source frame count (120), 1280×720, 24 fps, which is the format the benchmark expects.

## Note on the earlier Hugging Face script

The `inference_example.py` that was bundled with the weights on Hugging Face re-initialized the glyph encoder and zeroed the condition cross-attention output whenever the weights were loaded, so its outputs ignored the glyph video. `inference.py` here loads the checkpoint as trained; new modules are initialized only when their weights are missing (training from Wan2.1-VACE-14B). If you ran the old script, re-run with this code. `--no_glyph` switches the glyph stream off (e.g. for ablations or when no glyph video is available).

## Memory

| mode | where the DiT + VACE weights live | notes |
|---|---|---|
| default | GPU, bf16 (28 GB DiT + 8 GB VACE) | the paper setting: one 80 GB GPU per process at 720p × 121 frames |
| `--offload cpu` | CPU RAM, bf16, streamed to the GPU layer by layer | needs RAM for DiT + VACE + T5 (~47 GB); same result as default |
| `--offload disk` | the safetensors files, read layer by layer | T5 (11 GB) stays in RAM; same result as default, slowest |
| `--fp8` | combined with any mode above: weights stored in FP8 (14 GB + 4 GB), computed in bf16 | outputs differ slightly from bf16 |

T5 and the VAE are moved to the CPU between uses in every mode except the default. Activation memory grows with resolution and frame count, so the offload modes may still need `--height/--width/--num_frames` below 720p × 121 on smaller GPUs.

## Training

`train/train.sh` is the recipe behind the released checkpoint: 8× H100 80 GB with DeepSpeed ZeRO-3. Stage 1 runs 5 epochs at 720p × 49 frames, lr 5e-5, and takes ~22 h. Stage 2 runs 2 epochs at 720p × 121 frames, lr 1e-5, with CPU offload, starts from Stage 1, and takes ~50 h. The effective batch size is 64. Only the VACE branch, the glyph encoder and the condition cross-attention are trained.

```bash
hf download ViTeX-Bench/ViTeX-Dataset --repo-type dataset --include "train/*" --local-dir data
python vitex_edit/select_font.py  --records data/train/parsed_records.json --data_root data/train --output data/train/fonts.json
python vitex_edit/render_glyph.py --records data/train/parsed_records.json --data_root data/train \
    --fonts data/train/fonts.json --output_dir data/train/glyph_videos
python vitex_edit/train/make_metadata.py --data_root data/train --glyph_dir data/train/glyph_videos \
    --output data/train/metadata.csv
pip install deepspeed
bash vitex_edit/train/train.sh          # Wan2.1-VACE-14B is fetched into models/ if missing
```

The final checkpoint, `runs/vitex_edit_stage2/step-576.safetensors`, has the same layout as `vitex_14b.safetensors`; pass it to `inference.py --adapter`.

## License

Code: Apache-2.0. The bundled `diffsynth/` is Apache-2.0 (DiffSynth-Studio). Model weights: Apache-2.0 (Wan2.1 base model: see its license). ViTeX-Dataset: CC-BY-NC 4.0 (non-commercial research only); see the dataset card.
