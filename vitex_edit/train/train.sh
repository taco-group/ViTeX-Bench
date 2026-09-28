#!/bin/bash
# ViTeX-Edit-14B two-stage fine-tuning, as used for the released checkpoint.
# Hardware: 8x H100 80GB (ZeRO-3), ~2 TB CPU RAM for the Stage 2 offload.
#
#   Stage 1: 720p x 49 frames, 5 epochs, lr 5e-5, ZeRO-3 without offload     (~22 h)
#   Stage 2: 720p x 121 frames, 2 epochs, lr 1e-5, ZeRO-3 + CPU offload,
#            initialized from Stage 1, checkpoint every 20 steps              (~50 h)
# Effective batch 64 (8 GPUs x micro-batch 1 x grad-accum 8), dataset repeat 10.
# Only the VACE branch (VACE blocks + glyph encoder + condition cross-attention)
# is trained; the released vitex_14b.safetensors is the Stage 2 step-576 checkpoint.
#
# Run from the repository root after preparing data/train (see vitex_edit/README.md):
#   bash vitex_edit/train/train.sh
# Resume Stage 2 after a crash from its latest checkpoint:
#   STAGE=2 STAGE2_INIT=runs/vitex_edit_stage2/step-200.safetensors RESUME_STEP=200 bash vitex_edit/train/train.sh

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data/train}
METADATA=${METADATA:-$DATA_ROOT/metadata.csv}
# Wan2.1-VACE-14B is looked up at $MODEL_BASE/Wan-AI/Wan2.1-VACE-14B and downloaded
# there (ModelScope by default; DIFFSYNTH_DOWNLOAD_SOURCE=huggingface for HF) if absent.
MODEL_BASE=${MODEL_BASE:-models}
STAGE1_OUT=${STAGE1_OUT:-runs/vitex_edit_stage1}
STAGE2_OUT=${STAGE2_OUT:-runs/vitex_edit_stage2}
STAGE=${STAGE:-all}   # all | 1 | 2

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH=$MODEL_BASE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_DEEPSPEED_ZERO3_INIT=false
# NCCL 2.26 all-gather hang with ZeRO-3 (pytorch/pytorch#150381)
export NCCL_NVLS_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

WAN=Wan-AI/Wan2.1-VACE-14B
COMMON=(
  vitex_edit/train/train.py
  --dataset_base_path "$DATA_ROOT"
  --dataset_metadata_path "$METADATA"
  --data_file_keys "video,vace_video,vace_video_mask,glyph_video"
  --height 720 --width 1280
  --dataset_repeat 10
  --model_id_with_origin_paths "$WAN:diffusion_pytorch_model*.safetensors,$WAN:models_t5_umt5-xxl-enc-bf16.pth,$WAN:Wan2.1_VAE.pth"
  --tokenizer_path "$MODEL_BASE/$WAN/google/umt5-xxl"
  --remove_prefix_in_ckpt "pipe.vace."
  --trainable_models "vace"
  --extra_inputs "vace_video,vace_video_mask,glyph_video"
)

if [ "$STAGE" = all ] || [ "$STAGE" = 1 ]; then
  echo "== Stage 1: 720p x 49 frames, 5 epochs =="
  accelerate launch --config_file vitex_edit/train/accelerate_stage1.yaml "${COMMON[@]}" \
    --num_frames 49 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --output_path "$STAGE1_OUT" \
    --use_gradient_checkpointing
fi

if [ "$STAGE" = all ] || [ "$STAGE" = 2 ]; then
  STAGE2_INIT=${STAGE2_INIT:-$STAGE1_OUT/epoch-4.safetensors}
  [ -f "$STAGE2_INIT" ] || { echo "Stage 2 init checkpoint not found: $STAGE2_INIT" >&2; exit 1; }
  echo "== Stage 2: 720p x 121 frames, 2 epochs, from $STAGE2_INIT =="
  accelerate launch --config_file vitex_edit/train/accelerate_stage2.yaml "${COMMON[@]}" \
    --num_frames 121 \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --output_path "$STAGE2_OUT" \
    --use_gradient_checkpointing_offload \
    --save_steps 20 \
    --resume_step "${RESUME_STEP:-0}" \
    --model_checkpoint_path "$STAGE2_INIT"
  echo "Final checkpoint: $STAGE2_OUT/step-576.safetensors (use it as --adapter)"
fi
