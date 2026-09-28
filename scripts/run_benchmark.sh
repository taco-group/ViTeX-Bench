#!/bin/bash
# ViTeX-Bench end-to-end runner: OCR (CPU) + 13-metric evaluation (GPU).
#
# Single-baseline usage (most users):
#     bash scripts/run_benchmark.sh <method_name>
#
# All-baselines usage (paper-style sweep with CPU-OCR + GPU-eval pipelining):
#     bash scripts/run_benchmark.sh
#
# Configurable via environment variables (defaults shown):
#   ROOT              repository root                       (auto-detected)
#   DATA_ROOT         dataset root                          $ROOT/data/eval
#   RECORDS           records JSON                          $DATA_ROOT/parsed_records.json
#   BASELINES_ROOT    baseline output root                  $ROOT/baseline_output_videos
#   OUT               output / cache root                   $ROOT/outputs
#   PADDLE_PY         python in the paddleocr conda env     auto-detect
#   DSS_PY            python in the GPU-metrics env         auto-detect
#   WORKERS           OCR worker processes                  8
#   OCR_CONF          PP-OCRv5 confidence threshold         0.30
#   ORDER             baselines to run (sweep mode only)    full paper grid

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
BENCH=$ROOT/benchmark
DATA_ROOT=${DATA_ROOT:-$ROOT/data/eval}
RECORDS=${RECORDS:-$DATA_ROOT/parsed_records.json}
BASELINES_ROOT=${BASELINES_ROOT:-$ROOT/baseline_output_videos}
OUT=${OUT:-$ROOT/outputs}
SRC_CACHE=$OUT/source_ocr.json

PADDLE_PY=${PADDLE_PY:-$(conda run -n paddleocr which python 2>/dev/null || echo python)}
DSS_PY=${DSS_PY:-$(conda run -n vitex-bench which python 2>/dev/null || echo python)}

WORKERS=${WORKERS:-8}
OCR_CONF=${OCR_CONF:-0.30}
ORDER_DEFAULT="identity ViTeX-14B videopainter wan2.2vace14b kling fluxtext text_ctrl text_ctrl+anyv2v re-ste anytext2"
ORDER=${ORDER:-$ORDER_DEFAULT}

mkdir -p "$OUT"

# Auto-download the evaluation split of ViTeX-Dataset if absent.
ensure_data() {
    if [ -f "$RECORDS" ] && [ -d "$DATA_ROOT/original_videos" ] && [ -d "$DATA_ROOT/masks" ]; then
        return 0
    fi
    echo "[$(date '+%H:%M:%S')] Eval data not found at $DATA_ROOT, downloading from HF…"
    # `hf` is the CLI of huggingface_hub >= 1.0; older releases only ship `huggingface-cli`.
    local HF_CLI
    if command -v hf >/dev/null 2>&1; then
        HF_CLI=hf
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI=huggingface-cli
    else
        echo "ERROR: neither hf nor huggingface-cli found; pip install huggingface_hub, or download ViTeX-Dataset/eval manually to $DATA_ROOT" >&2
        exit 1
    fi
    "$HF_CLI" download ViTeX-Bench/ViTeX-Dataset \
        --repo-type dataset \
        --include "eval/*" \
        --local-dir "$ROOT/data"
    if [ ! -f "$RECORDS" ]; then
        echo "ERROR: download finished but $RECORDS still missing" >&2
        exit 1
    fi
    echo "[$(date '+%H:%M:%S')] Eval data ready: $(find "$DATA_ROOT/original_videos" -name '*.mp4' | wc -l) clips"
}

ensure_data
SUMMARY=$OUT/summary.tsv
if [ ! -f "$SUMMARY" ]; then
    printf "baseline\tn_clips\tSeqAcc\tCharAcc\tTTS\tFlicker_full\tFlicker_crop\tWarp_full\tWarp_crop\tMUSIQ_full\tMUSIQ_crop\tPSNR_loc\tSSIM_loc\tLPIPS_loc\tDreamSim_loc\n" > "$SUMMARY"
fi

ts() { date '+%H:%M:%S'; }

run_ocr() {
    local B=$1
    local BDIR=$BASELINES_ROOT/$B
    local OCR_OUT=$OUT/$B/ocr.json
    local LOG=$OUT/$B/log.txt
    mkdir -p "$OUT/$B"
    if [ -f "$OCR_OUT" ]; then
        echo "[$(ts)] == [$B] OCR cached"
        return 0
    fi
    echo "[$(ts)] >> [$B] OCR start"
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    "$PADDLE_PY" "$BENCH/ocr_extract.py" \
        --records   "$RECORDS" \
        --data_root "$DATA_ROOT" \
        --pred_dir  "$BDIR" \
        --output    "$OCR_OUT" \
        --src_cache "$SRC_CACHE" \
        --workers   "$WORKERS" \
        --ocr_conf  "$OCR_CONF" 2>&1 | tee -a "$LOG"
    "$DSS_PY" -c "
import json, sys
d = json.load(open('$OCR_OUT'))
n_total = len(d)
n_with_pred = sum(1 for r in d.values() if any(s.strip() for s in r['pred_ocr']))
print(f'[$B] OCR sanity: {n_with_pred}/{n_total} clips have at least one non-empty prediction frame')
if n_total < 100:
    print('FAIL: too few clips processed', file=sys.stderr); sys.exit(1)
"
    echo "[$(ts)] << [$B] OCR done"
}

run_eval() {
    local B=$1
    local BDIR=$BASELINES_ROOT/$B
    local OCR_OUT=$OUT/$B/ocr.json
    local EVAL_OUT=$OUT/$B/eval.json
    local LOG=$OUT/$B/log.txt
    if [ -f "$EVAL_OUT" ]; then
        echo "[$(ts)] == [$B] already evaluated"
        return 0
    fi
    echo "[$(ts)] >> [$B] evaluate start"
    "$DSS_PY" "$BENCH/evaluate.py" \
        --records     "$RECORDS" \
        --data_root   "$DATA_ROOT" \
        --pred_dir    "$BDIR" \
        --ocr_results "$OCR_OUT" \
        --output      "$EVAL_OUT" 2>&1 | tee -a "$LOG"
    echo "[$(ts)] << [$B] evaluate done"
    "$DSS_PY" -c "
import json
d = json.load(open('$EVAL_OUT'))
agg = d['aggregate']
n = len(d['per_clip'])
def m(k):
    v = agg[k]
    return 'N/A' if v.get('mean') is None else f'{v[\"mean\"]:.4f}'
keys = ['SeqAcc','CharAcc','TTS',
        'Flicker_full','Flicker_crop','Warp_full','Warp_crop','MUSIQ_full','MUSIQ_crop',
        'PSNR_loc','SSIM_loc','LPIPS_loc','DreamSim_loc']
row = ['$B', str(n)] + [m(k) for k in keys]
with open('$SUMMARY', 'a') as f:
    f.write('\t'.join(row) + '\n')
print(f'[$B] primary: SeqAcc={m(\"SeqAcc\")}  Warp_crop={m(\"Warp_crop\")}  DreamSim_loc={m(\"DreamSim_loc\")}')
"
}

run_one() {
    local B=$1
    local BDIR=$BASELINES_ROOT/$B
    if [ ! -d "$BDIR" ]; then
        echo "[$(ts)] !! $B: directory $BDIR missing"
        exit 1
    fi
    run_ocr "$B"
    run_eval "$B"
}

# Single-baseline mode: bash scripts/run_benchmark.sh <method_name>
if [ "$#" -ge 1 ]; then
    run_one "$1"
    echo
    echo "=================================================================="
    echo "Done. Summary so far:"
    column -t -s $'\t' "$SUMMARY"
    exit 0
fi

# All-baselines mode: bash scripts/run_benchmark.sh
PREV_EVAL_PID=""
for B in $ORDER; do
    BDIR=$BASELINES_ROOT/$B
    if [ ! -d "$BDIR" ]; then
        echo "[$(ts)] !! skipping $B: directory missing"
        continue
    fi
    run_ocr "$B"
    if [ -n "$PREV_EVAL_PID" ]; then
        wait "$PREV_EVAL_PID" || { echo "[$(ts)] !! previous eval failed"; exit 1; }
        PREV_EVAL_PID=""
    fi
    run_eval "$B" &
    PREV_EVAL_PID=$!
done

if [ -n "$PREV_EVAL_PID" ]; then
    wait "$PREV_EVAL_PID"
fi

echo
echo "=================================================================="
echo "All done. Summary:"
column -t -s $'\t' "$SUMMARY"
