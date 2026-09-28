"""PP-OCRv5 extraction for ViTeX-Bench (Axis 1 / detectability gating).

Multiprocess CPU pipeline. PaddlePaddle 3.x does not support Blackwell GPUs
(RTX 5090, compute 12.0) at the time of writing, so OCR runs on CPU with a
process-pool. Each worker initializes one PaddleOCR instance per language
on demand and reuses it across the rest of its task list. Source-video OCR
is cached on disk via --src_cache so it is computed once across baselines.

Usage
-----
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \\
    conda run -n paddleocr python benchmark/ocr_extract.py \\
        --records   data/eval/parsed_records.json \\
        --data_root data/eval \\
        --pred_dir  baseline_output_videos/ViTeX-14B \\
        --output    outputs/ViTeX-14B/ocr.json \\
        --src_cache outputs/source_ocr.json \\
        --workers   8 \\
        --ocr_conf  0.30
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_utils import mask_bbox
from lang_detect import annotate_records_with_lang, paddle_lang_code


def _src_resolution(src_path):
    cap = cv2.VideoCapture(src_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return h, w


def _read_synced(video_path, mask_path, target_h, target_w):
    cap_v = cv2.VideoCapture(video_path)
    cap_m = cv2.VideoCapture(mask_path)
    n_v = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT))
    n_m = int(cap_m.get(cv2.CAP_PROP_FRAME_COUNT))
    last = None
    for i in range(n_v):
        ok_v, frame = cap_v.read()
        if not ok_v:
            break
        if (frame.shape[0], frame.shape[1]) != (target_h, target_w):
            frame = cv2.resize(frame, (target_w, target_h),
                               interpolation=cv2.INTER_LANCZOS4)
        if i < n_m:
            ok_m, m = cap_m.read()
            if ok_m:
                last = m
        m = last
        if m is None:
            yield frame, None
            continue
        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        if (gray.shape[0], gray.shape[1]) != (target_h, target_w):
            gray = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        _, binm = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        yield frame, binm
    cap_v.release()
    cap_m.release()


def _ocr_one_video(ocr, video_path, mask_path, conf_thresh, target_h, target_w):
    out = []
    for frame, m in _read_synced(video_path, mask_path, target_h, target_w):
        if frame is None or m is None:
            out.append("")
            continue
        bbox = mask_bbox(m, pad_ratio=0.1)
        if bbox is None:
            out.append("")
            continue
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            out.append("")
            continue
        try:
            preds = ocr.predict(crop)
        except Exception as e:
            print(f"    OCR error: {e}", flush=True)
            out.append("")
            continue
        kept = []
        for r in preds:
            texts = r.get("rec_texts") or []
            scores = r.get("rec_scores") or [1.0] * len(texts)
            for t, sc in zip(texts, scores):
                if sc is None or sc >= conf_thresh:
                    kept.append(t)
        out.append(" ".join(kept))
    return out


_WORKER_OCRS = {}


def _worker_init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _get_worker_ocr(lang):
    if lang not in _WORKER_OCRS:
        from paddleocr import PaddleOCR
        code = paddle_lang_code(lang)
        _WORKER_OCRS[lang] = PaddleOCR(
            use_textline_orientation=True, lang=code, device="cpu"
        )
    return _WORKER_OCRS[lang]


def _process_one_clip(task):
    (vid, lang, src_path, pred_path, mask_path,
     do_src, do_pred, conf_thresh) = task
    ocr = _get_worker_ocr(lang)
    th, tw = _src_resolution(src_path)
    src_ocr = (_ocr_one_video(ocr, src_path, mask_path, conf_thresh, th, tw)
               if do_src else None)
    pred_ocr = (_ocr_one_video(ocr, pred_path, mask_path, conf_thresh, th, tw)
                if do_pred else None)
    return vid, src_ocr, pred_ocr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--src_cache", default=None)
    ap.add_argument("--ocr_conf", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.records) as f:
        records = json.load(f)
    records = annotate_records_with_lang(records)
    lang_count = {}
    for r in records:
        lang_count[r["lang"]] = lang_count.get(r["lang"], 0) + 1
    print(f"Loaded {len(records)} records; lang distribution: {lang_count}", flush=True)

    src_cache = {}
    if args.src_cache and os.path.exists(args.src_cache):
        with open(args.src_cache) as f:
            src_cache = json.load(f)
        print(f"Loaded source OCR cache for {len(src_cache)} clips", flush=True)

    tasks = []
    for r in records:
        vid = r["id"]
        src_path = os.path.join(args.data_root, r["original_video"])
        mask_path = os.path.join(args.data_root, r["mask_video"])
        pred_path = os.path.join(args.pred_dir, vid + ".mp4")
        if not (os.path.exists(src_path) and os.path.exists(mask_path)
                and os.path.exists(pred_path)):
            print(f"  SKIP {vid}: missing files", flush=True)
            continue
        do_src = vid not in src_cache
        tasks.append((vid, r["lang"], src_path, pred_path, mask_path,
                      do_src, True, args.ocr_conf))

    print(f"Dispatching {len(tasks)} clips across {args.workers} workers", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    rec_map = {r["id"]: r for r in records}
    results = {}
    t0 = time.time()
    with Pool(processes=args.workers, initializer=_worker_init) as pool:
        done = 0
        for vid, src_ocr, pred_ocr in pool.imap_unordered(_process_one_clip, tasks):
            done += 1
            if src_ocr is not None:
                src_cache[vid] = src_ocr
                if args.src_cache:
                    tmp = args.src_cache + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(src_cache, f, ensure_ascii=False)
                    os.replace(tmp, args.src_cache)
            results[vid] = {
                "lang": rec_map[vid]["lang"],
                "source_text": rec_map[vid].get("source_text", ""),
                "target_text": rec_map[vid].get("target_text", ""),
                "source_ocr": src_cache.get(vid, []),
                "pred_ocr": pred_ocr or [],
            }
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (len(tasks) - done) / max(rate, 1e-6)
            if done % 5 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] {vid} elapsed={elapsed/60:.1f}min "
                      f"eta={eta/60:.1f}min", flush=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved OCR for {len(results)} clips → {args.output}", flush=True)
    print(f"Total elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
