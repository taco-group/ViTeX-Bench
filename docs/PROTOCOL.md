# Evaluation Protocol

ViTeX-Bench reports **thirteen metrics** along **three orthogonal axes** — text correctness (3 metrics), visual quality (6 metrics), and edit locality (4 metrics). The full thirteen-metric vector is the unit of report; no cross-axis aggregate is computed, because no axis substitutes for another. For ranking purposes only, the public leaderboard sorts on **TextScore**, the geometric mean of the three text-correctness primitives (see the final section).

## Task

A task instance is a tuple $(V, M, s_{\mathrm{src}}, s_{\mathrm{tgt}})$ where:

| symbol | meaning |
|---|---|
| $V = \{f_t\}_{t=1}^{T}$ | source video, $T = 120$, $H \times W = 720 \times 1280$ |
| $M = \{m_t\}_{t=1}^{T}$ | binary mask video marking the editable text region |
| $s_{\mathrm{src}}$ | source string currently visible in the masked region |
| $s_{\mathrm{tgt}}$ | requested replacement string |

A submission outputs an edited video $\hat V = \{\hat f_t\}_{t=1}^{T}$ at the same resolution and length. Predictions at a different spatial resolution are auto-resampled to $1280 \times 720$ at evaluation time using Lanczos interpolation; predictions at a different temporal length are truncated to the source length.

A successful edit must (i) render $s_{\mathrm{tgt}}$ inside the mask, (ii) preserve the unmasked scene, and (iii) keep the rendered characters temporally consistent with the underlying motion and lighting.

---

## Axis 1 — Text correctness

### Recognizer

We use **PP-OCRv5** as the unified recognizer. Per-clip language is auto-routed from the source-text Unicode block over $\{\text{Latin}, \text{Chinese}, \text{Japanese}, \text{Cyrillic}\}$ — the four scripts that appear in the released test split — using the corresponding PP-OCRv5 weight set (`en`, `ch`, `japan`, `ru`). Recognized boxes below confidence **0.30** are dropped; the remaining strings are concatenated by space.

### Normalization

Both target strings and recognizer outputs are normalized before any string comparison:

1. NFKC fold (unifies full-width / half-width punctuation and digits).
2. Case fold to upper case.
3. Strip every non-letter, non-digit character — including ASCII punctuation (`. , ' " & ! ? / \ - _`), CJK punctuation (`， 。 ！ "`), and any whitespace.

Letters are kept across all scripts (Latin, CJK Unified Ideographs, Hiragana, Katakana, Hangul, Cyrillic). With this step `35,000` and `35 000` compare as the same character sequence, and `MIKE'S` and `MIKES` likewise. The same normalization is applied uniformly to the source-detectability gating, the static accuracy metrics, and the temporal stability metric.

### Fitting (semi-global) edit distance

Dilated text-region masks routinely pad beyond the original glyph, so a successful edit (target `BIG`) often appears inside a longer OCR string (`ABIGA`) because the pad picks up neighbouring scene pixels. Standard Levenshtein would penalize the two padding characters as substitutions; we instead use *fitting alignment*:

$$
d_{\mathrm{fit}}(\text{pattern}, \text{text}) = \min \{\text{edits aligning } \text{pattern} \text{ to a contiguous region of } \text{text}\}
$$

with a free prefix and suffix on `text`. Implementation: a standard $O(mn)$ dynamic program where the first row is initialized to zero and the answer is read off as the minimum of the last row.

Bounded similarity:
$$
\mathrm{Sim}(r, c) = 1 - \frac{d_{\mathrm{fit}}(r, c)}{\max(|r|, 1)}
$$
Note the asymmetry: $r$ is the reference, $c$ is the candidate; we use $|r|$ as the denominator so a candidate longer than the reference is not penalized on length alone.

### Detectability gating

OCR cannot read the source on every frame (camera motion, occlusion, blur, out-of-frame motion). Scoring all frames would conflate editing errors with intrinsic clip difficulty, so we evaluate text correctness only on **source-detectable** frames:

$$
\mathcal{D} = \{ t : \mathrm{Sim}(s_{\mathrm{src}}, s_t^V) \ge \tau \}, \qquad \tau = 0.5
$$

where $s_t^V$ is the recognizer's output on the source video at frame $t$. The adjacent-pair set used by TTS is

$$
\mathcal{P} = \{ (t, t+1) : t \in \mathcal{D},\ t+1 \in \mathcal{D} \}.
$$

### The three metrics

$$
\begin{aligned}
\mathrm{SeqAcc}  &= \mathop{\mathbb{E}}_{t \in \mathcal{D}} \mathbf{1}\bigl[d_{\mathrm{fit}}(s_{\mathrm{tgt}}, \hat s_t) = 0\bigr] \\
\mathrm{CharAcc} &= \mathop{\mathbb{E}}_{t \in \mathcal{D}} \mathrm{Sim}(s_{\mathrm{tgt}}, \hat s_t) \\
\mathrm{TTS}     &= \mathop{\mathbb{E}}_{(t, t+1) \in \mathcal{P}} \mathbf{1}[\hat s_t = \hat s_{t+1}]
\end{aligned}
$$

where $\hat s_t$ is the recognizer's output on the prediction at frame $t$. Equality in TTS is taken after normalization.

**Per-clip exclusion.** When $\mathcal{D} = \varnothing$ the clip is excluded from the SeqAcc / CharAcc test-split average and reported separately; when $\mathcal{P} = \varnothing$ the clip is excluded from the TTS average. The aggregate `n_clips` field in `eval.json` reports the support size for each metric.

---

## Axis 2 — Visual quality

We score visual quality with **six primitives** at two spatial scopes: the full output frame and the cropped text-edit region. The crop scope uses a single **union mask bbox** shared across the clip — the bbox covering every per-frame mask, padded by 10 % — so adjacent crops are spatially aligned for the temporal metrics and the canvas is consistent for MUSIQ_crop.

For a scope $S \in \{\text{full}, \text{crop}\}$ and prediction frames $x_t^S$:

$$
\mathrm{Flicker}_S = \mathop{\mathbb{E}}_{t} \mathrm{MAE}(x_{t+1}^S, x_t^S), \qquad
\mathrm{Warp}_S    = \mathop{\mathbb{E}}_{t} \mathrm{MAE}\bigl(x_t^S,\ \mathcal{W}(F^{\mathrm{src}}_{t \to t+1}, x_{t+1}^S)\bigr), \qquad
\mathrm{MUSIQ}_S   = \mathop{\mathbb{E}}_{t} \mathrm{MUSIQ}(x_t^S)
$$

where $F^{\mathrm{src}}_{t \to t+1}$ is the **RAFT** forward optical flow on the *source* video and $\mathcal{W}$ is the corresponding backward warp (sampled at valid pixels only).

- `Flicker` is the raw adjacent-frame MAE; lower is better. It is hackable by emitting smooth output that ignores source motion.
- `Warp` uses the source flow to predict $x_t^S$ from $x_{t+1}^S$ and only scores low when the prediction follows the same per-pixel motion as the source; lower is better. Flicker and Warp together discriminate motion-faithful edits from over-smoothed in-betweens.
- `MUSIQ` is the KonIQ-pretrained MUSIQ score from `pyiqa`; higher is better. `MUSIQ_crop` is restricted to detectable frames so glyph quality is not penalized on frames where the source text is unreadable; the temporal metrics use the full clip because adjacent-frame stability is meaningful even on hard frames.

---

## Axis 3 — Edit locality

We compose a **locality-only prediction** per clip — masked pixels are copied from the source, unmasked pixels come from the submission — and score it against the source video:

$$
\hat f_t^{\mathrm{loc}} = (1 - m_t) \odot \hat f_t + m_t \odot f_t
$$

$$
\{\mathrm{PSNR}, \mathrm{SSIM}, \mathrm{LPIPS}, \mathrm{DreamSim}\}_{\mathrm{loc}} = \{\mathrm{PSNR}, \mathrm{SSIM}, \mathrm{LPIPS}, \mathrm{DreamSim}\}(\hat V^{\mathrm{loc}}, V)
$$

The masked region is identical between $V$ and $\hat V^{\mathrm{loc}}$ by construction, so all four metrics reduce to a measurement of how well the *unedited* region is preserved.

- **PSNR / SSIM / LPIPS** are pixel-level: they penalize even the small pixel-value differences introduced by a full-frame VAE encode-decode round-trip, so a regenerative method can score low on these even when its output is perceptually identical to the source. PSNR is capped at 100 dB (the conventional IQA cap) to keep aggregates and bootstrap means well-defined when MSE = 0. SSIM and PSNR are computed by `skimage.metrics`; LPIPS uses the AlexNet reference implementation.
- **DreamSim** is a learned perceptual similarity metric trained on human "are these images the same?" judgments, which is robust to the codec/VAE noise that pixel-level metrics over-penalize. It tightens the locality reading by giving credit when the unedited region looks identical even if it is not byte-identical.

All four metrics are GPU-batched in our pipeline.

---

## Aggregation and confidence intervals

Per-clip values are averaged with **None entries skipped** (this is how the per-clip exclusion rule shows up in the aggregate). For each aggregate we report the **percentile 95 % bootstrap CI** over **1000 clip-level resamples** with replacement. CIs are written into `eval.json` alongside each metric's mean.

```json
"SeqAcc": { "mean": 0.341, "ci_lo": 0.265, "ci_hi": 0.418, "n": 157 }
```

Bootstrap captures sampling uncertainty over the 157-clip test split; if a CI is wide it tells the reader that the difference between two methods may not be reliable on this much data. The CIs make it easy to judge when method-vs-method comparisons are statistically well-supported.

---

## TextScore (leaderboard sort key)

A public leaderboard needs one column to sort on. Every cross-axis aggregate we considered hid more signal than it revealed: a ranking that mixes text correctness, visual quality, and edit locality lets one axis silently compensate for another, exactly the failure mode the protocol is built to expose. We therefore sort on a single-axis key instead.

$$
\mathrm{TextScore} = \sqrt[3]{\mathrm{SeqAcc} \cdot \mathrm{CharAcc} \cdot \mathrm{TTS}}
$$

All three primitives are natively in $[0, 1]$, so no normalization is applied. The geometric mean across the three text-correctness primitives means $\mathrm{SeqAcc} = 0$ collapses TextScore to zero — the intended semantics for methods that never produce the requested target string. The remaining ten primitives still appear next to TextScore on the leaderboard, so the full thirteen-metric vector is always visible to the reader.

TextScore is the only cross-metric aggregate emitted by the evaluator (`benchmark/text_score.py`). The aggregate is reported on the test-split aggregated text metrics with a 1000-resample clip-level percentile bootstrap CI. Because it covers only one axis, two methods with similar TextScore can still differ substantively on visual quality or edit locality — the per-metric vector remains the primary unit of comparison.
