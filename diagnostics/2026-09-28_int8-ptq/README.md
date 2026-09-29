# int8 post-training quantization of the shipped graph — 2026-09-28

IDEAS.md item 26, first pass, on `v4-ft-ps-e60-moderate` (the true best; `yamnet_trunk_pitchshift_depth12`,
moderate set, 60 epochs). Luke, 2026-09-28: quantization first, "worth trying even if it doesn't pay off on this box."

**Result: plain per-channel int8 PTQ is not shippable. It flips too many detections, and speed could not be
measured here.** Nothing retrained, nothing installed into `buzzdetect-train`.

## Parity metric

Detections are called at **logit > 0 on the exported graph**; the per-class centers (`activation_centers`) are
subtracted inside the fp32 head at export, so the exported output is what gets thresholded. The primary parity
number is therefore *detections gained and lost at > 0 versus the fp32 `model.onnx`*, per class, with `ins_buzz`
first. Raw logit error is secondary. (Memory: `detection-parity-metric`.)

Eval audio (`flips.py`): 13,125 frames = the 300 s export fixture (`04_deploy/fixtures/230808_1208_s89520.flac`,
Marysville/53, 43 annotated buzz events: 24 medium, 12 quiet, 6 loud) + 105 moderate-set snips from the 5 rotating
folds (up to 25 buzz-overlapping and 15 other snips per fold, seed 1, ≤120 s each). The fp32 reference has
**1,124 `ins_buzz` detections** in it. Flips count true and false detections alike; they are *not* yet a
sensitivity-at-fixed-FPR change.

## Results (`ins_buzz` at > 0, versus fp32; ref+ = 1,124)

| variant | gained | lost | mean abs logit err | logs |
|---|---|---|---|---|
| fp16 sibling (`model.fp16.onnx`) | 0 | 0 | 0.004 | round2 |
| weights-only int8 (per-channel, symmetric, all convs incl. depthwise) | 4 | 32 | 0.065 | round3b |
| static QDQ, entropy, 12 of the pointwise convs (skip-last variant, see caveat) | 29 | 9 | 0.066 | round3b |
| static QDQ, entropy, 12 of the pointwise convs (skip-first variant, see caveat) | 6 | 23 | 0.056 | round3b |
| static QDQ, entropy, pointwise convs only (u8 = s8, identical) | 36 | 77 | 0.20 | round2 |
| static QDQ, entropy, all trunk convs | 36 | 75 | 0.26 | flips_entropy |
| static QDQ, min/max, all trunk convs | 27 | 276 | 0.61 | flips_minmax |

Other classes move too (all-trunk entropy: `ins_trill` +183/-182 of 4,575; `mech_machinery` +1230/-273 of 11,496;
`mech_plane` 0/-15 of 25). `aambient_scraping` and `mech_quadcopter` sit above 0 on every frame in every graph
(no center recorded for them), so they carry no information here.

Readings, each one measurement deep:
- fp16 flips nothing, so the int8 loss is real quantization error, not numeric noise.
- Weight rounding alone loses ~3% of buzz detections (32/1124). Activation quantization on top raises the loss to
  ~7% of reference positives lost (75/1124; net of the 36 gained, ~3.5%). No calibration method got under the weights-only floor.
- Min/max calibration is unusable. Entropy is far better. Percentile calibration failed to build (numpy cannot
  histogram a near-constant tensor in this graph) and was dropped; with 120 clips x 20 s it also OOM-killed
  (exit 137), which is why entropy/percentile use 40 clips x 10 s.
- Keeping depthwise convs in fp32 barely helped (77 vs 75 lost), so depthwise layers are not the main source.
- The error is spread over the network: quantizing only 12 pointwise layers already flips 23-29.

## Caveats — read before reusing

- **The two `skip` variants are mislabeled.** In `quant.py`, the `pw_skiplastK` / `pw_skipfirstK` step runs
  *before* the pointwise filter and drops the last/first K *convs* of each branch (not pointwise convs). Each built
  graph quantizes 12 pointwise convs, not the 18 intended. Do not read skip-first vs skip-last as "which end of the
  trunk is sensitive."
- **Speed was not measured meaningfully.** This box is an i7-2600 (Sandy Bridge): AVX, no AVX2/FMA/VNNI. onnxruntime
  1.30 ran the first int8 graph at 0.45-0.54x of fp32 (1 and 4 threads, real 10-min clip). That says nothing about a
  modern x86 or ARM machine. Luke offered modern hardware; use it only for a variant that passes on flips.
- **Profile of fp32 `model.onnx` on this CPU, 1 thread, 10 min real audio:** 82x realtime; Conv 71% of kernel time,
  FusedMatMul (the STFT/mel front end) 19%. int8 of the convs cannot touch the 19%, so ~1.9x end-to-end is the
  ceiling from convs going infinitely fast. `model.fp16.onnx` is *slower* than fp32 on CPU (69x); it targets CoreML.
- **Flips are not sensitivity.** A lost detection may be a lost false positive. The next step for any passing
  variant is to score it against annotations at each fold's fixed FPR, and to re-derive `activation_centers` on the
  quantized outputs (a net-negative shift, as in the entropy runs, may be partly recoverable; the random part
  is not).
- One clip sample, one seed, calibration from 40 random moderate snips. Whether the calibration audio's
  distribution matters was not tested.

## Reproducing

The stock quantizer needs a newer `onnx` than `buzzdetect-train` pins (TF pin hazard), so it runs in its own venv:

```bash
uv venv .local/venv-quant --python 3.12 && uv pip install --python .local/venv-quant/bin/python onnx onnxruntime soundfile numpy
export INT8_WORK=.local/int8-ptq; mkdir -p $INT8_WORK
```

`quant.py <work> <minmax|entropy|percentile> <u8|s8> [n_clips] [clip_s] [all|pw|pw_skiplastK|pw_skipfirstK]`
turns `FusedConv(Relu)` back into Conv+Relu (the exported graph also carries two default-domain opset entries, which
the quantizer rejects; it dedupes them), then quantizes the trunk convs to QDQ, per-channel weights, and writes
`<work>/q_<method>_<act>_<mode>.onnx`. Calibration is the whole waveform-in graph on random moderate snips.
`flips.py <ref.onnx> <cand.onnx>...` prints the table above and caches the decoded audio and reference logits in
`$INT8_WORK/flips_cache.pkl`. `wonly.py` builds the weights-only graph; `prof.py`/`cmp.py` time graphs on a 10-minute
clip (`ffmpeg -t 600 -i <recording> -ac 1 -ar 16000 -f f32le $INT8_WORK/real10min.f32`). Long runs go through
`tools/launch_job.sh`; every script here was run that way. `logs/` holds the raw outputs quoted above (the
`round3.log` filter dropped the counts; `round3b.log` is the rerun that has them).

## Not done

Per-layer sensitivity with the corrected skip logic, AdaRound / bias correction / SmoothQuant-style rescaling,
QAT (needs a separate env or a PyTorch route), other calibration pools, the sensitivity-at-FPR and threshold
re-centering evaluation, and any speed measurement on AVX2/VNNI or ARM hardware.
