# yamnet-pitchshift-down

## Hypothesis

Luke's proposal: the mirror direction of `yamnet_pitchshift`'s x2 up-shift
rung. Whether shifting bee-buzz harmonics *down* an octave helps or hurts
wasn't assumed either way going in -- the recordings carry real content on
both sides (higher-frequency environmental noise some of the time, clearer
low-frequency content other times).

## Changes

New `embedders/yamnet_pitchshift_down/embedder.py`. Declares
`samplerate = 32000` (not the pipeline's usual 16000) so `02_set/extract.py`
resamples directly from the native 44.1kHz audio -- reaching real content up
to 16kHz that the plain `yamnet`/`yamnet_pitchshift` pipelines never see (they
are capped at 8kHz by their own 16kHz declaration). Unshifted block: the full
32kHz-rate frame downsampled to a true 16kHz representation of the whole
0.96s. Down-shifted block: the first 15360 samples of the native 32kHz frame
(0.48s of real time) fed to YAMNet as-is, no resample call -- same sample
count, shorter real span, YAMNet told it's the full 0.96s, halving perceived
pitch. 2048-d total (unshifted + down-shifted), same frame grid as plain
`yamnet`. `--fixed-epochs 400`, bare linear probe, `--translation general`,
compared against `cv_baseline_v3` (this is a standalone new representation,
not concatenated onto the x2/x4 cache).

**Deployability audited and confirmed** (see the audit note at the top of
this file from earlier in the session): the declared `samplerate=32000` is
honored end-to-end by the real inference path
(`tools/export_onnx.py` -> `config_model.json` -> `engine/src/stream/worker.py`'s
soxr resample), not just by extraction. `embed()` is per-frame and
self-contained; nothing reads outside the frame.

## Results

| fold | baseline (cv_baseline_v3) | down (this) | delta | buzz events |
|---|---|---|---|---|
| 1_150 | 0.266 | 0.352 | +0.086 | 14 |
| 1_37 | 0.369 | 0.424 | +0.055 | 5 |
| willard | 0.368 | 0.404 | +0.036 | 12 |
| 1_29 | 0.441 | 0.452 | +0.011 | 13 |
| 53 | 0.429 | 0.438 | +0.009 | 11 |
| 1_114 | 0.250 | 0.241 | -0.009 | 13 |
| 1_143 | 0.461 | 0.450 | -0.011 | 9 |
| 1_95 | 0.052 | 0.031 | -0.021 | 23 |

mean sens@fpr0.005: cv_baseline_v3 0.330 -> yamnet_pitchshift_down 0.349
(+0.019), inclusive 0.269 -> 0.285 (+0.016). 5/8 folds up, 3 down.
`tools/eval_sampling_sd.py` could not run against `cv_baseline_v3` -- that
model predates `predictions.csv` retention, so no per-fold SD is available
for this comparison. Read the deltas as investigation-grade, per the standing
~0.09-per-fold resolution, not as SD-backed.

Tiers, vs. cv_baseline_v3's own breakdown (loud 0.778, untagged 0.350,
background 0.322, quiet 0.070, faint 0.000): loud 0.795 (+0.017), untagged
0.368 (+0.018), background 0.336 (+0.014), quiet 0.076 (+0.006), faint 0.000
(flat). **All tiers move the same small amount, including loud/untagged** --
not a quiet-buzz-only shuffle.

## Conclusion

Small, broadly positive, and inside/near the ~0.027 headline MDE at n=1:
**+0.019, weak evidence for a real (if modest) gain**, not a null result and
not a confident win either -- exactly the "unsure, report the magnitude"
case LOOP.md asks for. The notable mover is `1_150` (+0.086), one of the
era's two named hard folds, alongside `1_37` and `willard`; `1_95` (the other
hard fold) is flat-to-down (-0.021), consistent with `1_95` failing on false
positives (a jet flyover) rather than on missed detection, a mechanism this
lever doesn't obviously address. No seed control exists; a repeat draw would
be the natural next step before trusting the `1_150` movement specifically.

Shipped model not trained (`--skip-cv` not run). Deployment already confirmed
safe; nothing further needed there.
