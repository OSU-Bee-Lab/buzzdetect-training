# gain4

## Hypothesis
Multiply the waveform by 4x before YAMNet's front end. YAMNet computes
`log(abs(stft) @ mel + 1e-3)`, so a 4x waveform is a 4x magnitude spectrogram:
a near-constant +log(4) ≈ +1.386 shift on log-mel wherever signal sits above the
1e-3 floor, and a smaller lift on bins near the noise floor. That asymmetry is
the point — quiet buzz content near YAMNet's log_offset floor gets pulled up
more, relative to loud broadband negatives, and the frozen BatchNorm/ReLU stack
turns the shift into a non-linear reshaping of the embedding rather than a pure
bias a linear probe could absorb.

Prior: bandpass and handcrafted-frequency front-end edits are logged dead ends,
but all on YAMNet and none were a pure level change. Weak prior; "dumb one" per
Luke. Cheap: audio cache (sr16000_fl0.96) is shared with yamnet, so only the
embedding pass is new.

## Changes
- New embedder `embedders/yamnet_gain4/` — subclasses `EmbedderYamnet`, scales
  `audio * 4.0` in `embed()`. Nothing else touched.

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | val frames |
|---|---|---|---|---|
| 1_29 (JamesU MustardBumbler) | 0.426 | 0.314 | -0.112 | 7617 |
| Marysville/53 (Lily Fit+Fast) | 0.425 | 0.362 | -0.063 | 7540 |
| 1_150 (Diel Drivers 04-08) | 0.021 | 0.007 | -0.014 | 4947 |
| 1_95 (Diel Drivers 05-06) | 0.037 | 0.055 | +0.018 | 7572 |
| willard/1_11 (One Hive) | 0.180 | 0.210 | +0.030 | 6930 |


- mean sens@fpr0.005: baseline 0.218 -> this 0.190 (-0.028), 2 folds up / 3 down.
- The two confident moves are LOSSES on the two buzz-rich folds (1_29 -0.112,
  Marysville -0.063). The two gains (1_95 +0.018, willard +0.030) are on hard
  folds but within per-fold bootstrap SD (0.010-0.037). 1_150 stays ~0.
- Not a stopping artifact: best_epoch moved only mildly vs cv_baseline
  (1_29 134 vs 162, Marysville 177 vs 174, 1_150 19 vs 26, 1_95 164 vs 147,
  willard 82 vs 65) — a scalar multiply barely reshapes log-mel past the first
  frozen BatchNorm, unlike the input-normalization experiments (6-8x epoch
  blowups). The -0.028 is real, not selection.

## Conclusion
Negative, trust clean. A flat +log(4) lift on log-mel compresses headroom at
the top of YAMNet's range, pushing loud broadband negatives and buzz together
exactly where there is the most buzz to lose. The small hard-fold gains do not
survive the losses and are within noise. Do not rerun a pure gain change; a
level move that *expanded* dynamic range (compression/AGC per frame) is a
different mechanism and is not addressed here.
