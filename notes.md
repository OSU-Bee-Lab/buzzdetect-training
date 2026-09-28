# ps-fast

## Hypothesis
`v4-ft-ps` (fine-tuned depth12 trunk + octave-up view, 0.452 on medium) costs
~2.3x a single-view trunk at inference. `yamnet_trunk_pitchshift_fast_depth12`
builds the up view cheaply: read off the plain view's STFT (every second frame,
mel matrix at 2x the sample rate) and passed through the CNN untiled (48-frame
patch, (3,4,512) map). Untrained prototype: ~1.34x single-view cost instead of
~2.27x. Hypothesis: the cheap up view keeps most of v4-ft-ps's gain over
`v4-ft` (0.375): headline within noise (~0.014) of 0.452. The question is how
much sensitivity, if any, the speed costs. Luke's request, 2026-09-28.

Comparators: `v4-ft-ps` (matched control, same 30-epoch config) and `v4-ft`
(no shift) for how much of the up-view gain survives.

## Changes
New embedder only (committed on main, 97f2c2e). Training config identical to
v4-ft-ps: 30 epochs, TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4,
TRUNK_BATCH=1024, TRUNK_FP16=1, BUZZDETECT_CHUNK_FRAMES=48.

## Results

Against the matched control `v4-ft-ps`:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.499 | 0.542 | +0.043 | 0.031 | 32 |
| 53 | 0.517 | 0.543 | +0.026 | 0.040 | 28 |
| 1_11 | 0.575 | 0.560 | -0.015 | 0.035 | 26 |
| 1_143 | 0.559 | 0.568 | +0.009 | 0.054 | 22 |
| 1_150 | 0.312 | 0.285 | -0.027 | 0.047 | 21 |
| 1_95 | 0.201 | 0.190 | -0.011 | 0.032 | 46 |
| 1_37 | 0.480 | 0.475 | -0.005 | 0.043 | 14 |
| 1_114 | 0.471 | 0.297 | -0.174 | 0.038 | 28 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.452 → this 0.432 (-0.020 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.372 → 0.355 (-0.017)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.843 | 0.786 | -0.057 | 122 |
| untagged | 0.474 | 0.450 | -0.024 | 2418 |
| background | 0.404 | 0.452 | +0.048 | 1874 |
| quiet | 0.110 | 0.109 | -0.001 | 627 |
| faint | 0.000 | 0.150 | +0.150 | 12 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Against `v4-ft` (no shift):

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.375 → this 0.432 (+0.057 ± 0.013)
- inclusive (sensitivity), same thresholds: 0.307 → 0.355 (+0.048)

Headline -0.020 ± 0.014 (eval SD; ~0.016 with training noise) against
v4-ft-ps: about 1.3 SD, not clearly a loss. It is almost entirely one fold:
1_114 -0.174 ± 0.038, well outside its SD, a real regression. The other seven
folds average +0.003, every one within its SD (1_29 +0.043 and 53 +0.026 up,
1_150 -0.027 down; all unsure). Against v4-ft (no shift) it keeps +0.057 ±
0.013 of v4-ft-ps's +0.077, with every fold level or up, 1_114 included
(+0.092). Tiers vs v4-ft-ps: loud -0.057 (122 frames, thin), untagged -0.024,
background +0.048, quiet flat.

1_114 has been sensitive to pitch-shift details before: in v3,
`trunk-ft-pitchshift` moved it down relative to `pitchshift-contrast`'s clean
1_114 gain (see the yamnet_trunk_pitchshift_depth12 docstring). What the cheap view
changes is frequency resolution (25 ms window on the original audio where the
resampled view had 50 ms) and the tiled 96-frame context; which of the two
1_114 depends on is untested.

## Conclusion
The cheap up view (shared STFT, untiled) runs ~1.7x faster than v4-ft-ps's
(prototype: 1.34x single-view cost vs 2.27x) and keeps ~75% of its gain over
v4-ft; the loss is concentrated in 1_114 (-0.174). Not a drop-in replacement
on accuracy; a speed/accuracy trade Luke decides. Next if pursued: isolate
which change costs 1_114 (shared-STFT tiled = prototype P1, vs untiled alone).
