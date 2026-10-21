# eq-aug-d8
## Hypothesis
IDEAS #32. Each deployment is one recorder, and recorders differ in frequency response (mic, housing, enclosure). Device-response augmentation is the strongest single lever for unseen recording devices in acoustic scene classification (Morocutti et al. 2023, arXiv 2305.07499; DCASE 2023-24 Task 1). No recorder-response augmentation has run in any era. Training the depth8 tail on a random-EQ copy of half of every training fold's snips should make it rely less on a recorder's spectral colouring and lift held-out deployments.

Prediction: headline >= +0.02 over the matched control `ps-depth8-r3` (0.497; depth8 three-draw mean 0.507), with `untagged` or `loud` up, and the movers being folds whose recorders differ most from the pool. The hard folds' confusers (aircraft at 1_95, trill at 1_114) are not a device effect, so they are not the expected movers. Falsifier: headline within 0.02 of the control and no tier up.

Control: `ps-depth8-r3` (same recipe, same embedder cache, no augmentation).

## Results
8-fold medium CV, depth8 recipe, augment dir `augment_eq3.0_6.0_p0.5_s0` (random tilt ±3 dB/oct + peaking ±6 dB, half of every training fold's snips; ~80k train frames per fold).

Headline sens@fpr0.5% (excl. quiet): **0.489** vs ps-depth8-r3 0.497 (**-0.008 ± 0.015**); -0.018 vs the depth8 three-draw mean 0.507. Inclusive 0.407 vs 0.411. No fold missed fpr 0.005.

| fold | control | eq-aug | delta ± SD |
|---|---|---|---|
| 1_29 | 0.486 | 0.516 | +0.030 ± 0.047 |
| 53 | 0.550 | 0.546 | -0.004 ± 0.050 |
| 1_11 (willard) | 0.573 | 0.591 | +0.018 ± 0.025 |
| 1_143 | 0.586 | 0.640 | +0.054 ± 0.041 |
| 1_150 | 0.435 | 0.315 | -0.120 ± 0.044 |
| 1_95 | 0.284 | 0.222 | -0.062 ± 0.028 |
| 1_37 | 0.648 | 0.643 | -0.005 ± 0.049 |
| 1_114 | 0.414 | 0.435 | +0.021 ± 0.052 |

Tiers: loud +0.005, untagged -0.006, background +0.014, quiet +0.016 (all within noise). Buzz logit SD per fold 0.51-1.21 (no shrinkage; 1_150 lowest at 0.51).

## Conclusion
Falsified: headline within 0.02 of the control and no tier up beyond noise. The SD is eval sampling only; the training-noise floor for a headline delta is ~0.014, so -0.008 is indistinguishable from zero. Per fold (investigation only): 1_150 -0.120 ± 0.044 and 1_95 -0.062 ± 0.028 are the largest moves and both hard folds went down, while 1_143 +0.054 ± 0.041 is the largest gain; no clear recorder pattern. Random parametric EQ at this strength does not help depth8 on held-out deployments; recorder colouring is not what limits the hard folds. Read alongside snr/gain/pitch-aug-d8 (same batch, same recipe).
