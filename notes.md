# ps12sf-d8
## Hypothesis
Fully seam-free three-view trunk: [plain, octave-up seam-free (1.92 s source), two-octaves-up seam-free (3.84 s source)]. `ps2-seamfree-d8` added the seam-free two-octave view beside the *tiled* octave-up view and gained at 1_114 (+0.256) but lost weakly at 1_150/1_95; `pssf-d8` (same batch) tests the seam-free octave-up view alone. If both shifted views help for different reasons, removing the last tiled view from the three-view model should keep the 1_114 gain and recover the hard-fold losses.

Prediction: headline >= +0.02 over ps-depth8-r3 (0.497) and above ps2-seamfree-d8 (0.509), with 1_114 above the ps-depth8 range (> 0.55) and 1_150/1_95 not below the control beyond their SDs. Falsifier: at or below ps2-seamfree-d8 and pssf-d8, which says the two seam-free views do not compound.

Control: `ps-depth8-r3`; comparators `ps2-seamfree-d8` (+ its repeat `ps2sf-d8-r2`) and `pssf-d8`.

## Changes
New embedder `yamnet_trunk_pitchshift12sf_depth8` (shared tree `embedders/`; copy in `embedder_copy/` on this branch): `context_frames = 2`; plain view from the parent recipe, both shifted views from `yamnet_trunk_pitchshiftsf_depth8.seamfree_view` (span 2 and 4). Checked on a random 12-frame buffer: plain and two-octave views equal `yamnet_trunk_pitchshift2sf_depth8`'s bit for bit, the octave-up view equals `yamnet_trunk_pitchshiftsf_depth8`'s.
Run: medium extraction, then the three-view settings `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512`, 30 epochs. Chain: `exp/pssf-d8:chain_b43.sh`.

## Results
vs ps-depth8-r3 (matched control). Buzz logit SD per fold 0.55-1.33 (no collapse).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.495 | +0.009 | 0.033 | 32 |
| 53 | 0.550 | 0.568 | +0.018 | 0.033 | 28 |
| 1_11 | 0.573 | 0.580 | +0.007 | 0.038 | 26 |
| 1_143 | 0.586 | 0.604 | +0.018 | 0.046 | 22 |
| 1_150 | 0.435 | 0.275 | -0.160 | 0.052 | 21 |
| 1_95 | 0.284 | 0.194 | -0.090 | 0.033 | 46 |
| 1_37 | 0.648 | 0.615 | -0.033 | 0.067 | 14 |
| 1_114 | 0.414 | 0.464 | +0.050 | 0.042 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.474 (-0.023 ± 0.016)
- inclusive (sensitivity), same thresholds: 0.411 → 0.392 (-0.019)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.893 | +0.037 | 122 |
| untagged | 0.524 | 0.500 | -0.024 | 2418 |
| background | 0.406 | 0.420 | +0.014 | 1874 |
| quiet | 0.130 | 0.140 | +0.010 | 627 |
| faint | 0.000 | 0.042 | +0.042 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Interpretation: headline -0.023 ± 0.016, weak negative, the lowest of the batch (pssf-d8 0.478, ps2sf-d8-r2 0.499, ps2-seamfree-d8 0.509). It fails both predictions: below the control, below ps2-seamfree-d8. The loss sits in the hard folds: 1_150 -0.160 ± 0.052 (beyond 3 SD) and 1_95 -0.090 ± 0.033; rich folds flat to weakly up (+0.007 to +0.018, all unsure). 1_114 +0.050 ± 0.042 matches the other seam-free runs (unsure). Tiers: untagged -0.024, background +0.014, loud +0.037, quiet +0.010.

Falsifier met: the two seam-free views don't compound. They make the 1_150/1_95 loss worse. Each run adds more real context (pssf 1.92 s, ps2sf 3.84 s, ps12sf both), and the more context, the larger the 1_150 loss (-0.11, -0.11, -0.16). That fits the context prior: extra seconds of real audio hurt the hard, noisy folds.

## Conclusion
Weak negative (-0.023 ± 0.016): fully seam-free three-view trunk [plain, octave-up 1.92 s, two-octave 3.84 s]. Hard folds fall most: 1_150 -0.160 ± 0.052, 1_95 -0.090 ± 0.033; 1_114 only +0.050 ± 0.042. Tiers: untagged -0.024, loud +0.037. The seam-free views don't compound, and the more real context, the bigger the 1_150 loss. Close the seam-free family.
