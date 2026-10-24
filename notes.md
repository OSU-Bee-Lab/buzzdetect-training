# bn-shrunk-d8
## Hypothesis
IDEAS #37 (c) shrunk BN re-estimation. Run together with (b) `bn-site-d8` from one job; code, chain and models live on `exp/bn-site-d8` (`.local/worktrees/bn-site-d8/`), whose notes.md holds the full rationale and the baseline's side result.

Each held-out fold's tail BN statistics (14 layers) re-estimated on that fold's own frames as in (b), then shrunk toward the pooled training statistics with lam = n_site / (n_site + 6000) (folds hold 5.0-7.6k frames, so lam 0.456-0.558; one value, not tuned), mixture moments. Shrinkage limits the small-sample and flyover problems of (b). Prediction: between (b) and the unadapted `bntrain-d8` if (b) overshoots; above both if site statistics help but are noisy. Falsifier: within 0.02 of bntrain-d8, or a loss.

**Policy caveat: uses the eval deployment's audio at test time (two-pass or streaming per-recording inference in deployment). Luke's call; results surface to him, no design work toward deploying.**

## Changes
None on this branch. `tools_exp/soup.py bn-shrunk-d8 --members models/bntrain-d8 --bn-site 20000 --bn-shrink 6000` on `exp/bn-site-d8`, applied to `bntrain-d8`'s saved fold weights (depth-8 pitch-shift trunk, trainable tail BN, 30 epochs, batch 512 x 2 accumulated because trainable BN at 1024 OOMed the 4 GB card). Model dir: `.local/worktrees/bn-site-d8/models/bn-shrunk-d8`.

## Results
Buzz logit SD 0.72-1.95 (1_114 1.95; no 1_150 blow-up, which reached 7.72 under unshrunk site stats).

`tools/results.py bntrain-d8 bn-shrunk-d8` (same weights, only BN stats differ):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.487 | 0.489 | +0.002 | 0.022 | 32 |
| 53 | 0.530 | 0.334 | -0.196 | 0.063 | 28 |
| 1_11 | 0.554 | 0.535 | -0.019 | 0.039 | 26 |
| 1_143 | 0.631 | 0.622 | -0.009 | 0.039 | 22 |
| 1_150 | 0.312 | 0.259 | -0.053 | 0.054 | 21 |
| 1_95 | 0.188 | 0.191 | +0.003 | 0.025 | 46 |
| 1_37 | 0.600 | 0.586 | -0.014 | 0.035 | 14 |
| 1_114 | 0.271 | 0.504 | +0.233 | 0.053 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.447 → this 0.440 (-0.007 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.364 → 0.366 (+0.002)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.845 | 0.838 | -0.007 | 122 |
| untagged | 0.466 | 0.467 | +0.001 | 2418 |
| background | 0.409 | 0.292 | -0.117 | 1874 |
| quiet | 0.112 | 0.138 | +0.026 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

Headline -0.007 ± 0.015: null. Shrinkage did what it was meant to on the mid folds: 1_29, 1_11, 1_143, 1_37 are back within their SDs of unadapted (unsure), and loud is no longer down (-0.007). But the two big movers of (b) survive almost whole: 1_114 +0.233 ± 0.053 (about 60% of (b)'s +0.380) and 53 -0.196 ± 0.063 (no better than (b)'s -0.174). 1_150 -0.053 ± 0.054 unsure; 1_95 flat. background tier still -0.117 (1874 frames, only 2 folds hold it), quiet +0.026. Neither between nor above: (c) lands below (b) on the headline (-0.007 vs +0.000), both within SD of zero.

## Conclusion
(c) shrunk site BN: null (-0.007 ± 0.015), falsifying both predictions. Shrinkage removed (b)'s small losses on mid folds and its loud-tier loss but kept the two large fold moves (1_114 +0.233 ± 0.053, 53 -0.196 ± 0.063). Whatever the site statistics carry for those two folds is not small-sample noise. Rests on bntrain-d8, which is -0.050 ± 0.017 below the frozen-BN ps-depth8-r3. Test-time adaptation is Luke's call.
