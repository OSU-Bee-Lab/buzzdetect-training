# ps2-seamfree-d8
## Hypothesis
IDEAS #36. `ps2-depth8` added a two-octave-up third view (0.96 s frame resampled 4:1, then tiled 4x) and lost -0.017 ± 0.014 against ps-depth8-r3, worst at 1_114, the fold most sensitive to tile seams (`ps-untiled`, `ps-sharedtiled`). The tiled view also only covers 0.24 s of audio. A seam-free view built from a 4x-longer source window centred on the frame (3.84 s of real neighbouring audio via `context_frames`, which does not leak) resampled 4:1 fills YAMNet's 0.96 s input with one continuous two-octave-up signal: no seams, and the same frame-centred span as the other views at its centre.

Prediction: headline >= +0.02 over ps-depth8-r3 (0.497) and above the three-draw mean (0.507), with `untagged` up and 1_114 not down beyond its SD. Falsifier: below the three-draw mean again, which closes the two-octave line. Risk: the 3.84 s window is a context lever, and context levers have a standing prior of lifting rich folds (1_29/53) and leaving hard folds flat; read 1_95/1_114/1_150 as the falsifier, not the headline.

Control: `ps-depth8-r3`; `ps2-depth8` (0.480) as the tiled comparator.

## Changes
New embedder `yamnet_trunk_pitchshift2sf_depth8` (shared tree, `embedders/`; untracked there; a copy is in `embedder_copy/` on this branch): the ps-depth8 recipe's [plain, octave-up] plus a third view from 4 frames (3.84 s) of real audio centred on the frame, `context_frames = 2`, resampled 2:1 twice (librosa, as `Resample('half')`) to one frame; zero-padded past a recording edge, never tiled. All three views share the depth8 tail. Checked: views 1-2 equal ps-depth8's embed() bit for bit on the same buffer. No ONNX form (research embedder).
`02_set/extract.py`: the context path allocated float32 regardless of the recipe's dtype (2x disk); now keeps it (float16 here; lossless, the values are float16 already). Reported as friction for main.
Run: medium extraction, then ps2-depth8's settings: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512`, 30 epochs. Chain: `exp/binary-d8:chain_b41.sh`.

## Results
Every fold's buzz logit SD is 0.62-1.16 (no collapse).

Against the matched control `ps-depth8-r3`:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.468 | -0.018 | 0.033 | 32 |
| 53 | 0.550 | 0.533 | -0.017 | 0.037 | 28 |
| 1_11 | 0.573 | 0.546 | -0.027 | 0.039 | 26 |
| 1_143 | 0.586 | 0.659 | +0.073 | 0.043 | 22 |
| 1_150 | 0.435 | 0.343 | -0.092 | 0.057 | 21 |
| 1_95 | 0.284 | 0.247 | -0.037 | 0.031 | 46 |
| 1_37 | 0.648 | 0.606 | -0.042 | 0.065 | 14 |
| 1_114 | 0.414 | 0.670 | +0.256 | 0.069 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.509 (+0.012 ± 0.017)
- inclusive (sensitivity), same thresholds: 0.411 → 0.420 (+0.009)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.876 | +0.020 | 122 |
| untagged | 0.524 | 0.539 | +0.015 | 2418 |
| background | 0.406 | 0.380 | -0.026 | 1874 |
| quiet | 0.130 | 0.142 | +0.012 | 627 |
| faint | 0.000 | 0.042 | +0.042 | 12 |

Against the tiled two-octave view `ps2-depth8` (0.480): 0.509, +0.029 ± 0.018. Per fold nearly all of it is 1_114, +0.366 ± 0.076; 1_29 -0.095 ± 0.043, 1_143 +0.046 ± 0.038, the rest within their SD.

Interpretation. The headline is +0.012 ± 0.017 (eval sampling; ~±0.02 with training noise): no detectable change, and it lands at the depth8 three-draw mean (0.507), short of the predicted +0.02 but not below it, so the falsifier doesn't fire. One fold moved a long way. 1_114 went to 0.670 (+0.256 ± 0.069). The highest of 11 earlier depth8-family runs was 0.512, so this is well outside the training-noise range. And 1_114 is the fold where the tiled view lost the most. So removing the seams fixed the 1_114 loss from the tiled view, and more besides. The cost lands on other hard folds: 1_150 -0.092 ± 0.057 (weak; 0.343 is at the low end of the depth8 draws' 0.30-0.46) and 1_95 -0.037 ± 0.031 (unsure). The rich folds 1_29, 53 and 1_11 are each down ~0.02, unsure. 1_143 +0.073 ± 0.043, weak. Tiers: untagged +0.015 and loud +0.020 are up slightly, and background is down -0.026. This is the opposite of the context-lever prior: the gain sits on a hard fold, not the rich ones.

## Conclusion
Headline flat (+0.012 ± 0.017 vs ps-depth8-r3; +0.029 ± 0.018 vs tiled ps2-depth8). One large hard-fold gain: 1_114 0.670 (+0.256 ± 0.069), above every earlier depth8 run (max 0.512), offset by weak losses on 1_150 (-0.092 ± 0.057) and 1_95 (-0.037 ± 0.031). The seam-free two-octave line is not closed. Per docs/judging-results.md, a large gain needs one repeat draw to check that 1_114 holds before anyone builds on it. A repeat would also show whether the 1_150 loss is real. Tiers: untagged/loud up slightly, background down.
