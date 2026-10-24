# pssf-d8
## Hypothesis
`ps2-seamfree-d8` built its two-octave view from real neighbouring audio instead of tiles and took the trill fold 1_114 from 0.41 to 0.67. The one-octave-up view that every depth8 run relies on (worth -0.079 when removed, `plain-d8`, almost all at 1_114) is still built the old way: the 0.96 s frame resampled 2:1 and tiled 2x, one seam mid-frame, 0.48 s of audio repeated. Seams were already shown to matter at 1_114 (`ps-untiled`/`ps-sharedtiled`, depth12), but only zero-padding or sharing was tried, never real context.

Prediction: replacing the tiled octave-up view with a seam-free one (1.92 s of real audio centred on the frame, resampled 2:1 to one frame) lifts the headline >= +0.02 over ps-depth8-r3 (0.497), with 1_114 up and untagged up, at two-view cost. Falsifier: within ~0.02 of the control with 1_114 flat, which would say the two-octave view's 1_114 gain came from the new pitch, not from seam removal. Risk: a 1.92 s window is a context lever; the standing prior says those lift rich folds (1_29/53) only, so 1_95/1_114/1_150 are the falsifier, not the headline.

Control: `ps-depth8-r3` (same tail, LR, epochs, batch; differs only in how view 2 is built).

## Changes
New embedder `yamnet_trunk_pitchshiftsf_depth8` (shared tree `embedders/`, untracked there; copy in `embedder_copy/` on this branch): [plain, octave-up seam-free], `context_frames = 1`, window `[tL - L/2, tL + 3L/2)`, zero-padded past a recording edge, resampled once with librosa (`Resample('half')`). Its helper `seamfree_view` is shared with `yamnet_trunk_pitchshift12sf_depth8`. Checked on a random 12-frame buffer: the plain view equals `yamnet_trunk_pitchshift2sf_depth8`'s bit for bit.
Run: medium extraction, then ps-depth8's settings `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, 30 epochs. Chain: `chain_b43.sh`.

## Results
vs ps-depth8-r3 (matched control). Buzz logit SD per fold 0.58-1.26 (no collapse).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.525 | +0.039 | 0.031 | 32 |
| 53 | 0.550 | 0.490 | -0.060 | 0.054 | 28 |
| 1_11 | 0.573 | 0.585 | +0.012 | 0.046 | 26 |
| 1_143 | 0.586 | 0.596 | +0.010 | 0.041 | 22 |
| 1_150 | 0.435 | 0.324 | -0.111 | 0.049 | 21 |
| 1_95 | 0.284 | 0.240 | -0.044 | 0.029 | 46 |
| 1_37 | 0.648 | 0.597 | -0.051 | 0.048 | 14 |
| 1_114 | 0.414 | 0.468 | +0.054 | 0.036 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.478 (-0.019 ± 0.015)
- inclusive (sensitivity), same thresholds: 0.411 → 0.396 (-0.015)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.884 | +0.028 | 122 |
| untagged | 0.524 | 0.503 | -0.021 | 2418 |
| background | 0.406 | 0.395 | -0.011 | 1874 |
| quiet | 0.130 | 0.130 | +0.000 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Interpretation: headline -0.019 ± 0.015, weak evidence of a loss, about the size of the combined noise floor (~0.014 incl. training). Below the prediction (>= +0.02) and below ps2-seamfree-d8's two draws (0.509, 0.499). 1_114 +0.054 ± 0.036 is the same weak rise every seam-free run shows here (0.465-0.468 in this batch), not the 0.67 of ps2-seamfree-d8's first draw. Losses: 1_150 -0.111 ± 0.049 (identical to ps2sf-d8-r2's), 1_95 -0.044 ± 0.029, 53 -0.060 ± 0.054 (unsure), 1_37 -0.051 ± 0.048 (unsure, 14 events). Tiers: untagged -0.021, background -0.011, loud +0.028, quiet flat.

Falsifier partly met: 1_114 is not flat but rises only weakly, and the headline is below the control rather than within 0.02 above. Replacing the tiled octave-up view with 1.92 s of real context doesn't help; if anything the context costs at the hard folds (1_150, 1_95), in line with the standing prior that wider context lifts rich folds at best.

## Conclusion
Weak negative (-0.019 ± 0.015): seam-free octave-up view (1.92 s source) in place of the tiled one. 1_114 rises weakly (+0.054 ± 0.036) as in every seam-free run; 1_150 (-0.111 ± 0.049) and 1_95 (-0.044 ± 0.029) fall, the same pattern as ps2sf-d8-r2. Tiers: untagged -0.021, loud +0.028. Seam removal through real context isn't a lever; the tiled views' seams appear not to be what limits 1_114.
