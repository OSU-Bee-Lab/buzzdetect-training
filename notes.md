# pssf-d8
## Hypothesis
`ps2-seamfree-d8` built its two-octave view from real neighbouring audio instead of tiles and took the trill fold 1_114 from 0.41 to 0.67. The one-octave-up view that every depth8 run relies on (worth -0.079 when removed, `plain-d8`, almost all at 1_114) is still built the old way: the 0.96 s frame resampled 2:1 and tiled 2x, one seam mid-frame, 0.48 s of audio repeated. Seams were already shown to matter at 1_114 (`ps-untiled`/`ps-sharedtiled`, depth12), but only zero-padding or sharing was tried, never real context.

Prediction: replacing the tiled octave-up view with a seam-free one (1.92 s of real audio centred on the frame, resampled 2:1 to one frame) lifts the headline >= +0.02 over ps-depth8-r3 (0.497), with 1_114 up and untagged up, at two-view cost. Falsifier: within ~0.02 of the control with 1_114 flat, which would say the two-octave view's 1_114 gain came from the new pitch, not from seam removal. Risk: a 1.92 s window is a context lever; the standing prior says those lift rich folds (1_29/53) only, so 1_95/1_114/1_150 are the falsifier, not the headline.

Control: `ps-depth8-r3` (same tail, LR, epochs, batch; differs only in how view 2 is built).

## Changes
New embedder `yamnet_trunk_pitchshiftsf_depth8` (shared tree `embedders/`, untracked there; copy in `embedder_copy/` on this branch): [plain, octave-up seam-free], `context_frames = 1`, window `[tL - L/2, tL + 3L/2)`, zero-padded past a recording edge, resampled once with librosa (`Resample('half')`). Its helper `seamfree_view` is shared with `yamnet_trunk_pitchshift12sf_depth8`. Checked on a random 12-frame buffer: the plain view equals `yamnet_trunk_pitchshift2sf_depth8`'s bit for bit.
Run: medium extraction, then ps-depth8's settings `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, 30 epochs. Chain: `chain_b43.sh`.
