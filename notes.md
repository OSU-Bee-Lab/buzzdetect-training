# ps2-seamfree-d8
## Hypothesis
IDEAS #36. `ps2-depth8` added a two-octave-up third view (0.96 s frame resampled 4:1, then tiled 4x) and lost -0.017 ± 0.014 against ps-depth8-r3, worst at 1_114, the fold most sensitive to tile seams (`ps-untiled`, `ps-sharedtiled`). The tiled view also only covers 0.24 s of audio. A seam-free view built from a 4x-longer source window centred on the frame (3.84 s of real neighbouring audio via `context_frames`, which does not leak) resampled 4:1 fills YAMNet's 0.96 s input with one continuous two-octave-up signal: no seams, and the same frame-centred span as the other views at its centre.

Prediction: headline >= +0.02 over ps-depth8-r3 (0.497) and above the three-draw mean (0.507), with `untagged` up and 1_114 not down beyond its SD. Falsifier: below the three-draw mean again, which closes the two-octave line. Risk: the 3.84 s window is a context lever, and context levers have a standing prior of lifting rich folds (1_29/53) and leaving hard folds flat; read 1_95/1_114/1_150 as the falsifier, not the headline.

Control: `ps-depth8-r3`; `ps2-depth8` (0.480) as the tiled comparator.

## Changes
New embedder `yamnet_trunk_pitchshift2sf_depth8` (shared tree, `embedders/`; untracked there; a copy is in `embedder_copy/` on this branch): the ps-depth8 recipe's [plain, octave-up] plus a third view from 4 frames (3.84 s) of real audio centred on the frame, `context_frames = 2`, resampled 2:1 twice (librosa, as `Resample('half')`) to one frame; zero-padded past a recording edge, never tiled. All three views share the depth8 tail. Checked: views 1-2 equal ps-depth8's embed() bit for bit on the same buffer. No ONNX form (research embedder).
`02_set/extract.py`: the context path allocated float32 regardless of the recipe's dtype (2x disk); now keeps it (float16 here; lossless, the values are float16 already). Reported as friction for main.
Run: medium extraction, then ps2-depth8's settings: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512`, 30 epochs. Chain: `exp/binary-d8:chain_b41.sh`.
