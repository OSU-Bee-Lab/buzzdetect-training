# ps2-seamfree-d8
## Hypothesis
IDEAS #36. `ps2-depth8` added a two-octave-up third view (0.96 s frame resampled 4:1, then tiled 4x) and lost -0.017 ± 0.014 against ps-depth8-r3, worst at 1_114, the fold most sensitive to tile seams (`ps-untiled`, `ps-sharedtiled`). The tiled view also only covers 0.24 s of audio. A seam-free view built from a 4x-longer source window centred on the frame (3.84 s of real neighbouring audio via `context_frames`, which does not leak) resampled 4:1 fills YAMNet's 0.96 s input with one continuous two-octave-up signal: no seams, and the same frame-centred span as the other views at its centre.

Prediction: headline >= +0.02 over ps-depth8-r3 (0.497) and above the three-draw mean (0.507), with `untagged` up and 1_114 not down beyond its SD. Falsifier: below the three-draw mean again, which closes the two-octave line. Risk: the 3.84 s window is a context lever, and context levers have a standing prior of lifting rich folds (1_29/53) and leaving hard folds flat; read 1_95/1_114/1_150 as the falsifier, not the headline.

Control: `ps-depth8-r3`; `ps2-depth8` (0.480) as the tiled comparator.
