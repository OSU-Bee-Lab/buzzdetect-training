# ps12sf-d8
## Hypothesis
Fully seam-free three-view trunk: [plain, octave-up seam-free (1.92 s source), two-octaves-up seam-free (3.84 s source)]. `ps2-seamfree-d8` added the seam-free two-octave view beside the *tiled* octave-up view and gained at 1_114 (+0.256) but lost weakly at 1_150/1_95; `pssf-d8` (same batch) tests the seam-free octave-up view alone. If both shifted views help for different reasons, removing the last tiled view from the three-view model should keep the 1_114 gain and recover the hard-fold losses.

Prediction: headline >= +0.02 over ps-depth8-r3 (0.497) and above ps2-seamfree-d8 (0.509), with 1_114 above the ps-depth8 range (> 0.55) and 1_150/1_95 not below the control beyond their SDs. Falsifier: at or below ps2-seamfree-d8 and pssf-d8, which says the two seam-free views do not compound.

Control: `ps-depth8-r3`; comparators `ps2-seamfree-d8` (+ its repeat `ps2sf-d8-r2`) and `pssf-d8`.

## Changes
New embedder `yamnet_trunk_pitchshift12sf_depth8` (shared tree `embedders/`; copy in `embedder_copy/` on this branch): `context_frames = 2`; plain view from the parent recipe, both shifted views from `yamnet_trunk_pitchshiftsf_depth8.seamfree_view` (span 2 and 4). Checked on a random 12-frame buffer: plain and two-octave views equal `yamnet_trunk_pitchshift2sf_depth8`'s bit for bit, the octave-up view equals `yamnet_trunk_pitchshiftsf_depth8`'s.
Run: medium extraction, then the three-view settings `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512`, 30 epochs. Chain: `exp/pssf-d8:chain_b43.sh`.
