# ps2sf-d8-r2
## Hypothesis
Repeat draw of `ps2-seamfree-d8` (0.509; +0.012 ± 0.017 vs ps-depth8-r3), whose conclusion asked for one before anyone builds on it. Its one large move was the trill fold 1_114 at 0.670 (+0.256 ± 0.069), above every earlier depth8-family run (max 0.512), offset by weak losses at 1_150 (-0.092 ± 0.057) and 1_95 (-0.037 ± 0.031).

Prediction: 1_114 holds above 0.55 (i.e. well above the 0.41-0.51 range of the two-view ps-depth8 draws), and the headline lands within ~0.02 of 0.509. Falsifier: 1_114 back inside the ps-depth8 range, which would make the first draw's 1_114 a lucky draw. The 1_150 loss is read the same way: real if it repeats.

Control: `ps-depth8-r3` (matched; the comparator ps2-seamfree-d8 used), with ps2-seamfree-d8 as the first draw.

## Changes
None: same embedder (`yamnet_trunk_pitchshift2sf_depth8`, cache already extracted) and settings (`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=512 TRUNK_ACCUM=2 SCORE_CHUNK=512`, 30 epochs). Chain: `exp/pssf-d8:chain_b43.sh`.
