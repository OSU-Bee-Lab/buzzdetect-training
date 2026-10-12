# ps-d8-ens-same5
## Hypothesis
Training arm (IDEAS 30a). `ps-d8-ensemble` found the logit mean of three plain `ps-depth8` draws at 0.558
(members 0.497-0.515) and a five-member mean at 0.575, but the five mixed depths (three depth8, one depth4,
one depth10), so two things are confounded: more members, and members of different depth. Two more plain
`ps-depth8` draws (`ps-depth8-r4`, `ps-depth8-r5`) give a 4- and a 5-draw same-recipe mean.

Prediction: the gain keeps growing with members of one recipe but with diminishing returns, as averaging
partly correlated scores does (negatives correlate 0.62 between draws): the 5-draw same-recipe mean lands
above `ps-d8-ens3` (0.558) and within 0.02 of the mixed-depth five (0.575), i.e. member count, not depth
diversity, is what the mixed five bought.
Falsified two ways: (a) the same-recipe five stays at the three's level (<= 0.558 + ~0.01): the curve is
flat past three and the mixed five's extra +0.017 was depth diversity; (b) it lands more than 0.02 under
the mixed five while above the three: both matter.

Matched controls: `ps-d8-ens3` (0.558; the same recipe at three members) and `ps-d8-ens5` (0.575, mixed
depth, same member count). Dose: every 3-, 4- and 5-subset mean of the five same-recipe draws, since which
three draws went into `ens3` is itself a draw.

New draws: main's recipe unchanged, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`,
`--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general (`chain.sh`).
