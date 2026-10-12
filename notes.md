# ps-d8-ktails
## Hypothesis
Training arm (IDEAS 30). The logit mean of three separately trained `ps-depth8` draws is 0.558 against
members at 0.497-0.515 (`ps-d8-ensemble`), and the ten same-recipe threes average 0.545 over singles that
average 0.492 (`ps-d8-ens-same5`). Those draws already share everything frozen: the stem (layers 1-7) is the
cached embedding, and every tail starts from the same AudioSet weights. A draw differs from another only in
(i) the class Dense's random init and (ii) the batch order. So "K independent tails on the shared stem, each
with its own data order" is the logged ensemble by construction, and the open question is narrower: **does
the decorrelation need an independent data order, or is the head's init enough?**

This run trains K = 3 tails (YAMNet layers 8-14 + GAP + Dense, each a full `build_head()` model with its own
AdamMultiLR and its own gradient clip) inside one job, **all three on the same batches in the same order**,
and scores the mean of their logits. One job, one data pipeline, one model object (stem -> 3 tails -> mean),
which is the form a deployable graph would take.

Prediction: the tails diverge from the head init alone (each tail's gradient passes through its own Dense),
so the 3-tail mean lands with the same-recipe threes: within 0.02 of their mean (0.545; range 0.535-0.558),
with members at single-draw level (0.464-0.515) and member-to-member correlations near the separate draws'
(negatives 0.62, buzz 0.88).
Falsified if the 3-tail mean lands at a single draw's headline (<= ~0.515, the best single), with members
correlating well above the separate draws: then shared batch order co-adapts the tails and an ensemble needs
separately ordered runs.

Matched controls: `ps-d8-ens3` (0.558) and the ten same-recipe 3-subsets `ps-d8-same3-*` (mean 0.545) for
the ensemble; the five single draws (mean 0.492) for the members. Recipe unchanged:
`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8
--epochs 30`, set medium, translation general. Also timed against a single draw (~2 h 07 min per CV).
