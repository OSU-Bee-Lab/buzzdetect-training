# binary-d8
## Hypothesis
On a frozen probe, collapsing the 14 non-buzz classes into one `nonbuzz` target was null once ins_buzz's loss weight was pinned (`binary-control`, E3, -0.006). With a fine-tuned depth8 tail the question is different: the auxiliary classes now shape shared features in seven trainable blocks, not just a readout. Two opposite mechanisms are live. (a) Capacity spent separating trill from plane from rain is capacity not spent on the buzz boundary, so binary lifts the headline. (b) The named confuser classes (`ins_trill` at 1_114, `mech_plane` at 1_95) supervise exactly the features that keep those negatives below threshold, so binary hurts the hard folds.

Prediction (for (a)): headline >= +0.02 over the matched control `ps-depth8-r3` (0.497; depth8 three-draw mean 0.507) with `untagged` up. Falsifier: within 0.02, or 1_114/1_95 down beyond their SDs, which would read as (b).

Control: `ps-depth8-r3` (same recipe and cache, `--translation general`).
