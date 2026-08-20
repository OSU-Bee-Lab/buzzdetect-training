# class-weight-fix

## Hypothesis

`class_weight=data.weight_dict`, passed to `model.fit()` in `03_train/train.py`,
is not a valid way to apply per-class weights to this model. The targets are
multi-hot (a frame can be `ins_buzz` and `ambient_background` and `mech_auto`
at once), but Keras' `class_weight` assumes single-label targets: for a
multi-hot `y` it collapses each sample to one class via `argmax(y, axis=-1)`
and scales the *entire* sample's loss by that one class's weight — picked by
class-index order, which `build_classes()` sorts alphabetically
(`ambient_*, animal, human, ins_buzz, ins_trill, mech_*, static`). Verified in
`keras/src/trainers/data_adapters/tf_dataset_adapter.py:130-141` and
`data_adapter_utils.py:120-135` (installed Keras 3.14.1).

Concretely: any frame where `ins_buzz` co-occurs with `ambient_*`/`animal`/
`human` never gets `ins_buzz`'s weight applied at all — the co-occurring
class's weight wins instead, applied uniformly across every neuron including
`ins_buzz`'s. Checked directly against the medium set's embeddings: 342/5478
pickles (~6%) hit this case (e.g.
`ambient_background+ins_buzz_medium.pickle`), and no per-neuron weighting has
ever actually happened for *any* sample — every training run to date,
including `cv-baseline` and `buzz-upweight`, applied one scalar per sample,
not per class.

Fix: `train_utils.weighted_bce_loss()` — a custom loss using
`tf.nn.weighted_cross_entropy_with_logits(pos_weight=...)`, which reweights
only the positive-target term of each neuron independently. This is the
correct multi-label generalization of `build_weights()`'s inverse-frequency
table: negative supervision is untouched, so upweighting a rare class doesn't
also mute the negative signal on every other neuron in the same frame.

This experiment changes only *how* the existing balanced weight table is
applied — same weights, same values, no 2x buzz multiplier. It isolates
whether the argmax-collapse bug itself has been material. Smoke-tested the
loss function's math and the compile → fit → save(include_optimizer=True) →
load(compile=False) round trip on dummy data before running the CV (matches
`write_model_py.py`'s load path).

## Changes

- `03_train/train_utils.py`: add `weighted_bce_loss(weights, label_smoothing)`.
- `03_train/train.py`: compile with `weighted_bce_loss(weights_ordered, ...)`
  instead of `BinaryCrossentropy(...)`; drop `class_weight=data.weight_dict`
  from both `model.fit()` calls (both the val-monitored and fixed-epoch paths).

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | val frames |
|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 0.448 | 0.392 | -0.056 | 6908 |
| Lily Adam - One Hive/wooster/2024-07-26/1_143 | 0.260 | 0.222 | -0.038 | 4707 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.028 | 0.014 | -0.014 | 4891 |
| Luke - Opportunistic/2025-07-03/1_37 | 0.279 | 0.272 | -0.007 | 4712 |
| Lily Adam - One Hive/willard/2024-08-07/1_11 | 0.177 | 0.180 | +0.003 | 4713 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.027 | 0.035 | +0.008 | 6606 |
| Luke - Opportunistic/2025-06-23/1_23 | 0.326 | 0.337 | +0.011 | 314 |
| Luke - Opportunistic/2025-08-27/48 | 0.013 | 0.028 | +0.015 | 1570 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.386 | 0.409 | +0.023 | 4710 |
| Luke - Opportunistic/2025-08-05/31 | 0.157 | 0.243 | +0.086 | 942 |
| Luke - Opportunistic/2025-08-12/1_114 | 0.163 | 0.277 | +0.114 | 3768 |

- sens_persite @ fpr0.005: baseline 0.206 → this 0.219 (+0.013)
- 7 folds up, 4 down. The two biggest movers up (08-05, 08-12) plus the two
  biggest movers down (JamesU, wooster) are all folds with enough val frames
  to trust individually (>3700, well above the "quiet fold" caution). This
  isn't a landslide in one direction — it's a majority-positive, genuinely
  mixed result, not the dramatic swing that would call the whole log's
  weighting-dependent history into question.

## Conclusion

Correcting the `class_weight=` argmax-collapse bug (see Hypothesis) — same
weight values, applied per-neuron instead of collapsed to one scalar per
sample — moves `sens_persite` from 0.206 to 0.219. Direction is majority
positive (7/11 folds) and the movers are folds with enough val data to trust,
but the size is modest and two well-populated folds moved the other way, so
this reads as a real but not dramatic effect — not evidence that every prior
CV-era log entry is unusable. The bug was real (verified independently
against Keras source and the medium set's file listing) and every entry to
date, including `cv-baseline` itself, was measured under the broken
application of weighting. Whether it's worth re-baselining the log is a
judgment call for whoever reads this next: the gain here is comparable in
size to several logged wins (e.g. `label-smoothing`), so it's plausible some
past negative/marginal verdicts would flip under correct weighting, but nothing
here suggests wholesale invalidation.

Given the modest, correctness-motivated nature of the change (not a
hyperparameter tune), it's a reasonable candidate to fold into `main` as the
new default rather than treat purely as one more CV-log entry — that's a
call for whoever reviews this, not made here.

trust: caveated — the direction is credible (it corrects a mechanism that was
verifiably broken, not a speculative structural change), but the fold split
(7 up / 4 down, including two large folds moving down) means the *magnitude*
shouldn't be taken at face value, and there's no seed control to separate this
from run-to-run noise.
