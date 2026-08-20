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

<filled in after the CV run>

## Conclusion

<filled in after the CV run>
