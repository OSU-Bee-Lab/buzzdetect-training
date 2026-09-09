# 03_train — leave-one-fold-out CV

One training call trains one model per rotating fold and writes the metrics
summary. The shipped model is opt-in (`--train-shipped`, implied by `--skip-cv`).

## Invariants

- **Validation is always a whole fold, never a split within one.** A within-fold
  split leaks site identity into the early-stopping signal. There is no
  snip-level splitter and there should not be one; README explains why.
- **No augmentation may cross a fold boundary.** Every augmented frame must be
  derivable from the source fold's audio alone: noise/volume transforms of one
  fold's frames, or mixes of two frames *from the same fold*. A `CombineSpec` /
  mixup pairing a source frame with a background or partner frame from another
  fold — or from a whole-pool shuffle — is forbidden, however tempting the "buzz
  over a different site's background" framing is. `load_augmented` loads augment
  dirs per training fold, so a cross-fold-paired frame carries an out-of-fold
  recording's embedding signature into the training pool, and when that fold
  rotates in as validation the early-stopping signal and the held-out metrics
  are both contaminated. This inflates the number purely through leakage (the
  models here are extremely sensitive to it) and is not a real gain. The rule
  also bars any augmentation whose parameters (noise floor, SNR target, mix
  ratio, class balance) are fit on statistics pooled across folds rather than
  within the fold being augmented. **If you catch a leak like this in existing
  code, stop and flag it rather than "fixing" it into a better score.**
- **Roles are training-time policy only** (`dataset.py::read_fold_roles`).
  `02_set` embeds every fold regardless, including `exclude`.
- **`folds_sx.csv` is the only metrics summary**, one row per (fold, FPR target)
  plus a `total` row, written by `sx.py::summarize_folds`. It replaced three
  files reporting the same per-fold sensitivity under three different NaN
  policies; `_fold_sens` holds the one that survived. Everything else derives
  from `folds/<fold>/predictions.csv` on demand — don't add a fourth summary
  file, add a column or a function.
- **Reruns resume, silently.** `train_utils.can_write()` skips any model
  directory already holding a `config_model.json`, and the CV summary is
  reassembled from disk so skipped folds still contribute. After a code change,
  use a fresh `--name` or clear the model dir, or you get a mix of both.

## Where to look

- Fold roles, CV loop, shipped-model epoch count — `train.py`, `dataset.py`
- Label translation semantics (`ignore` / `exclude` / missing row) —
  `dataset.py::translate_labels`
- What the headline sens@FPR number means, and the two readings deliberately not
  reported — `sx.py`; its module docstring is the argument, README's "Reading
  the results" is the operator-facing version
- `resummarize.py` rebuilds `folds_sx.csv` for an already-trained model, no
  TensorFlow
- Threshold sweeps (`metrics_by_group`, `metrics_at_fpr`, `metrics_at_precision`)
  — `metrics.py`
- Per-epoch sens@FPR monitor (reporting only; stopping is still on `val_loss`) —
  `callbacks.py`, plotted by `plot_history.py::plot_sens_history`
- Per-frame class activations + multi-label loss for finding bad annotations and
  hard negatives, on by default (`--no-surprisal`) — `surprisal.py`, written to
  `<model>/surprisal/<ident>_surprisal.csv`

## Do not modify

`metrics.py`. `03_train` imports `metrics_by_group` and `metrics_at_fpr` from
it, so editing it changes the metric itself and silently breaks comparability
with every number in the log.

## Smoke-testing a compile change

If a change touches `model.compile()` — a custom loss, a different output layer
— run `tools/smoke_model.py` before spending a CV on it. Its
`smoke_test_loss()` builds the same architecture as `train.py`, fits one step on
dummy data, and runs the exact `save(include_optimizer=True)` →
`load(compile=False)` round trip `write_model_py.py`'s inference path uses, so a
loss that isn't serializable the way that path expects fails in seconds instead
of after a full CV.
