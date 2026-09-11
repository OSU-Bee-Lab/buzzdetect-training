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
- Per-epoch sens@FPR monitor (reporting only; nothing stops on it) —
  `callbacks.py`, plotted by `plot_history.py::plot_sens_history`. It counts
  every buzz frame, quiet included, so the curves stay comparable with the ones
  already on disk.
- **The stopping rule is `--fixed-epochs`, and it is the default.** Every
  rotation trains exactly `--fixed-epochs` (400) with no early stopping and no
  restore-best, and ships its final weights — no epoch selection of any kind,
  so every arm of a comparison is scored at one identical epoch. This replaced
  `val_loss` early stopping on 2026-09-11. That rule carried no measurable
  *selection* optimism (-0.002 over 17 runs) but **undertrained unevenly**:
  `1_150` hit its `val_loss` argmin at epoch 2-32 under every embedder tried
  while its buzz curve climbed to e120-185, so that fold shipped a
  barely-trained probe. Removing it measured **+0.031 and +0.040** on two
  embedders. Evidence: `archive/2026-09-08_cv-medium-v2/README.md` and
  `exp/pairwise-rank:notes/new-era-audit.md`.

  **400 is provisional.** Every fixed-budget run from the last era is *still
  rising at its cap* — all but one capped at 150, and the one 400-epoch run
  peaked at e234 and cost 0.003 by running to 400. Whether a wide head keeps
  gaining past 250 was never tested. The budget ladder is an early experiment
  of this era, and it sets the number for everything after it.

  `--early-stop` restores the old rule, kept so the archived era reproduces.
  **Never mix the two rules in one comparison.** Each run records which it used
  in `config_model.json`'s `epoch_rule`. A cross-fold epoch rule stays
  available offline via `tools/honest_epoch.py`; on a fixed-budget run every
  fold's curve runs the full length, so that tool's truncation caveat does not
  bind.
- **The head is a bare linear probe — no dropout, no hidden layer.** `--dropout`
  defaults to 0.0. Dropout was 0.2 and hardcoded through the 2026-09 era; it is
  a regulariser tuned on YAMNet's 89.6%-sparse non-negative code, and on a
  dense signed code (AVES, Perch) it is heavy multiplicative noise instead. It
  is now something an experiment turns on, so that the anchor every result is
  read against is the simplest thing that could work.
- **Every sensitivity in `folds_sx.csv` shares that fold's one threshold.**
  Restricting *which positives count* leaves the negative pool and the FPR
  sweep untouched, so the whole family of columns is one operating point scored
  against different targets — directly comparable to each other, row by row.
  - `sensitivity_exclquiet` — **the headline.** Every buzz frame except the
    ones whose buzz is only `_quiet`-tagged.
  - `sensitivity` — the inclusive companion: every annotated buzz frame.
  - `sensitivity_<tier>` for `quiet` / `untagged` / `normal` / `loud`, with
    `<tier>_frames` counts beside them.
- **The loudness tiers are core instrumentation, not a diagnostic to opt into.**
  Every run prints them and every `folds_sx.csv` carries them. They are what
  says whether a hard fold is hard *because* its buzz is faint or hard on
  audible buzz too — different problems with different fixes, which the
  headline alone cannot distinguish. `1_150` and `1_95` sit near chance and
  that question has been open across three eras.
- **`train_utils.buzz_tier` is the definition.** A sample's tier is the
  **maximum** over its buzz labels under `quiet < untagged < normal < loud` — a
  frame is only as hard as its most audible buzz. `untagged` ranks above
  `quiet` so that "quiet" keeps meaning *every* buzz label on the frame was
  marked quiet (an untagged label is unknown, not faint), and below `normal` so
  an unknown never promotes a frame past a known one. Tagging is in progress,
  so the `untagged` bucket is how much of the corpus is not worked up yet; it
  should shrink to nothing. `predictions.csv` carries the per-frame `loudness`
  column, and runs from before 2026-09-11 get NaN in every tier column rather
  than a number pretending the split was made.
- **A `_quiet` buzz still trains, as an ordinary positive.** It is really
  there; calling faint buzz a negative would teach the model that faint buzz is
  background, which is worse than either scoring choice. The tier steers
  scoring only.
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
