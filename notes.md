# context-monitor

## Hypothesis

This era has produced exactly two `clean` gains over `cv_baseline` (0.218), and
neither has been combined with the other:

- **`context-embedder`** (+0.040 → 0.258). `yamnet_context` embeds
  `concat(t-1, t, t+1)` of real neighbouring audio, 3072-d. Its gain is carried
  entirely by the three folds with enough buzz to resolve an effect
  (+0.037 / +0.069 / +0.125 at 305–2144 buzz frames); the two near-chance Diel
  Drivers folds moved -0.007 and -0.023, i.e. nothing.
- **`--monitor val_sens`** (+0.031 → 0.257, `probe-grid`'s L1). Stopping and
  restoring on `val_sens_fpr0.005` instead of `val_loss`. Its gain is carried
  **entirely by `1_150`**, one of those same two near-chance folds
  (0.021 tight over three baseline draws → 0.089 / 0.158).

**The two pay on disjoint folds.** One improves the representation where there
is buzz to see; the other fixes which epoch gets shipped, and only shows up
where the `val_loss` argmin was badly placed. Nothing about either mechanism
routes through the other: `--monitor` changes no features, and the context
stack changes no stopping rule. So the prediction is **additive**, ~0.29, with
the context gain reappearing on the rich folds and the monitor gain on `1_150`.

The falsifier that matters: if the composed run lands at ~0.258 or ~0.257 —
i.e. at whichever component is larger, not their sum — then the two are
competing for the same headroom and at least one of them was really measuring
something about how the *stopping rule interacts with the input scale*, not two
independent effects. A 3072-d input reaches its `val_loss` argmin on a
different curve than a 1024-d one, and `--monitor val_sens` is precisely a
change to which curve is scored.

**Comparator is `models/context_embedder` (0.258), not `cv_baseline`.** One
variable moves: the monitor. The paired per-fold table against
`context_embedder` is the result; the comparison to `cv_baseline` is context
only.

Expected effect (+0.031, if it survives) is above `probe-grid`'s ~0.027
single-run MDE, but not by much — read the per-fold direction, and specifically
whether `1_150` moves, before the headline.

## Changes

Code, both ported unchanged from existing branches, no new logic:

- `embedders/yamnet_context/embedder.py` from `exp/context-embedder`. Not a
  symlink — main has no `yamnet_context` embedder, so it is a real directory on
  this branch. It is `.gitignore`d by `embedders/.gitignore`'s whitelist and
  needs `git add -f`.
- `03_train/{main.py,train.py}` from `exp/probe-grid` (`ProbeConfig` + the
  `--dropout/--label-smoothing/--learning-rate/--weight-decay/--batch-size/`
  `--monitor/--min-delta` flags). Every default is the previously hardcoded
  value, so only `--monitor val_sens` is actually exercised. Checked out by
  path rather than merged: both branches are *behind* main, so a merge would
  revert main's later doc and extractor commits.

No extraction. `02_set/sets/medium/embeddings/yamnet_context` already exists
(982 MB, shared through the symlink). Its ident set matches `yamnet`'s except
for a stale `Luke - External Data Sources/InsectSound1000` that is not in
`folds.csv` at all, so nothing in the run reads it.

Run: `--set medium --embedder yamnet_context --translation general
--monitor val_sens`.

## Results

## Conclusion
