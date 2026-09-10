# aves-mlp-head

## Hypothesis

A non-linear (one hidden ReLU layer) head over **frozen AVES** embeddings
recovers a large part of the AVES-vs-YAMNet frozen-probe gap, where a linear
probe cannot.

Motivation (IDEAS.md "Closed: AVES as a drop-in embedder", "What survives
unfreezing", memory `aves-geometry-diagnostic-retracted`):

- AVES's per-dimension buzz separation is *better* than YAMNet's on average
  (mean |Cohen d| 0.242 vs 0.155), but PCA-whitening shows YAMNet keeps 90% of
  its sensitivity in 16 principal directions while AVES keeps 28% and needs
  ~256 to reach 94%. AVES smears the buzz evidence over ~256 directions — the
  case where a linear readout does worst and a non-linear reader should help.
- `aves-probe` logged 0.218 -> 0.074 as a linear frozen probe; `aves-readout`
  showed a converged *linear* offline readout of the same embeddings reaches
  0.194, i.e. ~2.5x of that verdict was the readout/undertraining, not the
  representation.
- This is the cheap decision gate for whether an AVES fine-tune would pay: if
  an MLP closes most of the gap, the information is there and merely needs a
  better reader; if it doesn't, the representation itself needs to move.
- Head depth is explicitly outside the no-unfreezing injunction
  (`injunction-no-long-training-runs`): frozen embedder, ~1 s/epoch.

Counter-evidence: `archive/2026-06_fixed-test/notes/mlp-head-repro.md` is a
clear negative (-0.028, non-overlapping CIs) for Dense(128,relu) — but on
**YAMNet**, whose embedding is already linearly separable for this concept by
construction. The PCA result is why AVES is expected to answer differently.

## Changes

- `--hidden H` flag (cherry-picked verbatim from `exp/shared-trunk-head@4a90546`):
  `Input -> Dropout(0.2) -> [Dense(H,relu) -> Dropout(0.2)] -> Dense(n_classes)`.
  H=0 (default) is the shipped decoupled head, byte-identical path.
- No change to the loss, the stopping rule (`val_loss`, patience 50,
  `RestoreTrueBest`), the optimizer, or label smoothing. One thing changes:
  head depth.
- Embedder: **aves** (frozen, cached, no re-extraction). Translation `general`.

Stopping rule note: this era keeps the `val_loss` early-stopping rule
(`xfold-epoch` was not adopted). Both control and treatment are scored under
it, so the comparison is fair; the shipped h=0 number will understate the
linear readout's converged ceiling (0.194 offline) the same way `aves-probe`
did, which is the point of running a matched h=0 control here rather than
comparing to `aves_probe`'s logged 0.074.

## Runs

- `aves_h0`   — matched linear control (h=0) on AVES
- `aves_h256` — hidden width 256
- `aves_h1024`— hidden width 1024

## Results

| fold | aves_h0 | h256 | h1024 | YAMNet cv_baseline |
|---|---|---|---|---|

- mean sens@fpr0.005: h0 <val> -> h256 <val> / h1024 <val>; YAMNet baseline 0.218

## Conclusion
