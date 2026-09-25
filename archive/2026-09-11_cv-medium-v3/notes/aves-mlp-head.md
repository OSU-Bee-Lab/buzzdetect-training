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

### In-pipeline CV (cv-baseline head config: Dropout 0.2, no weight decay, val_loss early-stop)

| fold | aves_h0 (linear) | h256 | h1024 |
|---|---|---|---|
| 1_29 (rich)      | 0.064 | 0.013 | 0.193 |
| 53 / Fit+Fast (rich) | 0.274 | 0.224 | 0.197 |
| willard/1_11     | 0.179 | 0.030 | 0.053 |
| 1_150            | 0.000 | 0.000 | 0.007 |
| 1_95             | 0.022 | 0.017 | 0.007 |
| **mean sens@fpr0.005** | **0.108** | **0.057** | **0.091** |

- vs the linear h0 control: h256 -0.051 (0 up / 4 down), h1024 -0.017 (2 up / 3 down).
- All three far below YAMNet cv_baseline (0.218).
- h0 (0.108) is a fair AVES linear-probe draw, same order as aves-probe's logged 0.074.
- MLP best_epoch collapses to 6-40 on most folds vs h0's 18-150: more parameters
  reach the val_loss argmin even sooner, so the MLP is *more* undertrained under
  this stopping rule, not less. This is the aves-readout pathology, amplified.

### Offline converged readout (standardized input, sklearn, run to convergence)

`sweep_mlp.py` — same metric (`03_train/metrics.py:sens_at_fpr`) and same
leave-one-fold-out rotation as the pipeline, sklearn heads, standardized input,
run to convergence. lr = LogisticRegression(lbfgs, balanced); mlp = MLPClassifier
hidden (256,), adam, early_stopping on an internal split.

| readout | AVES | YAMNet |
|---|---|---|
| linear (converged)      | 0.182 | 0.248 |
| MLP(256) alpha 1e-3     | **0.231** | **0.272** |
| MLP(256) alpha 1e-1     | 0.229 | 0.267 |

- A converged non-linear head helps **both** embedders (AVES +0.049, YAMNet
  +0.024) — it is not an AVES-specific fix. Gains land on the three resolvable
  folds (1_29, 53, willard); 1_150 and 1_95 stay near chance under every readout.
- **It does not close the gap.** AVES-MLP 0.231 < YAMNet-linear 0.248 <
  YAMNet-MLP 0.272. The AVES->YAMNet representation gap (~0.04) survives adding
  non-linearity at both ends.
- Incidental (matches aves-readout): converged YAMNet linear offline is 0.248 vs
  cv_baseline's shipped 0.218 — the val_loss stopping rule leaves ~0.03 on the
  table for YAMNet too. Not pursued (era keeps the current stopping rule).

## Conclusion

**In-pipeline: clear negative.** A shared ReLU hidden layer over frozen AVES,
trained under cv-baseline's head config, scores 0.057 (h=256) / 0.091 (h=1024)
vs the linear control's 0.108 -- worse, and far below YAMNet's 0.218.

**But the negative is a training-pathology artifact, not a representation
verdict.** The MLP folds early-stop at epoch 6-40 (vs 18-150 linear): more
parameters hit the val_loss argmin sooner, so the head is *more* undertrained
under a rule tuned for a 768x15 linear probe on a sparse non-negative code
(dropout 0.2, no weight decay). Offline, the same (256,) architecture run to
convergence reaches **0.231 on AVES** (+0.049 over converged linear 0.182) and
**0.272 on YAMNet** (+0.024 over 0.248).

**Decision gate for an AVES fine-tune: not motivated.** The non-linear reader
does extract more from AVES (the PCA-predicted "information is there, spread
thin" holds), but it lifts YAMNet by a similar margin, and AVES-MLP still trails
YAMNet-*linear*. Beating frozen YAMNet by fine-tuning AVES means closing a ~0.04
gap that adding non-linearity does not touch. `trust: caveated` -- the logged
in-pipeline number is deflated by the stopping rule; the offline number is the
real signal but uses a different optimiser and convergence than the pipeline.
