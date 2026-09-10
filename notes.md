# buzz-selector-curves

## Hypothesis
Instrumentation only — no training change, stopping rule untouched (val_loss +
RestoreTrueBest). IDEAS.md ("A buzz-only, low-variance selection statistic")
asks for `val_auc_buzz` and `val_ce_buzz` to be logged alongside the existing
`val_sens_fpr*` curve "on the next run", then for the three curves' epoch-wise
signal-to-noise to be compared offline before any monitor swap is considered.

The premise under test is Cawley & Talbot's: for *selecting* an epoch, a
low-variance biased criterion can beat an unbiased high-variance one. The
bootstrap in IDEAS.md already showed AUC's per-frame SD is only ~1.2x smaller
than sens@fpr0.005's and larger on the two hardest folds, so raw SD is not the
argument. What matters is across-epoch SNR: how much the statistic moves as the
model genuinely improves vs how much it jiggles from eval-set sampling. That is
computable for sens@fpr0.005 today (curve persisted) and not for AUC/CE (no
curve). This run adds the curves.

Falsifier for "a buzz-only stable selector would help": if the pooled sens
curve plateau (IDEAS: flat epoch ~145-400) is where all three statistics' argmax
land anyway, a lower-variance selector buys nothing.

## Changes
- `03_train/callbacks.py`: `SensAtFPR.on_epoch_end` also writes `val_auc_buzz`
  (Mann-Whitney U over the buzz logit) and `val_ce_buzz` (buzz-only sigmoid
  cross-entropy) into `logs`, from the same forward pass. No steering.

## Results

## Conclusion
