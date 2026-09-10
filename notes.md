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

### Headline (4th baseline draw, instrumentation run)
| fold | cv_baseline | selcurves | delta | buzz frames |
|---|---|---|---|---|
| 1_29 (MustardBumbler) | 0.426 | 0.442 | +0.016 | 2144 |
| 53 (Fit+Fast) | 0.425 | 0.409 | -0.016 | 1031 |
| 1_11 (willard) | 0.180 | 0.194 | +0.014 | 305 |
| 1_150 (Diel Drivers 04-08) | 0.021 | 0.048 | +0.027 | 146 |
| 1_95 (Diel Drivers 05-06) | 0.037 | 0.037 | 0.000 | 433 |

- mean sens@fpr0.005: baseline 0.218 -> this 0.226 (+0.008), 3 up / 1 down / 1 flat.
  A fair draw: baseline is now 0.208 / 0.218 / 0.227 / 0.226 over four runs, mean
  0.2197. No drift; nothing to read into the +0.008.

### The curves, first read (offline, `models/selcurves`)
Each fold now persists `val_auc_buzz_curve` and `val_ce_buzz_curve` in
`summary.json` next to `val_sens_fpr0.005_curve` and `val_loss_curve`.

**Cross-fold epoch selection (non-leaking), 5-fold mean sens@fpr0.005, by which
statistic drives the pooled other-folds curve:**

| selector | truncated | untruncated |
|---|---|---|
| val_loss (15-class, the incumbent) | 0.222 | 0.236 |
| sens@fpr0.005 (buzz-only, tail) | 0.221 | 0.233 |
| auc_buzz (buzz-only, stable) | 0.223 | 0.236 |
| ce_buzz (buzz-only) | 0.223 | 0.236 |

own-peak leaky upper bound 0.237; shipped (val_loss + RestoreTrueBest) 0.226.

**All four selectors return the same per-fold sensitivities** — `[0.20, 0.41,
0.38, 0.04, 0.09]` — not just the same mean. The choice of selection statistic
is not a lever on this draw. Reason, and it is `xfold-epoch`'s: the pooled sens
curve is flat over e129-164 (max 0.443), so any selector that lands anywhere in
that plateau scores identically. A lower-variance buzz-only statistic can only
pay when the argmax is contested, and here it is not.

This fires the hypothesis's own falsifier. It also independently reproduces
`xfold-epoch`: early stopping was healthy on 3 folds (best 132/132/163) and
broke on 2 (1_11 e54, 1_150 e21); the cross-fold rule lifts 1_150 0.048 -> 0.09
and is flat-to-negative on the healthy folds.

### Conclusion
Instrumentation adopted (curves are cheap, survive data revisions, recompute
free). The buzz-only-selector lead is **closed, negative**: its premise (AUC
much lower variance) already failed the bootstrap in IDEAS.md, and the
across-epoch check it asked for now shows all three candidate statistics — and
the incumbent val_loss — are interchangeable as epoch selectors while the pooled
sens plateau is wide. The open epoch-selection question is `xfold-epoch`'s
(land in the plateau, repair stopping failures), and that is settled and
deferred, not a monitor choice. n=1, but the cross-statistic identity of the
per-fold results is structural (shared plateau), not stochastic.
