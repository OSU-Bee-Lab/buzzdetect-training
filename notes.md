# site-adv-w03
## Hypothesis
IDEAS.md item 27 (site-adversarial tail), reversal weight 0.3, one rung of the dose ladder 0.1 / 0.3 / 1.0 (`site-adv-w01`, `site-adv-w03`, `site-adv-w10`; same code, commit f1b6a56 cherry-picked).

Folds are deployments, so the hard folds are a site-shift problem. If the fine-tuned tail keeps site identity in its pooled code, a head that predicts which **training** fold a frame came from, behind a gradient-reversal layer, pushes the tail to drop it while the class head keeps buzz. Expected signature of a real effect: `1_95` / `1_114` / `1_150` and the `untagged` / `loud` tiers move up with dose. Falsifier: hard folds flat or down at every dose, or a headline that only moves through `1_29` / `53`'s background tier.

Known risk, stated before the run: site and label are not independent in the training pool (some folds, e.g. the Hard Negatives effort, hold no buzz at all), so an unconditional adversary also pushes against information the class head needs. A loss that grows with dose would point at that, not at "site invariance doesn't help".

Base recipe: `ps-depth8` (the era's best confirmed config): `yamnet_trunk_pitchshift_depth8`, 30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16. Control: `ps-depth8` (0.515) and `ps-depth8-repeat` (0.509), two draws of the identical config without the adversary; medium's annotations have not changed since they ran (annotations.csv 2026-09-25). The scored fold never reaches the adversary: its site index is -1 and its term is masked, so `val_loss` stays the class loss alone.

## Changes
`03_train/train.py`, gated by `TRUNK_ADV=<weight>` (unset: unchanged training):
- each training sample carries its fold's index; the target gains one trailing column holding it (-1 for the validation fold);
- the fit model is the trunk head plus `GradReverse(weight) -> Dense(256, relu) -> Dense(n_training_folds)` off the pooled code, its logits concatenated after the class logits; the loss is the usual weighted BCE on the class columns plus sparse softmax CE on the site columns;
- `site_acc` (the adversary's training-fold accuracy) is logged per epoch as instrumentation;
- after `fit`, scoring, surprisal and everything downstream use the class-logit model only.

## Results
Against `ps-depth8-repeat` (0.509):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.568 | 0.000 | -0.568 | 0.082 | 32 |
| 53 | 0.522 | 0.000 | -0.522 | 0.132 | 28 |
| 1_11 | 0.585 | 0.295 | -0.290 | 0.040 | 26 |
| 1_143 | 0.626 | 0.372 | -0.254 | 0.066 | 22 |
| 1_150 | 0.380 | 0.093 | -0.287 | 0.067 | 21 |
| 1_95 | 0.236 | 0.024 | -0.212 | 0.034 | 46 |
| 1_37 | 0.641 | 0.118 | -0.523 | 0.055 | 14 |
| 1_114 | 0.512 | 0.158 | -0.354 | 0.058 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.509 → this 0.132 (-0.377 ± 0.026)
- inclusive (sensitivity), same thresholds: 0.421 → 0.094 (-0.327)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.819 | 0.291 | -0.528 | 122 |
| untagged | 0.531 | 0.121 | -0.410 | 2418 |
| background | 0.444 | 0.000 | -0.444 | 1874 |
| quiet | 0.126 | 0.008 | -0.118 | 627 |
| faint | 0.000 | 0.062 | +0.062 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Against `ps-depth8` (0.515): mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.515 → this 0.132 (-0.383 ± 0.024)

The dose ladder (headline): weight 0.1 → 0.225, 0.3 → 0.132, 1.0 → 0.126, against 0.515 / 0.509 without the adversary. This rung is weight 0.3 = 0.132.

Adversary and class-loss curves for this run (chance for `site_acc` is 1/48 = 0.021; the no-adversary control's `val_loss` stays within 0.67-0.68 on `1_29` for all 30 epochs):

| fold | site_acc ep1 | min | max | ep30 | val_loss ep1 | max | ep30 |
|---|---|---|---|---|---|---|---|
| 1_29 | 0.098 | 0.038 | 0.125 | 0.114 | 0.68 | 4.5e+03 | 0.68 |
| 53 | 0.133 | 0.046 | 0.133 | 0.083 | 0.70 | 6.06e+05 | 0.86 |
| 1_11 | 0.102 | 0.044 | 0.116 | 0.116 | 0.66 | 1.57e+05 | 0.65 |
| 1_143 | 0.090 | 0.015 | 0.149 | 0.074 | 0.66 | 1.04e+04 | 0.61 |
| 1_150 | 0.099 | 0.052 | 0.133 | 0.092 | 0.70 | 4.53e+03 | 0.70 |
| 1_95 | 0.115 | 0.041 | 0.125 | 0.083 | 0.80 | 216 | 0.78 |
| 1_37 | 0.075 | 0.047 | 0.138 | 0.111 | 0.63 | 593 | 0.62 |
| 1_114 | 0.106 | 0.051 | 0.106 | 0.094 | 0.82 | 8.25e+06 | 0.83 |

**Interpretation.** Every fold fell by many times its SD at every dose, the hard folds (`1_95`, `1_114`, `1_150`) included, and every tier fell (`loud`, `untagged`, `background`; `faint`'s 12 frames are not readable). The falsifier is met as stated, but the curves say what was measured is a diverged optimisation, not site invariance:

- The held-out fold's class loss leaves its 0.6-0.8 basin and peaks between 1.6 and 4.5e2 at weight 0.1, and up to 8e6 at 0.3, before coming back as the learning rate decays. The control's never moves.
- At weight 0.1 the training loss on `1_29`'s rotation climbs from 3.9 (epoch 1) to 64 (epoch 9). A site adversary at chance costs ln 48 = 3.87, so the site term is ~15x past chance: the reversed gradient lets the tail *maximise* an unbounded cross-entropy, and it does so by driving the pooled code to large values that make the adversary confidently wrong (`site_acc` dips below chance, 0.014), not by removing site information.
- The adversary still names the training fold at 4-8x chance at epoch 30 at every dose (0.08-0.17), so the tail never became site-invariant; it was only kicked around. Sensitivity partly recovers late (per-epoch peaks at epochs 25-30 in most folds), consistent with a trunk re-learning after the blow-up rather than one that had converged.

So the ladder does not separate "site invariance doesn't help" from the stated site/label-dependence risk: neither was reached. What it shows is that plain gradient reversal on the pooled code is unstable in this recipe (head LR 2e-4 against a 1e-5 backbone, no warm-up of the reversal weight) from weight 0.1 up.

## Conclusion
Negative as implemented, and uninformative about the hypothesis: unbounded gradient reversal diverges the fine-tuned tail at weights 0.1 / 0.3 / 1.0 (headline 0.225 / 0.132 / 0.126 vs 0.515 / 0.509; all folds and tiers down). The adversary stays 4-8x above chance, so site identity was never removed. `site-conf-w01` reruns the idea with a bounded objective (the tail pushes the adversary's prediction toward uniform instead of maximising its cross-entropy).
