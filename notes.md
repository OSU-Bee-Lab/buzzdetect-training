# site-adv-w01
## Hypothesis
IDEAS.md item 27 (site-adversarial tail), first rung of a dose ladder (reversal weight 0.1 / 0.3 / 1.0; `site-adv-w03` and `site-adv-w10` are the other two).

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
| 1_29 | 0.568 | 0.021 | -0.547 | 0.077 | 32 |
| 53 | 0.522 | 0.216 | -0.306 | 0.078 | 28 |
| 1_11 | 0.585 | 0.281 | -0.304 | 0.042 | 26 |
| 1_143 | 0.626 | 0.532 | -0.094 | 0.048 | 22 |
| 1_150 | 0.380 | 0.155 | -0.225 | 0.037 | 21 |
| 1_95 | 0.236 | 0.024 | -0.212 | 0.034 | 46 |
| 1_37 | 0.641 | 0.467 | -0.174 | 0.047 | 14 |
| 1_114 | 0.512 | 0.106 | -0.406 | 0.051 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.509 → this 0.225 (-0.284 ± 0.019)
- inclusive (sensitivity), same thresholds: 0.421 → 0.173 (-0.248)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.819 | 0.653 | -0.166 | 122 |
| untagged | 0.531 | 0.221 | -0.310 | 2418 |
| background | 0.444 | 0.065 | -0.379 | 1874 |
| quiet | 0.126 | 0.054 | -0.072 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Against `ps-depth8` (0.515): mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.515 → this 0.225 (-0.290 ± 0.020)

The dose ladder (headline): weight 0.1 → 0.225, 0.3 → 0.132, 1.0 → 0.126, against 0.515 / 0.509 without the adversary. This rung is weight 0.1 = 0.225.

Adversary and class-loss curves for this run (chance for `site_acc` is 1/48 = 0.021; the no-adversary control's `val_loss` stays within 0.67-0.68 on `1_29` for all 30 epochs):

| fold | site_acc ep1 | min | max | ep30 | val_loss ep1 | max | ep30 |
|---|---|---|---|---|---|---|---|
| 1_29 | 0.139 | 0.014 | 0.358 | 0.093 | 0.69 | 5.88 | 0.71 |
| 53 | 0.136 | 0.025 | 0.166 | 0.087 | 0.66 | 4.82 | 0.69 |
| 1_11 | 0.128 | 0.023 | 0.159 | 0.146 | 0.64 | 49.1 | 0.68 |
| 1_143 | 0.113 | 0.026 | 0.263 | 0.089 | 0.61 | 1.56 | 0.61 |
| 1_150 | 0.120 | 0.026 | 0.140 | 0.081 | 0.69 | 110 | 0.68 |
| 1_95 | 0.128 | 0.028 | 0.166 | 0.166 | 0.76 | 1.83 | 0.83 |
| 1_37 | 0.140 | 0.030 | 0.144 | 0.098 | 0.62 | 34.9 | 0.61 |
| 1_114 | 0.181 | 0.050 | 0.181 | 0.115 | 0.78 | 453 | 0.75 |

**Interpretation.** Every fold fell by many times its SD at every dose, the hard folds (`1_95`, `1_114`, `1_150`) included, and every tier fell (`loud`, `untagged`, `background`; `faint`'s 12 frames are not readable). The falsifier is met as stated, but the curves say what was measured is a diverged optimisation, not site invariance:

- The held-out fold's class loss leaves its 0.6-0.8 basin and peaks between 1.6 and 4.5e2 at weight 0.1, and up to 8e6 at 0.3, before coming back as the learning rate decays. The control's never moves.
- At weight 0.1 the training loss on `1_29`'s rotation climbs from 3.9 (epoch 1) to 64 (epoch 9). A site adversary at chance costs ln 48 = 3.87, so the site term is ~15x past chance: the reversed gradient lets the tail *maximise* an unbounded cross-entropy, and it does so by driving the pooled code to large values that make the adversary confidently wrong (`site_acc` dips below chance, 0.014), not by removing site information.
- The adversary still names the training fold at 4-8x chance at epoch 30 at every dose (0.08-0.17), so the tail never became site-invariant; it was only kicked around. Sensitivity partly recovers late (per-epoch peaks at epochs 25-30 in most folds), consistent with a trunk re-learning after the blow-up rather than one that had converged.

So the ladder does not separate "site invariance doesn't help" from the stated site/label-dependence risk: neither was reached. What it shows is that plain gradient reversal on the pooled code is unstable in this recipe (head LR 2e-4 against a 1e-5 backbone, no warm-up of the reversal weight) from weight 0.1 up.

## Conclusion
Negative as implemented, and uninformative about the hypothesis: unbounded gradient reversal diverges the fine-tuned tail at weights 0.1 / 0.3 / 1.0 (headline 0.225 / 0.132 / 0.126 vs 0.515 / 0.509; all folds and tiers down). The adversary stays 4-8x above chance, so site identity was never removed. `site-conf-w01` reruns the idea with a bounded objective (the tail pushes the adversary's prediction toward uniform instead of maximising its cross-entropy).
