# site-adv-w10
## Hypothesis
IDEAS.md item 27 (site-adversarial tail), reversal weight 1.0, one rung of the dose ladder 0.1 / 0.3 / 1.0 (`site-adv-w01`, `site-adv-w03`, `site-adv-w10`; same code, commit f1b6a56 cherry-picked).

Folds are deployments, so the hard folds are a site-shift problem. If the fine-tuned tail keeps site identity in its pooled code, a head that predicts which **training** fold a frame came from, behind a gradient-reversal layer, pushes the tail to drop it while the class head keeps buzz. Expected signature of a real effect: `1_95` / `1_114` / `1_150` and the `untagged` / `loud` tiers move up with dose. Falsifier: hard folds flat or down at every dose, or a headline that only moves through `1_29` / `53`'s background tier.

Known risk, stated before the run: site and label are not independent in the training pool (some folds, e.g. the Hard Negatives effort, hold no buzz at all), so an unconditional adversary also pushes against information the class head needs. A loss that grows with dose would point at that, not at "site invariance doesn't help".

Base recipe: `ps-depth8` (the era's best confirmed config): `yamnet_trunk_pitchshift_depth8`, 30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16. Control: `ps-depth8` (0.515) and `ps-depth8-repeat` (0.509), two draws of the identical config without the adversary; medium's annotations have not changed since they ran (annotations.csv 2026-09-25). The scored fold never reaches the adversary: its site index is -1 and its term is masked, so `val_loss` stays the class loss alone.

## Changes
`03_train/train.py`, gated by `TRUNK_ADV=<weight>` (unset: unchanged training):
- each training sample carries its fold's index; the target gains one trailing column holding it (-1 for the validation fold);
- the fit model is the trunk head plus `GradReverse(weight) -> Dense(256, relu) -> Dense(n_training_folds)` off the pooled code, its logits concatenated after the class logits; the loss is the usual weighted BCE on the class columns plus sparse softmax CE on the site columns;
- `site_acc` (the adversary's training-fold accuracy) is logged per epoch as instrumentation;
- after `fit`, scoring, surprisal and everything downstream use the class-logit model only.
