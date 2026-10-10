# site-conf-w01
## Hypothesis
Follow-up to the `site-adv` ladder (IDEAS.md item 27, weights 0.1 / 0.3 / 1.0 → 0.225 / 0.132 / 0.126 against 0.515 / 0.509). Those runs did not test site invariance: gradient reversal lets the tail *maximise* the adversary's cross-entropy, which has no upper bound. Training loss ran to ~15x a chance-level adversary, the held-out class loss spiked to 1e2-1e6 mid-run, and the adversary still named the training fold at 4-8x chance at epoch 30.

This run keeps the idea and bounds the objective. The site head trains on a stop-gradient copy of the pooled code (ordinary cross-entropy, its own weights only). The tail is trained to make that head's prediction uniform: weight × KL(uniform || p), which is 0 when the adversary is at chance and cannot be driven further, so there is nothing to run away on. Weight 0.1, the least-damaged rung of the ladder.

Expected signature if site invariance helps: `val_loss` stays in the control's 0.6-0.8 basin (the precondition; without it the run is another artifact), `site_acc` falls toward chance (1/48 = 0.021) over training, and `1_95` / `1_114` / `1_150` with the `untagged` / `loud` tiers move up. Falsifier: training is stable, the adversary is pushed near chance, and the hard folds are flat or down. If training is stable but `site_acc` stays well above chance, the weight was too low to test it and the result is "no effect at this dose", not a verdict.

The risk named before the ladder still stands: site and label are not independent in the pool (the Hard Negatives folds hold no buzz), so an unconditional site term also pushes against information the class head uses. A stable run that loses sensitivity while reaching chance would point there, and the conditional form (non-buzz frames only) would be the next step.

Control: `ps-depth8` (0.515) and `ps-depth8-repeat` (0.509), identical config without the site head. The scored fold never reaches either site term (site index -1, masked), so `val_loss` is the class loss alone.

## Changes
On top of `site-adv`'s commit f1b6a56 (`TRUNK_ADV=<weight>`), `03_train/train.py` gains `TRUNK_ADV_MODE=confuse` (default `reverse`, the ladder's behaviour):
- `_SiteConfuse`: one `Dense(256, relu) -> Dense(n_training_folds)` applied twice, on `stop_gradient(code)` with live weights (adversary logits) and on the code with the weights stopped (confusion logits);
- loss = class BCE + site CE on the adversary logits + `TRUNK_ADV` × KL(uniform || softmax(confusion logits)), both site terms masked on the validation fold;
- `site_acc` reads the adversary logits, as before.

## Results
Against `ps-depth8-repeat` (0.509):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.568 | 0.002 | -0.566 | 0.082 | 32 |
| 53 | 0.522 | 0.016 | -0.506 | 0.126 | 28 |
| 1_11 | 0.585 | 0.005 | -0.580 | 0.102 | 26 |
| 1_143 | 0.626 | 0.000 | -0.626 | 0.064 | 22 |
| 1_150 | 0.380 | 0.000 | -0.380 | 0.064 | 21 |
| 1_95 | 0.236 | 0.003 | -0.233 | 0.037 | 46 |
| 1_37 | 0.641 | 0.000 | -0.641 | 0.077 | 14 |
| 1_114 | 0.512 | 0.002 | -0.510 | 0.061 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.509 → this 0.004 (-0.505 ± 0.029)
- inclusive (sensitivity), same thresholds: 0.421 → 0.005 (-0.416)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.819 | 0.021 | -0.798 | 122 |
| untagged | 0.531 | 0.006 | -0.525 | 2418 |
| background | 0.444 | 0.001 | -0.443 | 1874 |
| quiet | 0.126 | 0.008 | -0.118 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Against `ps-depth8` (0.515): mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.515 → this 0.004 (-0.511 ± 0.028)

Curves for this run, with the held-out fold's `ins_buzz` logit spread from `predictions_classes.csv` (chance for `site_acc` is 1/48 = 0.021; control is `ps-depth8-repeat`):

| fold | site_acc ep1 | min | ep30 | val_loss ep1 | max | ep30 | control val_loss ep30 | buzz logit SD | control |
|---|---|---|---|---|---|---|---|---|---|
| 1_29 | 0.278 | 0.121 | 0.169 | 0.68 | 0.76 | 0.68 | 0.67 | 0.10 | 1.20 |
| 53 | 0.261 | 0.139 | 0.150 | 0.69 | 0.77 | 0.68 | 0.66 | 0.20 | 1.09 |
| 1_11 | 0.256 | 0.123 | 0.145 | 0.65 | 0.75 | 0.64 | 0.64 | 0.08 | 0.72 |
| 1_143 | 0.290 | 0.125 | 0.167 | 0.63 | 0.73 | 0.62 | 0.61 | 0.11 | 0.75 |
| 1_150 | 0.266 | 0.126 | 0.191 | 0.70 | 0.79 | 0.69 | 0.66 | 0.11 | 0.65 |
| 1_95 | 0.278 | 0.142 | 0.177 | 0.77 | 0.84 | 0.76 | 0.75 | 0.12 | 0.76 |
| 1_37 | 0.270 | 0.118 | 0.171 | 0.65 | 0.74 | 0.62 | 0.60 | 0.09 | 0.77 |
| 1_114 | 0.262 | 0.115 | 0.170 | 0.77 | 0.82 | 0.77 | 0.74 | 0.11 | 0.87 |

**Interpretation.** The headline is 0.004 at a 0.005 FPR: every fold and every readable tier is at chance, each many SDs below the control. This is not the ladder's divergence and not a readable test either.

- **No divergence.** `val_loss` never leaves 0.62-0.84 in any fold (the ladder peaked at 1e2-1e6), and training loss stays at 3.8-4.0 throughout. The bounded term did what it was built to do on that front.
- **The class output collapsed instead.** The held-out `ins_buzz` logit has an SD of 0.08-0.20 across frames, against 0.65-1.20 for the control: the scored model emits nearly the same value for every frame. Sensitivity is already 0.03-0.05 at epochs 1-2 on `1_29` (control: 0.19, 0.28) and at chance from epoch 3, so the class signal went in the first epochs and never came back. The tail's cheapest way to make a frozen site head's prediction uniform is to flatten the pooled code, and that takes the class information with it.
- **The stated precondition could not see this.** A near-constant output scores 0.62-0.77 on the label-smoothed weighted BCE, within 0.00-0.03 of the trained control's epoch-30 `val_loss`. "`val_loss` stays in the 0.6-0.9 basin" separates divergence from non-divergence, not a working class head from a dead one. The per-epoch sensitivity monitor and the logit SD are the checks that do.
- **Site identity was not removed.** `site_acc` falls from 0.26-0.29 to 0.15-0.19 and stays there: 7-9x chance at epoch 30, on a code that no longer carries usable class signal. So what was lost first was class information, not site information.

Not measured: which term's gradient did the flattening. The two candidates are the KL's steepness when the site head is confident (KL(uniform || p) is bounded below but not above, and grows with every site the head rules out) and the site/label dependence in the pool named in the Hypothesis (Hard Negatives folds hold no buzz). The logged training loss is the sum of all three terms, so this run cannot split them.

## Conclusion
Uninformative about site invariance, for a different reason than the ladder: training is stable, but the tail flattens the pooled code and the class output goes near-constant (headline 0.004 vs 0.509 / 0.515; all folds and tiers at chance; `loud` 0.819 → 0.021, `untagged` 0.531 → 0.006, `background` 0.444 → 0.001). The adversary still names the training fold at 7-9x chance. Four runs on the pooled code (reversal 0.1 / 0.3 / 1.0, confusion 0.1) have now destroyed the class signal without removing site identity. A fifth should change where or on what the site term acts (non-buzz frames only; a term bounded above, such as the site head's negative entropy; a ramped weight), and should gate on the per-epoch sensitivity monitor and the logit SD, not on `val_loss`.
