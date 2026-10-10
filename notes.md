# site-ent-ramp
## Hypothesis
IDEAS.md item 30(b), the last named candidate in the site-invariance family. Five runs on `ps-depth8` lost the class signal without removing site identity. The two stable ones (`site-conf-w01`, all frames; `site-neg-w01`, non-buzz frames only) used weight 0.1 × KL(uniform || site head's prediction) and flattened the pooled code within two epochs (held-out `ins_buzz` logit SD 0.09-0.10 against the control's 1.2). `site-neg-w01`'s weight-zero control showed the site head, unopposed, at 84% training-fold accuracy: the head is confident, and KL(uniform || p) = -mean_k log p_k - ln n grows without limit as it rules sites out, so from the first step the tail faces a large gradient whose cheapest answer is a flat code.

This run changes the term's shape and its onset, not its target weight: the tail minimises ln(48) - H(p) (the site head's negative entropy, shifted to be 0 at chance; bounded above by ln 48 = 3.87, and its gradient vanishes when the head is certain rather than exploding), and the weight ramps linearly from 0 to 0.1 over the first 10 epochs, so the class head is trained before the term bites. All training frames, as in `site-conf-w01`.

Expected signature if the KL's shape and abrupt onset were the cause: fold 1's sensitivity monitor tracks the control through the ramp (`1_29`: 0.49 at epoch 10) and the logit SD stays in the control's 0.65-1.2. Then, if site invariance helps: `site_acc` falls well below the unopposed 0.84 and `1_95` / `1_114` / `1_150` with the `untagged` / `loud` tiers move up. Falsifiers: (i) sensitivity decays as the ramp climbs and the output collapses again: then no form of this term at weight 0.1 on the pooled code is compatible with the class head, and the family closes; (ii) the class output survives, `site_acc` drops materially, and the hard folds are flat or down; (iii) the class output survives but `site_acc` stays near the unopposed curve: no test at this dose.

Gate, set before the run: `TRUNK_GATE=10:0.3` stops the CV after the first rotation if its sens@fpr0.005 monitor is below 0.3 at epoch 10 (ramp at 0.9 of full weight by then). A tripped gate leaves a one-fold `folds_sx.csv`, which is not a CV headline. A collapse that only arrives after epoch 10 would pass the gate; the fold line's logit SD catches it.

Control: `ps-depth8-repeat` (0.509) and `ps-depth8` (0.515).

## Changes
None beyond `exp/site-neg-w01`'s `03_train/train.py` (`TRUNK_ADV_MODE=entropy`, `TRUNK_ADV_RAMP`, `TRUNK_GATE`), which that branch's smoke test exercised on lite.

Run: `TRUNK_ADV=0.1 TRUNK_ADV_MODE=entropy TRUNK_ADV_RAMP=10 TRUNK_GATE=10:0.3 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium.

## Results
**The gate tripped: the CV stopped after its first rotation (`1_29`).** `folds_sx.csv` holds that one fold, so its `total` row is not a CV headline.

Against `ps-depth8-repeat`, `1_29` only (`tools/results.py`):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.568 | 0.000 | -0.568 | 0.082 | 32 |

`1_29` in this run: untagged 0.001 (801 frames), background 0.000 (1256), loud 0.000 (4), quiet 0.000 (82). Held-out `ins_buzz` logit SD 0.08 (control 1.20); `val_loss` 0.670 at epoch 30 (control 0.67).

Fold-1 per-epoch curves against the ramp (weight during epoch e is 0.1 × (e-1)/10; `test_site_w0` is `site-neg-w01`'s control with the site head attached and no tail term):

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tail weight | 0 | 0.01 | 0.02 | 0.03 | 0.04 | 0.05 | 0.06 | 0.07 | 0.08 | 0.09 | 0.10 | 0.10 |
| sens@fpr0.005, this run | 0.271 | 0.416 | 0.521 | 0.189 | 0.460 | 0.090 | 0.004 | 0.079 | 0.031 | 0.008 | 0.003 | 0.000 |
| sens@fpr0.005, `test_site_w0` | 0.050 | 0.368 | 0.465 | 0.472 | 0.519 | 0.542 | 0.547 | 0.548 | 0.559 | 0.582 | 0.591 | 0.621 |
| site_acc, this run | 0.319 | 0.506 | 0.453 | 0.377 | 0.327 | 0.289 | 0.260 | 0.253 | 0.244 | 0.200 | 0.186 | 0.178 |
| site_acc, `test_site_w0` | 0.282 | 0.531 | 0.628 | 0.678 | 0.706 | 0.725 | 0.740 | 0.750 | 0.761 | 0.768 | 0.775 | 0.841 |

**Interpretation.**
- **The class head trains, then dies as the term ramps in.** Through epoch 3 (weight ≤ 0.02) the run tracks the weight-zero control (0.521 against 0.465). It becomes unstable at weight 0.03-0.04 (0.189, 0.460), is at 0.09 by weight 0.05 and at chance from weight 0.06, and never recovers. This is falsifier (i): the collapse is not an artefact of KL's unbounded shape or of hitting an untrained head at full weight. A term bounded on both sides, entered gradually onto a working class head, flattens the code all the same (logit SD 0.08).
- **A dose-response within one fold, read with care.** Epoch and weight are confounded and this is one fold, one draw, so the tolerated weight is "somewhere around 0.02-0.04", not a measured threshold. But the order of events is clear: `site_acc` is at 0.33-0.38, still 15-18x chance, when sensitivity first breaks, and it ends at 0.18, the same floor as `site-conf-w01` (0.17) and `site-neg-w01` (0.18).
- **So across three shapes of the term the tail gives up class signal before it gives up site.** The site head, unopposed, reads the training fold at 84%; no run in this family has pushed it below ~15% (7x chance), and every run that pushed it below ~35% had already lost the class output.

Not measured: a constant weight below 0.02, which the ramp passes through in two epochs. Epochs 2-3 here are the only evidence about it: at 0.01-0.02 the class head is unharmed and `site_acc` is 0.45-0.51 against the control's 0.53-0.63, so such a run would test a model that still names the site ~22x better than chance, i.e. not site invariance.

## Conclusion
Gated after one rotation, no CV headline: `1_29` 0.000 against 0.568 (-0.568 ± 0.082), every tier of that fold at chance, logit SD 0.08. A bounded entropy term ramped from 0 lets the class head train (0.52 at epoch 3) and then destroys it as the weight passes ~0.03-0.05, with the site head still 15x above chance when sensitivity breaks. With `site-conf-w01` (KL, all frames) and `site-neg-w01` (KL, non-buzz frames) this closes the family: on `ps-depth8`'s pooled code, a confusion term strong enough to move site identity materially removes the class signal first. Site identity is the dominant content of that code (84% unopposed), not a removable nuisance direction.
