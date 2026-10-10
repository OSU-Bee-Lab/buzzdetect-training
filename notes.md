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
