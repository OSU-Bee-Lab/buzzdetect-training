# site-neg-w01
## Hypothesis
IDEAS.md item 30(a), the untested remainder of the site-invariance family. Four runs on `ps-depth8` (`site-adv-w01/w03/w10`, gradient reversal; `site-conf-w01`, KL to uniform at weight 0.1) lost the class signal without removing site identity. `site-conf-w01` trained stably but flattened the pooled code (held-out `ins_buzz` logit SD 0.08-0.20 against the control's 0.65-1.20, headline 0.004). One named candidate cause: site and label are not independent in the pool (the Hard Negatives folds hold no buzz at all), so a site term on every frame also pushes against information the class head needs.

This run changes what the term acts on, not its weight: the tail's confusion term (KL(uniform || site head's prediction), weight 0.1, as `site-conf-w01`) is applied to **non-buzz training frames only**. Buzz frames carry no site term on the tail, so the tail can no longer lower it by erasing what separates buzz from non-buzz. The site head itself still trains on every frame (stop-gradient, its own weights only).

Expected signature if the site/label dependence was what collapsed `site-conf-w01`: the held-out `ins_buzz` logit SD returns to the control's range and fold 1's per-epoch sensitivity tracks the control (`1_29`: 0.49 at epoch 10). Then, if site invariance helps: `site_acc` falls toward chance (1/48 = 0.021) and `1_95` / `1_114` / `1_150` with the `untagged` / `loud` tiers move up. Falsifiers: (i) the class output collapses again, which rules the site/label dependence out as the sole cause and leaves the KL's shape (item 30(b)); (ii) the class output survives, `site_acc` nears chance, and the hard folds are flat or down.

Gate, set before the run (IDEAS.md item 30): stop after the first rotation (`1_29`) if its sens@fpr0.005 monitor is below 0.3 at epoch 10 (control 0.49). Built into the run as `TRUNK_GATE=10:0.3`; a tripped gate leaves a one-fold `folds_sx.csv`, which is not a CV headline.

Control: `ps-depth8-repeat` (0.509) and `ps-depth8` (0.515), the identical recipe without a site head. The scored fold never reaches either site term (site index -1, masked).

## Changes
`03_train/train.py`, on top of `exp/site-conf-w01`'s site code (cherry-picked onto current main):
- `TRUNK_ADV_SCOPE=nonbuzz`: the tail's confusion term is multiplied by (no `ins_buzz` label on the frame); default `all` is `site-conf-w01`'s behaviour.
- `TRUNK_ADV_MODE=entropy` and `TRUNK_ADV_RAMP=<epochs>` (for item 30(b); unused in this run): tail term ln(n) - H(p), bounded on both sides, and a linear weight ramp from 0.
- `TRUNK_GATE=<epoch>:<sens>`: after the first rotation, stop the CV if that fold's sens@fpr0.005 monitor was below `<sens>` at `<epoch>`; summarise the one fold and exit non-zero.

Run: `TRUNK_ADV=0.1 TRUNK_ADV_MODE=confuse TRUNK_ADV_SCOPE=nonbuzz TRUNK_GATE=10:0.3 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium.
