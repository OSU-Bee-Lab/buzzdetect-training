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

## Results
**The gate tripped: the CV stopped after its first rotation (`1_29`).** `folds_sx.csv` holds that one fold, so its `total` row is not a CV headline.

Against `ps-depth8-repeat`, `1_29` only (`tools/results.py`; the tier rows are this run's one fold against the control's eight, so only the fold row is a paired comparison):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.568 | 0.001 | -0.567 | 0.081 | 32 |

`1_29` in this run: untagged 0.004 (801 frames), background 0.000 (1256), loud 0.000 (4), quiet 0.000 (82).

Fold-1 curves, with a plumbing control run for this experiment (`test_site_w0`: the same site head attached, `TRUNK_ADV=1e-9`, so the tail's site term is switched off and everything else is identical; `--only-folds 1_29`):

| run (`1_29`) | sens@fpr0.005 ep1 / ep2 / ep10 / ep30 | site_acc ep1 / ep4 / ep30 | val_loss ep30 | buzz logit SD | final sens (excl. quiet) |
|---|---|---|---|---|---|
| `site-neg-w01` (non-buzz frames, KL, 0.1) | 0.074 / 0.007 / 0.000 / 0.001 | 0.316 / 0.177 / 0.183 | 0.664 | 0.09 | 0.001 |
| `site-conf-w01` (all frames, KL, 0.1) | 0.03-0.05 at ep1-2, chance from ep3 | 0.278 / - / 0.169 | 0.68 | 0.10 | 0.002 |
| `test_site_w0` (site head, no tail term) | 0.050 / 0.368 / 0.582 / 0.621 | 0.282 / 0.678 / 0.841 | 0.674 | 1.22 | 0.639 |
| `ps-depth8-repeat` (no site head) | 0.19 / 0.27 / 0.49 / - | - | 0.67 | 1.20 | 0.568 |

**Interpretation.**
- **Restricting the term to non-buzz frames changes nothing.** The class output collapses exactly as in `site-conf-w01`: sensitivity at chance from epoch 2, held-out `ins_buzz` logit SD 0.09 (control 1.20), `val_loss` in the normal basin (0.664). This is falsifier (i): the site/label dependence in the pool is not what flattened the code, or not alone.
- **The plumbing is not the cause.** With the same site head attached and the tail's term effectively off, `1_29` trains normally (0.639 against the control's 0.568, +0.071 ± ~0.08, one fold: within its SD). The collapse belongs to the tail's site term at weight 0.1, not to the wider fit model, the extra output columns or the optimizer.
- **Site identity is very strongly present in the pooled code.** Unopposed, the site head names the training fold 84% of the time by epoch 30 (chance 1/48 = 2.1%) and is still rising. This is the first measurement of what the term is up against: the earlier runs only showed the adversary under attack (0.15-0.19).
- **The term still does not reach chance.** With the term on, `site_acc` falls to 0.15-0.2 within four epochs and stays there (7-9x chance) while the class signal is already gone. So a tail that pays for the KL by flattening the code loses class information first and site information second, on buzz-free frames as much as on all frames.

Not measured: whether a term bounded above (entropy) or ramped in from 0 avoids the collapse (item 30(b), the next run); and whether any weight exists at which `site_acc` drops materially while the logit SD stays in the control's range.

## Conclusion
Gated after one rotation, so no CV headline: `1_29` 0.001 against 0.568 (-0.567 ± 0.081), every tier of that fold at chance, logit SD 0.09. Applying the KL site term to non-buzz frames only collapses the class output just as it did on all frames, so the site/label dependence is not the (sole) cause. A weight-zero control with the site head attached trains normally (`1_29` 0.639), which clears the plumbing, and shows the unopposed site head at 84% training-fold accuracy: site identity is a dominant feature of the pooled code, and the tail gives up class signal before it gives up site.
