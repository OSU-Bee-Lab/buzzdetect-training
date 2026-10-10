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
