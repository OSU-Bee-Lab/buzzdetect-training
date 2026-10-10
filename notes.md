# ps-d8-foldbal
## Hypothesis
The headline is a plain mean over deployments, but the training loss is a plain mean over frames, and deployments differ in size by about 100x (medium: 1 to 120 units of frames per fold, median 17; the eight rotating folds are among the largest and `1_95` is the largest of all). A rotation's pool is therefore dominated by a handful of big deployments, and the fine-tuned tail can spend its capacity on their backgrounds. Group-balanced ERM (each domain reweighted toward equal total weight) is the standard strong baseline for generalising to an unseen domain. So: scale each training frame's loss by its fold's `n_frames ** -0.5`, normalised to mean 1 over the pool; p = 0.5 halves the spread in log terms without handing a 1-unit fold 100x per-frame weight.

This reweights the class loss only. It adds no site head and no term on the pooled code, so it is not one of the ruled-out site-adversarial / site-confusion shapes. Class weights (`build_weights`) and the validation fold's loss are unchanged, so val_loss curves stay comparable.

Expected signature of a real effect: headline above the matched control by more than ~2 SD of a delta (~0.03), with gains on the hard folds (`1_95`, `1_150`, `1_114`) and in `untagged` / `loud`. Falsifiers: headline within its SD of the control; or the big-fold-rich tiers (`background`, `1_29` / `53`) falling while nothing else rises (the weights just took data away).

Control: `ps-depth8-r3` (this batch's redraw from the same main commit), with `ps-depth8` (0.515) and `ps-depth8-repeat` (0.509) as the earlier draws.

## Changes
`03_train/train.py`: `TRUNK_FOLDBAL=<p>` puts each training frame's fold weight in a trailing target column (`_targets`, `_load_data`) and the loss multiplies the per-frame class loss by it. Unset: unchanged.

Run: `TRUNK_FOLDBAL=0.5 TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general.
