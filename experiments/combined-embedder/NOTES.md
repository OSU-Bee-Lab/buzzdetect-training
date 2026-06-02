# combined-embedder (repurposed: full-val-set)

## Note on slug
This worktree was created for yamnet_combined but that embedder had no extracted embeddings. Repurposed for the full-validation-set experiment.

## Hypothesis
Validate on all classes instead of buzz-only during training. Currently `train.py` filters the validation set to only buzz samples (`labels_keep_raw=labels_buzz`), so early stopping monitors validation loss on buzz samples only. This doesn't provide signal about false positive behavior. Validating on all classes should calibrate early stopping against the full distribution, improving precision at high-recall operating points.

## Existing evidence
- Baseline (test_standard, buzz-only validation): 0.2506 sensitivity @ 95% precision
- deeper-head and deeper-std showed MLP head worse than linear probe
- TODO comment in train.py:50 explicitly flags this as open question

## Changes
- `03_train/train.py` line 50: `labels_keep_raw=labels_buzz` → `labels_keep_raw=None`
- Everything else unchanged from baseline (yamnet, standard, general translation, linear probe)

## Reproduction
No external artifacts. Change one line in train.py, retrain.
- Set: standard
- Embedder: yamnet
- Translation: general
- Epochs max: 300

## Results
Metric: sensitivity at 95% precision
- Baseline (test_standard): 0.2506
- This experiment (exp_fullval_std_v1): 0.2545

Additional metrics:
- sensitivity @ 90% precision: 0.2755 (baseline: 0.2763) — slightly worse
- precision @ 80% sensitivity: 0.2574 (baseline: 0.2563) — slight improvement

Small improvement at high-precision operating point; essentially neutral at medium-precision. The full validation set gives the early stopping callback a richer signal that marginally helps precision calibration, but the effect is small.

## Conclusion
Full-distribution validation gives a marginal improvement at 95% precision (+0.4pp) but is roughly neutral elsewhere. Worth using as a default since it's more principled, but not a major lever. Next experiments should focus on richer embeddings (yamnet_combined/yamnet_doublerate when the external drive is available) or data augmentation.
