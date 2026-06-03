# focal-loss-alpha

## Hypothesis
Focal loss with alpha=0.75 (vs. 0.25 in focal-loss experiment) improves sensitivity at 95% precision.

The focal-loss experiment showed that alpha=0.25 penalizes positive (buzz) examples too aggressively:
- 95% precision: 0.2499 (slightly worse than baseline 0.2506)
- 90% precision: 0.2937 (better than baseline 0.2763) — mechanism works, direction wrong

Alpha controls the class balance: alpha=0.25 weights positives at 25%, negatives at 75%.
For a recall-focused task like buzz detection, we want to weight the positive class more, not less.
Alpha=0.75 inverts this: weights positives at 75%, negatives at 25% — should push the PR curve
in the high-precision direction.

## Existing evidence
- focal-loss experiment: alpha=0.25 hurt primary metric, improved at 90% precision
- Combined-embedder: all-class validation marginally better
- deeper-head/deeper-std: linear probe outperforms MLP — loss function is key lever
- Lin et al. 2017: alpha is tuned per dataset; 0.25 was for COCO, not bioacoustics

## Config
- Embedder: yamnet, standard set, general translation, linear probe
- Validation: all classes (labels_keep_raw=None)
- Loss: FocalLoss(gamma=2.0, alpha=0.75)
- Everything else: same as focal-loss experiment
