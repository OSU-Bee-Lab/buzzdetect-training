# label-smooth

## Hypothesis
Label smoothing (ε=0.1) improves model calibration, lifting sensitivity at 95% precision.

The 95% precision threshold depends entirely on how well-calibrated the model's logit scores are.
With hard 0/1 targets, the model is trained to be maximally confident, potentially overfitting the
training distribution's calibration. Label smoothing replaces targets with:
  positive: 1 → 0.95
  negative: 0 → 0.05
This prevents overconfidence, which is known to improve the precision-recall curve at high thresholds.

The focal loss experiments showed the mechanism matters: the loss function changes calibration, not
just accuracy. Focal loss shifted the curve toward lower precision; label smoothing should improve
the score distribution without biasing it.

## Existing evidence
- Müller et al. 2019 "When Does Label Smoothing Help?": improves calibration (lower ECE) in classification
- Focal loss experiments: showed that loss function choice substantially affects PR curve shape
- Current BCE + class weights is well-optimized for accuracy but not necessarily calibration
- Linear probe is best head — loss function is the remaining lever with existing embeddings

## Config
- Embedder: yamnet, standard set, general translation, linear probe
- Validation: all classes (labels_keep_raw=None)
- Loss: BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
- Everything else unchanged from combined-embedder baseline
