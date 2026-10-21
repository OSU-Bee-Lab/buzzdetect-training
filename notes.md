# plain-d8
## Hypothesis
The octave-up view was this era's largest single lever at depth12 (`v4-ft-ps` +0.077 over plain `v4-ft`), and it was never re-measured once the trainable tail grew to depth8 (+0.06 over depth12). Two questions, one run:

1. **Is the up view still needed at depth8?** Seven trainable blocks may re-form low-frequency features the up view supplied. Prediction: plain depth8 (`yamnet_trunk_depth8`, single view) lands >= 0.03 below `ps-depth8-r3` (0.497), i.e. the up view still carries a real share. Falsifier of the up view's value: within 0.02.
2. **Does a cross-recipe member decorrelate an ensemble better than a same-recipe draw?** Same-recipe ensembles saturate at three (`ps-d8-ens-same5` = ens3, 0.558). If plain-d8's negatives decorrelate more from ps-depth8's than ps draws do from each other, the logit mean of {plain-d8, two ps-depth8 draws} beats the best same-recipe triple from those draws. Prediction: plain-d8 + ps-depth8 + ps-depth8-repeat >= ps-d8-ens3 (0.558) + 0.02. Falsifier: at or below ens3.

Control: `ps-depth8-r3` (same tail, LR, epochs, batch; differs only in the view). Ensembles via `exp/ps-d8-ensemble`'s `tools_exp/ensemble_predictions.py`.

## Changes
`03_train/train.py`: `TRUNK_ONLY_VIEW=<v>` trains on one view of a multi-view trunk cache. The model is the embedder's own single-view head (`n_ctx` 1: Reshape -> depth8 tail -> Dense), fed view v's 12288 columns cropped out of the cached row with standard Keras layers (Reshape, Cropping1D). So rows, labels, frames and thresholds are the control's exactly; no new cache. Run: ps-depth8-r3's recipe on `yamnet_trunk_pitchshift_depth8` with `TRUNK_ONLY_VIEW=0` (plain view). Chain: `exp/binary-d8:chain_b41.sh`.
