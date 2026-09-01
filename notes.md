# noise-floor-cv

## Hypothesis

Run-to-run variance on this pipeline (no seed control anywhere: TF nondeterministic
init + shuffle, plus val_loss-trajectory noise in early stopping) is large enough
that single-run per-fold deltas of ~0.02-0.05 carry little signal. Every recent
entry in `log.jsonl` (`trunk-ft` +0.046, `context-embedder` +0.022, the whole
`aug-*` cluster) is read against an unquantified denominator. `std-convergence`
and the `trunk-ft` stopping sweep both bumped into this wall: a same-config
3-fold rerun there moved the mean +0.018 and swung `best_epoch` by 15-50.

This experiment does the control LOOP.md names as expensive and never run:
repeat `cv-baseline`'s exact config as a full 11-fold CV under a fresh name,
join per fold against `models/yamnet_medium_general`, and report the per-fold
spread. That spread is the denominator for every delta in the log.

Prediction: per-fold |Δ| against `yamnet_medium_general` is ~0.02 median with
some quiet folds (Diel, Opp/08-27) swinging 0.05+. Headline mean sens@fpr0.005
lands within ~0.015 of 0.206.

## Changes

None. Stock `main` config, fresh `--name`. Linear probe on frozen YAMNet GAP
embeddings, Dropout(0.2), BinaryCrossentropy(label_smoothing=0.2), Adam 2e-3,
`general` translation, `medium` set, 11 rotating folds. No stage 2 (reuses the
existing GAP embedding cache).

## Results

_pending_

## Conclusion

_pending_
