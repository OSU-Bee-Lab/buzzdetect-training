# pseudo-label
## Hypothesis
IDEAS.md item 28. Annotation covers ~24 snips per fold; the rest of each recording is unlabelled. Self-training on that audio is a standard answer to cross-site shift (MAST, arXiv 2609.15221), and the 0.5% FPR threshold is read from the negative tail, which is where more site audio should help most.

Two-stage self-training inside each rotation, on the era's best confirmed recipe (`ps-depth8`): a teacher trains on the rotation's labelled pool exactly as a plain rotation does; it labels unannotated audio from that rotation's **training** folds; a student retrains from scratch on the labelled pool plus the teacher's confident frames, and the student is what is scored. A frame becomes a pseudo-positive when the teacher's `ins_buzz` logit is above 0 (the detection cutoff; on `ps-depth8-repeat`'s held-out folds 98.8% of such frames are buzz, at 16% recall) and a pseudo-negative below -3.5 (about the median held-out negative, -3.38); the band between is dropped. Both cutoffs were fixed before the run from that control's logit scale and are not tuned.

Expected signature of a real effect: the student beats its own teacher (same run, same pool, same held-out fold) on the headline, with the gain in `untagged` / `loud` and on the low-threshold-quality folds (`1_95`, `1_114`, `1_37`, whose thresholds are set by buzz-like negatives). Falsifiers: student within the headline delta's SD of the teacher (pseudo-labels the teacher already agrees with carry no new information: confirmation only); or a gain that lives only in `1_29` / `53`'s background tier.

Known limits, stated before the run: the unlabelled pool is 336 windows of 60 s (8 per train-role fold, 42 folds, ~20k frames against ~78k labelled), drawn uniformly at random over each recording, so much of it will be quiet night audio and easy negatives; confident negatives are by construction the frames the teacher already scores low, so they cannot be the hard negatives that set a threshold; and rotating folds contribute no unlabelled audio even when they are in a rotation's training pool.

No audio of a scored deployment is read: `medium_unlabeled` is built from train-role folds only (`build.py`), and each rotation labels only its own training folds with a teacher that never saw the held-out fold. Using the held-out fold's own unlabelled audio would be a protocol change and is not done.

Control: the run's own teacher (`pseudo-label_teacher`, a third draw of the `ps-depth8` recipe, scored on the same held-out folds), plus `ps-depth8-repeat` (0.509) and `ps-depth8` (0.515) for where the teacher draw sits.

## Changes
- `02_set/sets/medium_unlabeled/build.py`: a new set of unannotated windows (label `unlabeled`), each at least 120 s from any medium annotation of its file, seed 28. Extracted for real with `02_set/main.py --set medium_unlabeled --embedder yamnet_trunk_pitchshift_depth8` (the artificial-cache rule: every row is `embed()` on real audio).
- `03_train/train.py`, gated by `PSEUDO_SET=<set>` (unset: unchanged): per rotation, train the teacher into `models/<name>_teacher`, score the held-out fold with it, label `<set>`'s frames of the training folds (`PSEUDO_POS`, `PSEUDO_NEG`), reload the pool with those frames appended (class weights stay the labelled pool's) and train the student. The student's `summary.json` carries the pseudo-label counts.
