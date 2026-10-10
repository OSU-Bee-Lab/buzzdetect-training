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

## Results
Run 2026-10-10 07:09-11:37. The job died once, out of GPU memory in rotation 7's teacher (epoch 1, the per-epoch scoring callback) after six clean rotations; relaunched unchanged, it resumed at rotation 7 and finished. `train.rot1-6.log` holds the first six rotations' lines. Cause not established (memory building up across 12 fits in one process is the guess; reported to the loop as friction).

Every fold's `buzz logit SD` is in the trained range, teacher and student (teacher 0.54-1.14, student 0.57-1.17): no collapse. Pseudo-labels per rotation, from 41 folds and 20992 unlabelled frames: 295-397 positive, 5362-8374 negative; 60-70% of frames fell in the dropped band.

Student against its own teacher (`tools/results.py pseudo-label_teacher pseudo-label`), the matched pair:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.456 | 0.415 | -0.041 | 0.050 | 32 |
| 53 | 0.589 | 0.568 | -0.021 | 0.026 | 28 |
| 1_11 | 0.549 | 0.580 | +0.031 | 0.020 | 26 |
| 1_143 | 0.613 | 0.586 | -0.027 | 0.034 | 22 |
| 1_150 | 0.324 | 0.352 | +0.028 | 0.054 | 21 |
| 1_95 | 0.201 | 0.208 | +0.007 | 0.034 | 46 |
| 1_37 | 0.571 | 0.645 | +0.074 | 0.035 | 14 |
| 1_114 | 0.428 | 0.448 | +0.020 | 0.049 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.466 → this 0.475 (+0.009 ± 0.014)
- inclusive (sensitivity), same thresholds: 0.386 → 0.393 (+0.007)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.795 | 0.820 | +0.025 | 122 |
| untagged | 0.498 | 0.505 | +0.007 | 2418 |
| background | 0.402 | 0.374 | -0.028 | 1874 |
| quiet | 0.129 | 0.137 | +0.008 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only; training stochasticity is on top (~0.007 on a headline delta for a plain head, unmeasured for a fine-tuned trunk and likely larger).

**Headline:** +0.009 ± 0.014, inside its SD. This is the falsifier named in the hypothesis: the student is not distinguishable from its teacher.

**Folds:** `1_37` +0.074 ± 0.035 is the one delta near two eval SDs (14 buzz events; one of eight folds, so one such draw is not surprising). `1_11` +0.031 ± 0.020 is about 1.5 SD: unsure. Everything else is within its SD, both directions: `1_29` -0.041 ± 0.050, `53` -0.021 ± 0.026, `1_143` -0.027 ± 0.034, `1_150` +0.028 ± 0.054, `1_114` +0.020 ± 0.049. `1_95` did not move (+0.007 ± 0.034): the fold whose threshold is set by buzz-like negatives got nothing from the extra confident negatives, as the "known limits" paragraph expected.

**Tiers:** `loud` +0.025 on 122 frames and `untagged` +0.007: no detection gain to speak of. `background` -0.028 (the `1_29` / `53` losses). Nothing moved by more than a fold's noise.

**Where the teacher draw sits:** against `ps-depth8-repeat` the teacher is 0.509 → 0.466 (-0.043 ± 0.015), and the student 0.509 → 0.475 (-0.034 ± 0.015). The teacher is meant to be a third draw of the same recipe (same embedder cache, same 77777/7617 frames on `1_29`, same flags and env), yet it is ~3 eval SDs below both earlier draws (0.515, 0.509), lower on seven of eight folds. Not explained. Candidates, none checked: fine-tuned-trunk training noise is larger than the 0.007 measured on a frozen-embedder head; or something in 03_train drifted between 2026-10-01 and this branch's base; or the `PSEUDO_SET` path changes the teacher's training in a way I didn't intend. The student-vs-teacher delta is unaffected (both share the run), but **the student's 0.475 must not be read against 0.509 as a loss from pseudo-labelling**, and a later run on this recipe should check its control against all three draws.

## Conclusion
Null. Two-stage self-training on unlabelled audio of the training folds, with fixed cutoffs (logit > 0 positive, < -3.5 negative), leaves the headline where the teacher had it (+0.009 ± 0.014); no tier and no hard fold moved outside its noise, apart from `1_37` (+0.074 ± 0.035, weak). It matches the limit stated beforehand: confident pseudo-labels are frames the teacher already gets right, so they add ~6-8k easy negatives and ~350 easy positives and no new information at the threshold.

Not tested, and each a different experiment rather than a tuning of this one: unlabelled audio of the *held-out* deployment (a protocol change, Luke's call); a much larger unlabelled pool than 336 minutes; pseudo-labels from the band this run dropped (the frames that could move a threshold, and the ones a teacher mislabels).

Open thread for whoever next builds on `ps-depth8`: this run's teacher scored 0.466 where two earlier draws of the same recipe scored 0.515 and 0.509.
