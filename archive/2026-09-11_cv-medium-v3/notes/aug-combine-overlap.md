# aug-combine-overlap

## Hypothesis

The held-out-deployment endpoint fails hardest where a new site's background
differs from the training sites (willard, both Diel Drivers folds sit near
zero). `augment_specs.CombineSpec` already exists for exactly this — mix a
"source" class with an "augment" class in waveform space, re-embed, label with
the source — i.e. buzz-over-novel-background mixup. It has never been run under
CV.

Mixing each buzz frame with background (`ambient_*`) frames drawn from
elsewhere in the training pool synthesises "this buzz, over a different site's
background" examples. Prediction: sensitivity at fixed FPR rises on held-out
deployments, most on the low-scoring quiet-background folds.

## Why the scripts are "broken" as-is

- `_augment_combine_spec.collect_frames(label)` matches raw labels **exactly**
  (`label in _labels_from_path(p)`). `ins_buzz` as a raw label is on only 6
  files in `medium`; real buzz is `ins_buzz_medium/low/high/Episyrphus/...`. So
  `CombineSpec(class_source='ins_buzz', ...)` collects almost nothing. Same for
  any category ("ambient", "mech").
- `augment.py`'s `--fold train` default is the pre-CV single-fold name; there
  is no `raw/train/` under the CV layout.
- A fold with buzz but no `ambient_*` frames makes `collect_frames` return []
  and the function `raise`s, aborting the whole run.

## Changes

One experimental variable: add buzz+background combine augmentation to
training. Infrastructure to make the existing CombineSpec usable:

- `collect_frames` matches by **label prefix** (`l == label or
  l.startswith(label + '_')`), so `ins_buzz` collects all `ins_buzz_*` and
  `ambient` collects all `ambient_*`.
- Combined output is labelled with `class_source` only (filename
  `<class_source>.pickle`), not `source+augment` — the mixed-in background is
  acoustic texture, not a target, and bare `ambient` has no translation row.
- Empty source/augment pool for a fold → warn and skip that fold, not `raise`.
- `augment_set` loops over a `folds` list; `--all-folds` reads folds.csv
  (stdlib csv, keeps clear of the tf-before-pandas invariant); `--combine
  SRC AUG PROP` CLI.

Spec: `CombineSpec(class_source='ins_buzz', class_augment='ambient', prop=0.65,
limit=2)` — mixed = 0.65·buzz + 0.35·background, 2 augmented copies per buzz
frame.

## Results

Extra infra fix needed mid-run: `dataset.load_augmented` `raise`d when *any*
training fold lacked the augment dir. Label-selective combine aug only produced
output for **52 of 322 folds** (a fold needs buzz *and* `ambient_*` frames in
its own recording; most short clips have one or neither). Relaxed to skip
folds with no augment dir, still error if the dir is absent entirely or matches
no training fold. 14,780 combined frames added (~2.6x the 5,662 raw buzz
frames, on the folds that got any).

All 11 rotate folds' own buzz *did* get augmented copies into the other
rotations' pools — the 270 skipped folds are the small `train`-role
isolated-insect clips (InsectSound1000 etc.), which have no ambient anyway.

Baseline `models/yamnet_medium_general` = 0.199.

| fold | base | exp | delta | val frames | exp best_epoch |
|---|---|---|---|---|---|
| Fit+Fast/53 | 0.368 | 0.342 | -0.026 | 4712 | 125 |
| Various/2025-08-12/1_114 | 0.161 | 0.139 | -0.022 | 3768 | 135 |
| JamesU/1_29 | 0.447 | 0.426 | -0.021 | 6984 | 119 |
| Various/2025-08-27/48 | 0.035 | 0.023 | -0.012 | 1571 | 90 |
| willard/2024-08-07/1_11 | 0.191 | 0.184 | -0.007 | 4730 | 94 |
| Diel/2026-05-06/1_95 | 0.028 | 0.028 | 0.000 | 6628 | 110 |
| Various/2025-06-23/1_23 | 0.326 | 0.326 | 0.000 | 315 | 101 |
| Various/2025-08-05/31 | 0.148 | 0.159 | +0.011 | 942 | 140 |
| Diel/2026-04-08/1_150 | 0.014 | 0.034 | +0.020 | 4947 | 23 |
| Various/2025-07-03/1_37 | 0.241 | 0.264 | +0.023 | 4715 | 99 |
| wooster/2024-07-26/1_143 | 0.235 | 0.289 | +0.054 | 4708 | 79 |

- mean sens@fpr0.005: 0.199 → 0.201 (**+0.002**)
- 4 up, 5 down, 2 flat. Largest mover ±0.054. No direction.
- best_epoch is healthy across the board (79-140, one at 23) — unlike
  `exp/aug-snr-noise`, there is **no early-stopping-scale confound** here (the
  combine set adds ~20% frames, not ~100%), so this is a clean read.

## Conclusion

Neutral (+0.002, folds 4/5/2, all deltas small, best_epochs normal). Combine
("overlap"/mixup) augmentation of buzz over `ambient_*` backgrounds neither
helps nor hurts the held-out-deployment endpoint as configured. Like
`aug-snr-noise` this is a departure from "augmentation always hurts here", and
it's a cleaner null than that one (no stopping confound).

Trust: **caveated** — the mixup pairs buzz and background *within the same
recording*, which is the opposite of the hypothesis (buzz over a *different*
site's background). Cross-fold pairing (buzz from fold A + ambient from fold B)
is the version that actually tests the domain-robustness idea and would also
reach the 150 folds that have no ambient of their own. A within-fold null does
not rule that out.

Follow-ups: (a) cross-fold combine (draw `class_augment` frames from the whole
training pool, not `dir_audio_fold`); (b) sweep prop (0.5, 0.8) and limit;
(c) combine with `mech_*` / `animal` as well as `ambient`.

