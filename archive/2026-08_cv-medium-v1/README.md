# Era: CV on `medium`, v1 annotations

**30 runs, 2026-08-19 → 2026-09-08.** The first leave-one-fold-out era.
Closed on 2026-09-08 when the training data was revised.

## What made these comparable

Every entry, without exception and without being restated per-entry:

| | |
|---|---|
| Set | `medium` — day-long annotated recordings, `framehop_prop: 1`, `overlap_event_prop: 0.2` |
| Folds | 11 rotating (`role: rotate`), 675 train-only. A fold is one deployment = one ident |
| Metric | `total` row `sensitivity` at `fpr` 0.005 in `folds_sx.csv` — plain mean over folds, each deployment counted once, threshold set per fold on its own held-out audio |
| Translation | `general_v1` (17 classes). See the gotcha below |
| Embedder | YAMNet unless the experiment *was* the embedder |
| Validation | always a whole fold, never a split within one |

The metric is an **oracle ceiling**, not an operator number: placing a fold at
exactly 0.5% FPR uses that fold's labels. It is a fair ceiling for comparing two
models, since both get it.

## Baseline and best

`cv-baseline` — linear probe on frozen YAMNet, `Dropout(0.2)`,
`BinaryCrossentropy(label_smoothing=0.2)`, Adam 0.002 — **0.206**.

| Result | | Trust |
|---|---|---|
| `trunk-ft-restore-sens` | **0.298** | caveated |
| `trunk-ft-1e5-aug` | 0.265 | caveated |
| `trunk-ft-1e5` | 0.262 | caveated |
| `context-stack` | 0.256 | artifact |

`trill-vs-buzz-provenance-refresh` (0.277) is **incomplete** — 6 of 11 folds,
killed mid-run. Not comparable to the eleven-fold numbers above. Its own log
entry says so at length; believe that entry, not the float.

The `trust` field is doing real work in this log. Read it before quoting any
number.

## Gotchas

**`general_v1` is not `general`.** `general` was revised on 2026-09-03
(`0a2ef2a`, `b49cdaa`): 17 classes → 14, `mech_plane` folded into `mech_auto`.
Every run in this log resolved to the *pre-revision* table — the early ones
because their worktrees predated per-set translations, the later ones because
they passed `--translation general_v1` explicitly. `set/translations/` holds
both, so the distinction is checkable rather than remembered.

**The annotations moved under these results.** Folds were being annotated toward
24 snips each throughout the era. An August number and a September number sit on
different amounts of data even where nothing else changed. This is the main
reason the era is closed rather than extended — and the reason hyperparameter
results from it should not be banked.

**No seed control, and folds are not independent** (training pools overlap
~90%). A result worth acting on shows most folds moving the same way, not a
small mean shift with folds scattered either side. Measured noise floor:
~0.014 on the headline, ~0.017 median per-fold.

## Fold roster at cutover

Counts from `set/`, `ins_buzz` under `general_v1`. Annotation seconds are the
*annotated* span, not the recording length.

| rotating fold (deployment) | annots | buzz | buzz s | annot s | span h |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 1463 | 441 | 2391 | 9867 | 33.8 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 350 | 157 | 1148 | 5283 | 31.9 |
| Lily Adam - One Hive/recorders/willard/2024-08-07/1_11 | 660 | 191 | 184 | 4685 | 33.8 |
| Lily Adam - One Hive/recorders/wooster/2024-07-26/1_143 | 336 | 118 | 118 | 4623 | 29.3 |
| Luke - Diel Drivers/2026-04-08/1_150 | 1570 | 77 | 88 | 4973 | 21.4 |
| Luke - Diel Drivers/2026-05-06/1_95 | 1231 | 157 | 328 | 6927 | 27.7 |
| Luke - Various Opportunistic Recordings/2025-06-23/1_23 | 61 | 12 | 28 | 314 | 28.1 |
| Luke - Various Opportunistic Recordings/2025-07-03/1_37 | 323 | 149 | 250 | 4567 | 27.5 |
| Luke - Various Opportunistic Recordings/2025-08-05/31 | 150 | 129 | 525 | 1573 | 28.3 |
| Luke - Various Opportunistic Recordings/2025-08-12/1_114 | 375 | 199 | 346 | 7601 | 34.7 |
| Luke - Various Opportunistic Recordings/2025-08-27/48 | 102 | 19 | 275 | 1960 | 13.6 |
| **total** | **6621** | **1649** | **5681** | **52373** | **310.1** |

The spread is the point. `1_23` carries 12 buzz annotations and `1_29` carries
441; both count once in the mean. A fold that could not reach 0.5% FPR is blank
in `folds_sx.csv` rather than averaged in.

## Notes and code

`notes/` holds all 28 — every branch the log names, plus `backbone-slight-ft`,
`deployment-forensics`, `large-trunk-ft` and `prediction-provenance`, which were
worked but never logged.

Fourteen branches survive as `exp/*` on origin. **Twelve did not** —
`class-weight-fix`, `context-embedder`, `context-pooling`, `context-stack`,
`context-width`, `framehop-overlap`, `input-standardization`, `std-convergence`,
`tail-loss`, `tail-loss-retest`, `yamnet-combined`, `yamnet-native-buzz` — and
were recovered from dangling commits. Their code is at `refs/archive/<name>`
(local only; see `../README.md`).

`models/yamnet_medium_general/` is this era's baseline model, kept as an
artifact. Its `folds_sx.csv` is what a paired per-fold comparison joined
against.

## Reading these forward

Verdicts here are **leads, not settled answers** — and this era proved that
itself. `temporal-context` was a clear negative in the previous era; the
identical change, rerun here as `context-stack`, was the largest gain at the
time. A verdict inverted across an eval change.

Assume any of these could do the same under the new annotations. Rerun rather
than defer. The distilled version lives in `IDEAS.md`.
