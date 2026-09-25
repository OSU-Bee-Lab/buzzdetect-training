# Era: cv-medium-v3

**46 runs, 2026-09-11 → 2026-09-25.**
The first era under the fixed `--epochs` rule and the `_quiet` scoring split. Two levers
dominated: an octave-up pitch-shift concat, and fine-tuning YAMNet's last blocks ("trunk").

Closed on 2026-09-25 by the **Hard Negatives** annotation effort (10 new idents, 435
annotations of non-buzz audio that drew high `ins_buzz` activation) joining `medium` and
`moderate` as train-only data. The rotating folds are unchanged, but the training pool
isn't, so no number here can be compared with a later one.

## What made these comparable

| | |
|---|---|
| Set | `medium` — `overlap_event_prop: 0.2`, `framehop_prop: 1` |
| Folds | 8 rotating (`role: rotate`), 75 train-only |
| Metric | `total` row `sensitivity_exclquiet` at `fpr` 0.005 in `folds_sx.csv` — plain mean over folds, threshold set per fold on its own held-out audio |
| Translation | `general` (a few `binary` controls) |
| Embedder | varies per run; the anchor is frozen `yamnet` |

## Baseline and best

Anchor: `cv-baseline-v3`, **0.330** (bare linear probe on frozen `yamnet`, 400 epochs).
Later mid-era comparisons used `cv-baseline-v3-refresh` (0.329), the same config re-run.

| Result | | Trust |
|---|---|---|
| `trunk-pitchshift-depth12` | 0.472 | caveated |
| `perch-pitchshift-concat` | 0.459 | caveated |
| `trunk-pitchshift-depth12-repeat` | 0.454 | clean |
| `epoch-budget-700` | 0.437 | clean |
| `trunk-ft-pitchshift` | 0.434 | clean |

- `trunk-pitchshift-depth12` 0.472 was the high draw. Its repeat gave 0.454; use the two-run mean
  (~0.463). Its 1_114-recovery claim is amended as not replicating.
- `perch-pitchshift-concat` 0.459 has one draw only, since the Perch injunction rules out a repeat
  extraction.
- `epoch-budget-700` 0.437 is a frozen probe at 700 epochs, not a new architecture.

## Gotchas

- **Trunk runs use 60 epochs, frozen probes 400.** `trunk-ft-v3` found 60 vs 120 indistinguishable
  for fine-tuning. `epoch-budget-700` found 400 *too short* for wide frozen probes (+0.026 at
  700). Epoch counts differ across rows for these reasons.
- `_background` buzz (1_29, 53) is about half of all scored buzz seconds. Gains that land only on
  the `background` tier move the headline without touching isolated buzz; read the tiers.
- Hard folds 1_95 (jet flyover), 1_150 and 1_114 move a lot from run to run; per-fold claims on a
  single draw did not replicate twice (see the amended `trunk-pitchshift-depth12`).
- `aug-snr15-trunk` was never run (it died at startup, probably out of host RAM). Its notes are
  here but there is no log entry.

## Fold roster at cutover

Counts from `set/`, `ins_buzz` under the translation named above.

| rotating fold (deployment) | annots | buzz | buzz s | annot s | span h |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 1458 | 441 | 2391 | 12191 | 33.8 |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 430 | 157 | 1148 | 8533 | 35.0 |
| Lily Adam - One Hive/recorders/willard/2024-08-07/1_11 | 830 | 191 | 184 | 9507 | 36.8 |
| Lily Adam - One Hive/recorders/wooster/2024-07-26/1_143 | 341 | 118 | 118 | 5413 | 30.3 |
| Luke - Diel Drivers/2026-04-08/1_150 | 1253 | 83 | 84 | 6254 | 21.4 |
| Luke - Diel Drivers/2026-05-06/1_95 | 1280 | 154 | 289 | 7927 | 27.7 |
| Luke - Various Opportunistic Recordings/2025-07-03/1_37 | 417 | 152 | 252 | 4868 | 28.5 |
| Luke - Various Opportunistic Recordings/2025-08-12/1_114 | 507 | 294 | 480 | 9313 | 34.7 |
| **total** | **6516** | **1590** | **4945** | **64006** | **248.1** |

## Notes and code

`notes/` holds 117.

**32 recovered from dangling commits** (their branches were already deleted): `backbone-slight-ft`, `batchnorm`, `buzz-in-train`, `buzz-upweight`, `class-weight-fix`, `context-pooling`, `context-stack`, `context-width`, `deployment-forensics`, `dropout-repro`, `framehop-overlap`, `hyperparam-sweep`, `input-standardization`, `l2-only`, `l2-regularize`, `low-delta`, `ls02-repro`, `mlp-head-repro`, `no-reg-baseline`, `std-convergence`, `supp-freq`, `supp-freq-v2`, `tail-loss`, `tail-loss-retest`, `temporal-context`, `white-noise`, `with-dropout`, `yamnet-bandpass`, `yamnet-combined`, `yamnet-ft`, `yamnet-mask`, `yamnet-native-buzz`. Code at `refs/archive/<name>`.

Branches: `exp/*` on origin where they survive.

## Reading these forward

- **Rerun first:** trunk fine-tuning (+0.077, reproduced three times) with the pitch-shift concat,
  at the depth-12 cut. This was the era lead, and the v4 opening grid reruns it.
- **Worth a rerun on the new data:** temporal context with a retrained trunk (+0.002 at the
  layer-12 cut, never tried at depth 12), and a dense head on a retrained trunk (never tried).
- **Leads blocked by cost:** Perch concat (a real gain, but extraction is ~0.005x YAMNet's rate).
- **Likely dead:** noise augmentation on the frozen probe (flat twice, unconfounded the second
  time) and explicit contrast vs concat once the trunk is retrained (equivalent).
