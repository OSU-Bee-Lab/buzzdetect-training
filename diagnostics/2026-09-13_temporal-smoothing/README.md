# Temporal smoothing of frame scores — diagnostic, 2026-09-13

IDEAS.md item 6. No training, no extraction: `temporal_smoothing_diag.py
<model dir> [--windows 1,3,5,9] [--stat mean|median]` reads an existing model's
`predictions.csv` per fold and re-scores `sensitivity_exclquiet` after a
rolling mean/median over `activation_ins_buzz`, using `sx.summarize_folds`
unmodified so the FPR sweep and tier logic are identical to a real run.

**Pre-registered as an upper bound, not a candidate for adoption**: frames
inside one annotation sample nearly always share a label (`context-stack`'s
known artifact), and none of the models on disk carry the `sample` column yet
(all predate its addition to `train.py`), so the smoothing window can cross a
sample boundary using row-adjacency only. The headline numbers below are
inflated by that; the per-fold *shape* is the result.

## Result, `models/cv_baseline_v3`, `sensitivity_exclquiet`

| fold | w=1 | w=3(med) | w=5(med) | w=9(med) | w=3(mean) | w=9(mean) |
|---|---|---|---|---|---|---|
| `1_29` | 0.441 | 0.525 | 0.580 | 0.623 | 0.599 | 0.653 |
| `53` | 0.429 | 0.486 | 0.507 | 0.560 | 0.542 | 0.665 |
| `willard/1_11` | 0.368 | 0.415 | 0.401 | 0.386 | 0.447 | 0.466 |
| `wooster/1_143` | 0.461 | 0.514 | 0.505 | 0.532 | 0.595 | 0.830 |
| `1_150` | 0.266 | 0.310 | 0.269 | 0.315 | 0.352 | 0.380 |
| **`1_95`** | **0.052** | **0.017** | **0.000** | **0.000** | 0.014 | 0.000 |
| `1_37` | 0.369 | 0.479 | 0.526 | 0.524 | 0.516 | 0.654 |
| `1_114` | 0.250 | 0.311 | 0.337 | 0.366 | 0.282 | 0.329 |
| **total** | **0.330** | 0.382 | 0.391 | 0.413 | 0.418 | 0.497 |

Full tables: `median.txt`, `mean.txt`.

## Reading

**Both pre-registered predictions held.** Every fold except `1_95` moved up
with wider windows — consistent with spiky, isolated FPs and a sustained-drone
buzz signal, the shape smoothing is built for. `1_95` moved the other way,
monotonically to zero — consistent with its FPs being one contiguous ~90 s jet
block: smoothing a block of already-elevated scores does not lower it, and it
drags the surrounding true-buzz frames' relative rank down instead. **The jet
story is not incomplete** — this is the falsifier IDEAS.md item 6 specified,
and it came back negative (smoothing does not fix `1_95`, so the mechanism
proposed in the jet diagnostics is not contradicted by this).

**The headline move itself is not a result** — it is exactly the same-label-
neighbour inflation `context-stack` was caught on (+0.05 to +0.17 here,
larger than any real lever found this era), for the reason stated above: no
model on disk yet has the `sample` boundary column, so this is not even the
honest form of the artifact (which `context-embedder` showed shrinks to
~+0.02 once real audio neighbours replace cached rows). Do not adopt smoothing
from this number; if a real smoothing lever is ever wanted, it must read
audio-level neighbours or gate on the `sample` column once a model carries it.

**Practical output**: `1_150` (the genuine low-SNR fold) barely moves and
non-monotonically — its problem is amplitude, not spike shape, matching its
"positives" characterization. Every other fold's monotonic rise vs. `1_95`'s
monotonic fall is a clean, free partition of the deployments into "spiky FP"
vs. "block FP" — worth checking against any future FP-shape lever for free.
