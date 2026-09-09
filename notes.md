# probe-grid

## Hypothesis

`IDEAS.md`'s "Probe-convergence levers — grid these" (L1-L9). The shipped probe
takes ~2 gradient steps per epoch (`size_batch = 65568` against ~72k training
frames) and stops on a `val_loss` plateau while `val_sens` is still rising. If
that is a convergence bug rather than a tuning preference, the stopping/steps
levers (L1-L4) should recover most of the gap between `cv_baseline`'s 0.218 and
the 0.261 an offline `sklearn` readout reaches on the same YAMNet embeddings.
If they don't, the gap is in the loss/geometry levers (L5-L7) or in the
offline/pipeline mismatch itself.

One lever per run, each measured against `cv_baseline` paired per fold.

## Changes

`03_train/train.py`: the probe's hardcoded structure is lifted into a
`ProbeConfig` dataclass (dropout, label smoothing, learning rate, weight decay,
batch size, stopping monitor, min_delta) and threaded through `train_set` and
`_train_one`. `03_train/main.py` exposes each as a flag in a "probe structure"
argument group. **Every default is the previously hardcoded value**, so a run
passing none of the new flags takes cv_baseline's path unchanged.

Two things the monitor change needed:
- `RestoreTrueBest` now takes `mode`, since `val_sens_fpr0.005` is maximised
  and Keras' `mode='auto'` would infer `min` from the metric name.
- `SensAtFPR`'s prediction batch size is pinned to a new `BATCH_PREDICT`
  constant rather than following `data.size_batch`, so `--batch-size` moves the
  training lever only.

### L4 (epoch cap) — answered without a run

The cap is 400. No `cv_baseline` fold gets near it: n_epochs is 162 / 174 / 160
/ 70 / 83. The cap does not bind under the current stopping rule, so raising it
can do nothing on its own. It only becomes a live lever if L1 or L2 push runs
into it — noted per-run below.

### Not run

- **L8 (binary vs 15-class head)** and **L9 (input standardisation)** are not
  probe flags. L9 already has a pipeline result this era — `standardize-blocks`,
  0.261, clean — which is the same figure as the offline readout, so it is
  arguably already the answer the grid is chasing. L8 needs a translation file.
- **L3 note.** The epoch cap is left at 400 for the batch-size run so it stays a
  single-lever change; ~18 steps/epoch instead of 2 should stop it well short.

### Seed noise, measured by accident

`models/test_monitor_smoke/` and L1's first fold are the **same config on the
same fold** (`--monitor val_sens`, `JamesU - MustardBumbler/1_29`), run twice:

| run | epochs | best | sens@fpr0.005 |
|---|---|---|---|
| test_monitor_smoke | 261 | 215 | 0.449 |
| L1_sens fold 1     | 319 | 288 | 0.431 |

baseline on that fold: 0.426.

There is no seed control in the pipeline, so this is the first same-config
repeat this era has on a single fold: **0.018 apart**, which is most of the
distance from baseline to either of them. It is n=2 on one fold and not a noise
estimate, but it is a standing caution for every per-fold delta below — and it
means the smoke's apparent +0.023 for L1 on this fold was not a result.

### How to score a fold delta here (Luke, 2026-09-09)

**Gains on the low-sensitivity folds are the point, not a caveat.** `1_150`
(0.021 base) and `1_95` (0.037) are hard deployments, and their low sensitivity
makes them the target rather than something to discount. The model will be fine
in mustard (`1_29`, 0.426) whatever we do; the endpoint is a *new* deployment,
which may well be one of the hard ones. A lever that only moved the thick folds
would be close to worthless.

So the LOOP.md caution about thin folds is about **magnitude, not
importance** — and it does not license writing a hard-fold gain off as noise.
Read a delta table as: which direction did the hard folds move, and are the
losses elsewhere actually material? A -0.005 does not cancel a +0.068.

## Results

| lever | flag | mean sens@fpr0.005 | vs base (0.218) |
|---|---|---|---|
| L1 stopping metric | `--monitor val_sens` | **0.241** | **+0.023** |
| L2 stopping slack  | `--min-delta 0`      | 0.217 | -0.001 |

### L1 — `--monitor val_sens`

| fold | buzz frames | base | L1 | delta |
|---|---|---|---|---|
| JamesU MustardBumbler/1_29 | 2144 | 0.426 | 0.431 | +0.005 |
| Lily Fit+Fast/53 | 1031 | 0.425 | 0.410 | -0.015 |
| willard/1_11 | 305 | 0.180 | 0.243 | **+0.063** |
| Diel Drivers/1_150 | 146 | 0.021 | 0.089 | **+0.068** |
| Diel Drivers/1_95 | 433 | 0.037 | 0.032 | -0.005 |

The two large moves are both on hard deployments, which is the direction we
want. In frame terms: `willard` 55 -> 74 detected of 305, `1_150` 3 -> 13 of
146. `willard` is the more substantial of the two; `1_150`'s is ten frames.
The losses are -0.015 and -0.005, i.e. immaterial.

Direction believed, magnitude not yet banked — `L1_sens_r2` (`run_confirm.sh`)
re-runs the identical config for an independent init, since a same-config
repeat on `1_29` already came back 0.018 apart (above). If the hard-fold gains
reproduce, L1 is this grid's result.

### L2 — `--min-delta 0`

0.217, flat. Removing the patience-reset threshold changes nothing, so L1's
gain comes from **what is monitored**, not from training longer. That also
means the 400-epoch cap (L4) stays a non-issue: L2 ran a fold to 364 epochs
without the cap binding.

## Conclusion
