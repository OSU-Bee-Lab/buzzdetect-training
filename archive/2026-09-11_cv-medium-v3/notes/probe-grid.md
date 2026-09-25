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
| lever | flag | mean | delta | folds | `1_150` | `willard` | `1_95` |
|---|---|---|---|---|---|---|---|
| **L1** stopping metric | `--monitor val_sens` | **0.241** | **+0.023** | 3up 2dn | **+0.068** | **+0.063** | -0.005 |
| **L6** label smoothing | `--label-smoothing 0.05` | **0.233** | **+0.015** | **4up 1dn** | +0.013 | +0.040 | -0.005 |
| L3 batch size | `--batch-size 4096` | 0.221 | +0.003 | 2up 3dn | -0.007 | +0.030 | -0.005 |
| L6 label smoothing | `--label-smoothing 0` | 0.221 | +0.003 | 3up 2dn | +0.020 | +0.056 | **-0.037** |
| L5 dropout | `--dropout 0.1` | 0.219 | +0.001 | 1up 3dn | 0.000 | -0.006 | -0.002 |
| L2 stopping slack | `--min-delta 0` | 0.217 | -0.001 | 1up 3dn | -0.007 | 0.000 | -0.005 |
| L5 dropout | `--dropout 0` | 0.212 | -0.006 | 2up 3dn | -0.007 | +0.013 | -0.009 |
| L7 weight decay | `--weight-decay 1e-4` | 0.213 | -0.005 | 1up 4dn | -0.014 | +0.013 | -0.007 |

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

### L6 — `--label-smoothing 0.05` (+0.015)

The most *consistent* result in the grid: 4 folds up, 1 down, and the only
lever that lifts both thick folds and `1_150` and `willard` at once. Label
smoothing 0.2 is too aggressive at an 11.7% positive rate, as IDEAS predicted.

**It is an optimum, not a direction.** At `--label-smoothing 0` the mean is
0.221 but `1_95` collapses to **0.000** and Fit+Fast drops 0.043. Do not
extrapolate past 0.05.

### The framing hypothesis was wrong

The grid was motivated by "the probe does ~2 gradient steps per epoch, so it
never converges." **L3 (`--batch-size 4096`, ~18 steps/epoch instead of 2) is
+0.003, and L2 (stopping slack) is flat.** The probe was not step-starved. What
paid was *what is monitored* (L1) and *the shape of the loss* (L6) — not how
many steps it takes or how long it runs. The offline-readout gap is not a
convergence gap.

### Dead levers

L2 (flat), L5 dropout in both directions (0.0 is -0.006, 0.1 is +0.001), L3,
and L7 weight decay (-0.005, 4 folds down — consistent with both archived L2
results and the lowest prior in IDEAS).

### `willard` responds to almost everything

It rises under 6 of 8 levers, including mutually contradictory ones. That leans
toward its 0.180 baseline being a low draw rather than every lever helping it.
It does not undercut L1 or L6 (both move other folds too), but it is why
`L1_sens_r2` exists.

### L2 — `--min-delta 0`

0.217, flat. Removing the patience-reset threshold changes nothing, so L1's
gain comes from **what is monitored**, not from training longer. That also
means the 400-epoch cap (L4) stays a non-issue: L2 ran a fold to 364 epochs
without the cap binding.

### Confirmation runs

`L1_sens_r2` is `L1_sens` re-run with no changes (no seed control exists, so a
re-run is an independent init).

| fold | buzz frames | base | L1 run 1 | L1 run 2 | L1+L6 combo |
|---|---|---|---|---|---|
| JamesU MustardBumbler/1_29 | 2144 | 0.426 | 0.431 | 0.421 | 0.431 |
| Lily Fit+Fast/53 | 1031 | 0.425 | 0.410 | 0.445 | 0.430 |
| willard/1_11 | 305 | 0.180 | **0.243** | **0.226** | **0.243** |
| Diel Drivers/1_150 | 146 | 0.021 | **0.089** | **0.158** | **0.171** |
| Diel Drivers/1_95 | 433 | 0.037 | 0.032 | 0.035 | 0.034 |
| **mean** | | **0.218** | **0.241** | **0.257** | **0.262** |

**L1's direction reproduces, and it reproduces on the hard folds.** Both draws
lift `willard` (0.180 -> 0.243, 0.226) and `1_150` (0.021 -> 0.089, 0.158) well
clear of baseline. That is the lever working as intended, not a lucky fold.

**L1's magnitude does not reproduce**, and neither does `1_150`'s: 0.089 vs
0.158 between identical configs is 13 vs 23 detected frames of 146. Quote the
direction, not the size.

### What the repeat costs us: a minimum detectable effect

The identical-config pair puts the run-to-run SD on the CV mean at **~0.016**
(0.241 vs 0.257). So a difference between two *single* runs has SE ~0.016*sqrt2
= **0.023**, and the smallest effect a single-run comparison can resolve is
**~0.045**.

That reframes the whole grid: **only L1 and the combo clear it.** L6's +0.015
is under one spread, and everything else (<= +0.003) is indistinguishable from
zero. Resolving a +0.015 effect properly would need ~9 runs per arm, ~19 runs,
~4 h — which is exactly what LOOP.md means by not bootstrapping CIs. This
number recomputes for free whenever the data moves and is the useful, portable
part of the repeat.

### L1 and L6 do not stack

| comparison | delta |
|---|---|
| combo vs baseline | +0.044 (4 up, 1 down) |
| combo vs `L1_sens_r2` alone | **+0.005** |

L6 adds ~nothing on top of L1, which is mechanistically what you would expect:
`restore-on-sens` fingered the **label-smoothing overconfidence penalty** as
what makes `val_loss` diverge from sens once a model sharpens. L6 removes some
of that penalty; L1 stops reading the penalised curve at all. They are two fixes
to one problem, so the second is redundant given the first.

At 0.262 the combo also lands on the same figure as `standardize-blocks`
(0.261) and the offline `sklearn` readout (0.261) — three unrelated routes to
the same ceiling, which is worth someone's attention.

### Targeted repeats — the decisive table

`run_noise.sh`: two `cv_baseline` repeats so the era's denominator has n=3, plus
one `L6_ls005` repeat to resolve the single claim sitting at one spread.

| fold | base | base_r2 | base_r3 | L6 | L6_r2 | L1 | L1_r2 | combo |
|---|---|---|---|---|---|---|---|---|
| `1_29` | 0.426 | 0.440 | 0.421 | 0.443 | 0.334 | 0.431 | 0.421 | 0.431 |
| `53` | 0.425 | 0.408 | 0.372 | 0.434 | 0.417 | 0.410 | 0.445 | 0.430 |
| `willard` | 0.180 | 0.233 | 0.194 | 0.220 | 0.180 | 0.243 | 0.226 | 0.243 |
| **`1_150`** | 0.021 | 0.021 | 0.014 | 0.034 | 0.027 | **0.089** | **0.158** | **0.171** |
| `1_95` | 0.037 | 0.035 | 0.037 | 0.032 | 0.028 | 0.032 | 0.035 | 0.034 |
| **TOTAL** | 0.218 | 0.227 | 0.208 | 0.233 | 0.197 | 0.241 | 0.257 | 0.262 |

**Baseline n=3: mean 0.2177, SD 0.0095, range 0.019.** The earlier 0.016
estimate came from an unlucky L1 pair; 0.0095 is the better number, putting the
minimum detectable effect at **~0.027** rather than 0.045. It still rules out
every lever in the grid except L1.

**`cv_baseline` is a fair draw** — 0.218 against a 3-run mean of 0.2177. The
era's logged deltas are not biased by a lucky denominator, so nothing already in
`log.jsonl` needs re-baselining. That was the main risk the repeats bought off.

**L1 is real: +0.031 vs the baseline mean, and the draws separate completely.**
L1's worst run (0.241) beats baseline's best (0.227).

**L1's gain is `1_150`, not `willard`.** Baseline on `1_150` is tight across
three draws (0.021, 0.021, 0.014); L1 gives 0.089 and 0.158 and the combo 0.171
— 4-8x baseline on the hardest deployment, far outside baseline's own spread
there. `willard`, by contrast, ranges 0.180-0.233 in the **baseline** alone,
which covers most of what looked like a lever effect earlier. `willard` is not
evidence; `1_150` is.

**L6 is dead: mean 0.215, i.e. -0.003 vs baseline.** The +0.015 single run was
noise, as the MDE arithmetic predicted. The repeat came back 0.197 with `1_29`
collapsing to 0.334 — not merely neutral but unstable. Nine minutes well spent.

## Conclusion

**L1 — stop on `val_sens`, not `val_loss` — is the result. +0.031 over the
baseline mean (0.2177 -> 0.249, n=2 vs n=3), with no overlap between the two
configs' draws.** It should become the default, and it revives
`restore-on-sens`, which E2 measured at +0.014 and shelved on the grounds that
"the frozen probe's `val_loss` barely diverges from sens." On this data that
reasoning is wrong: the divergence is worth 0.03, and it is concentrated
entirely in the deployment we most need to fix.

**The gain is a hard-deployment gain.** `1_150` goes from 0.021 (tight over
three baseline draws) to 0.089/0.158/0.171. The thick folds are flat. That is
the shape of result this project wants — the model is already fine in mustard.

**The grid's motivating hypothesis was wrong, and that is worth as much as L1.**
It was premised on the probe being step-starved at ~2 gradient steps per epoch.
L3 gave it 9x the steps for +0.003; L2 let it train longer for -0.001; L4's cap
never binds. The probe converges fine. What was broken is that it was being
scored on the wrong curve while it converged. **The offline-readout gap is not a
convergence gap** — so the remaining distance to 0.261 is not going to come from
optimiser levers, and the next agent should not spend a CV looking for it there.

**L6 (`label_smoothing=0.05`) does not replicate and L1+L6 do not stack.** The
combo is +0.005 over L1 alone, inside noise. Mechanistically expected: label
smoothing's overconfidence penalty is *what makes* `val_loss` diverge from sens,
so L6 removes some of the divergence and L1 stops reading the divergent curve.
Two fixes to one problem; take the better one.

**Everything else is dead** at this resolution: batch size, both dropout
directions, stopping slack, weight decay. All are under the ~0.027 minimum
detectable effect, several with most folds moving down.

### What to do next

1. **Adopt `--monitor val_sens`** as the default and re-baseline the era on it.
   The flag is on this branch and defaults to the old behaviour, so merging is
   safe; flipping the default is the deliberate act.
2. **Do not re-run the loss/optimiser levers.** They are measured and dead.
3. The remaining gap to the offline readout is a **representation/geometry**
   question, not an optimisation one. `standardize-blocks` (0.261) and the
   combo (0.262) reaching the same ceiling by unrelated routes is the lead.
4. **Variance reduction beats variance measurement.** Averaging 2-3 seeds per
   fold inside one run would shrink the error on every future experiment for
   ~2x the compute of one run, and would have made this entire grid readable in
   8 runs instead of 13. It changes what a run means, so it is an era-boundary
   decision — parked in IDEAS.md, not done here.
