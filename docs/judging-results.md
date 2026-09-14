# Judging a result

Read at LOOP.md step 4, before writing an interpretation. `tools/results.py`
produces the numbers; this is how to read them and what each rule is based on.

## The comparator

**Compare against a matched control: the same stopping rule, the same
`--fixed-epochs` N, and your one variable.** Every fixed-budget run from the last
era was still rising at its cap, so N moves the score, and a run under
`--early-stop` scores 0.031-0.040 lower on the rule alone. Background:
`exp/pairwise-rank:notes/new-era-audit.md` and the archived era's README.

**Compare paired, per fold.** Both models ran on the same folds, so the per-fold
difference is the useful quantity. A headline gap that vanishes under pairing
was never a capability gap.

## What a number rests on

**A fold's sample size is `buzz_events_exclquiet`, not `buzz_frames`.** A buzz
event spans many frames (one bee heard for 26 s is one observation, not 26), so
a fold rests on single-digit to low-tens of events. `1_29` carries 1972 scored
buzz frames in 14 events; `53` has 900 in 9, one of which holds 79% of them; and
`1_37` has 5.

**Uncertainty from eval sampling, measured 2026-09-13 by event-blocked
bootstrap:**

| quantity | SD |
|---|---|
| one fold's sensitivity | 0.029-0.135 |
| one fold's paired delta | 0.017-0.052 |
| the headline delta | ~0.012 |

The paired delta is tighter than either model's own SD, because both models
score the same resampled events. The headline is tighter still, because eight
folds' independent wobbles partly cancel.

**Run the tool rather than quoting these figures.** This section's predecessors
were wrong twice. A "±0.25" figure was ~7x too large. Its replacement,
0.010-0.037, was 3-8x too small, because the tool resampled frames as if they
were independent.

**Training stochasticity is larger per fold, and no SD above includes it.**
`1_150` moved 0.105 between two identical `--hidden 1024` runs, and 0.055 between
two identical baseline runs, against a 0.041 eval-sampling delta SD. More
annotation won't fix that: 5x the labels on `1_150` only takes its eval SD from
0.042 to 0.021. Seed averaging is the only lever on it.

**Fold-to-fold spread is not a confidence interval.** Training pools overlap
~90% across rotations, so the fold models are correlated and the spread
understates the uncertainty about a genuinely new deployment.

## Hard folds are the target

The endpoint is a new deployment, and a new deployment may well be a quiet one.
A lever that moved only the rich folds would be close to worthless; one that
lifts a near-chance deployment is what this project is for. So when a gain is
concentrated in `1_150`, `1_95` or `willard`, **that concentration is the
result**. Read a delta table as: which way did the hard folds move, and are the
losses elsewhere material? A -0.005 does not cancel a +0.068, and calling that
"3 up, 2 down, inconclusive" is bad accounting.

**Writing a per-fold number down:**

- **Attach its uncertainty**, as in `+0.061 ± 0.041`.
- **Keep weak evidence weak, but not absent.** `+0.061 ± 0.041` is weak evidence
  for a positive effect. Don't collapse it to its sign, don't build a mechanism
  on it, and don't read the next draw's -0.044 as a refutation. Both are single
  samples from a distribution ~0.05 wide; the two `hidden-aves-verify` entries
  made exactly these errors, in opposite directions, on the same data.
- **Don't count signs or group folds.** The headline and the next experiment are
  what update.

**A repeat spread bounds noise *within* a treatment, not a difference *between*
two.** `1_150`'s 0.007-0.062 across identical baseline runs is the right caution
when reading two runs of one config against each other. It says nothing about a
gap produced by a change of configuration; used that way, it talks you out of
every hard-fold result the loop can find. When the gap is large, group the era's
runs by the variable you changed and check whether the groups overlap.

**Confirm a large hard-fold gain with one repeat run.** With no seed control, a
rerun is an independent draw. Frame it as confirmation, not as a check on a
suspect number.

## Selection on the reported metric

Before trusting any gain, ask whether the treatment could have selected on the
statistic it is scored by. `monitor-leakage` (2026-09-09) found that
`context-monitor`'s `1_150` gain was exactly that: with the held-out fold as its
own early-stopping monitor, a `val_sens` run reported the maximum over epochs of
the statistic it was scored on. Grouped by monitor, the `val_sens` runs beat
every `val_loss` run on that fold. Re-scored at a cross-fold epoch, the groups
overlapped completely.

## Trust

`trust` is what a later agent scans before building on your result. Flagging a
hard-fold gain `caveated` on magnitude grounds tells that agent to discount the
very movement the loop exists to produce. Reserve `caveated` for a defect in how
the number was measured, and put "this rests on one fold, here is the group
split" in `conclusion`.

`context-stack` (+0.050, `artifact`) is why the field exists. A same-label-
neighbour bug in the eval inflated it, and only `context-embedder`'s honest
rerun (+0.022, `clean`) showed the gap.
