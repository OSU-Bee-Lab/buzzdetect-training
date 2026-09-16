# Judging a result
There are two comparison to make:

The headline sensitivity at 0.005 FPR. This is the number we ultimately want to improve.
Per-fold sensitivities. These numbers help us dive deeper into headline sensitivity.
An improvement in the headline that only comes from one fold is an acceptable win.
An improvement in the headline that causes some folds' sensitivity to collapse is not. 

## What a number rests on

**A fold's sample size is measured in events, not frames.**
`buzz_events_exclquiet`, not `buzz_frames`
A buzz event spans many frames (one bee heard for 26 s is one observation, not 26), so a fold rests on single-digit to low-tens of events.

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

These numbers are demonstrative and will likely change between eras.

**Training stochasticity, measured 2026-09-15**, is not in the SDs above. Eight
identical CVs of plain YAMNet with `--hidden 1024` (no seed control):

| quantity | SD across runs | SD of a delta between two runs |
|---|---|---|
| one fold's sensitivity | 0.011-0.018 | 0.016-0.026 |
| the headline | 0.005 | 0.007 |

Per fold that is smaller than eval sampling; on the headline it is about half.
The two are independent, so a single-run comparison's full SD is their root sum
of squares: ~0.014 on the headline delta, and `1_150`'s 0.041 eval delta SD
becomes ~0.048. No fold stands out: the hard folds (`1_150`, `1_95`, `1_114`)
wobble no more than the rich ones. Eval sampling is the larger term, so more
annotated events shrink uncertainty more than seed averaging would.

Other configs (the bare linear probe, concatenated embedders) are unmeasured.
Branch `stoch/hidden1024-repeats` holds the eight runs and the recipe.

We're looking for significant gains, not just numerical ones. 

## Hard folds are not an artifact

If an experiment improves only one fold, but it's the hardest fold, don't dismiss it!
The endpoint is a new deployment, and a new deployment may well be a difficult one.
When a gain is concentrated in a hard fold (low sensitivity, high false positives, mostly quiet buzzes, lots of background noise, etc.),
this is an interesting and valuable result. Read a delta table as: which way did the hard folds move, and are the losses elsewhere material?
You may even find that the headline number didn't move but the per-fold sensitivity evened out.
This could reasonably be logged as a win, as it prevents pathological behavior in some deployments (e.g. systematic false positives).


## Stochasticity
- Keep uncertainty in the results, as in `+0.061 ± 0.041`.
- Weak evidence is legitimate, even if it's weak. +0.061 ± 0.041` is weak evidence
  for a positive effect. If the next draw is -0.044, both are single
  samples from a distribution ~0.05 wide. Give an honest interpretation and leave it
  up to future agents to decide what's good enough to follow up on.

Confirm a large gain with one repeat run to confirm that we aren't reading noise.

## Selection on the reported metric

Before trusting any gain, ask whether the treatment could have selected on the
statistic it is scored by. `monitor-leakage` (2026-09-09) found that
`context-monitor`'s `1_150` gain was exactly that: with the held-out fold as its
own early-stopping monitor, a `val_sens` run reported the maximum over epochs of
the statistic it was scored on. Grouped by monitor, the `val_sens` runs beat
every `val_loss` run on that fold. Re-scored at a cross-fold epoch, the groups
overlapped completely.

## Trust

`trust` is what a later agent scans before building on your result.

It judges the measurement separately from the delta:

- **`clean`:** nothing about how it was measured should discount it.
- **`caveated`:** a specific measurement defect means the direction is probably
  right but the size isn't. "It rests on one fold" is not such a defect; that
  goes in `conclusion`.
- **`artifact`:** the setup, not the effect, moved the number enough that
  comparing it at face value would mislead. `context-stack` (+0.050, `artifact`) is why the field exists. A same-label- neighbour bug in the eval inflated it, and only `context-embedder`'s honest rerun (+0.022, `clean`) showed the gap.


