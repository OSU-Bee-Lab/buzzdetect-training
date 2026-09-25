# context-monitor-r2

## Hypothesis

A confirmation repeat of `context-monitor` (0.307, the era's best), run because
LOOP.md's standing practice is to confirm a large hard-fold gain with an
independent draw — there is no seed control, so a rerun of the identical config
*is* the seed control. **This is confirmation, not a check on a suspect
number**; `context-monitor`'s amendment already settled that the `1_150` gain is
a treatment effect (the `val_loss`/`val_sens` groups do not overlap on that
fold, and `best_epoch` 5 -> 160 gives it a mechanism).

Two things it tests:

1. **Does the headline hold near 0.307?** Baseline run-to-run SD is 0.0095
   (`probe-grid`, n=3), so a second draw landing inside ~0.29-0.32 confirms;
   landing near 0.258 (context alone) would say the composition was a draw.
2. **The one genuinely n=1 part: `1_150` at 0.219.** That is the top of the
   era's four `val_sens` runs by 0.048 (previous max 0.158). If the repeat lands
   in 0.09-0.16, the extra lift was draw and the composed config is worth
   ~0.28-0.29; if it lands near 0.219 again, there is a real
   context x monitor interaction on that fold on top of the additive story.

Falsifier for the entry as a whole: a headline at or below 0.258 with `1_150`
back under 0.10.

## Changes

None. Identical config and identical code to `exp/context-monitor` (branched
from it at 31a9dc0), a fresh `--name`, no extraction — `yamnet_context`'s
`medium` cache is read through the shared symlink.

Run: `03_train/main.py --name context_monitor_r2 --set medium --embedder
yamnet_context --translation general --monitor val_sens`.

## Results

CV: 5 folds, 17:50:15 -> 18:15:34 (25 min). Shipped model not trained, as intended.

**Primary comparison — paired against `context_monitor`, the identical config (draw 1 vs draw 2):**

| fold | draw 1 | draw 2 (this) | delta | buzz frames | best_epoch (1 -> 2) |
|---|---|---|---|---|---|
| Fit+Fast/53 | 0.465 | 0.520 | +0.055 | 1031 | 17 -> 226 |
| Diel Drivers/1_150 | 0.219 | **0.219** | **0.000** | 146 | 160 -> 192 |
| Diel Drivers/1_95 | 0.044 | 0.039 | -0.005 | 433 | 18 -> 18 |
| JamesU/1_29 | 0.556 | 0.547 | -0.009 | 2144 | 107 -> 168 |
| willard/1_11 | 0.249 | 0.236 | -0.013 | 305 | 22 -> 25 |

- mean sens@fpr0.005: **0.307 -> 0.312 (+0.005)**, 1 up / 3 down / 1 flat.
- against `context_embedder`: **0.258 -> 0.312 (+0.054)**, 4 up / 1 down.
- against `cv_baseline`: **0.218 -> 0.312 (+0.094)**, **5 up / 0 down** — the
  second all-up run in the era, and both are this config.

**Both hypotheses confirmed.**

1. **The headline holds.** 0.312 vs 0.307 is +0.005 against a 0.0095
   single-run SD (`probe-grid`, n=3) — two draws of the same config
   indistinguishable from each other, and both ~0.09 clear of `cv_baseline`.
   The falsifier (a drop toward 0.258, context alone) did not fire; the composed
   config's mean over the two draws is **0.310**.

2. **`1_150` = 0.219 again, and it is an independent draw, not a copy.** The
   fold's granularity is 1/146 = 0.0068, so both runs detected exactly 32 of 146
   buzz frames at the fold's own 0.5% FPR threshold — from different training
   runs shipping different epochs (160 vs 192). The era's `val_sens` group on
   this fold is now 0.089, 0.158, 0.171, 0.219, **0.219**; the `val_loss` group
   is still 0.007-0.062, and the groups remain disjoint. The n=1 concern is
   resolved in the direction the first run suggested: **the composed config's
   0.219 sits above the top of the monitor-alone runs (0.158) twice**, so there
   is a real context x monitor lift on `1_150` on top of the additive story,
   not a lucky draw.

**Where the two draws actually differ is `Fit+Fast/53`, and that is the
stopping rule being genuinely bimodal.** Draw 1 shipped epoch 17 and scored
0.465; draw 2 shipped epoch 226 and scored 0.520. That is the same
early-vs-late `val_sens` ambiguity that `context-monitor` flagged running the
other way at this fold (144 -> 17 vs `context_embedder`). The fold has 1031 buzz
frames, so it is not thin — read it as run-to-run variance in which peak the
monitor lands on, and note that the *rich* folds are now the ones supplying most
of this config's spread, which is a reversal of the usual pattern.

Every fold reached the target FPR; `neg_frames` 24-35 as always. `1_95` at 0.039
is the fifth intervention not to move it.

## Conclusion

**CONFIRMED. `yamnet_context` + `--monitor val_sens` is a real +0.09 over
`cv_baseline` and it replicates.** Two independent draws at 0.307 and 0.312
(mean 0.310), each 5 folds up / 0 down against baseline. The one part that was
n=1 — `1_150` at 0.219, 0.048 above the best monitor-alone run — reproduced
exactly, so the composed config genuinely lifts the era's hardest resolvable
fold from near-chance (0.021 baseline) to 0.219, and that is the result this
project is for.

Standing follow-ups, unchanged in priority by this run: making `--monitor
val_sens` the default (it is load-bearing for the best config, and is Luke's
call plus probably an era boundary), and `1_95`, which nothing has moved.

One new observation worth carrying: **under `val_sens` the stopping epoch is
bimodal on rich folds** (`Fit+Fast/53` shipped 17 and 226 across two identical
runs, a 0.055 swing). The monitor's gain is not in question, but a
tie-breaking rule — e.g. preferring the later of two comparable `val_sens`
peaks, or smoothing the curve before taking the argmax — looks like a cheap
variance reduction on top of it, and every run's `val_sens_fpr0.005_curve` is
already persisted, so it can be replayed offline with no training.
