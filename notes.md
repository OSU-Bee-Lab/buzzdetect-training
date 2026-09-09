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

## Conclusion
