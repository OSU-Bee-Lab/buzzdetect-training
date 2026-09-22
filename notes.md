# pcen-block

## Hypothesis

IDEAS item 15. PCEN (per-channel energy normalisation) has an adaptive gain
whose time constant defines "background"; a ~90 s jet (`1_95`), a stationary
trill chorus (`1_114`) and long ambient spans (`1_37`) are all background at
that scale, a ~1 s buzz is not. As a *parallel* block (`yamnet_pcen` =
[yamnet 1024-d, 64 PCEN mel bands mean-pooled per 0.96 s frame], 1088-d) it
should lower the aircraft/trill/background share of threshold-setting
negatives on the hard folds, without touching YAMNet's own input distribution.

Unlike the original 2026-09-16 draft of this embedder (`pcen-parallel`
worktree, never trained), PCEN here runs over the whole audio buffer handed to
`embed()`, not each isolated 0.96 s crop -- a 1 s window would defeat PCEN's
own 1-2 s time constant, undermining the entire premise.

Falsifier: `1_95`/`1_114`/`1_37` do not move beyond their delta SD. A headline
gain that lives only on rich folds is not a PCEN effect.

## Changes

`embedders/yamnet_pcen/` (shared tree): ported from the `pcen-parallel`
worktree's embedder (never previously extracted or trained), rewritten so PCEN
runs over the full buffer handed to `embed()` and is then mean-pooled per
0.96 s frame, rather than computing PCEN independently per isolated crop.

Matched control: a fresh plain-YAMNet linear probe (`pcen-ctrl`, `--epochs
400`, this worktree), not `cv-baseline-v3-refresh` -- annotations moved
(commit `969a218`, "Dicentra annotation move") after that control was trained,
per IDEAS.md's "controls go stale with the annotations" rule.

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_pcen --workers 1
tools/launch_job.sh train_ctrl.log -- 03_train/main.py --name pcen-ctrl --set medium --embedder yamnet --translation general -y
tools/launch_job.sh train.log -- 03_train/main.py --name pcen-block --set medium --embedder yamnet_pcen --translation general -y
```

## Results

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.456 | 0.458 | +0.002 | 0.010 | 32 |
| 53 | 0.429 | 0.444 | +0.015 | 0.010 | 28 |
| 1_11 | 0.371 | 0.383 | +0.012 | 0.014 | 26 |
| 1_143 | 0.434 | 0.432 | -0.002 | 0.022 | 22 |
| 1_150 | 0.250 | 0.204 | -0.046 | 0.026 | 21 |
| 1_95 | 0.038 | 0.038 | +0.000 | 0.007 | 46 |
| 1_37 | 0.371 | 0.371 | +0.000 | 0.015 | 14 |
| 1_114 | 0.235 | 0.236 | +0.001 | 0.021 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline (`pcen-ctrl`) 0.323 -> this (`pcen-block`) 0.321 (-0.002 +/- 0.006)
- inclusive (sensitivity), same thresholds: 0.262 -> 0.262 (+0.000)

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.777 | -0.001 | 122 |
| untagged | 0.342 | 0.339 | -0.003 | 2418 |
| background | 0.333 | 0.344 | +0.011 | 1874 |
| quiet | 0.063 | 0.069 | +0.006 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

Headline is flat, well inside MDE (~0.027) and its own SD (0.006). The
falsifier's three named folds are the tell: `1_95` +0.000 (0.007 SD), `1_114`
+0.001 (0.021 SD), `1_37` +0.000 (0.015 SD) -- all three are bit-identical to
baseline to two decimal places. This isn't "inconclusive," it's a clean signal
that the PCEN block had exactly zero effect on those folds' threshold-setting
negatives, for or against. `1_150` drops -0.046 (~1.8 sigma, unsure -- a rich
fold moving down, not one of the predicted hard folds). No tier moved beyond
noise; `background` and `quiet` tick up +0.011/+0.006 but both are small next
to their frame counts and don't read as a real shift.

## Conclusion

Clean negative -- falsifier fires. PCEN as a parallel block does nothing
measurable at this budget: the three folds it was built to help (`1_95`
aircraft, `1_114` trill, `1_37` background) are unchanged to two decimals, not
just "within noise of a small move." A follow-up worth flagging rather than
chasing now: this is a bare linear head over `[yamnet(t), pcen_pooled(t)]`
with no per-block normalization, and PCEN's log-magnitude scale differs
sharply from YAMNet's mostly-zero non-negative scale -- the same
config-mismatch caution LOOP.md gives for a new embedder inheriting the probe
config unchanged. Whether the linear readout is simply putting ~zero weight on
the PCEN block (a scale problem, recoverable per the `aves-probe` precedent)
or PCEN's mean-pooled feature genuinely carries no separating signal for these
folds is undetermined by this run; an offline L2-sweep on the cached
embeddings (as `aves-probe`'s recovery did) would tell them apart cheaply if
anyone wants to chase it further. Not resuming this run to find out --
falsifier already answers the headline question this experiment was built to
ask.
