# hidden-aves-verify-r2

## Hypothesis

Repeat draw of `hidden-aves-verify`: is the +0.022 headline gain from
`--hidden 1024` on `yamnet_aves`, and specifically the coherent both-hard-folds-up
pattern (1_150 +0.061, 1_95 +0.051), a real effect or a lucky draw? No seed
control exists in this pipeline, so an independent rerun of the identical
config is the only way to check. Per `LOOP.md`: confirm a hard-fold gain with a
repeat run rather than discounting it on a within-treatment spread argument.

## Changes

None — identical command to `hidden-aves-verify`, different `--name` only:

```
03_train/main.py --name hidden_aves_verify_r2 --set medium --embedder yamnet_aves \
                 --translation general --hidden 1024 --dropout 0 --fixed-epochs 400 -y --verbose
```

## Results

Against the matched control (`yamnet-aves-verify`, hidden=0). `± SD` columns are
paired per-fold delta SDs from `tools/eval_sampling_sd.py --other`, evaluation
sampling only — training stochasticity is larger again per fold and is not in
them:

| fold | control | r1 | r2 | r1 delta | r2 delta | ± SD (r2) | buzz events |
|---|---|---|---|---|---|---|---|
| 1_114 | 0.156 | 0.075 | 0.112 | -0.081 | -0.044 | 0.024 | 14 |
| 1_150 | 0.304 | 0.365 | 0.260 | +0.061 | -0.045 | 0.046 | 14 |
| 1_37 | 0.382 | 0.347 | 0.367 | -0.035 | -0.015 | 0.043 | 5 |
| 1_95 | 0.098 | 0.149 | 0.120 | +0.051 | +0.022 | 0.015 | 23 |
| 53 | 0.493 | 0.521 | 0.556 | +0.028 | +0.064 | 0.041 | 9 |
| 1_29 | 0.465 | 0.495 | 0.550 | +0.030 | +0.085 | 0.041 | 14 |
| wooster/1_143 | 0.487 | 0.533 | 0.581 | +0.046 | +0.094 | 0.054 | 9 |
| willard/1_11 | 0.444 | 0.526 | 0.533 | +0.082 | +0.089 | 0.028 | 14 |

- headline vs control: r1 **+0.022 ± 0.012**, r2 **+0.031 ± 0.014** — the
  direction and rough size replicate
- r1 vs r2 directly: +0.008, inside the ~0.012 headline delta SD — the two
  draws agree

## Conclusion

**The headline replicated. `--hidden 1024` on `yamnet_aves` is worth ~+0.027,
and that is the whole result.**

**What this run does *not* show — and the first version of this file claimed it
did — is that r1's `1_150` gain was an artifact.** That reading took the
0.105 range on `1_150` (+0.061 -> -0.045) as a refutation, downgraded r1 to
`caveated`, and called its mechanism note wrong. It was over-reading in the
mirror image of r1's own error.

`1_150` holds **14 buzz events**, not 137 independent frames. Its paired delta
SD is **0.046** from evaluation sampling alone, and training stochasticity is
the larger term again (this pair of identical runs is itself the best estimate
of it). A 0.105 range across two draws is about one combined sigma: entirely
ordinary, and evidence for nothing in either direction. A threshold-free read
(pAUC over FPR<=0.005: control 0.217, r1 0.270, r2 0.208) confirms the two
models do differ on that fold — this is not a thresholding artifact — but not
by a margin this fold can resolve.

So: r1's +0.061 was weak evidence *for* a positive effect on `1_150`, which is
not the same as no evidence and not the same as a mechanism. r2's -0.045 is a
second draw from the same wide distribution. Both entries' headline numbers are
sound; both entries' `1_150` narratives should never have been written. r1 is
restored to `trust: clean` — its measurement was right, only its per-fold story
overreached.

`1_95` stayed on the correct side of control in both draws (+0.051 ± 0.020,
+0.022 ± 0.015), which is the more interesting standing observation, and still
short of a mechanism. Do not build a `1_150` story on this config; a third draw
would not fix it either, since the fold's resolution is ~0.09 and the effects in
play are a third of that.

## Instrumentation that came out of this

The loop had no error bar on a per-fold delta, which is how two consecutive
agents built opposite stories out of the same ±0.045. Now on main:

- `folds_sx.csv` carries `buzz_events` / `buzz_events_exclquiet`;
  `predictions.csv` carries a `sample` column so event boundaries are exact
  rather than inferred from row adjacency.
- `tools/eval_sampling_sd.py` resamples buzz **events**, not frames, and takes
  `--other` for paired per-fold delta SDs. Its previous frame bootstrap
  understated per-fold SD by up to ~8x (`1_29`: 0.015 reported, 0.115 actual).
- `tools/compare_folds.py` prints the event count beside each delta.
- LOOP.md: per-fold deltas are reported with their SD, and "unsure" is a normal
  verdict for a fold.
