# trunk-pitchshift-depth12-repeat

## Hypothesis

`trunk-pitchshift-depth12` (this batch, 2026-09-23) found a real, clean gain
cutting the fine-tuned trunk tail one layer earlier (layer11 cut, tune 12-14)
while keeping `trunk-ft-pitchshift`'s octave-up dual-view concat: headline
0.434 -> 0.472 (+0.038 +/- 0.011, ~3.5σ), 7/8 folds up, and notably `1_114`
(trill) recovered (+0.064) where plain fine-tuning's own depth swap
(`trunk-depth-headtohead`) found the two cuts statistically equivalent. This
is now the era lead. LOOP.md: "confirm a large gain with one repeat run" --
there is no seed control in this pipeline, so a second CV of the identical
config is the check against TF's run-to-run nondeterminism before leaning on
this result further (the same check `harmonic-comb`'s repeat run served, and
`pitchshift-aves-mid-repeat`, `aves-mid-repeat`, `pitchshift-decimate-up-repeat`
before it).

**Falsifier:** if the repeat's headline lands well below 0.472 (inside
`trunk-ft-pitchshift`'s 0.434, or with `1_114`'s gain evaporating), the first
run was a favorable draw, not a real depth effect, and the "new era lead"
claim should be walked back to "matched, not ranked" -- the same outcome
`trunk-depth-headtohead` reached for the plain-fine-tuning version of this
comparison.

## Changes

None -- identical config, same embedder (`yamnet_trunk_pitchshift_depth12`,
already extracted, cache reused), same command, new `--name` only:

```bash
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_ft.log -- \
  03_train/main.py --name trunkpsd12-ft-1e5-r2 --set medium \
  --embedder yamnet_trunk_pitchshift_depth12 --translation general --epochs 60 -y
```

Comparator: `trunk-pitchshift-depth12`'s own `trunkpsd12-ft-1e5` (paired, same
annotation snapshot) and `trunk-ft-pitchshift`'s `trunkps-ft-1e5` (the prior
lead, for the same paired comparison the first run made).
