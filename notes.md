# trunk-pitchshift-contrast

## Hypothesis

Two results this era point the same direction but disagree on `1_114`
(trill confusion): `pitchshift-contrast` (frozen YAMNet, explicit signed
contrast `[e_t, e_t - e_shifted(t)]`, +0.048 over a fresh matched control,
`1_114` **+0.153**, the largest single-fold gain logged this era) and
`trunk-ft-pitchshift` (fine-tuned trunk tail, plain CONCAT `[code_plain,
code_shifted]` of the same two views, +0.040 over its own frozen concat
control, but `1_114` **-0.038**, the wrong direction).

The mechanism `asym-context-yamnet` established (a linear readout of a raw
concat cannot compute a contrast; an explicit contrast channel beats it,
+0.046 vs +0.030 at the same budget) predicts the concat design is throwing
away exactly the information the contrast channel would hand it directly.
This embedder applies the contrast explicitly downstream of the SAME
fine-tuned trunk tail `trunk-ft-pitchshift` uses, instead of the frozen
YAMNet embedding `pitchshift-contrast` used: `[code_plain, code_plain -
code_shifted]`, where both codes come from the one shared, jointly
fine-tuned layers-13-14 tail (`TimeDistributed`, inherited from
`yamnet_trunk_context`'s mechanism).

Prediction: if the contrast channel's benefit is orthogonal to fine-tuning,
this should recover `1_114`'s clean win while keeping fine-tuning's own
gain over the frozen baseline -- beating both `pitchshift-contrast`'s frozen
number and `trunk-ft-pitchshift`'s concat number on `1_114` specifically.
If fine-tuning already learns to extract the contrast internally (the tail
is shared and jointly trained across both views, unlike the frozen case),
the explicit contrast should buy little on top of concat.

## Changes

New embedder `embedders/yamnet_trunk_pitchshift_contrast/embedder.py`:
inherits `embed()`/`initialize()`/`_pitch_up_octave` unchanged from
`yamnet_trunk_pitchshift` (same 2 x 12288 float16 cache), overrides only
`build_head()` to join the tail's two 1024-d codes with `Subtract()` +
`Concatenate()` (`[plain, plain - shifted]`) instead of `Flatten()`
(`[plain, shifted]`). Same head width (2048-d), same trainable tail, same
optimizer/LR-multiplier mechanism -- only the join changes.

Smoke test on `lite` first (`03_train`'s TimeDistributed + slicing path is
new; check the graph builds and trains before committing to `medium`).

```bash
tools/launch_job.sh --cpu smoke_extract.log -- 02_set/main.py --set lite --embedder yamnet_trunk_pitchshift_contrast --workers 1 --verbose
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh smoke_train.log -- 03_train/main.py --name test_trunkps_contrast_smoke --set lite --embedder yamnet_trunk_pitchshift_contrast --translation general --epochs 5 -y
```

Then, if the smoke test is clean:

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_trunk_pitchshift_contrast --workers 1
TRUNK_LR_BACKBONE=0 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_frozen.log -- 03_train/main.py --name trunkpscontrast-frozen --set medium --embedder yamnet_trunk_pitchshift_contrast --translation general --epochs 60 -y
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_ft.log -- 03_train/main.py --name trunkpscontrast-ft-1e5 --set medium --embedder yamnet_trunk_pitchshift_contrast --translation general --epochs 60 -y
```

Comparators (already trained, read directly): `trunk-ft-pitchshift`'s
`trunkps-frozen`/`trunkps-ft-1e5`e (`.local/worktrees/trunk-ft-pitchshift/models/`)
-- same tail depth, same pitch-shift mechanism, concat instead of contrast.

*Falsifier:* `1_114` is the named test. If `trunkpscontrast-ft-1e5`'s
`1_114` delta (vs `trunkpscontrast-frozen`, or vs `trunkps-ft-1e5` directly)
is not clearly positive, the contrast mechanism's benefit does not survive
being placed behind a jointly fine-tuned shared tail -- concat and
contrast are equivalent once the tail can adapt. A clean `1_114` win **and**
a headline at or above `trunkps-ft-1e5`'s 0.434 would confirm the explicit-
contrast lever generalizes past the frozen case.
