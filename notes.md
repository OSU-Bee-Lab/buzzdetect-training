# trunk-pitchshift-depth12

## Hypothesis

`trunk-ft-pitchshift` combined this era's two largest single levers (pitch-shift
concat, trunk fine-tuning) at the layer-12 cut (fine-tune 13-14) and got a
mixed hard-fold signature: 1_95 (jet) up, 1_114 (trill) down -- unlike
pitchshift-contrast's clean sweep in the frozen regime. Separately,
`trunk-depth-headtohead` found the layer-11 cut (depth12, fine-tune 12-14)
statistically equivalent to the layer-12 cut on **plain** fine-tuning alone
(-0.020 +/- 0.011 paired delta, inside the era's ~0.012-0.016 noise floor).

Whether that depth equivalence survives once pitch-shift concat is stacked on
top is untested: the extra trainable block (layer12) now has two co-tuned
views (plain + octave-up) feeding it jointly, not one, so the interaction
could differ from the plain-audio case. New embedder
`yamnet_trunk_pitchshift_depth12` is a straight recombination of two already-
merged embedders: `yamnet_trunk_depth12`'s cut/tail (layer11 cut, tune 12-14)
and `yamnet_trunk_pitchshift`'s dual-view octave-up concat + shared
TimeDistributed tail mechanism. No new audio transform, no new architecture
piece -- same structural move `trunk-ft-depth12` itself made relative to
`trunk-ft-v3`.

**Comparator:** `trunk-ft-pitchshift`'s `yamnet_trunk_pitchshift` model
(layer-12 cut + concat), same fine-tuning regime (`lr_backbone=1e-5`,
`TRUNK_FP16=1` per-fold RAM fix), same annotation snapshot as closely as same-
week timing allows. No fresh same-day control is run for this side branch;
read the headline against the era's ~0.012-0.016 noise floor and the paired
depth-headtohead SD as a proxy, per LOOP.md guidance to compare against the
matched prior experiment.

**Falsifier:** if the depth cut is genuinely inert here too, 1_114 stays down
(or flat) and the headline sits within ~0.02 of trunk-ft-pitchshift's 0.434,
matching the depth-headtohead equivalence finding. A depth cut that instead
recovers 1_114 while holding the 1_95 gain would mean depth interacts with
pitch-shift concat differently than with plain fine-tuning -- worth a real
paired rerun to confirm.

## Changes

New embedder `embedders/yamnet_trunk_pitchshift_depth12/embedder.py`:
`EmbedderYamnetTrunkPitchshiftDepth12(EmbedderYamnetTrunkDepth12)`, `n_ctx=2`,
`n_embeddings=24576`. Smoke-tested: 3 synthetic frames -> `(3, 24576)`
float16, `build_head(15, lr_backbone=1e-5)` accepts `(None, 24576)` input.
No changes to `03_train/` or any shared file.
