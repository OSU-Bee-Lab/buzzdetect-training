# trunk-ft-pitchshift

## Hypothesis

New proposal, not from IDEAS.md's queue. This era's two largest confirmed
non-AVES single levers have never been combined: `yamnet-pitchshift` (concat
`[e_t, e_shifted(t)]`, +0.069, frozen probe) and `trunk-ft-v3` (unfreezing
YAMNet's last two conv blocks, layers 13-14, +0.077, plain audio).
`trunk-ft-v3` already tested ONE combination -- temporal context
(`yamnet_trunk_context`, a shared fine-tuned tail run over `[t-1, t, t+1]`)
-- and found it did not stack with fine-tuning (+0.029 over its own frozen
control vs +0.077 for plain fine-tuning; +0.002 against `trunk3-ft-1e5`
directly). But pitch-shift is a different axis: same timestamp, transformed
audio, not a temporal neighbour, and it has already been shown additive with
a second lever this era (`pitchshift-aves-mid` beat both `yamnet-pitchshift`
and `aves-mid` individually). This is the first test of whether it is
additive with backbone fine-tuning too.

`embedders/yamnet_trunk_pitchshift` reuses `yamnet_trunk_context`'s shared-
weight `TimeDistributed(tail)` mechanism (already generic over `n_ctx` in
`yamnet_trunk.build_head()`), but with `n_ctx=2` and the second "context slot"
being the octave-up pitch-shifted twin of the SAME frame (ported unchanged
from `yamnet_pitchshift._pitch_up_octave`) instead of a temporal neighbour.
One shared, fine-tuned copy of layers 13-14 processes both views before their
1024-d codes concatenate into the Dense head.

Prediction: if pitch-shift and fine-tuning are independent levers (as
pitch-shift and AVES depth were), this should beat both `yamnet_trunk`'s
frozen linear probe on the concat (a natural comparator) and `trunk3-ft-1e5`
alone. If the shared-tail TimeDistributed mechanism itself is what stalls
stacking (not something specific to *temporal* context), this should be flat
like the context result.

Comparators: `trunk3-ft-1e5` (0.398, plain fine-tuned trunk, the confirmed
depth) is the primary one. A frozen control on this exact embedder
(`lr_backbone=0`, i.e. the concat with NO fine-tuning) isolates what
fine-tuning itself adds on top of the concat, mirroring how `trunk3-frozen`
isolated fine-tuning's contribution to the plain trunk.

## Changes

New embedder `embedders/yamnet_trunk_pitchshift/embedder.py`: subclasses
`EmbedderYamnetTrunk`, sets `n_ctx=2`, and overrides only `embed()` to run the
frozen layer-12 trunk over both the plain frame and its octave-up twin
(`_pitch_up_octave`, ported unchanged from `yamnet_pitchshift`), concatenating
the two 12288-d codes into one 24576-d row. `build_head()` is inherited
unchanged from `yamnet_trunk` -- its `n_ctx` path already does the shared
TimeDistributed tail + Flatten + Dense.

`03_train/train.py` needed `trunk-ft-v3`'s `build_head`/fp16-batches hook,
which was never merged to main; ported byte-identical (base commit unchanged
on main since).

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_trunk_pitchshift --workers 1
TRUNK_LR_BACKBONE=0 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_frozen.log -- 03_train/main.py --name trunkps-frozen --set medium --embedder yamnet_trunk_pitchshift --translation general --epochs 60 -y
TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train_ft.log -- 03_train/main.py --name trunkps-ft-1e5 --set medium --embedder yamnet_trunk_pitchshift --translation general --epochs 60 -y
```

(`TRUNK_FP16=1 TRUNK_BATCH=1024` are required, not optional defaults --
`trunk-ft-depth12`'s run found the plain path OOMs the 4 GB card immediately
at this width without them.)

## Results

**Memory, not GPU, was the operational hazard here.** Host RSS grew roughly
linearly across folds (~4 GB/fold at this 24576-d width, worse than
`trunk-ft-depth12`'s ~1.9 GB/fold at 12288-d -- scales with embedder width)
even with the existing GPU-session-clear fix in place, since that fix
addresses GPU allocator pileup, not host RAM. With only 23 GB RAM and swap
already thrashing by fold 3 each time, both the frozen and fine-tuned runs
were restarted every 2-3 folds (plain relaunch of the same command; resumes
cleanly via `can_write()`) rather than run start-to-finish in one process.
Flagged as friction with a concrete fix suggestion (an explicit `del` on the
per-fold `TrainingData`/`_Fp16Batches` object, which the current fix doesn't
reach) for whoever picks it up next.

| fold | frozen (concat, no ft) | ft (`lr_backbone=1e-5`) | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.469 | 0.454 | -0.015 | 0.046 | 32 |
| 53 | 0.432 | 0.573 | +0.141 | 0.057 | 28 |
| 1_11 (willard) | 0.437 | 0.565 | +0.128 | 0.038 | 26 |
| 1_143 | 0.532 | 0.550 | +0.018 | 0.027 | 22 |
| 1_150 | 0.296 | 0.287 | -0.009 | 0.042 | 21 |
| 1_95 | 0.073 | 0.156 | +0.083 | 0.028 | 46 |
| 1_37 | 0.480 | 0.493 | +0.013 | 0.061 | 14 |
| 1_114 | 0.436 | 0.398 | -0.038 | 0.042 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): `trunkps-frozen` 0.394 ->
  `trunkps-ft-1e5` 0.434 (**+0.040 ± 0.016**, ~2.5σ), 5/8 folds up, 3/8 down
- inclusive (sensitivity), same thresholds: 0.323 -> 0.359 (+0.036)
- folds that missed fpr 0.005: none

| tier | frozen | ft | delta | frames |
|---|---|---|---|---|
| loud | 0.837 | 0.817 | -0.020 | 122 |
| untagged | 0.426 | 0.459 | +0.033 | 2418 |
| background | 0.321 | 0.403 | +0.082 | 1874 |
| quiet | 0.087 | 0.108 | +0.021 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

**The frozen control itself is the bigger story.** `trunkps-frozen` (concat,
no fine-tuning) scores 0.394 against plain `trunk-ft-v3`/`trunk-ft-depth12`
frozen controls of 0.321 (+0.073) -- the pitch-shift concat alone, through an
*unfine-tuned* trunk tail, recovers most of what plain fine-tuning gets on
its own. Fine-tuning on top of the concat then adds a smaller, moderate
+0.040.

**Against plain fine-tuned trunk** (headline-only, no committed per-fold data
for `trunk-ft-v3`'s `trunk3-ft-1e5` to pair against): 0.398 -> 0.434 (+0.036),
and against this era's other fine-tuned depth (`trunk-ft-depth12`,
0.412): +0.022. Both sit close to this era's ~0.012-0.016 headline-delta
noise floor -- weak evidence of a real edge, not a clean win.

## Conclusion

**Falsifier does not cleanly fire, but does not cleanly confirm either.**
Fine-tuning does add something on top of the pitch-shift concat (+0.040 ±
0.016, ~2.5σ), but the hard-fold signature is mixed, unlike
`pitchshift-contrast`'s (this same batch's other run) clean sweep: `1_95`
(jet) is the one clear hard-fold win here too (+0.083 ± 0.028, ~3σ), but
`1_114` (trill) moves the WRONG way (-0.038, weak) and `1_150` is flat-to-down
(-0.009, inside its own SD). This is the opposite pattern from
`pitchshift-contrast`, where `1_114` was the largest gain of the whole batch
-- consistent with concat and contrast pulling different information out of
the pitch-shifted view, not a contradiction, but it means "pitch-shift stacks
with fine-tuning" is not the clean story predicted. The comparison against
plain fine-tuned trunk (+0.036, no paired SD available) and against
depth12's fine-tuned run (+0.022) both sit at or inside the era's noise
floor -- "at least as good as either single lever, not clearly additive."

The bigger, more confident finding is that the concat itself, even fully
frozen, recovers most of fine-tuning's own gain (0.394 vs plain trunk's
frozen 0.321, +0.073) -- a genuinely large, one-line-of-code-cheap win worth
noting for future trunk-family proposals, though it wasn't this run's
pre-registered question.
