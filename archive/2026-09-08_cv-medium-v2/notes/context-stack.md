# context-stack

## Hypothesis

The probe sees one 0.96 s YAMNet frame at a time and nothing either side of it.
A buzz is a sustained sound; the false positives that hurt in deployment are
loud transients — a bang, a truck door, a bird. Those are separable from buzz by
duration alone, and that information is present in the neighbouring frames and
currently thrown away.

Widening each frame's input with its immediate neighbours (t-1, t, t+1) should
raise sensitivity at a fixed FPR, because the FPR-setting hard negatives are
disproportionately transient while true buzz frames sit inside a run of similar
frames.

Cost: the input goes from 1024 to 3072 dimensions, tripling the dense layer's
parameters against the same data. If the gain is real it has to beat that.

Baseline: `yamnet_medium_general`, sens_persite @ fpr0.005 = 0.206.

## Changes

- `03_train/train_utils.py`: `stack_context(embeddings, k)` — widens
  (n_frames, n_dim) to (n_frames, n_dim*(2k+1)), edges clamped by repetition.
  `CONTEXT_FRAMES = 1` is the knob; 0 restores the baseline.
- `03_train/train.py`: applied per snip in both `_to_tf` (training) and
  `_score_fold` (scoring), never across snips; input layer widened to match.

One change, nothing else touched. No re-extraction — this reshapes cached
embeddings at load.

Deviation from LOOP.md: run on `medium`, not `lite`, so the comparison is
against `yamnet_medium_general` on identical folds.

## Results

`sens_persite` @ fpr0.005: baseline **0.206** -> this **0.256** (+0.050).

| fold | val frames | baseline | context-stack | delta |
|---|---|---|---|---|
| 2025-06-23/1_23 | 314 | 0.326 | 0.514 | +0.188 |
| 2025-07-03/1_37 | 4712 | 0.279 | 0.430 | +0.151 |
| 2024-07-26/1_143 | 4707 | 0.260 | 0.407 | +0.147 |
| 2023_R3_Marysville/53 | 4710 | 0.386 | 0.490 | +0.104 |
| MustardBumbler/1_29 | 6908 | 0.448 | 0.545 | +0.097 |
| 2026-04-08/1_150 | 4891 | 0.028 | 0.069 | +0.041 |
| 2025-08-12/1_114 | 3768 | 0.163 | 0.180 | +0.017 |
| 2025-08-27/48 | 1570 | 0.013 | 0.020 | +0.007 |
| 2026-05-06/1_95 | 6606 | 0.027 | 0.023 | -0.004 |
| 2024-08-07/1_11 | 4713 | 0.177 | 0.103 | -0.074 |
| 2025-08-05/31 | 942 | 0.157 | 0.040 | -0.117 |

8 of 11 folds up, mean +0.051, median +0.041. Restricting to the eight folds
with >=3000 validation frames — the ones whose per-fold sensitivity is worth
believing — 6 of 8 up, mean +0.060. The largest single loss is the
second-smallest fold in the set (942 frames), where per-fold sensitivity is
close to unmeasurable either way. The one substantial loss in a large fold is
willard (-0.074).

Also worth noting: the folds that gain most are the buzz-dense ones, and the
two folds that sit near zero for both models (2026-05-06, 2025-08-27) barely
move. Context doesn't rescue a deployment the probe can't hear at all.

## Conclusion

Positive, and larger than anything in the pre-CV log. Three frames of context
is the single cheapest structural change tried so far: no re-extraction, no
embedder change, one helper function.

Three caveats, in descending order of how much they should temper the number.

**Not shippable as it stands.** `model.py`'s `predict()` feeds raw 1024-dim
embeddings straight into the model, which now expects 3072. Deploying this
means mirroring `stack_context` in `write_model_py`'s generated wrapper and in
buzzdetect's inference path, including a decision about what to do at the edges
of an inference buffer. Until then the shipped `model.keras` here is scoring
artifact only.

**The eval geometry is friendlier than continuous audio.** Every frame in a snip
carries that snip's label, so a stacked frame's neighbours always agree with it,
in training and in scoring alike. On continuous audio a buzz starts and stops
mid-stream and boundary frames get mixed context. The comparison against
baseline is fair — both models saw the same snips — but the deployment gain will
be smaller than +0.050. It should not vanish: duration genuinely separates a
sustained buzz from a bang, which is the mechanism being tested.

**No seed control.** Run-to-run variance is unmeasured pipeline-wide. A single
fold flipping would prove nothing; 8 of 11 moving the same way, and 6 of 8 among
the trustworthy folds, is harder to explain as noise. A repeat under a fresh
`--name` would quantify it.

Next, in order of value: (1) mirror the stacking in the inference path so the
result can actually ship; (2) sweep k in {2, 3} — if the mechanism is duration,
more context should keep helping until it starts smearing onset; (3) ask why
willard lost, since it is a large fold and the only substantial regression.
