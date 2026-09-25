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

Also fixed a real bug in shared code, hit mid-run: `03_train/surprisal.py`'s
`write_fold_surprisal()` called `model()` on a whole fold's embeddings
unbatched (unlike `train.py::_score_fold`, which already batches the
identical shape of call in 1024-row chunks). Fold 3/8 finished training
cleanly, then GPU-OOM'd inside the surprisal export. Fixed on main
(`03_train: batch write_fold_surprisal's model() call, matching _score_fold`,
`e4f0da7`) and cherry-picked into this worktree (`fe9fd97`, since a full
`git merge main` chokes on the worktree's symlinked embedders -- applied the
one file directly instead). Training was retried on `--cpu` first per
CLAUDE.md's documented GPU-OOM fallback, but that fallback assumes a genuine
capacity OOM and this wasn't one: CPU ran at ~15 min/*epoch* (not fold) on
this 24576-d width, ~90 h projected for the remaining folds. Killed and
resumed on GPU once the real fix landed; folds 3-8 then trained normally
(~15 min/fold).

## Results

```
tools/results.py <trunk-ft-pitchshift worktree>/models/trunkps-ft-1e5 trunkpsd12-ft-1e5
```

| fold | trunk-ft-pitchshift (layer12 cut) | this (layer11 cut) | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.454 | 0.520 | +0.066 | 0.035 | 32 |
| 53 | 0.573 | 0.543 | -0.030 | 0.018 | 28 |
| 1_11 | 0.565 | 0.594 | +0.029 | 0.028 | 26 |
| 1_143 | 0.550 | 0.586 | +0.036 | 0.022 | 22 |
| 1_150 | 0.287 | 0.312 | +0.025 | 0.050 | 21 |
| **1_95** | 0.156 | 0.205 | **+0.049** | 0.021 | 46 |
| 1_37 | 0.493 | 0.551 | +0.058 | 0.031 | 14 |
| **1_114** | 0.398 | 0.462 | **+0.064** | 0.040 | 28 |

Headline: 0.434 → 0.472 (+0.038 ± 0.011, ~3.5σ), 7/8 folds up. Inclusive
(with `_quiet`): 0.359 → 0.393 (+0.034). Tiers: untagged +0.045 (n=2418,
real detection gain, not a background artifact), loud +0.028 (thin,
n=122), background +0.010, quiet +0.021.

## Conclusion

**A real gain, and it clears the falsifier's stronger branch.** +0.038 ±
0.011 is roughly 3.5x its own eval-sampling SD and well outside the era's
~0.012-0.016 headline noise floor -- not the "inside noise, matched" result
`trunk-depth-headtohead` found for the *same* depth-cut swap under **plain**
fine-tuning (-0.020 ± 0.011, paired, statistically equivalent). Depth
equivalence does not survive once pitch-shift concat is stacked on: with two
co-tuned views feeding the trainable block instead of one, cutting one layer
earlier is a real, positive lever here.

**Both hard folds move up, not just `1_95`.** `1_95` (jet) +0.049 ± 0.021
(~2.3σ) continues the direction `trunk-ft-pitchshift` itself already showed
over its own frozen control. The more notable move is `1_114` (trill):
+0.064 ± 0.040 (~1.6σ, moderate but real direction) -- `trunk-ft-pitchshift`'s
own conclusion flagged `1_114` moving the *wrong* way relative to its frozen
concat (-0.038), the one blemish on that experiment's otherwise clean story.
The depth12 cut recovers it while holding the `1_95` gain, exactly the
"more surprising, stronger result" branch of this experiment's pre-registered
falsifier. Only `53` moves down, and weakly (-0.030 ± 0.018, ~1.7σ).

**This is the new era lead** (0.472, up from `trunk-ft-pitchshift`'s 0.434,
which was itself the prior lead) via a pure structural swap -- one layer
earlier on the fine-tuning cut -- with no new audio transform, no new head
architecture, and no hyperparameter search. `untagged` moving with the
headline (+0.045 on 2418 frames) says this is a broad detection gain, not a
background-tier artifact. Worth a repeat run to confirm before leaning on it
further (LOOP.md: confirm a large gain with one repeat run) and a natural
next base for future pitch-shift/fine-tuning work in this era.

`trust`: clean. Real effect size, consistent direction on 7/8 folds, both
target hard folds move the right way, mechanism is a straightforward
recombination of two already-validated pieces.
