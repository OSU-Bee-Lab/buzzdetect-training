# pitchshift-dropout

## Hypothesis

`dropout-verify` (2026-09-12) tested `Dropout(0.2)` on plain YAMNet and found
it dead flat (-0.001, 4/8 up 4/8 down). But `03_train/CLAUDE.md`'s sparsity
argument for why dropout shouldn't matter is about the *representation*
(89.6% exact zeros, non-negative), and every structural lever this era has
shown effects can be representation-dependent (`--hidden 1024` was null on
plain YAMNet, +0.022-0.031 on `yamnet_aves`). `yamnet_pitchshift` concatenates
a pitch-shifted copy of the same YAMNet code onto itself, so it's still
89.6%-sparse non-negative in each half -- the sparsity argument predicts
dropout should still do nothing, but it had never been tested on this
representation. Matched control: `pitchshift-repeat` (the era leader's
confirmed second draw), not `cv-baseline-v3` or `yamnet-pitchshift` r1,
so the only variable is `--dropout 0.2` vs `0.0`.

## Changes

`--dropout 0.2` added to an otherwise identical `yamnet_pitchshift` run.
`embedders/yamnet_pitchshift` copied into the worktree (it's on the shared
tree already from `exp/yamnet-pitchshift`'s promotion, so this was a
plain symlink pickup, no code change).

## Results

| fold | pitchshift-repeat (control) | +dropout 0.2 | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| 1_150 | 0.331 | 0.343 | +0.012 | 0.021 | 21 |
| 1_95 | 0.076 | 0.073 | -0.004 | 0.010 | 46 |
| 1_29 | 0.468 | 0.465 | -0.003 | 0.010 | 32 |
| 53 | 0.436 | 0.440 | +0.003 | 0.008 | 28 |
| willard/1_11 | 0.430 | 0.451 | +0.021 | 0.012 | 26 |
| wooster/1_143 | 0.533 | 0.541 | +0.007 | 0.011 | 22 |
| 1_37 | 0.439 | 0.446 | +0.007 | 0.019 | 14 |
| 1_114 | 0.414 | 0.437 | +0.023 | 0.016 | 28 |

- mean sens@fpr0.005: 0.391 -> 0.399 (headline delta +0.008, delta SD 0.005,
  eval sampling only)

6/8 folds up, 2 down. The three largest movers (willard +0.021 ± 0.012,
1_114 +0.023 ± 0.016, 1_150 +0.012 ± 0.021) are each under 2x their own
eval-sampling SD -- suggestive but not individually resolvable, and no
repeat draw exists to separate this from training stochasticity (larger
again per fold, per every other dropout/hidden comparison this era).

## Conclusion

**Confirms `dropout-verify`'s null result on a second representation.**
Headline +0.008 ± 0.005 is an order of magnitude below the ~0.027 MDE,
same shape as plain YAMNet's -0.001. The sparsity argument in
`03_train/CLAUDE.md` predicted this: `yamnet_pitchshift` is still two copies
of the same 89.6%-sparse, non-negative YAMNet code, and dropout on that kind
of input is mild regularisation with nothing much to regularise at this
model size. Weak positive drift across most folds (6/8 up) is noted but not
claimed as a fold-level result -- every named mover is within ~2 sigma of
its own SD, well short of the bar `1_150`-style hard-fold claims need per
LOOP.md. Settles the representation-dependence question raised for item 1d:
dropout is null on both plain YAMNet and `yamnet_pitchshift`; no reason left
to expect it matters on any YAMNet-derived block without cross-frame mixing.
