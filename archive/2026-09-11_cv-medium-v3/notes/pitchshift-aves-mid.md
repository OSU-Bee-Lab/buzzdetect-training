# pitchshift-aves-mid

## Hypothesis

IDEAS.md item 17: stack this era's two strongest, mechanistically distinct
confirmed levers -- `yamnet_pitchshift` (+0.069, YAMNet frequency register)
and `aves-mid` (+0.026 to +0.032, AVES representation depth, concentrated on
the hard folds unlike item 16's context stack). Unlike item 16, `aves-mid`'s
own gain is itself hard-fold-concentrated, so this pairing has a real chance
of being additive on the folds that matter. Falsifier: if `1_150`/`1_95` do
not move beyond what either lever gets alone, this is redundant capacity, not
a stack.

## Changes

New embedder `embedders/yamnet_pitchshift_aves_mid/` (shared tree). Shares
one 1.0 s frame grid the way `yamnet_aves` does: two YAMNet passes on a
centre-cropped 0.96 s (unshifted, and pitch-shifted up an octave -- same
mechanism as `yamnet_pitchshift`), plus AVES layers 6/9/12 mean-pooled over
the full 1.0 s. `[yamnet(t), yamnet(pitch_up(t)), aves_l6/9/12(t)]` = 4352-d.
Unit-verified: unshifted block matches `yamnet_aves`'s own YAMNet half,
shifted block differs from it, AVES layer blocks match `yamnet_aves_mid`'s
own blocks exactly. `to_onnx()` deferred -- the whole `yamnet_pitchshift`
family has no working export yet, a pre-existing gap, not new here. One
`medium` extraction (reused the shared 1.0 s/16kHz raw-audio cache; no
crashes). Config matched to `pitchshift-repeat`/`aves-mid-repeat`:
`--translation general --fixed-epochs 400`, no dropout.

**Note on frame grid:** this embedder's `framelength_s = 1.0` (not
`yamnet_pitchshift`'s 0.96 s), so its `folds_sx.csv` frame counts don't align
1:1 with the plain pitch-shift family's -- `tools/eval_sampling_sd.py`
errors (index mismatch) when paired against `pitchshift-repeat` directly for
that reason. It aligns fine against `aves-mid-repeat` (same 1.0 s grid).
Aggregate `sensitivity_exclquiet` deltas from `compare_folds.py` are still
valid either way -- only the frame-indexed bootstrap needs a same-grid
partner.

## Results

vs `pitchshift-repeat` (plain `yamnet_pitchshift`):

| fold | pitchshift-repeat | this run | delta |
|---|---|---|---|
| **1_150** | 0.331 | **0.468** | **+0.137** |
| **1_95** | 0.076 | **0.170** | **+0.094** |
| willard/1_11 | 0.430 | 0.515 | +0.085 |
| 1_29 | 0.468 | 0.551 | +0.083 |
| 53 | 0.436 | 0.484 | +0.048 |
| 1_143 | 0.533 | 0.571 | +0.038 |
| 1_37 | 0.439 | 0.410 | -0.029 |
| **1_114** | 0.414 | **0.210** | **-0.204** |

mean sens@fpr0.005 (excl. quiet): 0.391 -> 0.422 (+0.031). 6/8 folds up.
`1_114` is the one clear casualty -- `yamnet_pitchshift`'s own standout fold
(previously logged as its best single-fold gain, +0.181 over the plain
anchor) reverts toward `aves-mid`'s own (lower) `1_114` value in the combo,
suggesting the AVES side dominates that fold's behaviour rather than the two
adding.

vs `aves-mid-repeat` (same 1.0 s frame grid, bootstrap-comparable):

| fold | aves-mid-repeat | this run | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| **1_150** | 0.346 | 0.468 | **+0.122** | 0.053 | 20 |
| 1_37 | 0.348 | 0.410 | +0.061 | 0.026 | 14 |
| **1_95** | 0.123 | 0.170 | +0.047 | 0.019 | 42 |
| 1_114 | 0.169 | 0.210 | +0.042 | 0.021 | 31 |
| 1_143 | 0.543 | 0.571 | +0.029 | 0.053 | 21 |
| 1_29 | 0.527 | 0.551 | +0.023 | 0.022 | 32 |
| willard/1_11 | 0.500 | 0.515 | +0.015 | 0.022 | 28 |
| 53 | 0.479 | 0.484 | +0.005 | 0.022 | 25 |

**8/8 folds up against `aves-mid-repeat`.** mean sens@fpr0.005 (excl. quiet):
0.379 -> 0.422 (**+0.043 +/- 0.012**, ~3.6 sigma, clears the ~0.027 MDE
cleanly). `1_150` (+0.122, ~2.3 sigma) and `1_95` (+0.047, ~2.5 sigma) --
both named hard folds -- move further still on top of `aves-mid`'s own gain.

Tiers (vs `pitchshift-repeat` / `aves-mid-repeat` / this run):
`sensitivity_exclquiet` 0.391 / 0.379 / **0.422** (beats both), `untagged`
0.422 / 0.408 / **0.453** (beats both), `background` 0.324 / 0.390 / **0.401**
(beats both), `loud` 0.837 / 0.685 / 0.712 (between the two, thin tier
~115-122 frames).

## Conclusion

**The falsifier does not fire -- both hard folds move beyond either parent
lever alone.** `1_150` (+0.137 over pitch-shift, +0.122 over aves-mid) and
`1_95` (+0.094 over pitch-shift, +0.047 over aves-mid) are additive, not
redundant: the combination beats both single-lever results on the exact
folds this project is for. The headline beats both parents outright (0.422
vs 0.398 plain pitch-shift and 0.386 aves-mid) and both `untagged` and
`background` tiers beat both parents too -- a broad detection gain, not a
tier-specific artifact.

**The one real cost is `1_114`**, previously `yamnet_pitchshift`'s own best
fold, which drops to roughly `aves-mid`'s own (lower) level in the combo
rather than keeping pitch-shift's edge there. Whatever mechanism gave
pitch-shift its `1_114` strength does not survive stacking with AVES depth --
worth a mechanism note if this config is pursued further, but not disqualifying
given the size of the gain everywhere else.

**This is provisionally the best config of the era (0.422 vs the prior best
single lever's 0.398), and given `1_150`'s size and its documented
training-stochasticity range, it needs a repeat draw before being trusted at
that magnitude** -- launching `pitchshift-aves-mid-repeat` immediately, same
protocol as `aves-mid`.
