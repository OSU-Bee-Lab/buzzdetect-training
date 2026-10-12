# ps-d8-ens-same5
## Hypothesis
Training arm (IDEAS 30a). `ps-d8-ensemble` found the logit mean of three plain `ps-depth8` draws at 0.558
(members 0.497-0.515) and a five-member mean at 0.575, but the five mixed depths (three depth8, one depth4,
one depth10), so two things are confounded: more members, and members of different depth. Two more plain
`ps-depth8` draws (`ps-depth8-r4`, `ps-depth8-r5`) give a 4- and a 5-draw same-recipe mean.

Prediction: the gain keeps growing with members of one recipe but with diminishing returns, as averaging
partly correlated scores does (negatives correlate 0.62 between draws): the 5-draw same-recipe mean lands
above `ps-d8-ens3` (0.558) and within 0.02 of the mixed-depth five (0.575), i.e. member count, not depth
diversity, is what the mixed five bought.
Falsified two ways: (a) the same-recipe five stays at the three's level (<= 0.558 + ~0.01): the curve is
flat past three and the mixed five's extra +0.017 was depth diversity; (b) it lands more than 0.02 under
the mixed five while above the three: both matter.

Matched controls: `ps-d8-ens3` (0.558; the same recipe at three members) and `ps-d8-ens5` (0.575, mixed
depth, same member count). Dose: every 3-, 4- and 5-subset mean of the five same-recipe draws, since which
three draws went into `ens3` is itself a draw.

New draws: main's recipe unchanged, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`,
`--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general (`chain.sh`).

## Changes
No code change to the pipeline. `chain.sh` trained two more plain `ps-depth8` draws (`ps-depth8-r4`,
`ps-depth8-r5`, main's recipe, 8 rotations each, no shipped model). `tools_exp/ensemble_predictions.py`
(from `exp/ps-d8-ensemble`) then wrote every 2-, 3-, 4- and 5-subset logit mean of the five same-recipe
draws: a = `ps-depth8`, b = `ps-depth8-repeat`, c = `ps-depth8-r3`, d = `ps-depth8-r4`, e = `ps-depth8-r5`
(`models/ps-d8-same{2,3,4}-<letters>`, `models/ps-d8-same5`). No training on eval folds, no selection: every
subset is reported.

Buzz logit SD, both new draws: 0.57-1.34 (`1_150` lowest at 0.57 / 0.60, `1_29` highest), no collapse.

## Results
### The new draws alone
`ps-depth8-r4` 0.464 and `ps-depth8-r5` 0.473, against the first three at 0.515 / 0.509 / 0.497
(r3 -> r4 -0.033 ± 0.014, r3 -> r5 -0.024 ± 0.012, eval SD only). Five draws of one recipe now span
0.464-0.515, mean 0.492, SD 0.022: wider than the 0.005 run-to-run SD measured on the frozen hidden-1024
head (docs/judging-results.md), so a single fine-tuned-trunk draw is a noisier estimate than the docs'
table suggests. Both low draws lose on `1_150` and `1_114` and gain on `1_29` and in the background tier.

### Headline by member count (same recipe, every subset)
| members | subsets | mean | range |
|---|---|---|---|
| 1 | 5 | 0.492 | 0.464-0.515 |
| 2 | 10 | 0.531 | 0.512-0.556 |
| 3 | 10 | 0.545 | 0.535-0.558 |
| 4 | 5 | 0.553 | 0.544-0.560 |
| 5 | 1 | 0.558 | |

3-subsets: abc 0.558, abd 0.550, abe 0.555, acd 0.545, ace 0.549, ade 0.535, bcd 0.542, bce 0.544,
bde 0.538, cde 0.536. 4-subsets: abcd 0.554, abce 0.560, abde 0.559, acde 0.546, bcde 0.544.
`ps-d8-ens3` (abc) was the best of the ten threes: it held the three strongest draws.

### `ps-d8-ens3` (three same-recipe) -> `ps-d8-same5`
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.619 | 0.638 | +0.019 | 0.013 | 32 |
| 53 | 0.581 | 0.582 | +0.001 | 0.013 | 28 |
| 1_11 | 0.632 | 0.608 | -0.024 | 0.016 | 26 |
| 1_143 | 0.640 | 0.650 | +0.010 | 0.019 | 22 |
| 1_150 | 0.470 | 0.500 | +0.030 | 0.026 | 21 |
| 1_95 | 0.312 | 0.309 | -0.003 | 0.012 | 46 |
| 1_37 | 0.701 | 0.688 | -0.013 | 0.014 | 14 |
| 1_114 | 0.509 | 0.489 | -0.020 | 0.023 | 28 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.558 → this 0.558 (+0.000 ± 0.006)
- inclusive (sensitivity), same thresholds: 0.467 → 0.464 (-0.003)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.861 | 0.861 | +0.000 | 122 |
| untagged | 0.580 | 0.579 | -0.001 | 2418 |
| background | 0.507 | 0.519 | +0.012 | 1874 |
| quiet | 0.162 | 0.156 | -0.006 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

### `ps-d8-ens5` (five, mixed depth) -> `ps-d8-same5`
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.613 | 0.638 | +0.025 | 0.016 | 32 |
| 53 | 0.600 | 0.582 | -0.018 | 0.017 | 28 |
| 1_11 | 0.648 | 0.608 | -0.040 | 0.020 | 26 |
| 1_143 | 0.676 | 0.650 | -0.026 | 0.030 | 22 |
| 1_150 | 0.472 | 0.500 | +0.028 | 0.028 | 21 |
| 1_95 | 0.312 | 0.309 | -0.003 | 0.015 | 46 |
| 1_37 | 0.702 | 0.688 | -0.014 | 0.016 | 14 |
| 1_114 | 0.576 | 0.489 | -0.087 | 0.025 | 28 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.575 → this 0.558 (-0.017 ± 0.008)
- inclusive (sensitivity), same thresholds: 0.479 → 0.464 (-0.015)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.876 | 0.861 | -0.015 | 122 |
| untagged | 0.596 | 0.579 | -0.017 | 2418 |
| background | 0.514 | 0.519 | +0.005 | 1874 |
| quiet | 0.163 | 0.156 | -0.007 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only; training stochasticity is not in it.

### Reading
By the test set before training, this is falsifier (a): the same-recipe five is at the three's level
(0.558 against 0.558, +0.000 ± 0.006), not above it. Against a single draw it is still a large gain
(r3 -> same5 +0.061 ± 0.011, every fold up, background +0.113, untagged +0.055).

The test's reference point was itself a lucky draw, though. `ps-d8-ens3` is the top of the ten
three-subsets; the subset means run 0.492 / 0.531 / 0.545 / 0.553 / 0.558 at 1-5 members, so the
same-recipe curve does keep rising, by +0.014 from 2 to 3, +0.008 from 3 to 4 and +0.005 from 4 to 5.
Those last two steps are within the headline's eval SD: past three members the return is small and
unsure.

The mixed-depth five's -0.017 ± 0.008 lead is not evidence for depth diversity. Its members average
0.506 (0.515, 0.509, 0.497, depth4 0.523, depth10 0.487) and the same-recipe five's average 0.492, a
0.014 gap in the members alone; the ensemble lift over the member mean is +0.069 mixed and +0.066
same-recipe. So the two fives pool about equally well, and the mixed five was ahead because r4 and r5
came out weak, which is a draw and not a property of the recipe. One comparison, single draws of each
depth: weak evidence that diversity adds nothing, not proof.

Folds, `ens3` -> `same5`: none moves by two of its own SDs. `1_29` +0.019 ± 0.013, `1_150` +0.030 ± 0.026
and `1_11` -0.024 ± 0.016 are each about 1-1.5 SD, unsure. Against the mixed five, `1_114` is
-0.087 ± 0.025 (0.576 -> 0.489) and `1_11` -0.040 ± 0.020; `1_114` is where the single low draws lost
most (r4 0.283, r5 0.340 against r3 0.414), so that is the weak members showing through, on a hard
fold. `1_95` does not move with any member count (0.309-0.312): averaging does not touch the jet.

Tiers, `ens3` -> `same5`: background +0.012, everything else within 0.006. Against the mixed five:
loud -0.015, untagged -0.017, background +0.005.

## Conclusion
Past three members a same-recipe ensemble buys little: subset means 0.545 / 0.553 / 0.558 at 3 / 4 / 5,
and the five equals the logged three (0.558, +0.000 ± 0.006). The mixed-depth five's 0.575 is accounted
for by its stronger members (member mean +0.014, ensemble +0.017), so neither member count beyond ~3 nor
depth diversity is shown to add; member quality does, about one-for-one. Useful side result: five draws
of the plain recipe span 0.464-0.515 (SD 0.022), so single-draw deltas on the fine-tuned trunk under
~0.03-0.04 deserve a repeat before they are believed. Next: IDEAS 30 (K tails on a shared stem), with
K = 3.
