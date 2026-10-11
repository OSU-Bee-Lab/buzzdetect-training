# fused-blocks
## Hypothesis
Distillation arm. On the GPU path buzzdetect runs (onnxruntime CUDA), a student's time is not in its
multiply-adds. An onnxruntime per-node profile of `fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd56000` at
200 s puts ~60% of kernel time in the 13 depthwise convolutions and ~9% in the 13 pointwise ones, although
the pointwise layers hold nearly all the MACs; a depthwise layer's cost follows its channel (group) count,
not its map size (layer 14, a 2x1 map, is the most expensive). That is the standing fact "14x fewer MACs
buys 2.0x" seen from the other side.

So replace depthwise + pointwise blocks with one full 3x3 convolution ("fused" block, as in
Fused-MBConv) where the map is small: half the layers on the GPU path, more capacity per block, and MACs
that grow only where maps are tiny. Prediction: at matched front end, width, classes, rung and step
budget, a fused student is faster than its separable twin at an equal or higher headline, i.e. a new
non-dominated point. Falsified if the random-weight bench shows no speed gain (the profile misattributes
asynchronous GPU time), or if the trained student's headline falls by more than the speed is worth
(lands on or under the existing line).

Matched control: the separable student of the same front end, alpha, classes, rung C WSD budget and seed
already in `05_distill/log.jsonl`.

### Speed bench before training (random weights, 200 s, GPU, x YAMNet; `bench.log`)
| trunk on fast32h16 | x YAMNet | MMACs |
|---|---|---|
| a0.50 (separable) | 2.51 | 5.9 |
| a0.50_d8 | 2.56 | 2.9 |
| a0.25 | 2.75 | 1.6 |
| a0.50_f8 / _f4 / _f2 (fused from layer 8 / 4 / 2) | 1.91 / 1.81 / 1.80 | 33 / 45 / 49 |
| a0.375_f2 | 2.09 | 27.6 |
| a0.25_f2 / a0.25_f8 | 2.49 / 2.53 | 12.3 / 8.3 |
| a0.50_d8_f2 | 2.36 | 23.1 |
| **a0.25_d8_f2** | **2.75** | 5.8 |

On the YAMNet front end: a0.50_d8 1.51, a0.50_d8_f2 1.32, a0.25_d8_f2 1.62. On fast32: a0.50_d8 1.63,
a0.50_d8_f2 2.03 (the logged trained a0.50_d8 student times 2.34x, so this pair was read under contention;
not used).

**The speed half of the hypothesis is false at equal width.** Fusing costs speed at every width tried
(a0.50: 2.51 -> 1.80), so the profile's per-node times misattribute asynchronous GPU work and MACs do
matter once they are large. What the bench does show is a ceiling: every trunk under ~6 MMACs sits at
2.5-2.75x, whatever its blocks. Under that ceiling a fused trunk is free: `a0.25_d8_f2` (8 full 3x3
convs, widths 8-128, 5.8 MMACs) times the same 2.75x as separable `a0.25` (14 layers, 1.6 MMACs).

**Revised, before training:** at the speed ceiling, spend the free compute on full convolutions.
`fast32h16:a0.25_d8_f2` should hold `a0.25`'s speed (2.75x) and beat its headline (0.612) by more than
seed noise (> 0.02), a new fastest frontier point; the second comparison is `a0.50_d8` (0.646 / 0.642,
two seeds, at 2.66x). Falsified if it lands within 0.02 of `a0.25` or times slower than 2.66x.

## Changes
- `05_distill/student.py`: arch `a<alpha>[_d<depth>][_f<k>]` (`parse_arch`); layers k on are fused blocks (one
  full conv with the layer's kernel and stride, named `layer{i}_conv`); `fused_from_of`, MAC count.
- `distill_train.py`, `export_student.py`, `bench_arch.py`, `student_init.py`: carry it through. Fused layers
  keep their random init (YAMNet channel selection only covers the separable layers before them).
- Run: `05_distill/main.py --rung C --wsd 7000,14000,28000,56000 --runs
  "fast32h16:a0.25_d8_f2:classes=ins_buzz+ambient_rain+human"`, seed 1.

## Results
Step-budget curve (headline; speed x YAMNet at 200 s):

| budget | a0.25_d8_f2 (this) | a0.25 (control) | a0.50_d8 (seed 1) |
|---|---|---|---|
| 7000 | 0.594 @2.71x | 0.546 | 0.624 |
| 14000 | 0.615 @2.73x | 0.569 | 0.640 |
| 28000 | 0.629 @2.72x | 0.603 | 0.648 |
| 56000 | **0.630 @2.73x** | 0.612 @2.75x | 0.646 @2.64x (seed 2: 0.642 @2.66x) |

hit@K 56.6 / 60.8 / 61.2 / 63.0; the last doubling added +1.8 hit@K and +0.001 headline.

Against the matched control `fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd56000` (`tools/results.py`):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.670 | 0.690 | +0.020 | 0.023 | 34 |
| 53 | 0.641 | 0.644 | +0.003 | 0.014 | 32 |
| 1_11 | 0.586 | 0.608 | +0.022 | 0.012 | 32 |
| 1_143 | 0.567 | 0.581 | +0.014 | 0.020 | 26 |
| 1_37 | 0.597 | 0.629 | +0.032 | 0.016 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.612 → this 0.630 (+0.018 ± 0.008)
- inclusive (sensitivity), same thresholds: 0.505 → 0.527 (+0.022)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.730 | 0.766 | +0.036 | 603 |
| untagged | 0.616 | 0.636 | +0.020 | 7485 |
| background | 0.607 | 0.616 | +0.009 | 9412 |
| quiet | 0.167 | 0.198 | +0.031 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |

Against `fe_C_fast32h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000` (0.646 @2.64x): -0.016 ± 0.007; `1_29`
-0.038 ± 0.019, `1_143` -0.026 ± 0.018, `53` -0.003 ± 0.011, `1_11` +0.000 ± 0.011, `1_37` -0.013 ± 0.021;
tiers loud -0.012, untagged -0.013, background -0.026, quiet +0.005.

± SD is eval sampling only; seed noise on a student's headline (~0.01-0.02) is not in it.

Interpretation. Speed held as the bench said: 2.73x against the control's 2.75x, inside timing noise (the
control's own branches time 2.73-2.75x). Headline +0.018 ± 0.008 over separable `a0.25`: about 2 eval SDs,
but under the 0.02 bar set before the run and the size of seed noise, on one seed each. So the prediction
as written is not met; it is weak evidence for a small gain, not a confirmed step. Every fold leans up;
`1_11` (+0.022 ± 0.012) and `1_37` (+0.032 ± 0.016) are about 2 SDs, `1_29`, `1_143` and `53` are within
theirs (unsure). Tiers: `loud` +0.036, `untagged` +0.020, `quiet` +0.031, `background` +0.009, so what
moved is isolated buzz rather than the drones. The clearest difference is early: +0.048 at 7k and +0.046
at 14k, closing to +0.018 at 56k, where the fused trunk has flattened (28k to 56k: +0.001) and the
separable one was still rising (+0.009). Against `a0.50_d8` it is -0.016 ± 0.007 for +0.07-0.09x of
speed, which is inside the timing spread between that student's own branches (2.64-2.67x).

## Conclusion
Mixed: the mechanism is falsified, the student is a marginal frontier point. (1) Fusing depthwise +
pointwise into one full 3x3 conv does not speed a student up at equal width; it slows it (a0.50 on
fast32h16: 2.51x to 1.80x, random weights). The onnxruntime per-node GPU profile that suggested otherwise
misattributes asynchronous work; whole-graph timing (`bench_arch.py`) is the only reading to trust. (2)
What the bench found instead is a speed ceiling on fast32h16: every trunk under ~6 MMACs times 2.5-2.75x.
Under it, `a0.25_d8_f2` (8 full convs, widths 8-128) holds separable `a0.25`'s speed (2.73x vs 2.75x) at
0.630 against 0.612 (+0.018 ± 0.008, incl. quiet 0.527 vs 0.505; loud +0.036, untagged +0.020, quiet
+0.031, background +0.009). That is under the repeat rule's 0.02 and one seed, so it is the fastest
non-dominated point by a margin the size of seed noise, and it sits on the line to `a0.50_d8` (0.646 /
0.642 at 2.64-2.66x), not above it. It reaches its level in a quarter of the steps (0.615 at 14k, where
`a0.25` needs 56k for 0.612). Where to dig: the ceiling itself. Past ~2.75x the trunk no longer sets the
speed on this front end, so the next speed step has to come from the front end or the fixed per-call
cost, not from a smaller or differently built trunk; and a fused trunk with more width inside the
ceiling's MAC budget is untested.
