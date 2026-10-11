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
