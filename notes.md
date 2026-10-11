# conv-stft
## Hypothesis
Distillation arm. `fused-blocks` (2026-10-11) found a speed ceiling on fast32h16: fifteen trunks from 0.8 to
16 MMACs, separable or fused, all time 2.46-2.84x YAMNet at 200 s (random weights, `exp/fused-blocks`
bench2.log), so past ~2.7x the trunk is not what the time is spent on. What every one of those graphs
shares is the exported front end: `tf.signal.stft` becomes an index Gather over the whole waveform (one
int64 index per sample of every frame), a window Mul, two Transposes and a MatMul DFT, then a second
Gather to cut patches.

The same spectrogram can be computed by one strided 1-D convolution whose kernels are the Hann window
times the DFT's cos / sin rows (only the bins the mel matrix reads), and the patch cut is a Reshape when
patches do not overlap. Same numbers up to float rounding, so **no retraining**: an existing student's
weights go into a graph with the conv front end.

Prediction: re-exported frontier students keep their detections (parity as in the int8 work: no ins_buzz
detection flips at logit > 0 on the fixture, headline equal to the third decimal) and time faster at
200 s on the GPU, most at the fast end where the front end is the largest share. A frontier-wide shift
right at zero headline cost. Falsified if the front end alone does not time faster with the conv form
(then the ceiling is per-call overhead, not the front end's ops), or if parity fails.

Matched control: each student's own existing export (same weights), from `05_distill/log.jsonl`.

## Changes
- `frontends.make_features_layer(fe, impl='conv')`: the STFT as one strided Conv1D (Hann x DFT cos/sin rows
  for the bins the mel matrix reads), the patch cut as a Reshape. `export_student.py --fe-impl conv`,
  `bench_arch.py` names with `+conv`, `ladder_record.py record --run` (a row that takes its V readouts
  from a source run), `reexport.py` (export, score, time and record `<run>_cfe`).
- No training. Six existing rung-C students (56000 branch) re-exported with the conv front end as
  `<run>_cfe`: the fastest student (`fast32h16 a0.25`) and the five yamnet-front-end students.
- Numerical check, CPU, before any timing: conv vs STFT log-mel within 1e-5 on every front end, 3e-5
  against YAMNet's own feature layer; `fast32h16 a0.25` re-export flips 0 of 313 fixture frames at logit
  > 0, max |logit diff| 2.6e-6, graph 130 -> 90 nodes, no Gather.

## Results
Front end alone and whole graphs, random weights, 200 s on the GPU, one session (`bench.log`):

| graph | STFT s/s | conv s/s | conv / STFT |
|---|---|---|---|
| front end only, fast32h16 | 11728 | 10858 | 0.93 |
| front end only, fast32 | 10806 | 9375 | 0.87 |
| front end only, yamnet | 7044 | 8622 | 1.22 |
| a0.25 @ fast32h16 | 10725 | 10118 | 0.94 |
| a0.50_d8 @ fast32 | 6449 | 5952 | 0.92 |
| a0.50_d8 @ yamnet | 5972 | 6773 | 1.13 |

(YAMNet 3953 s/s in the same session. CPU, informational on this i7-2600: fast32h16 front end 6840 ->
6206, but a0.25@fast32h16 3757 -> 5684.)

Trained students, each `_cfe` row against its own source export (`05_distill/log.jsonl`):

| student (rung C, 56000) | headline, source / conv | x YAMNet at 200 s, source -> conv |
|---|---|---|
| fast32h16 a0.25 | 0.612 / 0.612 | 2.75 -> 2.55 |
| yamnet a0.25_d8 | 0.713 / 0.713 | 1.69 -> 1.96 |
| yamnet a0.25 | 0.707 / 0.707 | 1.62 -> 1.86 |
| yamnet a0.375_d8 | 0.716 / 0.716 | 1.58 -> 1.83 |
| yamnet a0.50_d8 | 0.735 / 0.735 | 1.52 -> 1.70 |
| yamnet a0.75_d8 | 0.731 / 0.731 | 1.33 -> 1.47 |

Parity: all six headlines equal to the third decimal. `tools/results.py` on the `yamnet a0.50_d8` pair:
every fold delta +0.000 (`1_29` 0.829, `53` 0.654, `1_11` 0.713, `1_143` 0.717, `1_37` 0.764), every tier
+0.000 (loud 0.828, untagged 0.742, background 0.704, quiet 0.257). Same weights, same detections.

Speed. The source timings are from 2026-10-05 to 10-09 and the conv ones from today, so a row's ratio
mixes the change with day-to-day timing; the same-session bench is the clean read, and it agrees:
a0.50_d8@yamnet +13% in the bench, +12% in the rows. The prediction was wrong about where: the fast
front ends (32 mel, hop 16-ish frames) are 6-13% *slower* with the conv form, front end alone and whole
graph, so the ~2.7x ceiling `fused-blocks` found is not the Gather/MatMul STFT's ops. On the yamnet
front end (64 mel, 10 ms hop: many more frames) conv is 22% faster alone and 11-16% faster in every
trained student.

Frontier (rung C, one seed each, headline unchanged so no repeat is owed): the five yamnet-front-end
students move right at no cost. `yamnet a0.25_d8` at 1.96x (0.713) and `a0.375_d8` at 1.83x (0.716) now
sit above `twofast32 a0.50` (0.691 at 1.76x), which they dominate; `yamnet a0.50_d8` is 0.735 at 1.70x,
the highest headline on the line, where it was 1.52x. Faster than ~2.2x the line is unchanged (fast32
and fast32h16 students keep the STFT export). Tiers do not move by construction.

## Conclusion
Half right. A conv STFT front end is a free re-export (identical detections, no retraining) and makes
every yamnet-front-end student 11-16% faster at 200 s on the GPU (a0.50_d8: 0.735 at 1.52x -> 1.70x;
a0.25_d8: 0.713 at 1.69x -> 1.96x), pushing the 1.5-2.0x band of the frontier right and dominating
`twofast32`. It is 6-13% slower on fast32 / fast32h16, so it does not lift the fast end's ~2.7x ceiling:
the hypothesis's mechanism (the ceiling is the STFT's Gather + MatMul) is false there, and the ceiling is
still unexplained (per-call overhead or the mel/log/patch ops are the candidates, unmeasured). Use
`--fe-impl conv` for yamnet-front-end exports only; the `_cfe` rows are the ones to compare against in
that band. Timing caveat: source rows were timed on earlier days; the same-session random-weight bench
gives the same sign and size.
