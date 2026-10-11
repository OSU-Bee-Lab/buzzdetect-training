# matmul-stft
## Hypothesis
Distillation arm, no training. `fused-blocks` and `conv-stft` left the fast end's speed ceiling unexplained:
on fast32h16 every trunk under ~6 MMACs times 2.5-2.84x YAMNet at 200 s, and the front end alone is 2.97x
(11728 s/s against YAMNet's 3953), so the trunk is ~5% of a fast student's time and the front end is the
rest. `conv-stft` showed the STFT's Gather + MatMul are not it (a strided Conv1D is 7% slower there).
Two candidates remain, never measured: the fixed cost of one `session.run` on 200 s (3.2M float32 copied
to the card), or the front end's remaining ops.

fast32h16's window equals its hop (256), so its frames do not overlap: the framing is a Reshape, and the
windowed DFT of the bins the mel matrix reads is one MatMul (256 x 2*bins). No index Gather over the
waveform, no Transpose, no convolution. (For a window of m hops it is m shifted MatMuls summed.) Same
numbers up to float rounding, so existing students re-export without retraining, as `conv-stft` did.

Predictions, in order:
1. A floor graph (waveform -> one ReduceSum) on the same harness times well above the front end alone
   (>= 2x its rate), i.e. the ceiling is the front end's ops, not the per-call cost. If the floor is within
   ~20% of the front end alone, the ceiling is the call itself and no graph change can move the fast end;
   the experiment ends there with that as its result.
2. If (1) holds: the Reshape + MatMul front end ("mm") times faster than both the STFT and the conv form
   on fast32h16, front end alone and with a trunk, by more than timing noise (> 5%).
3. If (2) holds: re-exported fast32h16 frontier students keep their detections (headline equal to the third
   decimal, no ins_buzz flips on the fixture) and move right on the frontier.

Matched control: each student's own existing export (same weights), and the same-session random-weight
bench for speed (rows timed on different days mix the change with day-to-day timing).

## Changes
- `05_distill/frontends.py`: two new graph forms of the same front end, both equal to the STFT form up to
  float rounding (checked on all 14 specs at 20 lengths, the edges of the padding rule included: max
  log-mel difference 2e-5, identical patch counts).
  - `mm`: frames by Reshape, windowed DFT by MatMul; windows that are a whole number of hops only; keeps
    the waveform Pad.
  - `mmt`: the same for any window (kernel zero-padded to whole hops) and **no Pad over the waveform**:
    complete patches come from a Slice, the zero-padded last 0-2 patches from the waveform's short tail
    (a Pad over <= 31k samples), joined at the patch level.
- `export_student.py --fe-impl mm|mmt`, `student.py` (the YAMNet front end takes any impl through the
  `yamnet` spec), `bench_arch.py` (`+mm`, `+mmt`; `floor`: waveform -> ReduceSum, the cost of a call).
- `05_distill/prefix_bench.py` (new): times prefixes of a graph, cut at each float node and reduced to a
  scalar on the device. Whole-graph timings, so it is safe where a per-node CUDA profile is not.
  `micro/`: hand-built one-op graphs, timed with the input on the host and already on the device.
- Brought over from `exp/conv-stft` as a patch: the `--fe-impl` plumbing, `reexport.py`, fused blocks.
- No training. 24 rung-C students (every final branch in the log) re-exported as `<run>_mmt`, scored on
  the rotating folds, timed, recorded.

## Results
**1. The ceiling is one node.** GPU, 200 s per call (3.2M samples), ms per call:

| graph | ms |
|---|---|
| waveform -> ReduceSum (the call itself: host-to-device copy + one op) | 2.5 (0.15 with the input already on the device) |
| + Mul or Relu over the waveform | 2.7 |
| + Reshape + MatMul(256 x 244) | 3.2 |
| + **Pad** by 15120 constant zeros | **13.1** (10.8 on device; 2.4 on the CPU EP) |
| + **Concat** with 15120 constant zeros | **21.9** (20.0 on device; 2.6 on the CPU EP) |
| Pad or Concat of 60 rows onto a (12600, 32) tensor | +0.02 / +0.17 |

Prefix timing of the exported graphs agrees: in `fe@fast32h16` the Pad is 13.2 of 16.5 ms and nothing
else exceeds 1.3 ms; in `a0.25@fast32h16+mm` the Pad is 13.1 of 17.0 ms, the whole trunk 1.7. Prediction 1
holds (the call is 2.5 ms against the front end's 17), but not as written: the cost is not "the front
end's ops", it is the single Pad over the waveform, which the CUDA EP runs ~5x slower than the CPU EP.
A Concat is worse, so swapping the node does not help; not touching the whole waveform with either does.

**2. Random-weight bench**, same session, 200 s, s of audio per s (`bench.log`, `bench3.log`; YAMNet 3969):

| graph | STFT | conv | mm (Pad kept) | mmt (no Pad) |
|---|---|---|---|---|
| front end only, fast32h16 | 11717 | 10881 | 12716 | **36578** |
| front end only, twofast32 | 7853 | 9377 | 11288 | **27454** |
| front end only, fast32 | 10819 | | | **26766** |
| front end only, yamnet | | 8642 | | **14348** |
| a0.25 @ fast32h16 | 10948 (2.76x) | | 11702 (2.95x) | **29086 (7.33x)** |
| a0.50_d8 @ fast32h16 | 10554 (2.66x) | | 11285 (2.85x) | **26865 (6.77x)** |
| a0.50 @ twofast32 | 6957 (1.75x) | | 9573 (2.41x) | **18884 (4.76x)** |
| a0.50_d8 @ yamnet | | 6785 (1.71x) | | **10523 (2.65x)** |

Prediction 2 holds: `mm` alone is +7-8% on fast32h16 and +38% on twofast32 (whose 512-sample window made
the STFT's Gather twice as large); removing the Pad is the large step. At 20 s per call the gain is
smaller (a0.25@fast32h16 2.93x -> 3.68x, a0.50_d8@fast32h16 2.9x -> 4.99x): Pad's cost scales with the
samples, the fixed per-call costs do not.

**3. Trained students, re-exported** (rung C, same weights; `05_distill/log.jsonl` rows `<run>_mmt`):

| student | headline, source / mmt | x YAMNet at 200 s: source | conv (`_cfe`) | **mmt** | at 20 s (mmt) |
|---|---|---|---|---|---|
| fast32h16 a0.25 s1 / s2 | 0.612 / 0.612, 0.616 / 0.616 | 2.75, 2.73 | 2.55 | **7.28, 7.30** | 2.98, 2.90 |
| fast32h16 a0.25_d8_f2 | 0.630 / 0.630 | 2.73 | | **6.83** | 5.60 |
| fast32h16 a0.50_d8 s1 / s2 | 0.646 / 0.646, 0.642 / 0.642 | 2.64, 2.66 | | **6.67, 6.77** | 4.19, 4.34 |
| fast32h16 a0.375 | 0.639 / 0.639 | 2.63 | | **6.39** | 2.91 |
| fast32h16 a0.50 s1 / s2 (28k) | 0.650 / 0.650, 0.650 / 0.650 | 2.50, 2.52 | | **5.69, 5.74** | 2.88, 2.87 |
| fast32h16 a0.75_d8 | 0.656 / 0.656 | 2.46 | | **5.57** | 4.04 |
| fast32 a0.50_d6 | 0.664 / 0.664 | 2.38 | | **4.98** | 4.59 |
| twofast32 a0.50 | 0.691 / 0.691 | 1.76 | | **4.81** | 2.39 |
| fast32 a0.50_d8 s1 / s2 | 0.685 / 0.685, 0.678 / 0.678 | 2.34, 2.33 | | **4.74, 4.77** | 3.82, 4.16 |
| fast32 a0.50_d10 | 0.696 / 0.696 | 2.28 | | **4.55** | 3.61 |
| fast32 a0.375 | 0.675 / 0.675 | 2.27 | | **4.52** | 2.87 |
| fast32 a0.50_d12 | 0.697 / 0.697 | 2.21 | | **4.33** | 3.23 |
| fast32 a0.50 | 0.687 / 0.687 | 2.15 | | **4.13** | 2.77 |
| yamnet a0.25_d8 | 0.713 / 0.713 | 1.69 | 1.96 | **3.24** | 3.00 |
| yamnet a0.25 | 0.707 / 0.707 | 1.62 | 1.86 | **2.96** | 2.78 |
| yamnet a0.375_d8 | 0.716 / 0.716 | 1.58 | 1.83 | **2.89** | 2.93 |
| yamnet a0.50_d8 s1 / s2 (28k) | 0.735 / 0.735, 0.725 / 0.725 | 1.52, 1.51 | 1.70 | **2.60, 2.59** | 2.70, 2.42 |
| yamnet a0.50 (28k) | 0.725 / 0.725 | 1.40 | | **2.17** | 2.11 |
| yamnet a0.75_d8 | 0.731 / 0.731 | 1.32 | 1.47 | **2.06** | 2.12 |

Parity: over the 24 students, every headline, every fold and every tier value in the log row equals the
source row's (240 values, largest difference 0.000). Each export's fixture check passes (max |onnx - keras|
2e-6 to 3e-6, same buzz detections). So tiers and folds do not move, by construction; no tier reading
applies and no repeat is owed (same weights). Prediction 3 holds.

Speed caveats. Source rows were timed on 2026-10-05 to 10-11 and the `_mmt` rows today; the same-session
random-weight bench gives the same factors (a0.25@fast32h16 2.76x -> 7.33x bench, 2.75x -> 7.28x trained).
"x YAMNet" divides by the deployed YAMNet graph, which carries the same Pad (below), so part of every
multiple is YAMNet's own avoidable cost. The 20 s column is noisier (single short calls; the full-depth
a0.25 / a0.375 / a0.50 trunks time lower there than their d8 cuts) and is not the frontier's axis.

**The frontier it leaves** (rung C, non-dominated, one seed unless two are given):

| x YAMNet | student | headline |
|---|---|---|
| 7.3 | fast32h16 a0.25 | 0.612 / 0.616 |
| 6.8 | fast32h16 a0.25_d8_f2 | 0.630 |
| 6.7 | fast32h16 a0.50_d8 | 0.646 / 0.642 |
| 5.7 | fast32h16 a0.50 | 0.650 / 0.650 |
| 5.6 | fast32h16 a0.75_d8 | 0.656 |
| 5.0 | fast32 a0.50_d6 | 0.664 |
| 4.8 | twofast32 a0.50 | 0.691 |
| 4.55 | fast32 a0.50_d10 | 0.696 |
| 4.3 | fast32 a0.50_d12 | 0.697 |
| 3.2 | yamnet a0.25_d8 | 0.713 |
| 2.9 | yamnet a0.375_d8 | 0.716 |
| 2.6 | yamnet a0.50_d8 | 0.735 / 0.725 |

Two things changed besides the scale. The order: `twofast32 a0.50` (0.691) was the slowest of the fast
front ends at 1.76x and is now 4.81x, above every fast32 d8 student (0.678-0.685 at 4.74-4.77x). And the
trunk counts again: under the Pad, a0.25 to a0.75_d8 on fast32h16 spanned 2.46-2.75x; without it 5.57-7.30x.
The standing fact "every trunk under ~6 MMACs times 2.5-2.75x, whatever its blocks" described the Pad.

**The deployed graphs carry the same Pad** (`deployed.log`, prefix timing at 200 s): in
`yamnet_large_general` it is 11.0 of 50.6 ms (the whole front end ~24 ms, the STFT's DFT MatMul 6.4); in the
teacher `v4-ft-ps-e60-moderate` both branches have one, 10.8 ms each of 140 ms (and the octave-up branch's
resampling Conv is 54 ms). Not changed here: those graphs are `04_deploy`'s.

## Conclusion
Positive, and it rewrites the speed axis: the fast end's ~2.7x "ceiling" was one `Pad` node over the
waveform (11-13 ms of a 15-17 ms call at 200 s; a Concat costs 20 ms), not the STFT, the trunk or the
call. A front end that never pads the full waveform (`--fe-impl mmt`: Slice + Reshape + MatMul, the padded
tail computed apart and joined at the patch level) is numerically the same graph, so all 24 final rung-C
students re-export with identical headlines, folds and tiers and time 1.5-2.7x faster: fast32h16 a0.25
0.612 at 2.75x -> 7.28x, twofast32 a0.50 0.691 at 1.76x -> 4.81x, yamnet a0.25_d8 0.713 at 1.96x -> 3.24x,
yamnet a0.50_d8 0.735 at 1.70x -> 2.60x. Tiers do not move by construction. `mmt` supersedes `conv`
(`_cfe`) everywhere; compare against the `_mmt` rows from here on, and make `mmt` the export default when
this is merged. The lite floor (>= 1.5x) is now met by every student in the log, a0.75_d8 on the YAMNet
front end included (2.06x).
Where to dig: (1) trunk and front-end choices should be re-read on the new axis, since students that
tied in speed under the Pad no longer do (width and depth cuts on fast32h16 now buy 5.6x -> 7.3x; fast32
at hop 10 ms is dominated by twofast32 and the YAMNet front end at d8 around 3-5x); a 64-band hop-16 front
end between `yamnet` and `fast32h16` is the obvious untried point. (2) The deployed YAMNet and teacher
graphs have the same Pad (11 of 51 ms; 2 x 11 of 140 ms): a `04_deploy` change with exact parity, queued in
IDEAS. (3) At 20 s per call the gains are smaller and per-call costs dominate; buzzdetect's chunk length
decides which column a user sees.
