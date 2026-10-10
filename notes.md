# yamnet-a075-d8
## Hypothesis
`yamnet-depth` showed layers 9-14 buy nothing on the YAMNet front end at a0.50. Does width bought back on the 8-layer trunk buy sensitivity at the frontier's high end (the standard-tier side)? Full depth, a0.50 over a0.25 bought +0.018. Random-weight bench (bench_arch, 20 s, GPU, one run, x YAMNet; within-bench ratios only): a0.50 1.50, a0.375 1.67, a0.25 1.88, a0.50_d8 1.76, a0.50_d6 1.82, a0.375_d8 1.88, a0.25_d8 2.00, a0.75_d8 1.39, a0.75_d6 1.62, a0.375_d6 1.92, a1.00_d6 1.48. Logged/bench ratio on this front end: a0.50 0.93, a0.25 0.86, a0.50_d8 0.86. a0.75_d8 benches at 1.39 (a0.50 full 1.50), so expect ~1.25-1.30x logged. Student: yamnet a0.75_d8, seed 1. Rung C, buzz+rain+human (select), WSD to 56k with the stop rule (--wsd-max 56000 --wsd-stop 1.3), teacher v4-ft-ps-e60-moderate. Matched control: yamnet a0.50_d8 (0.735 @1.52x, width only). Gain: headline > 0.735 by more than seed noise (~0.02), since a0.50_d8 is faster and anything slower must beat it. Falsifier: <= 0.735, i.e. width beyond a0.50 buys nothing on the 8-layer trunk (shallow-wide on fast32h16 traded along the frontier: +0.010 ± 0.006 over a0.50_d8 for -0.18x).

## Run log
- 2026-10-09 10:17: the batch-26 chain (pid 399369) died at step 0 of this student: OOM in `val_flips`' eager
  batch-512 inference, right after the select init. TF only ever gets ~2.4 GB of the 4 GB card here. Measured:
  after the init only 17 MB stays resident (no leak); a random-weight a0.75_d8 trains at batch 512 with a 2.22 GB
  peak; the eager val call at 512 straight after the init fills all 2.4 GB. Fix (commit a56376c on
  exp/yamnet-d8-repeat): `val_flips` batch 512 -> 128. It is inference only, so results are unchanged; training
  batch, schedule and step counts are untouched. A 30-step `test_a075_d8_mem` run (init, val, batch-512 training,
  full-pool final val) ran clean and was deleted. Relaunched alone at 10:28: pid 1377627, log
  `.local/worktrees/yamnet-d8-repeat/distill.log`.
- 2026-10-09 20:20: resumed after the second park with the trunk at 55000/56000. Remaining time was ~15 min, under
  the 1.5 h park threshold, so the session watched it to the end rather than parking a third time (the resume
  prompt said to park if still running; the park rule's own threshold was taken as the intent). Exit 0 at 20:31.

## Changes
None to the student code: `yamnet:a0.75_d8:select` is an existing arch spec. The only code touched for this run is
the `val_flips` batch 512 -> 128 fix above (a56376c on exp/yamnet-d8-repeat; inference only).

## Results
Student `fe_C_yamnet_a0.75_d8_s1_select_c-buzz-rain-human`, all four budgets ran (the stop rule did not fire: dK
+1.43, +1.17, +0.58, so only the last doubling followed one under 1.3).

| steps | hit% | hit@K | headline | x YAMNet |
|---|---|---|---|---|
| 7000 | 56.87 | 68.64 | 0.728 | 1.32 |
| 14000 | 58.81 | 70.07 | 0.730 | 1.32 |
| 28000 | 61.65 | 71.25 | 0.729 | 1.32 |
| 56000 | 62.14 | 71.83 | 0.731 | 1.32 |

Control a0.50_d8 s1: hit@K 65.97 / 68.39 / 69.49 / 69.89, headline 0.727 / 0.731 / 0.735 / 0.735 @1.52x.

### vs a0.50_d8 seed 1, wsd56000 (matched control)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.829 | 0.827 | -0.002 | 0.007 | 34 |
| 53 | 0.654 | 0.658 | +0.004 | 0.010 | 32 |
| 1_11 | 0.713 | 0.692 | -0.021 | 0.010 | 32 |
| 1_143 | 0.717 | 0.729 | +0.012 | 0.013 | 26 |
| 1_37 | 0.764 | 0.751 | -0.013 | 0.015 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.735 → this 0.731 (-0.004 ± 0.005)
- inclusive (sensitivity), same thresholds: 0.618 → 0.615 (-0.003)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.828 | 0.823 | -0.005 | 603 |
| untagged | 0.742 | 0.735 | -0.007 | 7485 |
| background | 0.704 | 0.706 | +0.002 | 9412 |
| quiet | 0.257 | 0.255 | -0.002 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

### vs a0.50_d8 seed 2, wsd28000 (its last budget; the stop rule ended it there)
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.820 | 0.827 | +0.007 | 0.009 | 34 |
| 53 | 0.660 | 0.658 | -0.002 | 0.010 | 32 |
| 1_11 | 0.686 | 0.692 | +0.006 | 0.010 | 32 |
| 1_143 | 0.706 | 0.729 | +0.023 | 0.015 | 26 |
| 1_37 | 0.752 | 0.751 | -0.001 | 0.014 | 15 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.725 → this 0.731 (+0.006 ± 0.005)
- inclusive (sensitivity), same thresholds: 0.610 → 0.615 (+0.005)

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.825 | 0.823 | -0.002 | 603 |
| untagged | 0.730 | 0.735 | +0.005 | 7485 |
| background | 0.702 | 0.706 | +0.004 | 9412 |
| quiet | 0.259 | 0.255 | -0.004 | 2231 |
| faint | 0.067 | 0.067 | +0.000 | 35 |

± SD is eval sampling only; seed noise on a student headline is ~0.01-0.02 and is not in it.

Interpretation. Headline 0.731 @1.32x against the matched control's 0.735 @1.52x: -0.004 ± 0.005, and +0.001
against the two-seed a0.50_d8 mean (0.730). That is no difference, at a 0.20x speed cost. The hypothesis needed
> 0.735 by more than seed noise; the falsifier (<= 0.735) is met. Folds, against seed 1: `1_11` -0.021 ± 0.010 is
the only one outside its eval SD, but the two a0.50_d8 seeds differ on `1_11` by 0.027 themselves and against
seed 2 it is +0.006 ± 0.010, so it is seed noise, not a width effect. `1_143` is up against both seeds (+0.012 ±
0.013, +0.023 ± 0.015): weak evidence of a small gain there. `1_29`, `53`, `1_37`: unsure, all within SD or seed
spread. Tiers: none moved (all within ±0.007 against either seed).

The wider trunk does fit the teacher better on the V pool (hit@K 71.83 vs 69.89 at 56k, MAE 0.165 vs 0.171), and
that does not reach the headline: at this point the student's teacher-fit is not what limits eval sensitivity.
Logged speed 1.32x, in the predicted 1.25-1.30x band's neighbourhood (bench ratio 0.95).

## Conclusion
Negative. Width beyond a0.50 on the 8-layer YAMNet-front-end trunk buys nothing: a0.75_d8 is 0.731 @1.32x, level
with a0.50_d8 (0.735 s1, 0.730 two-seed mean @1.52x) and slower, so it is dominated and adds no frontier point;
it is also below the lite 1.5x speed floor. No tier moved; `1_143` is the one fold with a weak positive lean.
With `yamnet-a0375-d8` and `yamnet-a025-d8`, the d8 width series on this front end is 0.713 @1.69x (a0.25),
0.716 @1.58x (a0.375), 0.730 @1.52x (a0.50), 0.731 @1.32x (a0.75): it saturates at a0.50. Better V-pool fit
(hit@K +1.9) without a headline gain says the high end of the frontier is not capacity-limited here; further
standard-tier gains would have to come from the teacher, the front end or the distillation data, not trunk width.
