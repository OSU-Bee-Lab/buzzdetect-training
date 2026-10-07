# fast32-d8d6
## Hypothesis
`fast32-depth` found that cutting fast32 a0.50's trunk to 12 or 10 layers costs no sensitivity (0.697 / 0.696 vs 0.687 full depth at 56k) but buys little speed (2.15x -> 2.28x). If depth is genuinely not where this student's sensitivity lives, the cut can go further: a0.50_d8 (layers 9-14 removed: ends two layers into the 512-wide block, 256 ch at a0.50) and a0.50_d6 (layers 7-14 removed: ends before the 512 block, 128 ch at a0.50, one stride-2 fewer). Rung C, buzz+rain+human, seed 1, WSD to 56k with the stop rule. Matched controls: fe_C_fast32_a0.50_s1_c-buzz-rain-human (0.687 @2.15x) and fe_C_fast32_a0.50_d10 (0.696 @2.28x). Gain: either student above the frontier line at its speed (fast32h16 a0.50 0.650 @2.51x; fast32h16 a0.375 0.639 @2.63x). Falsifier: headline falls below d10's by more than ~0.03 with < 0.1x of speed gained, or speed stays under 2.4x (front end alone is 2.70x, so the ceiling is near).

## Changes
`a0.50_d10`, `a0.50_d8`, `a0.50_d6` added to `distill_train.ARCHS` and `bench_arch.CANDIDATES`.
Random-weight bench (bench_arch, 200 s, GPU, x YAMNet; fast32 rows read ~0.6x low against trained students' speed_200, so only ratios count): a0.50 1.55, _d10 1.61, _d8 1.64, _d6 1.67, front end alone 2.72. Expected logged speeds by ratio: d8 ~2.3x, d6 ~2.35x.
Run as the first part of one chained job (`chain.sh` in this worktree) that then runs `fast32h16-depth` and `shallow-wide` from their own worktrees.
