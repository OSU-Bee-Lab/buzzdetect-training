# fast32-depth
## Hypothesis
IDEAS.md distillation "Untried": dropping layers beyond d12. On YAMNet's front end, a0.50_d12 tied a0.50 (0.623 vs ~0.625, rung B, contaminated era) at 1.61x vs 1.47x. On fast32 the front end is cheap, so the trunk is a larger share of wall time and cutting its last layers should buy proportionally more speed. Students fast32 a0.50_d12 (layers 13-14 removed) and a0.50_d10 (11-14 removed), rung C, buzz+rain+human, WSD to 56k with the stop rule, seed 1. Matched control: fe_C_fast32_a0.50_s1_c-buzz-rain-human (0.680/0.687 at 28k/56k, 2.16x). Gain: either student above the frontier line at its speed (fast32 a0.375 0.675 at 2.28x; fast32h16 a0.50 0.650 at 2.51x). Falsifier: speed barely moves (< 2.3x) or the headline drops more than a0.375 does.

## Changes
`a0.50_d10` added to `distill_train.ARCHS` and `bench_arch.CANDIDATES`.

## Results
## Conclusion
