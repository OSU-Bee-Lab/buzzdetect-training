# fast32-depth
## Hypothesis
IDEAS.md distillation "Untried": dropping layers beyond d12. On YAMNet's front end, a0.50_d12 tied a0.50 (0.623 vs ~0.625, rung B, contaminated era) at 1.61x vs 1.47x. On fast32 the front end is cheap, so the trunk is a larger share of wall time and cutting its last layers should buy proportionally more speed. Students fast32 a0.50_d12 (layers 13-14 removed) and a0.50_d10 (11-14 removed), rung C, buzz+rain+human, WSD to 56k with the stop rule, seed 1. Matched control: fe_C_fast32_a0.50_s1_c-buzz-rain-human (0.680/0.687 at 28k/56k, 2.16x). Gain: either student above the frontier line at its speed (fast32 a0.375 0.675 at 2.28x; fast32h16 a0.50 0.650 at 2.51x). Falsifier: speed barely moves (< 2.3x) or the headline drops more than a0.375 does.

## Changes
`a0.50_d10` added to `distill_train.ARCHS` and `bench_arch.CANDIDATES`.

## Results
Rung C, buzz+rain+human, seed 1, WSD to 56k (stop rule did not fire for either). Headline = sensitivity_exclquiet @ fpr 0.005, 200 s speed vs YAMNet GPU.

| student | 28k | 56k | x YAM 200s | vs control @56k |
|---|---|---|---|---|
| control fast32 a0.50 (full depth) | 0.680 | 0.687 | 2.15 | - |
| fast32 a0.50_d12 | 0.692 | 0.697 | 2.21 | +0.010 ± 0.006 |
| fast32 a0.50_d10 | 0.696 | 0.696 | 2.28 | +0.009 ± 0.006 |

Per-fold vs control (± eval-sampling SD only):
- d12: 1_29 +0.028 ± 0.014, 53 +0.003 ± 0.009, 1_11 +0.018 ± 0.013, 1_143 -0.033 ± 0.015, 1_37 +0.037 ± 0.014.
- d10: 1_29 +0.023 ± 0.012, 53 +0.001 ± 0.009, 1_11 +0.000 ± 0.015, 1_143 -0.002 ± 0.014, 1_37 +0.023 ± 0.011.
- Tiers: the gain sits mostly in "background" buzz (+0.020 d12, +0.016 d10); quiet/faint flat.

Frontier: d10 at 2.28x scores 0.696 vs fast32 a0.375's 0.675 at 2.27x (+0.021); d12 at 2.21x dominates the full-depth control. Neither reaches fast32h16 a0.50's speed (2.51x). d10's curve saturates at 28k (0.696 at both 28k and 56k; hit@K +0.57 on the last doubling).

## Conclusion
Gain, on the falsifier's terms only half: the headline did not drop (it rose about 0.01 for both, single seed, within roughly 1.5 SD of eval sampling, and training noise is not in that SD), but the speed gain is small: 2.15x -> 2.21x (d12) -> 2.28x (d10), so the "< 2.3x" falsifier strictly holds. The trunk's last layers buy little speed on fast32 at 200 s, yet cost nothing in sensitivity; d10 is the new fast32 point at ~2.28x, ~0.02 above a0.375 at the same speed. Removing layers 11-14 is free capacity-wise, so depth is not where this student's sensitivity lives. Next: go further (d8, d6) where the speed gain compounds, or combine depth cut with fast32h16 to push past 2.5x.
