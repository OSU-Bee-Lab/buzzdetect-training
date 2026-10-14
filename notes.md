# yam64h32
## Hypothesis
Distillation arm, the lead `yam64h16` left. On YAMNet's 64 bands the 16 ms hop cost almost nothing against
the 10 ms twins (-0.002 / -0.008 +- 0.007), while frame rate is the largest speed term. `yam64h32` halves
the frame rate again: 64 bands, 125-7500 Hz, hop 32 ms (30 frames per patch), the window grown to the hop
(32 ms, fft 512) so no audio falls between frames (as `fast32h32` did).

Prediction: `yam64h32:a0.25_d8` / `a0.50_d8` keep within ~0.01-0.02 of their `yam64h16` twins (0.711 / 0.727)
and time well above them (front end alone should approach fast32h32's 8.4x territory, minus the 64-band
mel), so they land above the current line between `yam64h16:a0.25_d8` (0.711 at 4.57x) and
`twofast32:a0.25_d8` / `fast32h16` (0.668 at ~6x, 0.646 at 6.7x).
Falsifier: the hop-32 cost on 64 bands is ~0.03 or more against the hop-16 twins (frame rate stops being
free below ~30 frames per patch), or the bench puts them no faster than `yam64h16`.
Matched comparators (same teacher `v4-ft-ps-e60-moderate`, rung C, WSD 7000-56000, seed 1, buzz+rain+human,
same trunks, mmt): `fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt`, `..._a0.50_d8_...`.

## Changes
- `05_distill/frontends.py`: `yam64h32` registered. The rest of the `05_distill/` diff is `exp/yam64h16`'s
  code as a patch (the `mmt` front end, `reexport.py`, bench), not in main.
- `bench.sh`: random-weight speed (pad-free `mmt`, 200 s, GPU), run as the chain's first step.
- `chain.sh`: bench, then `main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3` on `yam64h32:a0.25_d8`
  and `a0.50_d8` (buzz+rain+human), then `reexport.py --fe-impl mmt --suffix _mmt` on the judged branches.
- Smoke: `a0.25_d8@yam64h32+mmt` builds and exports on CPU (40k params, 0.82 MMACs).

### Speed bench before training (random weights, pad-free `mmt`, 200 s, GPU; `bench.log`)
| graph | x YAMNet | MMACs |
|---|---|---|
| front end alone, yam64h32 | 6.62 | |
| **yam64h32 a0.25_d8** | **6.70** | 0.8 |
| **yam64h32 a0.50_d8** | **6.00** | 2.9 |
| front end alone, yam64h16 | 5.37 | |
| yam64h16 a0.25_d8 | 4.20 | 1.6 |
| yam64h16 a0.50_d8 | 3.79 | 5.8 |
| twofast32 a0.50_d8 | 5.33 | 3.0 |
| twofast32 a0.25_d8 | 6.12 | 0.8 |
| front end alone, fast32h16 | 8.90 | |
| fast32h16 a0.25 | 6.80 | 1.6 |

Reading before training: halving the frame rate makes `yam64h32:a0.50_d8` faster than `twofast32:a0.50_d8`
(6.00x vs 5.33x) and `a0.25_d8` level with `fast32h16:a0.25` (6.70x vs 6.80x), the 64-band front end
(6.6x alone) now the ceiling. (`a0.25_d8` timing above its own front end is session noise, ~+-0.3x.)
So `a0.50_d8` is above the line if it holds ~0.70 (`twofast32:a0.50_d8` 0.694-0.701 is slower), and
`a0.25_d8` if it beats ~0.67 (`twofast32:a0.25_d8` 0.668 at 6.1x, `fast32h16:a0.50_d8` 0.646 at 6.7x).

## Results
Judged budget: 56000 for both (the stop rule fired there: hit@K +3.84 / +1.21 / +0.59 for a0.25_d8,
+1.88 / +0.56 / +0.39 for a0.50_d8). Speed is the `_mmt` row's x YAMNet at 200 s on the GPU.

| student | headline | x YAM | hop-16 twin | twin x YAM |
|---|---|---|---|---|
| `yam64h32:a0.25_d8` | 0.658 | 6.81 | 0.711 | 4.57 |
| `yam64h32:a0.50_d8` | 0.687 | 6.06 | 0.727 | 3.77 |

`a0.25_d8`, baseline = `fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt`,
this = `fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt`:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.801 | 0.760 | -0.041 | 0.017 | 34 |
| 53 | 0.663 | 0.646 | -0.017 | 0.018 | 32 |
| 1_11 | 0.688 | 0.589 | -0.099 | 0.021 | 32 |
| 1_143 | 0.680 | 0.607 | -0.073 | 0.020 | 26 |
| 1_37 | 0.723 | 0.687 | -0.036 | 0.024 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.711 → this 0.658 (-0.053 ± 0.009)
- inclusive (sensitivity), same thresholds: 0.594 → 0.542 (-0.052)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.817 | 0.806 | -0.011 | 603 |
| untagged | 0.717 | 0.662 | -0.055 | 7485 |
| background | 0.692 | 0.658 | -0.034 | 9412 |
| quiet | 0.231 | 0.192 | -0.039 | 2231 |
| faint | 0.000 | 0.000 | +0.000 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

`a0.50_d8`, baseline = `..._yam64h16_a0.50_d8_...`, this = `..._yam64h32_a0.50_d8_...`:

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.824 | 0.792 | -0.032 | 0.014 | 34 |
| 53 | 0.657 | 0.653 | -0.004 | 0.018 | 32 |
| 1_11 | 0.678 | 0.626 | -0.052 | 0.023 | 32 |
| 1_143 | 0.715 | 0.644 | -0.071 | 0.020 | 26 |
| 1_37 | 0.760 | 0.722 | -0.038 | 0.014 | 15 |
- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.727 → this 0.687 (-0.040 ± 0.008)
- inclusive (sensitivity), same thresholds: 0.613 → 0.572 (-0.041)
- folds that missed fpr 0.005: none
| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.815 | 0.813 | -0.002 | 603 |
| untagged | 0.731 | 0.690 | -0.041 | 7485 |
| background | 0.704 | 0.682 | -0.022 | 9412 |
| quiet | 0.255 | 0.218 | -0.037 | 2231 |
| faint | 0.000 | 0.014 | +0.014 | 35 |
± SD is eval sampling only (tools/eval_sampling_sd.py); training stochasticity is larger per fold and not in it.

Reading. The hop-32 cost on 64 bands is real: -0.053 ± 0.009 (a0.25_d8) and -0.040 ± 0.008 (a0.50_d8)
against the hop-16 twins, every fold down in both, most on 1_11 and 1_143 (-0.05 to -0.10, several SD);
53 is within its SD in both. Untagged carries it (-0.055 / -0.041), background less (-0.034 / -0.022), loud
barely moves (-0.011 / -0.002). The falsifier (a hop-32 cost of ~0.03 or more) is met: frame rate stops
being free between 60 and 30 frames per patch.

Against the frontier, by speed it still pays:
- `a0.50_d8` 0.687 at 6.06x: above `twofast32:a0.25_d8` (0.668 at 6.14x, +0.019) and `fast32h16:a0.50_d8`
  (0.646 at 6.7x); below `twofast32:a0.50_d8` (0.694-0.701) but 1.14x faster. On the line, gain over
  everything at its speed or faster +0.019, just under the repeat rule's 0.02.
- `a0.25_d8` 0.658 at 6.81x: above everything at its speed or faster (`fast32h16:a0.25_d8_f2` 0.630 at
  6.83x, `fast32h16:a0.25` 0.612-0.616 at 7.3x), +0.028; the repeat rule fires (`yam64h32-s2`). Roughly
  level with `twofast32:a0.25_d8` (0.668) at 1.11x its speed.
Speed sits on the front end now: the bench put the front end alone at 6.6x and a0.25_d8 at 6.7x.

## Conclusion
Halving the frame rate again on YAMNet's 64 bands costs 0.04-0.05 headline against hop 16 (falsifier met,
the cliff is between 60 and 30 frames), loss concentrated in untagged and on 1_11 / 1_143, loud flat. Both
students still join the frontier by speed: `a0.50_d8` 0.687 at 6.06x (+0.019 over anything as fast) and
`a0.25_d8` 0.658 at 6.81x (+0.028, single seed; repeat `yam64h32-s2` queued). Next: hop 24 ms
(`yam64h24`, 40 frames), which the bench puts at ~5.9x / ~7.0x, to find where between 60 and 30 frames the
cost starts.
