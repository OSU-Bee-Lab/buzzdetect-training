# yam64h24
## Hypothesis
Distillation arm, the lead `yam64h32` left. On 64 bands, hop 16 ms cost almost nothing against 10 ms
(`yam64h16`), but hop 32 ms cost 0.04-0.05 against hop 16 (`yam64h32`: a0.25_d8 0.658 vs 0.711, a0.50_d8
0.687 vs 0.727), so the frame-rate cliff is somewhere between 60 and 30 frames per patch. `yam64h24` sits
between them: 64 bands, 125-7500 Hz, hop 24 ms (40 frames per patch), window = hop (24 ms, fft 512) so no
audio falls between frames.

Speed before training (random weights, mmt, 200 s, GPU, one session that read ~7% slow throughout:
twofast32 a0.50_d8 timed 4.99x against its logged 5.33x): `yam64h24` a0.25_d8 6.59x, a0.50_d8 5.51x, front
end alone 7.24x; same session `yam64h32` a0.50_d8 5.77x. Scaled to the log's timings, about 7.0x and 5.9x.

Prediction: `yam64h24:a0.50_d8` keeps >= 0.71 (most of h16's 0.727) at ~5.9x, above `twofast32:a0.50_d8`
(0.694-0.701 at 5.33x) and `yam64h32:a0.50_d8` (0.687 at 6.06x); `a0.25_d8` lands ~0.68-0.69 at ~6.5-7x,
above `yam64h32:a0.25_d8` (0.658 at 6.81x).
Falsifier: the h24 students land no better than their h32 twins (within ~0.01), i.e. the cliff is
already at 40 frames, or they time no faster than `twofast32:a0.50_d8`.
Matched comparators (same teacher `v4-ft-ps-e60-moderate`, rung C, WSD 7000-56000 + stop rule, seed 1,
buzz+rain+human, same trunks, mmt): the `yam64h32` and `yam64h16` twins,
`fe_C_yam64h32_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt`, `..._a0.50_d8_...`, and the same for `yam64h16`.

## Changes
- `05_distill/frontends.py`: `yam64h24` registered. The rest of the `05_distill/` diff is `exp/yam64h32`'s
  code as a patch (`mmt` front end, `reexport.py`, bench), not in main.
- `chain.sh`: `main.py --rung C --seed 1 --wsd-max 56000 --wsd-stop 1.3` on `yam64h24:a0.50_d8` and
  `a0.25_d8` (buzz+rain+human), then `reexport.py --fe-impl mmt --suffix _mmt` on the judged branches.
  Launched from `../yam64h32-s2/chain.sh` after that repeat, in one job.

## Results
Chain exit 0 (job of `../yam64h32-s2`, pid 1075598), comparators in `cmp.log` (`cmp.sh`).

WSD curves (hit@K): a0.50_d8 62.99 / 65.04 / 65.37 / 65.52 at 7000-56000; the stop rule fired on
28000 and 56000 (+0.32, +0.15), last branch run 56000. a0.25_d8 57.86 / 60.08 / 62.67 / 63.10, never
fired. Both judged at 56000. Headlines by budget: a0.50 0.680 / 0.685 / 0.690 / 0.689, a0.25 0.636 /
0.641 / 0.651 / 0.658.

Judged rows (mmt, speed = x_yamnet200):
| student | headline | inclusive | speed |
|---|---|---|---|
| yam64h24:a0.50_d8 | 0.689 | 0.577 | 5.59x |
| yam64h24:a0.25_d8 | 0.658 | 0.552 | 6.69x |
| yam64h32:a0.50_d8 (s1) | 0.687 | 0.572 | 6.06x |
| yam64h32:a0.25_d8 (s1 / s2) | 0.658 / 0.676 | | 6.81x / 7.05x |
| yam64h16:a0.50_d8 / a0.25_d8 | 0.727 / 0.711 | | 3.77x / 4.57x |
| twofast32:a0.50_d8 (s1) | 0.701 | 0.590 | 5.33x |

Paired (tools/results.py, ± eval sampling SD):
- vs hop-32 twins: a0.50_d8 +0.002 ± 0.008, a0.25_d8 +0.000 ± 0.008. Every tier within 0.02 except
  faint (35 frames).
- vs hop-16 twins: a0.50_d8 -0.038 ± 0.008, a0.25_d8 -0.053 ± 0.009. Untagged carries most of it
  (-0.045 / -0.049), loud is flat (-0.003 / -0.020).
- vs twofast32:a0.50_d8 (s1): -0.012 ± 0.009 at 1.05x its speed.

Speed: the h24 students timed *slower* than their h32 twins (5.59x vs 6.06x, 6.69x vs 6.81x), and only
~5% faster than twofast32:a0.50_d8. The pre-training guess (~5.9x / ~7.0x) was high.

## Conclusion
Falsifier met on both counts. Hop 24 ms (40 frames) scores the same as hop 32 ms: +0.002 and +0.000,
within sampling. The whole hop-16→32 loss is already there at hop 24 (-0.038 / -0.053 against h16).
So the frame-rate cliff sits between 60 and 40 frames per patch, not between 40 and 30. And h24 is
slower than h32, so it is dominated: the same sensitivity at lower speed. Nothing new on the frontier.
a0.50_d8 sits at 0.689 / 5.59x, just under twofast32:a0.50_d8 (0.694-0.701 / 5.33x). If the frame-rate
axis gets probed again, it is between hop 16 and hop 24 (e.g. hop 20, 48 frames). Below 60 frames
there is nothing left to buy. No repeat (`ladder_record.py repeats` flags none).
