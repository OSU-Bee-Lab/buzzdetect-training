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

## Conclusion
