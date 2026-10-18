# mmt-overlap
## Hypothesis
yam64h20-cliff found that student speed tracks the mmt front end's `m = ceil(window / hop)` -- the number of
MatMuls over shifted row slices `a[j:j+n]` summed by add_n -- more than the trunk's frame count: hop 20 ms
(48 frames, m=2) is only 1.05x faster than hop 16 (60 frames, m=2), while window = hop front ends (h24, h32; m=1)
are ~1.4x faster than h20, and h32 (30 frames) is no faster than h24 (40). FLOP-wise the m=2 front end is tiny
next to the trunk, so the gap should be graph overhead (sliced copies of the reshaped waveform, the add_n, the
zero rows of the padded kernel), removable in export only, as matmul-stft removed the Pad.
Test: prefix_bench on yam64h20 a0.25_d8 (m=2) vs yam64h24 a0.25_d8 (m=1) to locate the gap, then an alternative
mmt formulation for m > 1 with identical outputs (e.g. one MatMul against the concatenated per-shift kernels,
shifts applied to the small output instead of the waveform; or frames gathered once), re-exported for the
overlapping-window students (yam64h16, yam64h20, yamnet, twofast32 ...) with identical headlines.
Prediction: the h20 a0.25_d8 graph's front end costs >= 25% of its 200 s time vs <= 10% for h24, and the new
form brings yam64h16 a0.25_d8 (0.711) from 4.57x to >= 5.5x -- onto the frontier above twofast32 a0.50_d8
(0.701 @ 5.33x).
Falsifier: the front end is < 10% of h20's time in prefix_bench (the gap is in the trunk, not the m=2 graph),
or the new form changes speed by < 5%.
Comparator: the same students' existing `_mmt` rows (same weights; headline must match exactly).
## Changes
`05_distill/frontends.py`: new impl `mmtz` = `mmt` with each hop of the window multiplying only its live samples
(the last, partial hop is a column Slice of the reshaped waveform against the kernel's non-zero rows), so a frame
costs `window` kernel rows, not `ceil(window / hop) * hop`. `export_student.py` / `student.py` accept it. Checked
equal to `mmt` (max |diff| 0.0) on yam64h16/h24, twofast32, fast32, fast32h16 and yamnet front ends at two lengths.
`yam64h20` front end brought over from exp/yam64h20-cliff. `chain.sh`: `reexport.py --fe-impl mmtz --suffix _mmtz`
for yam64h20 a0.25_d8/a0.50_d8, yam64h16 a0.25_d8/a0.50_d8, yamnet a0.25_d8/a0.50_d8, fast32 a0.50_d8 and
yam64h24 a0.25_d8 (window = hop: nothing to trim, a timing-noise control). Rows saved per frame: h20 640->400,
h16 512->400, yamnet 480->400, fast32 320->256, twofast32/h24/h32/fast32h16 none.
## Results
**Headlines identical** for all 8 re-exports (`_mmtz` rows = `_mmt` rows: 0.693, 0.658, 0.711, 0.706, 0.727, 0.713,
0.735, 0.685), as the equality check predicted.

**Where the time went (prefix_bench, 200 s, yam64h20 a0.25_d8, before):** of 11.39 ms, the front end is 9.2 ms
(incl. ~2.3 ms for the first Slice of the input, the same in every graph); its two shifted MatMuls 1.80 + 1.79 ms
and AddN 0.55 ms; the whole trunk + head 2.2 ms. yam64h24 (m=1): one MatMul 1.11 ms, total 7.40 ms. So the
front end MatMuls are FLOP-bound and the m=2 graph paid for its kernel's zero rows. **After (mmtz):** total 9.51 ms;
first MatMul 1.82 ms, the trimmed second 0.48 ms, AddN 0.51; front end still ~7.5 of 9.5 ms.

**Speed, paired (one session, 3 interleaved rounds, rounds within 1%; `retime.sh`):**

| student | mmt | mmtz | ratio | rows/frame |
|---|---|---|---|---|
| yam64h20 a0.25_d8 | 4.89x | 5.73x | 1.172 | 640 -> 400 |
| yam64h20 a0.50_d8 | 4.11x | 4.68x | 1.138 | 640 -> 400 |
| yam64h16 a0.25_d8 | 4.97x | 5.26x | 1.060 | 512 -> 400 |
| yam64h16 a0.50_d8 | 4.02x | 4.21x | 1.049 | 512 -> 400 |
| yamnet a0.25_d8 | 3.47x | 3.65x | 1.052 | 480 -> 400 |
| yamnet a0.50_d8 | 2.77x | 2.87x | 1.036 | 480 -> 400 |
| fast32 a0.50_d8 | 4.97x | 5.10x | 1.024 | 320 -> 256 |
| yam64h24 a0.25_d8 (control) | 7.40x | 7.40x | 1.000 | 384 -> 384 |

**Timing drift between sessions:** the same mmt graphs time 5-10% faster today than their logged rows
(yam64h16 a0.25_d8 4.57x -> 4.97x, yam64h24 a0.25_d8 6.69x -> 7.40x, twofast32 a0.50_d8 5.33x -> 5.72x;
YAMNet's own rate is steady at ~3650 s/s, so it is the students' timing that moved). Logged speeds from different
days are only good to ~10%, so frontier calls between students within ~1.1x need a same-session re-time.

**Frontier, same-session (`retime2.sh`, 2 rounds):** twofast32 a0.50_d8 s1/s2 5.72x / 5.73x (0.701 / 0.694),
yam64h20 a0.25_d8 mmtz 5.67x (0.693), yam64h16 a0.25_d8 mmtz 5.27x (0.711), yam64h24 a0.50_d8 5.82x (0.689),
yam64h32 a0.50_d8 6.34x (0.687), twofast32 a0.25_d8_l5x0.50 6.32x (0.683), yam64h32 a0.25_d8_l5x0.50 7.10x
(0.683). So h20 a0.25_d8 is still a tie with twofast32 a0.50_d8 (-0.008 +- 0.009 at 0.99x its speed), and h16
a0.25_d8 is +0.010 over it at 0.92x the speed: neither is a step. The logged `_mmtz` rows carry the
re-export session's speeds (5.66x, 5.28x, ...), which match these.

## Conclusion
Falsifier not met: the front end is ~80% of a yam64h20 student's GPU time, its MatMuls are FLOP-bound, and
dropping the kernel's zero rows (`--fe-impl mmtz`, identical outputs and headlines) speeds h20 students 14-17%,
h16 5-6%, yamnet 4-5%, fast32 2%, window = hop front ends 0%. Prediction partly failed: h16 a0.25_d8 reaches 5.27x,
not >= 5.5x, and no student moves onto the frontier (the h20 tie with twofast32 a0.50_d8 remains a tie). Use
`mmtz` over `mmt` for every re-export from now on (equal for window = hop, never slower). What remains: the front end
is still ~75% of the h20 graph and FLOP-bound in the window x bins MatMul (frames/s x window x 2*bins), so the
next speed lever for 25 ms-window students is fewer DFT bins or less overlap, which changes the features and needs
training. Also: logged speeds drift 5-10% between sessions; compare close students in one session.
