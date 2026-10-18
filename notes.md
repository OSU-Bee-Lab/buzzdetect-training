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
## Results
## Conclusion
