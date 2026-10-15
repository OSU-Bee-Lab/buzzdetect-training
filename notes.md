# nonuniform-width
## Hypothesis
Distillation arm, "never tried: non-uniform widths" (docs/distill-proposer-facts.md). On every fast front end the
a0.25_d8 -> a0.50_d8 width step buys headline (twofast32 0.668/0.668 -> 0.701/0.694, yam64h32 0.658/0.676 -> 0.687,
yam64h16 0.711 -> 0.727) and costs speed. Random-weight timing (mmt, 200 s, GPU, this session) says the cost is
not spread evenly: widening only the late, low-resolution layers is nearly free. On yam64h16, a0.25_d8 4.41x,
layers 7-8 at a0.50 4.42x, layers 5-8 at a0.50 4.28x, layers 3-8 at a0.50 3.84x (= a0.50_d8's 3.78x). On
twofast32: a0.25_d8 5.74x, `a0.25_d8_l5x0.50` 5.87x, `a0.25_d8_l7x1.0` 5.77x, a0.50_d8 5.39x (session noise
~5%; logged a0.25_d8 6.14x / 5.85x, a0.50_d8 5.33x).

If the width gain lives in the late layers (more channels for the code/readout at low resolution), a trunk that
is a0.25 early and wide late gets a0.50_d8's headline at a0.25_d8's speed.

Students (teacher `v4-ft-ps-e60-moderate`, rung C, WSD 7000-56000 + stop rule 1.3, seed 1, buzz+rain+human,
twofast32, judged on the mmt re-export):
- `twofast32:a0.25_d8_l5x0.50` (filters 8,16,32,32,128,128,256,256)
- `twofast32:a0.25_d8_l7x1.0` (filters 8,16,32,32,64,64,512,512)

Prediction: both >= 0.69 (most of the +0.03 a0.25 -> a0.50 step) at ~5.8-6.0x, above everything at that speed
(yam64h32:a0.50_d8 0.687 at 6.06x, twofast32:a0.25_d8 0.668 at 6.14x).
Falsifier: within 0.01 of twofast32:a0.25_d8 (0.668, both seeds), i.e. the width gain is in the early,
high-resolution layers, which is where the cost is too.
Matched comparators (same teacher, rung, schedule, seed 1, front end, classes, mmt):
`fe_C_twofast32_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt` (0.668) and `..._a0.50_d8_...` (0.701).

## Changes
- `05_distill/student.py`: arch names take `_l<j>x<alpha2>` (layers j on at alpha2); `arch_filters(name)` gives
  the per-layer widths. `parse_arch` keeps its (alpha, depth, fused) return.
- `distill_train.py` builds from `arch_filters(a.arch)`; `bench_arch.py` times such names. Export reads widths from
  the trained model, YAMNet channel-selection init reads them from the built model, so neither changed.
- `chain.sh`: main.py on the two students, then the mmt re-export; then `../nonuniform-width-yam64h32/chain.sh`.

Disk note: the root disk has 15 GB free, so no new front end (34 GB pack) could be run this batch; both
experiments reuse existing packs (friction reported).

## Results
