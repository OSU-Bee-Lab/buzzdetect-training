# Distillation proposer facts

Moved out of `IDEAS.md` (which holds candidate experiments only). Read before
proposing a `05_distill` experiment, after `python 05_distill/ladder_record.py
frontier` (the current frontier; it is not repeated here). Detail and evidence
for each fact: `05_distill/FRONTENDS.md` (closed 2026-10-06) and the `notes.md`
of the experiment named. Speeds are x YAMNet at 200 s on the GTX 1650; headlines
are as in the training arm. How to export and time a student is in
`05_distill/CLAUDE.md`.

## What sets speed

- **Front end and trunk both set the speed.** Front end alone, x YAMNet at
  200 s (`mmt`): fast32h16 9.2, fast32h32 8.4, twofast32 6.9, fast32 6.7,
  yamnet 3.6. Frame rate (hop 16 ms against 10 ms) is the large term; trunks on
  fast32h16 span 5.6x (a0.75_d8) to 7.3x (a0.25). Readings from before
  2026-10-11 (FFT length sets the speed, shrinking the trunk buys little, a
  ~2.7x ceiling) were the waveform Pad (`matmul-stft`). Band count and band
  placement (`lo` twins) were not re-timed.
- **The front end's DFT MatMul is FLOP-bound and dominates 25 ms-window students** (mmt-overlap, 2026-10-18):
  ~80% of a yam64h20 a0.25_d8 graph's GPU time, trunk + head ~2 ms of ~11. Its cost is frames/s x rows x
  2 x DFT bins kept by the mel range, so it scales with the overlap ratio (window / hop) and the band range,
  not the frame rate alone: window = hop front ends (h24, h32) time alike whatever the hop. `mmt` paid for
  `ceil(window/hop) * hop` kernel rows; `--fe-impl mmtz` (exp/mmt-overlap, not in main) pays `window` rows,
  identical outputs: h20 +14-17%, h16 +5-6%, yamnet +4-5%, fast32 +2%, window = hop 0%.
- **Logged speeds drift 5-10% between timing sessions** (same graph, same day-to-day YAMNet rate). Compare
  students within ~1.1x of each other in one `bench_arch.py time` session: `--candidates` takes their model
  dirs directly, `--out <tmp>` (see its docstring); interleave a few rounds as exp/mmt-overlap `retime.sh` did.

## What sets sensitivity

- **No narrow buzz band.** Buzz frames differ from clean negatives nearly flat
  across frequency (`band_profile.py`); no reason to expect a band-limited front
  end to win big.
- **Distil only `ins_buzz`, `ambient_rain`, `human`:** +0.05 at every width on
  fast32h16 (rung B), same speed. `lam=0` (no code regression) hurts. Every live
  frontier student is such a subset student.
- **Steps vs data.** At equal steps rung C ≈ rung B; past ~14k steps the gain
  needs C's data (B overfits: 0.616 at 14k, 0.588 at 56k; C keeps climbing to
  0.650). YAMNet-trunk students are flat from 7k; fast-front-end students keep
  learning to 28-56k. A 7k WSD branch matches a 7k cosine run.
- **Jets at `1_95`:** 36-76% of the threshold-setting false positives in every
  student; none rejects them. Class subsets help on that fold (0.134 to 0.200,
  fast32h16 a0.50). `eval_folds.py probe` scores it (a training fold, not
  held-out). The teacher has not been probed.

## Disk

- **A new front end needs ~23-46 GB free on the root disk** (`df -h /`) for its
  rung-C pack in `05_distill/data/<teacher>/shards/`. If it is short, finished
  packs go to the server disk with `tools/cold_packs.sh` (`05_distill/README.md`,
  "Disk space: cold packs"); don't train from a moved pack (spinning disk).

## Reading a result

- **Seed noise on the headline is ~0.01-0.02**; the repeat rule's bar is 0.02.
  Read single-seed gaps under ~0.03 as ties.
- **hit@K tracks the headline** (r 0.94 over buzz+rain+human students) and is
  calibration-blind, which is why it is the WSD stop signal. hit% at logit 0 is
  as good a proxy but also reads the student's calibration offset.
- **Comparison points for the teacher:** honest rotation 0.574; its ONNX through
  the harness 0.692 (inflated: trained on those folds). Students inherit fold
  knowledge through the teacher, so beating 0.574 is not "better than the
  teacher".
- **Contamination:** every rung-B student and the yamnet rung-C runs before
  2026-10-04 trained on test-set audio; their rows are in
  `05_distill/log.contaminated_2026-10-02.jsonl`. Directions survive (one retrain
  landed inside seed noise), numbers are not comparable.
- **Eval is slightly pessimistic for long windows** (each frame scored alone,
  zero-padded), and the headline is the student ONNX's, not the full pipeline's.

## Never tried

Pruning dead channels, non-uniform widths, a learned (conv) front end, int8
(needs modern hardware to time).
