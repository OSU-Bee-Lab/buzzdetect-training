# yamnet-a075-d8
## Hypothesis
`yamnet-depth` showed layers 9-14 buy nothing on the YAMNet front end at a0.50. Does width bought back on the 8-layer trunk buy sensitivity at the frontier's high end (the standard-tier side)? Full depth, a0.50 over a0.25 bought +0.018. Random-weight bench (bench_arch, 20 s, GPU, one run, x YAMNet; within-bench ratios only): a0.50 1.50, a0.375 1.67, a0.25 1.88, a0.50_d8 1.76, a0.50_d6 1.82, a0.375_d8 1.88, a0.25_d8 2.00, a0.75_d8 1.39, a0.75_d6 1.62, a0.375_d6 1.92, a1.00_d6 1.48. Logged/bench ratio on this front end: a0.50 0.93, a0.25 0.86, a0.50_d8 0.86. a0.75_d8 benches at 1.39 (a0.50 full 1.50), so expect ~1.25-1.30x logged. Student: yamnet a0.75_d8, seed 1. Rung C, buzz+rain+human (select), WSD to 56k with the stop rule (--wsd-max 56000 --wsd-stop 1.3), teacher v4-ft-ps-e60-moderate. Matched control: yamnet a0.50_d8 (0.735 @1.52x, width only). Gain: headline > 0.735 by more than seed noise (~0.02), since a0.50_d8 is faster and anything slower must beat it. Falsifier: <= 0.735, i.e. width beyond a0.50 buys nothing on the 8-layer trunk (shallow-wide on fast32h16 traded along the frontier: +0.010 ± 0.006 over a0.50_d8 for -0.18x).

## Run log
- 2026-10-09 10:17: the batch-26 chain (pid 399369) died at step 0 of this student: OOM in `val_flips`' eager
  batch-512 inference, right after the select init. TF only ever gets ~2.4 GB of the 4 GB card here. Measured:
  after the init only 17 MB stays resident (no leak); a random-weight a0.75_d8 trains at batch 512 with a 2.22 GB
  peak; the eager val call at 512 straight after the init fills all 2.4 GB. Fix (commit a56376c on
  exp/yamnet-d8-repeat): `val_flips` batch 512 -> 128. It is inference only, so results are unchanged; training
  batch, schedule and step counts are untouched. A 30-step `test_a075_d8_mem` run (init, val, batch-512 training,
  full-pool final val) ran clean and was deleted. Relaunched alone at 10:28: pid 1377627, log
  `.local/worktrees/yamnet-d8-repeat/distill.log`.
