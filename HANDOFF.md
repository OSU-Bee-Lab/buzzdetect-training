# HANDOFF: step-budget curve for distillation (warmup-stable-decay)

The code is in: `main.py --wsd`, `distill_train.py --schedule wsd`, `ladder_record.py wsd`
(05_distill/README.md, "Step budget", has the design and the choices made). What is left is the experiment.
Delete this file when the curves are reported.

## Done: first rung-C curves (2026-10-02, old stop rule on hit% at logit 0, tolerance 1.3)

    run (rung C, buzz+rain+human)      steps  hit%@0  headline
    yamnet a0.50 wsd7000                7000   61.2    0.723   (rung-B cosine 7k: 0.705)
    yamnet a0.50 wsd14000              14000   59.8    0.722   -> stopped here (dhit -1.38, inside the 1.31 noise)
    fast32h16 a0.50 wsd7000             7000   45.4    0.605   (rung-B cosine 7k: 0.610)
    fast32h16 a0.50 wsd14000           14000   51.9    0.621
    fast32h16 a0.50 wsd28000           28000   53.8    0.639
    fast32h16 a0.50 wsd56000           56000   52.2    0.650   (hit% dipped, headline and mae_live improved)

## Stop rule changed to hit@K (2026-10-03)

hit% at logit 0 also measures the student's calibration offset: a small student trained by Huber regression
under-calls a rare class (fast32h16 at 56k: 5,195 buzz calls on V vs the teacher's 7,685). hit@K (share of the
teacher's K V buzz calls among the student's K top scores) reads ranking only, as the headline does. README,
"Step budget", has it; `backfill_hitk.py` adds it to older runs; `ladder_record.py proxy` checks it against
the headline. Calibration does not matter for deployment (buzzdetect users set the threshold); only the rule.

## Running: `05_distill/chain_wsd.sh` (launched 2026-10-03 23:16 from the main checkout, pid 2046228)

    tools/launch_job.sh 05_distill/data/chain_wsd.log -- bash 05_distill/chain_wsd.sh

Nobody is watching it: Luke is away until about 2026-10-05 05:00 and asked for no ongoing monitoring. A fresh
agent checks it once, fixes what is broken, resumes, and does not arm a recurring watch unless asked.

Steps, in order, with rough wall time (GPU; ~20 h total if no stop fires):

    0 backfill-hitk            ~80 min  59 runs x ~80 s; then proxy, both rung-C curves, `--wsd-stop` = rung-A hit@K spread
    1 equivalence              ~1.2 h   rung B wsd7000, yamnet + fast32h16 a0.50 (the yamnet trunk resumes from step 2000)
    2 yamnet-C-56k             up to 6 h  rung C, new rule; may stop right away if hit@K 7k -> 14k gained < the spread
    3 fast32h16-C-112k         ~4.5 h   rung C, NO stop: extends the trunk 47600 -> 95200, then the 112k branch
    4 yamnet-a0.25-B-curve     ~3.5 h   rung B 7k/14k/28k/70k, new rule
    5 fast32h16-a0.25-C-56k    ~5 h     rung C, new rule

**Checking it** (one look, not a loop):

    ps -p 2046228 >/dev/null && echo running || echo ended
    grep -E "\[wsdchain\]|RECOUNT DIFFERS|Traceback|launch_job\] exit" 05_distill/data/chain_wsd.log | tail -30

- `[wsdchain] <step>: done` for each step; `[wsdchain] failed steps: none` and `[launch_job] exit 0` at the end.
- `attempt 2 FAILED` means that step was skipped; read the Traceback above it, fix, and relaunch the same
  command: every stage skips when its artifact exists, so the rerun redoes only what is missing.
- `RECOUNT DIFFERS` on a backfill line: the saved model did not reproduce the recorded teacher/lost counts.
  Small differences (a few frames near logit 0) are GPU nondeterminism; large ones mean the wrong V pack or
  class list (backfill_hitk.py takes both from the run's curve.json args). None in the first 7 runs.
- `[wsdchain] no hit@K spread; stopping`: the rung-A repeats (lad_A_s1, lad_A_s2) were not backfilled.
- `plateau` / `skipped (WSD plateau, --wsd-stop X)` lines are the rule working, not a fault.

**Killing it:** killing launch_job's pid alone does not stop the python under it (2026-10-03: main.py and
distill_train.py kept training). Kill the trainer and main.py too: `pgrep -af "[d]istill_train|[0]5_distill/main"`,
then `kill` those pids, and confirm the card is free with `nvidia-smi --query-compute-apps=pid --format=csv`.
Training checkpoints every 2000 steps and at each branch point, so a kill loses at most 2000 steps.

**Early readout (backfill, first 7 runs).** hit@K vs hit% at logit 0, same runs:

    fast32h16 a0.50 C   7k 60.39 (45.39)   14k 63.54 (51.92)   28k 64.62 (53.81)   56k 64.76 (52.24)
    fast32h16 trunk @56k 62.34 (41.01; undecayed)   yamnet a0.50 C 14k 70.53 (59.80)

hit@K does not dip at 56k, so calibration explains hit% at 0's dip. But hit@K is flat from 28k to 56k (+0.14)
while the headline rose 0.011 (inside seed noise): the new rule at ~1.3 would have stopped fast32h16 after
28k. Step 3's 112k branch, run without a stop, says whether that stop is right; Luke also has a truly
held-out test set to check the 7k/28k/56k/112k branches on when he is back.

Known, unrelated: `test_distill.py`'s `test_migrate` "dry run moves nothing" fails on main as well.

## Still to do

1. Read the chain log: `ladder_record.py proxy` (does hit@K track the headline at least as well as hit% at 0,
   r = 0.86 / 0.96?), each curve with `ladder_record.py wsd --name <trunk>`, and the equivalence (a `wsd7000`
   branch near the cosine 7k run on `mae_live` and lost%; if clearly worse, try `lr=2e-3` / `lr=5e-4`).
   The noise floor is two repeats only; say plainly where a curve is flat or still rising at its last point.
2. Revisit LADDER.md's "B to C did not advance": the ladder was confounded by passes (A/B/C got ~19/5/1.2).
3. Then a new sensitivity-speed frontier at rung C, like rung B's (`tools/human/frontier_*`): each student at
   its own budget under the stop rule, not a fixed 7k (Luke, 2026-10-03: "eventually").
