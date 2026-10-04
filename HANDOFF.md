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

## Running: `05_distill/chain_wsd.sh` (launched 2026-10-03 from the main checkout)

    tools/launch_job.sh 05_distill/data/chain_wsd.log -- bash 05_distill/chain_wsd.sh

Backfill + proxy check, then: rung-B equivalence (wsd7000 vs cosine 7k, yamnet + fast32h16 a0.50); yamnet
a0.50 rung C to 56k under the new rule (the old rule cut it at 14k); fast32h16 a0.50 rung C to 112k with no
stop (a check on the new rule); the rung-B a0.25 curve; fast32h16 a0.25 rung C to 56k. `--wsd-stop` is the
rung-A hit@K spread (`ladder_record.py spread`). Same command resumes; a failed step is retried once, then skipped.

## Still to do

1. Read the chain log: `ladder_record.py proxy` (does hit@K track the headline at least as well as hit% at 0,
   r = 0.86 / 0.96?), each curve with `ladder_record.py wsd --name <trunk>`, and the equivalence (a `wsd7000`
   branch near the cosine 7k run on `mae_live` and lost%; if clearly worse, try `lr=2e-3` / `lr=5e-4`).
   The noise floor is two repeats only; say plainly where a curve is flat or still rising at its last point.
2. Revisit LADDER.md's "B to C did not advance": the ladder was confounded by passes (A/B/C got ~19/5/1.2).
3. Then a new sensitivity-speed frontier at rung C, like rung B's (`tools/human/frontier_*`): each student at
   its own budget under the stop rule, not a fixed 7k (Luke, 2026-10-03: "eventually").
