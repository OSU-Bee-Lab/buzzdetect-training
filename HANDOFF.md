# HANDOFF — batch 23 (covers all 4 experiments)

Logged so far: **3 of 4** (d1-yamnet-a050-clean, d2-fast-56k, fast32-a0375-c). Remaining: 4 (fast32h16-a0375-c). All four are distillation-arm runs, no code changes beyond a dpaths fix.

| # | slug (worktree / branch exp/<slug>) | job |
|---|---|---|
| 1 | d1-yamnet-a050-clean | DONE, logged |
| 2 | d2-fast-56k | DONE, logged. The chain (pid **2594037**) and its log `.local/worktrees/d2-fast-56k/chain.log` still live in that worktree |
| 3 | fast32-a0375-c | DONE, logged (chain exit 0 at 18:32) |
| 4 | fast32h16-a0375-c | same chain |

The chain (`d2-fast-56k/chain_b23.sh`, committed) waits for pid 2589974 to exit, then runs 2, 3, 4
sequentially, each from its own worktree, printing `[chain] <slug> exit N` after each.
Each worktree's `notes.md` has the Hypothesis, matched control and exact command.

Measured rate (2026-10-06): yamnet a0.50 trunk ~7 min / 1000 steps after a ~15 min load.

## Resume
1. `tools/watch_job.sh --adopt 2594037`, then ONE Monitor on
   `tools/watch_job.sh`. **If still running, report progress (and park again if > 1.5 h remains) and stop.**
2. As each experiment finishes (its rows are in main's `05_distill/log.jsonl`):
   - Read: `tools/results.py <control eval dir> <student eval dir>` (dirs under
     `05_distill/data/v4-ft-ps-e60-moderate/eval/<name>`), `ladder_record.py wsd --name <trunk>_wsd`.
     NB `ladder_record.py frontier` is rung-B-only and prints nothing useful (friction filed); place the
     student against the rung-C rows in log.jsonl / FRONTENDS.md's 2026-10-05 table by hand.
   - Controls: 1 -> `fe_C_yamnet_a0.25_s1_select_c-buzz-rain-human_wsd*`; 2 -> each trunk's own 28k branch;
     3 -> `fe_C_fast32_a0.50_s1_c-buzz-rain-human_wsd*`; 4 -> `fe_C_fast32h16_a0.50/a0.25_s1_c-buzz-rain-human_wsd*`.
   - Fill notes.md Results/Conclusion, delete D1/D2 from main's IDEAS.md for 1/2, then from main:
     `tools/finish_experiment.sh <slug> --arm distill --summary "..." --runs "<student names>" [--commit-also IDEAS.md]`.
     For 1, also append a dated update to FRONTENDS.md if the standard-tier reading changed.
3. When all 4 are logged: `loop_signal.sh done "<slugs>"`.

## If a job died
Every stage resumes on rerun. Launch_job's --verbose breaks distill main.py (friction filed), so launch as:
- 1: from `.local/worktrees/d1-yamnet-a050-clean`:
  `tools/launch_job.sh distill.log -- bash -c 'source /home/luke/projects/buzzdetect-training/tools/python_path.sh; exec env MALLOC_ARENA_MAX=2 "$PY" -u 05_distill/main.py --rung C --runs "yamnet:a0.50:select:classes=ins_buzz+ambient_rain+human" --wsd-max 56000 --wsd-stop 1.3'`
- 3-4: from `.local/worktrees/d2-fast-56k`: `tools/launch_job.sh chain.log -- bash chain_b23.sh`
  (edit its first `while kill -0 2589974` line out if pid 1 is gone; finished stages skip themselves).
Each worktree carries the dpaths.py/student.py `.local/worktrees` fix; without it main.py can't find the onnx venv.

## Progress at third park (2026-10-06 18:55)
Exp 4 (fast32h16 a0.375) trunk started 18:34, at ~5k/5950 (first branch) by 18:50, ~5.9 step/s.
Trunk to 47.6k ~= 2.2 h plus four decays/evals: done ~21:30 if it runs to 56k, ~20:00 if the stop rule fires at 28k.

## Progress at re-park (2026-10-06 16:33)
Exp 3: 28k branch recorded 16:27 (stop rule did not fire at 28k), trunk continuing 23.8k->47.6k at ~3.7 step/s,
so trunk ends ~18:20, 56k decay + record ~18:45. Exp 4 then starts from scratch (+2-4 h).

## Progress at park (2026-10-06 15:05)
Exp 2 finished 14:07 (exit 0) and is logged. Exp 3 fast32 a0.375: 7k branch recorded at 14:43, then trunk at ~10k/11.9k at 15:00 (4.7 step/s).
With the 1.3 stop rule, exp 3 should end ~16:30 if it stops at 28k, ~18:30 if it runs to 56k. Exp 4 (fast32h16 a0.375) follows from scratch, +2-4 h.

## Earlier progress (2026-10-06 11:15)
fast32 a0.50 trunk at ~45k/47.6k (3.6 step/s); then its 56k decay, then twofast32 trunk 23.8k->47.6k + decay,
then experiments 3 and 4 from scratch. Exp 2 ETA ~15:00; all ETA ~20:00+. When the chain is past exp 1 the
first `while kill -0 2589974` line is a no-op (pid gone), so a relaunch needs no edit.

NB: this file now lives in fast32h16-a0375-c (moved off fast32-a0375-c so it could be finished). finish_experiment.sh
refuses while the slug's worktree has HANDOFF.md: `git rm` it in fast32h16-a0375-c before finishing exp 4.
