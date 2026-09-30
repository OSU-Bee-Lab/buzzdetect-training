# HANDOFF — exp/ps-featdrop (batch 16, parked)

Batch 16 already logged **ps-gmp** (1 of 4). The other three, ps-featdrop, ps-bntrain and ps-nosmooth, run
sequentially from ONE detached chain: pid **2395427**, log `.local/worktrees/ps-gmp/chain16.log`
(driver `.local/worktrees/ps-gmp/run_chain.sh`, order: ps-gmp, ps-featdrop, ps-bntrain, ps-nosmooth; ~60 min per CV,
started 11:01, expected done ~15:00). Each worktree's `notes.md` already has Hypothesis/Changes; models land in
`.local/worktrees/<slug>/models/<slug>`. This handoff covers all three; the ones not yet in `log.jsonl` are still yours.

## 1. Resume
`tools/watch_job.sh --adopt 2395427`, then arm the ONE `tools/watch_job.sh` Monitor (30 min, re-arm at expiry).

## 2. If it's still running
Report progress (folds done per CV, from the Monitor's first lines) and **park again**:
`tools/loop_signal.sh park 120 "chain16 still running"`. Do not busy-wait, do not tail logs.

## 3. When a CV finishes
For each slug with a finished `models/<slug>/folds_sx.csv`, in its worktree:
`python tools/results.py ../../../models/v4-ft-ps models/<slug>`, then fill notes.md Results (headline against SD,
fold movers against their SD, tiers) and Conclusion, then from main
`tools/finish_experiment.sh <slug> --summary ... --model .local/worktrees/<slug>/models/<slug> --baseline-model v4-ft-ps --hypothesis ... --trust clean --conclusion ...`
(see `exp/ps-gmp`'s notes.md and the log entry as the template; ps-gmp's was headline +0.001 ± 0.011).
When all four are logged: `loop_signal.sh done "ps-gmp, ps-featdrop, ps-bntrain, ps-nosmooth"`.
IDEAS.md needs no deletion (these were agent-designed, not queue items).

## 4. If the chain died
`tools/launch_job.sh chain16.log -- /home/luke/projects/buzzdetect-training/.local/worktrees/ps-gmp/run_chain.sh`
(finished folds/CVs are skipped on rerun; run it from the ps-gmp worktree). Check the log's last lines for an
OOM (then `BUZZDETECT_NO_GPU` is NOT the fix; see CLAUDE.md gotchas). If the loop killed the job at park, this is the case.
