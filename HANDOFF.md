# HANDOFF: nonuniform-width (batch 34, parked 2026-10-15 ~05:20)

**2026-10-16 16:30, machine idle (rebooted), chain RELAUNCHED: launch_job pid 18975, same distill.log. Adopt that pid, not 1955094. ~8.5 h remain; parked.**

**2026-10-16 12:58, resumed and re-parked:** Luke's analysis was still running (buzzdetect-engine pid 2193467,
heavy_v1 on "Lily - Clover", GPU at 3.4 GB, load ~4.5), so the chain was NOT relaunched. A placeholder job
(launch_job pid 2197197, `tail --pid=2193467`, log wait_engine.log) only waits for the engine to exit so the park
has a job to wake on. When you resume: re-run the three checks above; if all idle, relaunch the chain per
"If it died" below; if another analysis is running, launch a fresh waiter on its pid and park again.

**STOPPED BY LUKE, 2026-10-15 ~14:30 (not a crash): he needed the computer.** Both the agent_loop driver
(SIGTERM while parked, so no session was running) and the chain (SIGTERM to launch_job pgroup 1955094) were
killed on purpose. The chain was in the last decay of the first half: `twofast32:a0.25_d8_l7x1.0`, decay
47600-56000, at train step ~51500/56000 (the trunk to 47600 and the `_l5x0.50` student were done). Nothing
else was lost. The batch is still parked: `.local/agent_loop/park` is in place, so rerunning
`tools/human/agent_loop.sh` relaunches an agent for batch 34 (0 of 4 logged) which reads this file.
**Before relaunching anything, wait until the machine's resources are free.** Luke runs buzzdetect analyses
(`heavy_v1` engine) on this box, and the loop is queued to start only after one exits, but he may start
another. Check first: `pgrep -af "[b]uzzdetect-engine"` (a hit means an analysis is running), `nvidia-smi`
for GPU memory beyond the ~300 MiB desktop, and `uptime` for load. If anything is busy, don't launch the
chain (the 4 GB card fits one GPU job, and the analysis needs the CPU): park again with
`tools/loop_signal.sh park <minutes> "waiting for Luke's buzzdetect analysis"`, or wait on the engine's pid.
Relaunch only once all three are idle.
Treat the job as "died" in the branch below: no fix needed, just relaunch. main.py resumes at the training
checkpoint, so expect it to pick up near step 51500 (check the log's first lines to confirm), then the
mmt re-export, then the yam64h32 pair (`../nonuniform-width-yam64h32`, see its HANDOFF.md, unchanged).
Remaining chain time was ~0h20m for l7x1.0, then the re-export, then ~8 h for the yam64h32 pair.

Batch 34's experiment count: 0 of 4 logged when parked. This batch's two experiments are `nonuniform-width`
(twofast32) and `nonuniform-width-yam64h32`, chained in one job; the remaining two slots are unchosen
(candidates: a seed-2 repeat if `ladder_record.py repeats` flags one; IDEAS D2 if root-disk space is freed).

This worktree's part: the twofast32 pair (main.py --runs twofast32:a0.25_d8_l5x0.50, twofast32:a0.25_d8_l7x1.0) and their mmt re-export, the first half of the chain.

- Job: launch_job pid **1955094**, log `/home/luke/projects/buzzdetect-training/.local/worktrees/nonuniform-width/distill.log`
  (one job for both experiments). Resume: `tools/watch_job.sh --adopt 1955094`, then ONE Monitor on `tools/watch_job.sh`.
- If it's still running: report progress (the `[chain] left:` line) and stop, parking again if > 1.5 h remain.
- When this part finishes (its `_mmt` rows are in `05_distill/log.jsonl`):
  1. `python 05_distill/ladder_record.py frontier`, `ladder_record.py wsd --name <student>_wsd`, `ladder_record.py repeats`.
  2. `tools/results.py <control> <student>` (launch via launch_job: >2 min) for controls fe_C_twofast32_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt and fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt
     against fe_C_twofast32_a0.25_d8_l5x0.50_s1_c-buzz-rain-human_wsd<last>_mmt fe_C_twofast32_a0.25_d8_l7x1.0_s1_c-buzz-rain-human_wsd<last>_mmt (<last> = the last WSD budget recorded, 56000 unless the stop rule fired earlier).
  3. Fill notes.md Results/Conclusion (headline, speed, tiers; falsifier in Hypothesis), delete HANDOFF.md, commit,
     then from main: `tools/finish_experiment.sh nonuniform-width --arm distill --summary "..." --runs "<the _mmt student names>"`.
- If it died: read the log's tail, fix, and relaunch (main.py and reexport.py resume):
  `cd .local/worktrees/nonuniform-width && tools/launch_job.sh distill.log -- ./chain.sh   # resumes; runs the yam64h32 pair after`
