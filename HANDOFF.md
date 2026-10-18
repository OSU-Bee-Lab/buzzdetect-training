# HANDOFF yam64h16lo (batch 36; experiment 4 of 4. Logged so far: wsd112-yam64h32, yam64h20-cliff, mmt-overlap. This is the last one.)
- Job: launch_job pid 997871, log .local/worktrees/yam64h16lo/distill.log (chain.sh: cache_fe C+V yam64h16lo, main.py rung C
  WSD 56000 for yam64h16lo:a0.25_d8 and a0.50_d8 (buzz+rain+human), reexport `--fe-impl mmtz --suffix _mmtz`, then a
  same-session timing, 2 rounds, into .local/worktrees/yam64h16lo/.local/bench/results_{1,2}.json).
  Resume: `tools/watch_job.sh --adopt 997871`, then the one `tools/watch_job.sh` Monitor.
- If still running: report progress (the `[chain] left:` ETA) and stop; park again if >1.5h left.
- When finished: compare the `_mmtz` rows fe_C_yam64h16lo_{a0.25_d8,a0.50_d8}_s1_c-buzz-rain-human_wsd<last>_mmtz against the
  yam64h16 twins' `_mmtz` rows (wsd56000_mmtz: a0.25_d8 0.711, a0.50_d8 0.727) for headline, and take SPEED from the
  same-session bench json (x YAMNet = rate / yamnet_large_general rate, mean of the 2 rounds; logged speeds drift 5-10%
  between sessions, see docs/distill-proposer-facts.md). Frontier neighbours timed in that bench: twofast32 a0.50_d8
  (0.701), yam64h32 a0.25_d8_l5x0.50 (0.683; two-seed mean 0.670), yam64h32 a0.50_d8 (0.687).
  Run `05_distill/ladder_record.py frontier`, `wsd --name <trunk>_wsd`, and via launch_job
  `tools/results.py <h16 twin _mmtz> <h16lo _mmtz>` for both widths. Fill notes.md Results/Conclusion (prediction
  a0.25_d8 0.70+-0.02 at ~7.5x, a0.50_d8 0.715+-0.02 at ~5.8x; falsifier a0.25_d8 < 0.683 or < 6.5x). Check the repeat
  rule (`ladder_record.py repeats`; a gain > 0.02 over everything at its speed or faster needs a seed-2 repeat: note it in
  notes.md / IDEAS.md for the next batch rather than running it here). Delete HANDOFF.md, commit, then from main:
  `tools/finish_experiment.sh yam64h16lo --arm distill --summary "..." --runs "<both _mmtz row names>"`, then
  `tools/loop_signal.sh done "wsd112-yam64h32 yam64h20-cliff mmt-overlap yam64h16lo"`.
- If died: from this worktree `/home/luke/projects/buzzdetect-training/tools/launch_job.sh distill.log -- bash chain.sh`
  (cache_fe, main.py and reexport resume; the bench reruns).
