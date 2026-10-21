# HANDOFF: ps2-seamfree-d8 (batch 41)

Batch 41 runs four experiments in ONE chain: binary-d8, plain-d8, sam-d8, ps2-seamfree-d8, in that order.
binary-d8, plain-d8 and sam-d8 are logged (3 of 4 done; re-park 10-21 ~19:15). Only ps2-seamfree-d8 remains: its CV started 18:59, ~53 s/epoch (~27 min/fold), so expect it done ~22:45. This worktree's part: 4th and last CV (smoke passed). Also compare against ps2-depth8 (0.480, the tiled two-octave view).

- **Job:** pid 1636291 (launch_job.sh), log `.local/worktrees/binary-d8/chain_b41.log`; script `exp/binary-d8:chain_b41.sh`. Per-run log: `.local/worktrees/ps2-seamfree-d8/train.log`.
  Resume watching: `tools/watch_job.sh --adopt 1636291`, then ONE Monitor on `tools/watch_job.sh`.
- **If it's still running:** report progress (folds done per slug from `grep "\[chain\]\|^\[.\/8\]" chain_b41.log`) and stop (park again if > 1.5 h remain).
- **When this slug's CV is done** (`models/ps2-seamfree-d8/folds_sx.csv` exists): check each fold's `buzz logit SD` in train.log, then LOOP.md step 4 against the matched control **ps-depth8-r3** (`$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 ps2-seamfree-d8`), read docs/judging-results.md, fill Results + Conclusion in notes.md, delete this HANDOFF.md (git rm), then step 5 `tools/finish_experiment.sh ps2-seamfree-d8 ... --model .local/worktrees/ps2-seamfree-d8/models/ps2-seamfree-d8 --baseline-model ps-depth8-r3`. No IDEAS.md item used, except ps2-seamfree-d8 = IDEAS #36 (delete its section in main's IDEAS.md, pass --commit-also IDEAS.md).
- **If it died:** read the end of chain_b41.log and the slug's train.log. Relaunch (resumes: extraction skips done idents, smokes skip via .smoke_ok, CVs skip finished folds):
  `cd /home/luke/projects/buzzdetect-training/.local/worktrees/binary-d8 && /home/luke/projects/buzzdetect-training/tools/launch_job.sh chain_b41.log -- bash chain_b41.sh`
