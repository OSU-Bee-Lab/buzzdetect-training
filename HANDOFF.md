# HANDOFF: ps-depth4 (batch 24, experiment 3 of 4)

Batch 24 status: 1 of 4 logged (ps-depth8-repeat). ps-depth6 (experiment 2) runs first; see ../ps-depth6/HANDOFF.md, which also says how to pick experiment 4.

- Job: launch_job pid **3211300**, log `.local/worktrees/ps-depth4/train.log`. It runs `bash run_depth4.sh 3210686`, which sleeps until ps-depth6's pid 3210686 exits and then trains depth4 (`TRUNK_STREAM=1`, batch 256 x accum 4, `SCORE_CHUNK=128`). Adopt with `tools/watch_job.sh --adopt 3211300`; the same single Monitor covers both jobs.
- **If it's still running, report progress and stop.** It has no measured fold time yet. Expect >= 1.5x depth6's ~24 min/fold (the cache is 2x larger), about 5 h after depth6 ends.
- **When it finishes (exit 0):** `python tools/results.py v4-ft-ps ps-depth4`, and also against ps-depth6 and ps-depth8-repeat. Fill notes.md, delete this file, commit, and run `tools/finish_experiment.sh ps-depth4 --model .local/worktrees/ps-depth4/models/ps-depth4 --baseline-model v4-ft-ps --trust clean ...`.
- **If it died:** read train.log. A GPU OOM means halving `TRUNK_BATCH` and doubling `TRUNK_ACCUM` in run_depth4.sh (keep the product 1024), or halving `SCORE_CHUNK` if it died in scoring. Host-RAM trouble (the per-fold RSS line climbing) is a different problem. Relaunch from this worktree, with no pid argument once depth6 is done: `tools/launch_job.sh train.log -- bash run_depth4.sh`. Finished folds are skipped.
