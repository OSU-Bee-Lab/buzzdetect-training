# HANDOFF — batch 18, IDEAS.md item 27 (stage-5 distill frontier grid)

Batch 18 so far: 0 of 4 experiments logged. Item 27 is a single `05_distill/main.py` batch (not a CV, no log.jsonl line), ~10.5 h from 2026-09-30 15:30. First cell finished 15:53.

- Job: launch_job pid **2953820**, log `05_distill/data/main_stage5_grid.log`. Run from main checkout.
- Resume: `tools/watch_job.sh --adopt 2953820`, then one `tools/watch_job.sh` Monitor (30 min). If still running: report progress and stop (park again per ETA).
- When it finishes (exit 0): `ladder_record.py frontier` and `frontier_svg.py` in 05_distill; report filled grid + per-cell times (wall.txt) to Luke in notes / signal; delete IDEAS.md section 27 (commit). Then continue with remaining queue (only low-priority items 26/25 remain; nothing else) and signal done.
- If it died: rerun the same command (resumable; cells with presence markers are skipped):
  `tools/launch_job.sh 05_distill/data/main_stage5_grid.log -- /home/luke/anaconda3/envs/buzzdetect-train/bin/python 05_distill/main.py --teacher v4-ft-ps-e60-moderate --rung B --runs "<the --runs string in IDEAS.md item 27>"`
  Check OOM location first (CLAUDE.md); one GPU job at a time.
