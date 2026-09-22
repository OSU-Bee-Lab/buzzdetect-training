# HANDOFF — exp/perch-centred

**Halted for a loop restart Luke requested (2026-09-22), not a technical
blocker.** Nothing is broken. The extraction below IS running — a `halt`
signal exits the driver before it kills any job, so this should still be
alive when you pick this up.

## What's running

Medium extraction, launched this batch (2026-09-22 12:03): pid **1092642**,
log `.local/worktrees/perch-centred/extract_medium.log`, `--workers 2`,
`.local/venv-perch-extract/bin/python`, CPU. Watch with
`tools/watch_job.sh 1092642 .../extract_medium.log` in a Monitor, re-armed
every 30 min. As of 2026-09-22 17:28 it was on ident 3/82 (~5.4 h elapsed),
each ident taking roughly 1-2.5 h depending on snip count — this is a long
job, expect several more hours. Memory has fluctuated (available dropped as
low as ~2 GB / swap as high as 7 GB at one point when a concurrent GPU
training job was also running) but never crashed; with the GPU job now also
possibly still running (see `exp/trunk-depth-headtohead`'s HANDOFF.md),
check `free -h` on your first re-arm and don't launch a third heavy job
alongside both if memory looks tight.

## What to do

1. If it's still running when you pick this up, report progress and stop —
   don't launch anything else against this worktree.
2. If it died, read the log's tail. Re-running the same command resumes:
   idents are fingerprinted per-ident, and a half-written ident rebuilds.
   ```bash
   /home/luke/projects/buzzdetect-training/tools/launch_job.sh --cpu extract_medium.log -- \
     /home/luke/projects/buzzdetect-training/.local/venv-perch-extract/bin/python -u \
     02_set/main.py --set medium --embedder perch_centred --workers 2 --verbose
   ```
3. When it finishes, train with the pinned env:
   ```bash
   /home/luke/projects/buzzdetect-training/tools/launch_job.sh cv.log -- \
     /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u 03_train/main.py \
     --name perch-centred --set medium --embedder perch_centred --translation general -y
   ```
   Compare against the `cv-baseline-v3` family anchor (see notes.md
   Comparator). `1_95` is the pre-registered falsifier. Then do LOOP.md
   steps 4-5.
