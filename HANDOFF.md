# HANDOFF — exp/perch-centred

**Unblocked (batch-11 fixer). Nothing is running**: no pid to watch. Read
the `## Fix` section at the end of `notes.md`. The memory blowup was the
whole annotation chunk going into Perch as one batch. `embed()` now uses
fixed 16-window batches (main, `embedders/perch_centred/embedder.py`), and
`lite` held a flat ~2.7 GB RSS. The experiment has not been started:
no medium embeddings exist yet.

## What to do

1. Launch the medium extraction as an ordinary long job from this worktree
   (CPU, the TF 2.21 venv; `BUZZDETECT_CHUNK_FRAMES` doesn't reach the context
   path, so it's not set). Don't use `run_perch_centred.sh`, which is obsolete:

   ```bash
   /home/luke/projects/buzzdetect-training/tools/launch_job.sh --cpu extract_medium.log -- \
     /home/luke/projects/buzzdetect-training/.local/venv-perch-extract/bin/python -u \
     02_set/main.py --set medium --embedder perch_centred --workers 2 --verbose
   ```

   Watch it with a Monitor on `tools/watch_job.sh <pid> <log>` and re-arm it
   every 30 min. It will likely take many hours: lite ran ~1.2-2 min per 360 s
   snip on one worker. Check `free -h` on the first re-arms; each worker
   should sit near 3 GB.
2. If it's still running when you pick this up, report progress and stop.
3. If it died, read the log. Re-running the same command resumes: idents
   are fingerprinted, and a half-written ident rebuilds.
4. When it finishes, train with the pinned env:

   ```bash
   /home/luke/projects/buzzdetect-training/tools/launch_job.sh cv.log -- \
     /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u 03_train/main.py \
     --name perch-centred --set medium --embedder perch_centred --translation general -y
   ```

   Compare against the `cv-baseline-v3` family anchor (see notes.md
   Comparator). `1_95` is the pre-registered falsifier. Then do LOOP.md
   steps 4-5.
