# HANDOFF — exp/perch-pitchshift-concat

**Halted for a normal batch end, not a technical blocker.** Luke: "We can
keep that pitchshift concat job running, but after that we should be done"
(2026-09-23) -- run it to completion, don't kill it, and this is the last
Perch experiment (LOOP.md's new Perch injunction, same commit range).

## What's running

Medium extraction of `perch_centred_pitchshift`, launched this batch
(2026-09-23 11:40:59): pid **1622567**, log
`.local/worktrees/perch-pitchshift-concat/extract.log`, `--workers 2`,
`.local/venv-perch-extract/bin/python`, CPU. Watch with
`tools/watch_job.sh --adopt 1622567` in a Monitor, re-armed every 30 min.
As of 2026-09-23 18:51 it was on ident 7/82 (~7.2 h elapsed, 6 idents
written) -- `perch-centred`'s own medium extraction (same Perch cost, this
embedder just adds a cheap YAMNet pass on top) took **~22.6 h wall-clock**
end to end, so expect roughly that order of magnitude total; this is a long
job, likely spanning multiple more batches.

## What to do

1. If it's still running when you pick this up, report progress and stop --
   don't launch anything else against this worktree.
2. If it died, read the log's tail. Re-running the same command resumes:
   idents are fingerprinted per-ident, and a half-written ident rebuilds.
   ```bash
   /home/luke/projects/buzzdetect-training/tools/launch_job.sh --cpu extract.log -- \
     /home/luke/projects/buzzdetect-training/.local/venv-perch-extract/bin/python -u \
     02_set/main.py --set medium --embedder perch_centred_pitchshift --workers 2 --verbose
   ```
3. When it finishes, train (frozen linear probe, fast -- not a trunk
   fine-tune, so single-digit minutes on GPU):
   ```bash
   /home/luke/projects/buzzdetect-training/tools/launch_job.sh cv.log -- \
     /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u 03_train/main.py \
     --name perch-pitchshift-concat --set medium --embedder perch_centred_pitchshift \
     --translation general -y
   ```
4. Pull results against both comparators named in notes.md's Hypothesis:
   ```bash
   python tools/results.py <perch-centred worktree>/models/perch-centred perch-pitchshift-concat
   python tools/results.py <yamnet-pitchshift comparator, if one is findable> perch-pitchshift-concat
   ```
   `1_95` and `1_114` are the falsifier (must hold at or above `perch-centred`'s
   own deltas for this to be a real stack, not a wash) -- see notes.md for the
   full falsifier and comparator list. Then LOOP.md steps 4-5 (write up
   Results/Conclusion in notes.md, `tools/finish_experiment.sh`).
5. **This is the last Perch experiment** (LOOP.md's Perch injunction,
   committed 2026-09-23, `333799b`): once this one is logged, do not start
   another Perch extraction. The injunction and its rationale (Perch's CPU
   rate measured at ~0.005x YAMNet's) are in LOOP.md's "Special instructions"
   and `perch-injunction` project memory.
