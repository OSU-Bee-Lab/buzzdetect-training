# HANDOFF — exp/shift-ratio-ladder

Not started. `notes.md`'s Hypothesis and Changes sections are written; the two
new embedders (`yamnet_pitchshift_decimate_x126`, `_x150`) are already
committed to main's shared tree (commit `3ad01e9`) and smoke tested — nothing
left to build. What's left is running two extractions and two CVs, one at a
time, all on **CPU** (GPU was busy with another batch's job when this was
scoped — check `nvidia-smi` before assuming that's still true; if the GPU is
now free, drop `--cpu` from the commands below and it'll be faster).

**Run everything sequentially, one job at a time** — CPU/GPU is fully
saturated by a single job, so concurrency buys nothing and only makes both
jobs slower and harder to read logs for.

## 1. Progress check — one command

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/shift-ratio-ladder
tail -5 extract_x126.log extract_x150.log train_x126.log train_x150.log 2>/dev/null
pgrep -af "[0]2_set/main.py --set medium --embedder yamnet_pitchshift_decimate_x1"
pgrep -af "[0]3_train/main.py --name (x126|x150)"
```

Remember: `pgrep -f`/`pkill -f` match Claude Code's own `bash -c` wrapper —
bracket the first letter as above, or you'll see a "running" job that isn't.

## 2. If a job is still running, STOP

Report which stage (extraction/training), which rung (x126/x150), and
progress from the log tail. Do not poll on a timer — if you started the job,
`tools/notify_job.sh <pid> --log <log>` was (or should be) run so this
session gets pinged on completion; wait for that.

## 3. If nothing has started yet, launch x126 first

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/shift-ratio-ladder
tools/launch_job.sh --cpu extract_x126.log -- 02_set/main.py --set medium --embedder yamnet_pitchshift_decimate_x126 --workers 4
```

Wait for its DONE ping, then:

```bash
tools/launch_job.sh --cpu train_x126.log -- 03_train/main.py --name x126 --set medium --embedder yamnet_pitchshift_decimate_x126 --translation general -y
```

Wait for its DONE ping, **then and only then** do the same pair for x150:

```bash
tools/launch_job.sh --cpu extract_x150.log -- 02_set/main.py --set medium --embedder yamnet_pitchshift_decimate_x150 --workers 4
# wait for DONE
tools/launch_job.sh --cpu train_x150.log -- 03_train/main.py --name x150 --set medium --embedder yamnet_pitchshift_decimate_x150 --translation general -y
```

`--fixed-epochs` defaults to 400. This is a 2048-d config (same width as
`yamnet_pitchshift`/`yamnet_pitchshift_decimate`), not the 4352-d lead, so
`epoch-budget-700`'s finding (400 too short at 4352-d) does **not**
automatically apply here — 400 is the matched-control default and is what
`pitchshift-decimate-up-repeat` (the comparator) used. Don't second-guess it
without a reason specific to this width.

Extraction on CPU: `--workers 4` (this is a plain-YAMNet-class embedder, no
GPU-worker cap needed — that cap is only for embedders that load a GPU model
per worker). Each rung does two YAMNet forward passes per frame, roughly the
same cost as `yamnet_pitchshift_decimate`'s own extraction (that one's `notes.md`
says ~35 min, but that was likely GPU — budget more on CPU; the `lite`-set
smoke test for x126 took ~5.5 min for 7 idents, `medium` is much larger).

## 4. When both rungs have trained

Comparator: `.local/worktrees/pitchshift-decimate-up-repeat/models/pitchshift-decimate-up-repeat`
(headline 0.393, baseline `yamnet-pitchshift` 0.398) — the seamless x2 rung,
**not** the tiled `yamnet_pitchshift`.

```bash
cd /home/luke/projects/buzzdetect-training
python tools/results.py .local/worktrees/pitchshift-decimate-up-repeat/models/pitchshift-decimate-up-repeat .local/worktrees/shift-ratio-ladder/models/x126
python tools/results.py .local/worktrees/pitchshift-decimate-up-repeat/models/pitchshift-decimate-up-repeat .local/worktrees/shift-ratio-ladder/models/x150
```

Read `docs/judging-results.md` before interpreting. Fill in `notes.md`'s
Results (both rungs, one table each or combined) and Conclusion per the
falsifier already stated there. This is **one experiment, one log entry**
covering both rungs (like `probe-grid`'s multi-run format) — use
`tools/log_entry.py`'s `--model`/`--baseline-model` for whichever rung is the
headline finding (or the better of the two if both are informative), and say
the other rung's result in the conclusion text.

Then, if you used the idea (you did): delete IDEAS.md's `## 24. Shift-ratio
ladder below one octave` section entirely (in main, not the worktree copy),
and:

```bash
cd /home/luke/projects/buzzdetect-training
tools/finish_experiment.sh shift-ratio-ladder --summary "<one line>" \
  --model .local/worktrees/shift-ratio-ladder/models/<better rung> \
  --baseline-model .local/worktrees/pitchshift-decimate-up-repeat/models/pitchshift-decimate-up-repeat \
  --baseline-name pitchshift-decimate-up-repeat \
  --hypothesis "..." --trust clean --conclusion "..." --commit-also IDEAS.md
```

Then `git push origin main` (this workflow doesn't auto-push main; check
`git status --short --branch` shows `ahead N` before assuming it's already
pushed).

## 5. If a job died

`[launch_job] exit N` at the end of the log — `N != 0` means it failed;
check the log for the traceback. Both extraction and training resume safely
on relaunch with the **same command** (extraction skips finished idents,
training skips finished folds) — just rerun the launch command from step 3
that corresponds to whichever stage/rung died.

## Things a fresh agent would get wrong

- `embedders/*` and each set's `audio/`+`embeddings/` here are symlinks back
  to main. Don't delete or write through them.
- Both new embedder dirs are already committed and pushed to main
  (`3ad01e9`) — don't recreate them.
- LOOP.md had an uncommitted edit on main from another session when this
  worktree was set up (a note about running experiments sequentially). Not
  this experiment's concern; don't touch it, don't worry if it's still there
  or already committed by someone else.
