# HANDOFF — decimate-lead (IDEAS item 19)

Session wrapped up (second time) on Luke's Ctrl+C to `tools/agent_loop.sh`
while the extraction was still running. Do not wait for it in the same turn
you read this — check once, act accordingly.

## Progress check (one command)

```bash
ps -p 2417327 -o pid,stat,etime,cmd
tail -5 /home/luke/projects/buzzdetect-training/.local/worktrees/decimate-lead/extract.log
```

**If it's still running:** report progress (which ident, how many of 73 done
this relaunch — `grep -c "ident took" extract.log`) and stop. Don't relaunch,
don't wait inline; re-arm a Monitor if you want to keep watching:

```bash
HEARTBEAT=1200 bash tools/watch_job.sh 2417327 --log '.local/worktrees/decimate-lead/extract.log'
```
(pid may differ if it was relaunched — always check `ps` first.)

**State when this was written (2026-09-14, ~21:30 local):** pid 2417327,
~21m elapsed, on the 4th ident this relaunch (`Chia - Bee Audio 2022
Original/7-13-22_SouthCharleston/1`), 3/73 idents fully extracted this
relaunch. This is a **relaunch**, not the original run: the first attempt
(pid 2384834) had reached 7/82 idents (through `1_37`) when it was found dead
with no traceback and no `[launch_job] exit` line on the next session's
check-in — almost certainly killed by a prior loop-cleanup on session end,
not a real crash. It was relaunched per this file's "If it died" section; the
9 already-cached idents (including the 7 from the first run) were skipped on
resume, leaving 73 to go. Per-ident pace so far this relaunch: 12.2min, 8.0min,
0.75min (varies a lot by ident size) — the Monitor's own linear ETA is not
trustworthy; read `extract.log` directly if you need a real sense of
remaining time.

If you find it dead again with no traceback, don't assume a real crash before
relaunching — check `driver.log`/session history for a Ctrl+C around the time
it stopped first.

## When it finishes

1. Confirm `[launch_job] exit 0` at the end of `extract.log`.
2. Launch the CV from the worktree root:
   ```bash
   tools/launch_job.sh train.log -- 03_train/main.py --name decimate-lead \
     --set medium --embedder yamnet_pitchshift_decimate_aves_mid \
     --translation general -y
   ```
   Arm `watch_job.sh`'s printed command (with `HEARTBEAT=1200` prefix) as a
   persistent Monitor.
3. Read results against **both** matched controls, per notes.md's Hypothesis:
   ```bash
   python tools/results.py yamnet_pitchshift_aves_mid decimate-lead   # the tiled-shift lead (main comparator)
   python tools/results.py yamnet_pitchshift_decimate decimate-lead   # optional: vs the plain decimated shift
   ```
4. Fill in notes.md's Results/Conclusion sections. **Read
   `diagnostics/2026-09-14_trill_1_114/README.md` first** — it predicts this
   run's `1_114` behavior (AVES blocks confuse trill for buzz; the decimated
   shift alone is only a marginal improvement over the tiled shift on that
   axis, so don't be surprised if `1_114` stays down here too). Falsifier from
   IDEAS item 19: `1_114` AND `1_95` must both rise above
   `yamnet_pitchshift_aves_mid`, headline no worse than -0.027.
5. `tools/finish_experiment.sh decimate-lead --summary "..." \
     --model .local/worktrees/decimate-lead/models/decimate-lead \
     --baseline-model yamnet_pitchshift_aves_mid --hypothesis "..." \
     --trust clean --conclusion "..." --commit-also IDEAS.md` (only pass
   `--commit-also IDEAS.md` if `git diff IDEAS.md` shows nothing but item 19's
   deletion).
6. Delete this HANDOFF.md in the same commit that records the result (or a
   follow-up commit) — it's a resumption aid, not part of the permanent record.

## If it died

Check the tail of `extract.log` for a traceback. Relaunch with:
```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium \
  --embedder yamnet_pitchshift_decimate_aves_mid --workers 1
```
Reruns resume incrementally (idents already extracted are skipped) unless the
embedder code changed since — it hasn't; `embedders/yamnet_pitchshift_decimate_aves_mid/`
was smoke-tested and committed to main before this run started (unit-checked
byte-identical against `yamnet_pitchshift_aves_mid` on the unshifted/AVES
blocks, see notes.md's Changes section).

## What's already committed and pushed to main (not just this branch)

- `embedders/yamnet_pitchshift_decimate_aves_mid/` (new embedder, IDEAS item 19)
- `embedders/yamnet_aves_mid_avesshift/` (new embedder, IDEAS item 22 — built
  and smoke-tested during this extraction's wait, NOT yet extracted or run;
  next experiment after this one, same falsifier logic as item 22's IDEAS
  entry — extraction not yet launched)
- `diagnostics/2026-09-14_trill_1_114/` (free diagnostic for item 18, already
  complete with README — no further action needed on it)

This branch (`exp/decimate-lead`) holds only this experiment's `notes.md` and
this `HANDOFF.md`.
