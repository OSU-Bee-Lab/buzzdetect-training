# HANDOFF: hidden-context-verify

Written 2026-09-12 ~03:47 EDT, at fold 1/8, because the first fold took **78
min** — over LOOP.md's ~50-min-per-fold handoff line, so no Monitor can keep
one agent's cache warm across the rest of this run. At 8 folds this projects
to **~10.4 h total**; that number is from one fold, not a promise.

Luke gave a standing instruction (2026-09-11 night) to run the LOOP.md
lifecycle unattended and keep chaining `IDEAS.md` queue items without
checking in, only surfacing data/policy calls, unresolved crashes, or an empty
queue. This handoff exists *within* that mandate — pick this back up and keep
going the same way, no need to ask Luke first.

## 1. One-command progress check

```
cd /home/luke/projects/buzzdetect-training/.local/worktrees/hidden-context-verify
pgrep -af "[0]3_train/main.py"
find models/hidden_context_verify/folds -name summary.json | wc -l
tail -3 train_hidden_context_verify.log
```

## 2. If it's still running, STOP

If `pgrep` shows the trainer alive and fold count is < 8: report `fold N/8`
and stop there. Arm a fresh per-fold Monitor (same pattern as the last one —
see `git log -p` on this branch or `exp/hidden-head-verify:notes.md`'s CLAUDE.md
recipe) rather than tailing the log by hand; each fold event is ~78 min apart
here so a fresh Monitor per wake is the right shape, not one 10h Monitor
(timeout caps at 3600000ms anyway). Do not read the rest of the repo while
waiting — that's the cost LOOP.md is warning about.

## 3. What to do when it finishes

`folds_sx.csv` lands at `models/hidden_context_verify/folds_sx.csv` once all 8
folds are done. Then:

```
cd /home/luke/projects/buzzdetect-training
python tools/compare_folds.py \
  .local/worktrees/context-verify/models/context_verify_v1 \
  .local/worktrees/hidden-context-verify/models/hidden_context_verify
```

**The comparator is `context_verify_v1` (yamnet_context, no hidden layer,
0.415 headline), NOT `cv_baseline_v3`.** This is testing whether `--hidden
1024` adds anything *on top of* the context concat — see IDEAS.md item 1c's
current text. Also read the tier breakdown (`sensitivity_background` /
`_untagged` / `_loud` columns in both `folds_sx.csv`s) the way
`hidden-head-verify`'s and `context-verify`'s log.jsonl entries did — say which
tiers moved, and whether the two named hard folds (`1_150`, `1_95`) moved.

Then the normal lifecycle: `notes.md` in this worktree (hypothesis section is
already implicit in IDEAS.md item 1c and `exp/hidden-head-verify`'s notes —
mirror that structure), `tools/log_entry.py --name hidden-context-verify
--model .local/worktrees/hidden-context-verify/models/hidden_context_verify
--baseline-model .local/worktrees/context-verify/models/context_verify_v1
--write` (run from repo root, main checkout, so `main_commit` is right),
`git add -A && git commit` in this worktree, `git push origin
exp/hidden-context-verify`, then commit `log.jsonl` + `IDEAS.md` on `main`.
**Delete this `HANDOFF.md` file as part of that commit** — it's a marker for
mid-run pickup, not a permanent artifact.

After this, if `--hidden` still hasn't shown a real gain (on either plain
YAMNet or `yamnet_context`), the remaining leg is `--hidden` on `yamnet_aves`
(the other confirmed concat) before writing off the whole idea — same recipe,
`--embedder yamnet_aves`, comparator `.local/worktrees/yamnet-aves-verify`'s
model (check the exact model name under its `models/` dir). If `yamnet_context`
*did* show a gain, that becomes the new best config and the next queue items
(1d dropout, item 2 Perch, item 9 per-block standardization, etc. — see
`IDEAS.md`) should probably be re-run against it as the new comparator rather
than against `cv_baseline_v3`, but that's a judgment call for whoever picks
this up, not a rule.

## 4. If it died

Check `train_hidden_context_verify.log` for a traceback. If the process is
gone (`pgrep` empty) with no traceback and fold count is 0 or partial, it may
have been killed by something external (OOM, machine restart) — this box is a
4GB GTX 1650 but the run is `CUDA_VISIBLE_DEVICES=""` (CPU-only) so GPU memory
isn't the risk; check `dmesg`/system logs if truly mysterious. Relaunch with
the identical command (same `--name`, so a partial run resumes rather than
restarting from scratch — `can_write()` skips folds already trained):

```
cd /home/luke/projects/buzzdetect-training/.local/worktrees/hidden-context-verify
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  03_train/main.py --name hidden_context_verify --set medium --embedder yamnet_context \
  --translation general --hidden 1024 --dropout 0 --fixed-epochs 400 --verbose -y \
  >> train_hidden_context_verify.log 2>&1 &
disown; echo "pid $!"
```

## Notes for whoever picks this up

- `--hidden` is **not on `main`** — it was ported from `exp/hidden-head-verify`
  onto this worktree by hand (`git diff main exp/hidden-head-verify --
  03_train/main.py 03_train/train.py` applied as a patch, commit `37a9bba` on
  this branch). Don't expect to find it on a fresh worktree from `main`.
- Symlinked caches (`02_set/sets/medium/embeddings/*`, `embedders/*`) are
  shared with `main` — do not delete or break them. `yamnet_context`'s
  embeddings already exist on disk from `context-verify`; this run needed no
  extraction.
- One benign warning in the log ("no embedding files under
  .../yamnet_context/raw/.../2026-07-27/1_99") — that ident/fold combo has no
  usable frames under this translation, same as it presumably showed for
  `context_verify_v1` too; not a bug in this run.
- Do not touch the `hidden_yamnet_verify` model/branch (`exp/hidden-head-verify`,
  already finished, written up, pushed) or the `context-verify` /
  `yamnet-aves-verify` branches — they're comparators, not to be retrained or
  modified.
