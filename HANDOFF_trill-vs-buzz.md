# Handoff — trill-vs-buzz

## What this is

A merge of `exp/trunk-ft` (best structural config, `trunk_ft_1e5`, +0.046 vs
frozen) and `exp/prediction-provenance` (adds `path`/`frame_index`/
`labels_raw` to `predictions.csv`, never run through a real CV). Training
this reconfirms `trunk_ft_1e5` on the current annotation set *and* produces
the first model with provenance columns, unlocking `IDEAS.md#trill-vs-buzz`
step 1 (tabulate raw labels of the negatives above each fold's threshold).
See `notes.md` for the merge details (two small additive conflicts,
resolved) and the full hypothesis.

Launched 2026-09-07, PID at launch time 1570646 (process may have restarted
under a different PID if it crashed and was relaunched — check `pgrep`, not
this number).

## 1. One-command progress check

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/trill-vs-buzz
pgrep -af "03_train/main.py"
find "models/trill_vs_buzz_ft1e5/folds" -name summary.json 2>/dev/null | wc -l   # of 11
tail -5 train_trill_vs_buzz.log
```

**If it's still running (pgrep shows a process, fold count < 11), STOP.**
Report `fold N/11` and end the turn. Do not tail the log on a timer, do not
open a Monitor on it, do not read the rest of the repo while waiting — this
is a CPU CV on `medium`, budget ~40h. See CLAUDE.md's "Running long jobs" for
why waiting in-session burns tokens for nothing.

## 2. When it finishes (11/11 fold summaries + `models/trill_vs_buzz_ft1e5/folds_sx.csv` exists)

1. Compare paired against the two things that matter:
   ```bash
   python tools/compare_folds.py .local/worktrees/trunk-ft/models/trunk_ft_1e5 \
     .local/worktrees/trill-vs-buzz/models/trill_vs_buzz_ft1e5
   ```
   (run from the main checkout, or adjust paths — either bare model dir path
   works per `tools/compare_folds.py`'s own doc). This tells you whether the
   annotation set moved enough since `trunk_ft_1e5` was trained
   (2026-09-01ish) to change the structural conclusion. Expect close to
   0 delta, most folds flat — that confirmation is a secondary result, not
   the point of this run.

2. **Do the actual diagnostic** (the real point of this run):
   `predictions.csv` under each `models/trill_vs_buzz_ft1e5/folds/<fold>/`
   now has `path`, `frame_index`, `labels_raw` alongside
   `activation_ins_buzz`/`correct`. For each fold: take `_fold_sens`'s
   threshold at fpr 0.005 (in `folds_sx.csv`, the `threshold` column, row
   where `fpr==0.005`), filter `predictions.csv` to
   `correct == False & activation_ins_buzz >= threshold`, and tabulate
   `labels_raw` value counts. Do this per fold and pooled. `IDEAS.md#trill-
   vs-buzz` has the full hypothesis and what a trill-dominated result would
   imply (a pairwise buzz-vs-trill margin loss, smoke-tested with
   `tools/smoke_model.py` first — see `tail-loss` in `IDEAS.md` as the
   cautionary tale before writing a custom loss).

3. Write up both the CV reconfirmation and the trill-vs-buzz tabulation in
   `notes.md`'s `## Results` / `## Conclusion`. The CV number needs a
   `log.jsonl` entry (`tools/log_entry.py`, baseline `trunk_ft_1e5` if you
   want the comparison explicit, though the log's convention baselines
   against `cv-baseline`/`yamnet_medium_general` — use your judgement,
   note which in `conclusion`). The trill tabulation is a diagnostic, not a
   metric — write its finding into `IDEAS.md#trill-vs-buzz` (mark step 1
   done, state whether it narrows the research program) rather than
   `log.jsonl`.

## 3. If it died

Check `train_trill_vs_buzz.log`'s tail for a traceback vs. an OOM-style
silent kill (`dmesg`/no traceback — this box's GPU is disabled via
`CUDA_VISIBLE_DEVICES=""` for this run, so OOM is much less likely than on
`large`, but not impossible on CPU RAM for the 12288-d pool). Reruns resume:
`can_write()` skips any fold dir that already has `config_model.json`, so
relaunching the exact same command below is always safe and picks up where
it left off.

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/trill-vs-buzz
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  03_train/main.py --name trill_vs_buzz_ft1e5 --set medium --embedder yamnet_trunk \
  --translation general_v1 --batch 1024 --lr-backbone 1e-5 --lr-head 2e-4 \
  --min-delta 0.002 --patience 50 --verbose -y \
  >> train_trill_vs_buzz.log 2>&1 &
disown
```

## Things a fresh agent would get wrong here

- **`02_set/sets/medium/embeddings` is NOT the symlink `setup_worktree.sh`
  or a prior commit left in place.** It was replaced (real directory) with
  two hand-made symlinks:
  `embeddings/yamnet -> <main>/02_set/sets/medium/embeddings/yamnet` and
  `embeddings/yamnet_trunk -> <trunk-ft worktree>/02_set/sets/medium/embeddings/yamnet_trunk`.
  This reuses `exp/trunk-ft`'s already-extracted `yamnet_trunk` cache
  (2.8 GB) instead of re-extracting. **Do not delete or "fix" this back to
  a single symlink** — `main`'s own `medium/embeddings/` has no
  `yamnet_trunk` subdir at all, only `yamnet`.
- **This directory is deliberately NOT committed.** `git status` will show
  `D 02_set/sets/medium/embeddings` (the old tracked symlink, itself an
  accidental commit from an earlier experiment's `git add -A` — see
  `notes.md`). Do not `git add -A` at the end; stage code/notes/log files by
  name. Committing the deletion would break other worktrees that still rely
  on `02_set/sets/medium/embeddings` resolving as a symlink into main.
- The `02_set/sets/large/audio` and `02_set/sets/large/embeddings` dirs
  showing as untracked (`??`) in `git status` are pre-existing, unrelated to
  this experiment (from `large-trunk-ft`/main's own state) — leave them.
- `03_train/train.py::_score_fold` was reconciled by hand from two
  divergent versions (trunk-ft's batched `_predict_buzz`, for VRAM/RAM
  safety on 12288-d embeddings, vs prediction-provenance's provenance
  columns) — both are present now; don't revert one for the other if you
  see the git history and think it looks like a partial merge.
- Branch `exp/trill-vs-buzz` is based on current `main` (c57468f), then
  merges `exp/trunk-ft` then `exp/prediction-provenance` on top — three
  parents deep in history, which is unusual for this repo's normal
  one-branch-per-experiment convention. That's intentional here since the
  diagnostic needs both branches' code; don't try to rebase it flat.
