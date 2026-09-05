# HANDOFF — exp/unfreeze-one

A full 11-fold CV was launched **2026-09-05 00:00** in this worktree
(`/home/luke/projects/buzzdetect-training/.local/worktrees/unfreeze-one`).
Expect **~40 h on CPU**. Read `notes.md` for the hypothesis.

## 1. One-command progress check

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/unfreeze-one
pgrep -af 03_train/main.py
find models/unfreeze_one/folds -name summary.json | wc -l   # of 11
tail -3 train_unfreeze_one.log
```

(Launch PID was 1371611, but trust `pgrep` — PIDs get reused.)

## 2. If it's still running, STOP

Report `fold N/11 done` and end your turn. Do not tail the log on a timer, do
not open a Monitor, do not read the rest of the repo while waiting — everything
read while waiting is paid for twice.

## 3. When it finishes (`models/unfreeze_one/folds_sx.csv` exists)

The comparator is **`trunk_ft_1e5`, not `cv-baseline`** — same trunk pipeline,
same LRs, same batch, differing only in that layer 13 is frozen here:

```bash
cd /home/luke/projects/buzzdetect-training
python tools/compare_folds.py \
  .local/worktrees/trunk-ft/models/trunk_ft_1e5 \
  .local/worktrees/unfreeze-one/models/unfreeze_one
```

Secondary reads, for placing it on the depth curve (frozen 0.216 → 13-14 0.262
→ 12-14 0.229): `.local/worktrees/trunk-ft/models/trunk_frozen` and
`.local/worktrees/unfreeze-more/models/`.

Reading it:
- The noise floor is **~0.017 median per-fold, ~0.014 headline** (`noise-floor-cv`).
  A split-fold result under that is inconclusive — say so and move on.
- Check `buzz_frames` / `neg_frames` in `folds_sx.csv` before believing any
  per-fold delta. Thin folds (`Various Opportunistic/48`, `Diel/2026-05-06`,
  `1_23`) swing ±0.25 on their own. `trunk-ft-restore-sens` got burned by
  exactly this — one fold with 6 negative frames carried its whole headline.
- Also compare median `best_epoch` (in each fold's `summary.json`): 13-14 ran
  35, 12-14 ran 25. If 14-only runs *longer*, that supports the
  capacity/overfit reading of the turnover.

Then: fill in `notes.md` Results + Conclusion → `tools/log_entry.py --write`
(run `git rev-parse --short HEAD` in **main**, not here) → commit both.
Update `IDEAS.md`'s `lora-adapter` entry, which currently records 13-14 as the
settled depth. **Do not merge into main; do not delete this worktree.**

## 4. If it died

```bash
grep -aE "Traceback|Error|Killed|MemoryError|Exception" train_unfreeze_one.log | tail -20
```

Reruns resume: any fold with a `config_model.json` is skipped, so a relaunch
picks up where it stopped — the same command is safe to re-issue as-is:

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/unfreeze-one
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  BUZZDETECT_TRUNK_FT_BLOCKS=14 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  03_train/main.py --name unfreeze_one --set medium --embedder yamnet_trunk \
  --translation general --lr-backbone 1e-5 --lr-head 2e-4 --batch 1024 --verbose -y \
  >> train_unfreeze_one.log 2>&1 &
disown; echo "pid $!"
```

**`BUZZDETECT_TRUNK_FT_BLOCKS=14` is the whole experiment.** Relaunching
without it silently retrains 13-14 and mixes two configs into one model dir —
if that ever happens, delete `models/unfreeze_one/` and start over.

## Things a fresh agent would get wrong

- **This branch is based on `exp/trunk-ft` @ 38ee546, not on main.** main has no
  trunk embedder and no `--lr-backbone` plumbing.
- **`02_set/sets/medium/embeddings/yamnet_trunk` is a symlink into
  `.local/worktrees/trunk-ft/`** (2.8 GB, shared). Never delete or re-extract
  through it — you would destroy the comparator's cache. Stage 3 only reads it.
  `embedders/*` are symlinks into main for the same reason; `yamnet_trunk` is
  the one real directory, because the experiment edits it.
- The `EXCLUDE`/dropped-embedding warnings at startup are normal and appear in
  every trunk run.
