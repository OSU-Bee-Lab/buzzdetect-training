# HANDOFF — exp/probe-grid

Eight one-lever CVs (`IDEAS.md` L1-L7) running **sequentially** from one
detached driver, `./run_grid.sh`, in this worktree.

## 1. Progress check — one command

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/probe-grid
tail -5 grid_driver.log            # which run is up, and the sx tail of each finished one
ls -d models/L*/                   # runs started
pgrep -af run_grid.sh              # driver alive?
```

`grid_driver.log` prints `== <time> <name> : <flags>` when a run starts and
`== <time> <name> done rc=` plus its `folds_sx.csv` total row when it ends. The
last line is `== GRID COMPLETE` when all eight are through.

## 2. If it's still running, STOP.

Report which run it's on and how many of the eight are finished, and quit. Do
not tail the training logs on a timer, do not Monitor it, do not read the rest
of the repo. Everything read while waiting is paid for twice.

Budget, measured not guessed: L1's fold 1 took **4.5 min** (319 epochs — it
does early-stop, it does not hit the 400 cap), so L1 ≈ 20-22 min. L3
(`--batch-size 4096`) is the other slow one. The rest should be ~9-10 min like
cv_baseline. **Whole grid ≈ 1 h 40 m from the 10:21 relaunch, i.e. done ~12:00.**

## 3. When it finishes

Comparator is **`models/cv_baseline`** (0.218) for every run — each is one lever
off that config and nothing else, which is the entire point of the grid.

```bash
cd /home/luke/projects/buzzdetect-training
for m in .local/worktrees/probe-grid/models/L*/; do
  echo "== $m"; python tools/compare_folds.py models/cv_baseline "$m"; done
```

Reading it:
- **Folds `Luke - Diel Drivers/2026-04-08/1_150` (146 buzz frames) and
  `.../2026-05-06/1_95` (433) are the thin ones** — baseline sens 0.021 and
  0.037. A big delta there is probably fold noise. The two folds worth
  believing are `JamesU - MustardBumbler/1_29` (2144) and
  `Lily - Fit+Fast/.../53` (1031).
- The bar is direction + count, not the mean: 4-5 folds moving the same way.
  Sub-0.02 mean shifts with folds split either side are inconclusive — say so
  and move on.
- **For L1 and L2 specifically, also check whether folds hit the 400-epoch
  cap** (`n_epochs` in `models/<name>/folds/*/summary.json`). If they do, L4
  (epoch cap) stops being a free no-op and needs its own run at `--epochs 3000`.

Then: fill in `notes.md`'s Results and Conclusion (the Hypothesis and Changes
sections are already written), append **one** `log.jsonl` line in **main** with
`tools/log_entry.py`, commit, push `exp/probe-grid`.

The grid is one experiment, so it is one log entry — the attribution table goes
in `notes.md`, and the log line says which lever paid and by how much. Use
`--baseline-model models/cv_baseline`. Remember `main_commit` is main's HEAD at
logging time, not the worktree's base.

### Do not wrap the driver in a waiter

The first launch attempt put `run_grid.sh` behind a detached
`while pgrep ...; do sleep; done` loop and **the waiter was killed within
minutes**, so the grid never started. That is the exact failure CLAUDE.md's
"Running long jobs" documents. Relaunch `run_grid.sh` directly.

## 4. If it died

`rc=` on the last `done` line in `grid_driver.log` says whether a run failed;
`grid_<name>.log` has the traceback. **`run_grid.sh` is resumable and safe to
relaunch** — it skips any run whose `models/<name>/folds_sx.csv` already exists:

```bash
nohup ./run_grid.sh > grid_driver.log 2>&1 & disown
```

A run that died *part-way* through its folds is the one hazard: 03_train
resumes silently into a half-populated model dir, but here that is fine, since
the flags are identical on relaunch. If you change a flag, `rm -rf
models/<name>` first.

## 5. Things a fresh agent would get wrong

- `embedders/*` and each set's `audio/`+`embeddings/` are **symlinks back to
  main**. Do not delete them, do not write through them. Nothing here
  re-extracts; all eight runs read main's cached `yamnet` embeddings.
- Models land in this worktree's own `models/`, not main's.
- `models/test_monitor_smoke/` is a one-fold throwaway that verified the
  `val_sens` monitor key populates. **Read `notes.md`'s "Seed noise" section
  before deleting it** — it and L1's fold 1 are the same config on the same
  fold and landed 0.018 apart, which is the only same-config repeat this era
  has. Don't log it as a result.
- The branch is based on main@57de23f. `standardize-blocks` (0.261) is *not*
  the comparator — it is a different lever (L9) that already has its own entry.
