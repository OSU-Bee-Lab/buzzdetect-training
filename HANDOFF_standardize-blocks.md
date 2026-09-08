# HANDOFF — standardize-blocks

A 5-fold CV is running detached. It is the long side of a paired comparison;
the control is already finished and on disk.

## 1. Progress check (one command)

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/standardize-blocks
pgrep -af "name combined_standardized"
find models/combined_standardized/folds -name summary.json | wc -l   # of 5
tail -3 train_standardized.log
```

## 2. If it's still running, STOP.

Report `fold N/5` and quit. Do not tail the log on a timer, do not open a
Monitor, do not read the rest of the repo while waiting — everything read while
waiting is paid for twice. It was launched 2026-09-08 ~19:20 EDT with a
**3000-epoch cap**, which is expected to be slow: standardization pushed the
probe fold to 393 best-epoch where the control stops at 19-160, so budget on
the order of hours, not the ~13 min a normal frozen-probe CV takes.

## 3. When it finishes

Comparator is **`models/combined_control`** (0.230), *not* `cv_baseline`
(0.218) and not `combined_revalidate` (whose model dir is gone). The control is
the same embedder, same data, same code, flag off — that pairing is the whole
design.

```bash
python /home/luke/projects/buzzdetect-training/tools/compare_folds.py \
  models/combined_control models/combined_standardized
```

Then check the thing the run exists to answer, **before** reading the headline:

```bash
python - <<'PY'
import json, glob
for f in sorted(glob.glob('models/combined_standardized/folds/**/summary.json', recursive=True)):
    d = json.load(open(f))
    print(f.split('folds/')[1].rsplit('/summary',1)[0][:48], d['n_epochs'], 'best', d['best_epoch'])
PY
```

**If any fold shows `n_epochs` 3000, the cap bound and that fold did not
converge** — say so and treat the headline as a floor, not a measurement.

Folds too thin to trust: `1_150` (146 buzz frames — it moved 0.007 -> 0.062
between two *identical* runs) and `1_95` (433). `willard` at 305 is thin too.
The two rich folds (`1_29` 2144, `Fit+Fast` 1031) are where a real effect has
to show.

Then: notes.md Results/Conclusion -> `log.jsonl` in **main** (via
`tools/log_entry.py`, `--baseline-model` = the control) -> commit -> push
`exp/standardize-blocks` -> prune the worktree.

## 4. If it died

`tail -40 train_standardized.log`. A NaN blowup is the failure mode this design
guards against (`VAR_FLOOR`, 554 of 1545 dims floored on a real fold) and it
would raise `training loss went non-finite`. Reruns **resume**: any fold
directory already holding `config_model.json` is skipped, so relaunch the exact
same command and it picks up where it stopped.

```bash
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  03_train/main.py --name combined_standardized --set medium \
  --embedder yamnet_combined --translation general --standardize --epochs 3000 -y \
  > train_standardized.log 2>&1 &
disown
```

## Things a fresh agent would get wrong

- **`02_set/sets/*/embeddings` and `embedders/*` are symlinks into main.** Never
  delete anything through them. `yamnet_combined`'s embeddings (506 MB) and the
  rebuilt `yamnet_combined.keras` are shared, and re-extracting them is 7 min
  wasted.
- The interesting early signal is on **1_150**, the fold that has been pinned
  near zero all era: the one-fold probe put it at **0.185** standardized against
  0.014-0.034 for the control. That is also the noisiest fold in the set, so it
  needs the full run and the rich folds before anyone believes it.
- Don't train the shipped model. It is not part of `folds_sx.csv`.
