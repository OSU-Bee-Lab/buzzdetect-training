# Autoresearch Loop

## Goal
Improve sensitivity at 95% precision for `ins_buzz` detection — the value reported by `compare_metrics.py`.
Production standard: 28% (`model_general_v3`).

## Constraints
- Embedder: YAMNet only. Others have proven wasteful.
- Set: `medium` only.
- Augmentation: has hurt training so far — avoid without strong reason.
- Err against hyperparameter tuning, unless you have a strong reason for doing so. We're looking for structural gains; hyperparameters can be tuned in one large sweep after a good structure is identified.


## Experiment lifecycle

### 0. Orient and hypothesize
```bash
cat log.jsonl
conda run -n buzzdetect-train python compare_metrics.py
```

Propose a hypothesis for a change or set of changes that could improve model performance.

Check `log.jsonl` and IDEAS.md before proposing a hypothesis. If you use an IDEA, remove it after testing.

Entries in log.jsonl with `null` commit predate tracking — treat their metric values as directional only. Read `notes.md` on the relevant `exp/<slug>` branch for details on any entry.

### 1. Create worktree
```bash
bash setup_worktree.sh <slug>
```
Confirm the sanity check passes. Uncommitted changes to tracked files won't appear in the worktree — commit or stash first if they matter.

To modify the YAMNet embedder (it's a symlink), replace it with a real copy first:
```bash
cp -rL embedders/yamnet embedders/yamnet_copy && rm embedders/yamnet && mv embedders/yamnet_copy embedders/yamnet
```

All code changes go in the worktree. Do not touch main's tracked files.

### 2. Make changes

| What | Where |
|------|-------|
| YAMNet embedder | `embedders/yamnet/embedder.py` |
| Extraction | `02_set/extract.py` |
| Training | `03_train/train.py`, `03_train/dataset.py`, `03_train/train_utils.py` |
| Config/paths | `config.py` |

### 3. Run the pipeline

All stages run from the worktree. `cd` there first using the path printed by `setup_worktree.sh`:
```bash
WT=/Users/luke/Documents/bioacoustics/buzzdetect-training/.local/worktrees/<slug>
cd $WT
```

```bash
# Stage 2 — only if embedder or extraction changed
conda run -n buzzdetect-train python 02_set/main.py --set medium --embedder yamnet --workers <N>

# Stage 3 — train N runs (default 5); loads data once, trains v1–vN sequentially
conda run -n buzzdetect-train python 03_train/main.py \
  --name <modelname> --set medium --embedder yamnet --translation general

# Stage 4 — test N runs; embeds test audio once, runs all classifiers
conda run -n buzzdetect-train python 04_test/main.py --name <modelname>
```

After training and testing, report results (`evaluate_set.py` anchors to its own `models/`; must run from the worktree via `$WT`):
```bash
conda run -n buzzdetect-train python $WT/evaluate_set.py <modelname>
```

To compare against a baseline set, copy its per-run metrics into the worktree first, then run `compare_sets.py`. The baseline models are in their own experiment worktree, not in main:
```bash
BASELINE=$(git worktree list | grep "exp/<baseline-slug>" | awk '{print $1}')
for v in 1 2 3 4 5; do
  mkdir -p $WT/models/<baseline>_v$v/tests
  cp "$BASELINE/models/<baseline>_v$v/tests/metrics.csv" $WT/models/<baseline>_v$v/tests/
done
conda run -n buzzdetect-train python $WT/compare_sets.py
```

### 4. Record results

Create `notes.md` in the worktree root. Write the hypothesis section *before* touching code; fill in the rest after. Commit it.

```
# <slug>
## Hypothesis
## Changes
## Results
- Baseline (<set>): <val>, ..., <val>  mean=<mean>  median=<median>  95% CI=[<lo>, <hi>]
- This experiment (<modelname> v1–v5): <val>, ..., <val>  mean=<mean>  median=<median>  95% CI=[<lo>, <hi>]
<interpretation: do CIs overlap? by how much?>
## Conclusion
```

Append one line to `log.jsonl` in **main** and commit it. Be very brief; the log only serves as a summary to guide agents to dig deeper. Use the mean as `sensitivity_at_95pct_precision` and include `ci_95`.
```json
{"name": "<slug>", "branch": "exp/<slug>", "date": "<YYYY-MM-DD>", "main_commit": "<git rev-parse --short HEAD>", "hypothesis": "...", "metrics": {"sensitivity_at_95pct_precision": 0.0, "n_runs": 5, "range": [0.0, 0.0], "ci_95": [0.0, 0.0]}, "baseline": {"model": "<name>", "sensitivity_at_95pct_precision": 0.0, "ci_95": [0.0, 0.0]}, "conclusion": "..."}
```

### 5. Commit worktree
```bash
git add -A && git commit -m "exp/<slug>: <what was tried and outcome>"
```

Then stop. Do not merge into main. Do not delete the worktree or branch. Do not proceed to another experiment. You're done! Thank you!


## Restoring a worktree
```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

## Prohibited
Do not modify: `compare_metrics.py`, `summarize_metrics.py`, `04_test/`, `01_annotate/`
