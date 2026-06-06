# Autoresearch Loop

## Goal
Improve sensitivity at 95% precision for `ins_buzz` detection — the value reported by `compare_metrics.py`.
Production standard: 28% (`model_general_v3`).

## Constraints
- Embedder: YAMNet only. Others have proven wasteful.
- Set: `medium` only.
- Augmentation: has hurt training so far — avoid without strong reason.

## Experiment lifecycle

### 0. Orient
```bash
cat log.jsonl
conda run -n buzzdetect-train python compare_metrics.py
```

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

All stages run from the worktree.

```bash
# Stage 2 — only if embedder or extraction changed
conda run -n buzzdetect-train python 02_set/main.py --set medium --embedder yamnet --workers <N>

# Stage 3 — train 5 independent runs (v1–v5); epochs arg is optional, defaults to 400
for i in 1 2 3 4 5; do
  conda run -n buzzdetect-train python 03_train/main.py \
    --model <modelname>_v$i --set medium --embedder yamnet --translation general
done

# Stage 4 — test all 5
for i in 1 2 3 4 5; do
  conda run -n buzzdetect-train python 04_test/main.py --model <modelname>_v$i
done
```

Report the **median** sens@95prec across the 5 runs as the canonical metric. Include the full set of values and the range in notes.md.

To run `compare_metrics.py` against baseline models, copy their metrics into the worktree first:
```bash
MAIN=$(git worktree list | awk 'NR==1{print $1}')
mkdir -p models/<baseline>/tests
cp "$MAIN/models/<baseline>/tests/metrics.csv" models/<baseline>/tests/
```

### 4. Record results

Create `notes.md` in the worktree root. Write the hypothesis section *before* touching code; fill in the rest after. Commit it.

```
# <slug>
## Hypothesis
## Changes
## Results
- Baseline (<model>): <sens@95prec>
- This experiment (<modelname> v1–v5): <val>, <val>, <val>, <val>, <val>  median=<median>  range=[<min>, <max>]
<interpretation>
## Conclusion
```

Append one line to `log.jsonl` in **main** and commit it. Be very brief; the log only serves as a summary to guide agents to dig deeper. Use the median as `sensitivity_at_95pct_precision`.
```json
{"name": "<slug>", "branch": "exp/<slug>", "date": "<YYYY-MM-DD>", "main_commit": "<git rev-parse --short HEAD>", "hypothesis": "...", "metrics": {"sensitivity_at_95pct_precision": 0.0, "n_runs": 5, "range": [0.0, 0.0]}, "baseline": {"model": "<name>", "sensitivity_at_95pct_precision": 0.0}, "conclusion": "..."}
```

### 5. Commit worktree
```bash
git add -A && git commit -m "exp/<slug>: <what was tried and outcome>"
```

Then stop. Do not merge into main. Do not delete the worktree or branch.

## Reading prior experiments

Check `log.jsonl` before proposing a hypothesis. To check if the codebase has changed meaningfully since a prior run:
```bash
git log --oneline <commit>..HEAD -- 03_train/ 02_set/sets/ translations/
```
Entries with `null` commit predate tracking — treat their metric values as directional only. Read `notes.md` on the relevant `exp/<slug>` branch for details on any entry.

## Restoring a worktree
```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

## Prohibited
Do not modify: `compare_metrics.py`, `summarize_metrics.py`, `04_test/`, `01_annotate/`
