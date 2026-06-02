# Autoresearch Loop

This document guides research agents through the experiment loop for improving buzzdetect models.

## Goal

Improve model performance on the precision/sensitivity tradeoff for buzz detection.
The primary metric is the sensitivity at 95% precision; this is the value reported by eval.py and this is what we're seeking to improve.
The current production standard is 28% sensitivity at 95% precision. Any improvement to this is valuable.

## Experiment lifecycle
### 0. Think
First, check out what previous agents have done in the log file.
Run eval.py with no model as an argument to see the best performing loops.
What did those loops do that made them so successful?
What hasn't been tried yet?
To understand a model's configuration (embedder, set, translation), read `models/<modelname>/config_model.json`.


### 1. Create a worktree

Each experiment lives on its own branch in an isolated worktree:

```bash
ROOT=$(git rev-parse --show-toplevel)
EXP=<short-slug>   # e.g. birdnet-embedder, deeper-head, augment-pitch
git worktree add .local/worktrees/$EXP -b exp/$EXP

# Replace git-checked-out embedder dirs with symlinks to main worktree
# (so binary weights and gitignored files are available)
for d in $ROOT/embedders/*/; do
    name=$(basename "$d")
    rm -rf ".local/worktrees/$EXP/embedders/$name"
    ln -s "$d" ".local/worktrees/$EXP/embedders/$name"
done
ln -sf "$ROOT/embedders/embedding.py" ".local/worktrees/$EXP/embedders/embedding.py"

# Symlink gitignored data dirs (embeddings, test audio) — read-only shared data
for setname in lite standard; do
    ln -s "$ROOT/02_set/sets/$setname/embeddings" \
          ".local/worktrees/$EXP/02_set/sets/$setname/embeddings"
done
ln -s "$ROOT/04_test/audio"      ".local/worktrees/$EXP/04_test/audio"
ln -s "$ROOT/04_test/embeddings" ".local/worktrees/$EXP/04_test/embeddings"
```

All code changes go in the worktree. Do not touch the main worktree's tracked files during an experiment.

**Embedder binaries**: the symlinks above point to the main worktree's embedder dirs (weights, `.pb` files). If you want to modify an existing embedder or download a new one, replace the symlink with a real copy first:

```bash
cp -rL embedders/yamnet embedders/yamnet_copy
rm embedders/yamnet
mv embedders/yamnet_copy embedders/yamnet
```

For a brand-new embedder, just create the dir directly — no symlink to remove.

### 2. Propose a hypothesis

Before touching code, create NOTES.md and begin it with the following:
- What you're changing and why you expect it to help
- What existing evidence (from prior experiments or the literature) supports it


### 3. Make changes

Scope is nearly unlimited: new embedders, rewritten training logic, new dependencies, new data preprocessing, etc.. Relevant entry points:

| What | Where |
|------|-------|
| Embedder interface | `embedders/embedding.py` |
| Add a new embedder | `embedders/<name>/embedder.py` (subclass `BaseEmbedder`) |
| Training logic | `03_train/train.py` |
| Data loading / folding | `03_train/dataset.py`, `03_train/train_utils.py` |
| Extraction pipeline | `02_set/extract.py` |
| Test/inference | `04_test/inference.py`, `04_test/metrics.py` |
| Shared config/paths | `config.py` |

If you download an external model or artifact (embedder weights, pretrained backbone, etc.), record the exact source URL and version in your experiment notes (see step 5). The artifact itself won't be in git; the code that fetches or uses it must be.

### 4. Run the pipeline

Run only the stages you need. All commands should be run from inside the worktree, with the worktree root as the working directory. Use `conda run -n buzzdetect-train python <script>` for each stage.

**Stage 2 — extract embeddings** (only needed if you changed the embedder or extraction logic):
```bash
conda run -n buzzdetect-train python 02_set/main.py \
  --set <setname> --embedder <embeddername>
```

**Stage 3 — train** (almost always needed):
```bash
conda run -n buzzdetect-train python 03_train/main.py \
  --model <modelname> --set <setname> --embedder <embeddername> \
  --translation general --epochs 300
```

**Stage 4 — test/eval** must be run manually after training:
```bash
conda run -n buzzdetect-train python -c "
import sys
sys.path.insert(0, '$(pwd)/04_test')
sys.path.insert(0, '$(pwd)')
import importlib.util
spec = importlib.util.spec_from_file_location('test_main', '04_test/main.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.test_model('<modelname>')
"
```
Note: `04_test/main.py` cannot be invoked as `python 04_test/main.py` because `main` collides with the root `main.py` on sys.path. Use the spec loader snippet above.

Use a descriptive `--model` name that won't collide with existing models (e.g. `exp_birdnet_v1`). Use `--set lite` for a fast smoke test before committing to a full run on `standard`.

Model output goes to `models/<modelname>/` in the worktree (gitignored; only code is tracked).

### 5. Record results

Write two artifacts when the run finishes:

**A. Experiment notes** — create `experiments/<slug>/notes.md` in the worktree and commit it:

```
# <slug>

## Hypothesis
<what you expected and why>

## Changes
<brief description of what you changed; key files and what each does>

## Reproduction
<exact steps to reproduce from scratch, including any external downloads>
- Embedder source: <URL, version, checksum if available>
- Additional packages: <any conda/pip installs>

## Results
Metric: precision at 80% sensitivity (or sensitivity at 90% precision)
- Baseline (<modelname>): <value>
- This experiment (<modelname>): <value>

<interpretation — did it work, partially work, fail? why do you think?>

## Conclusion
<what the next agent should know from this experiment>
```


Copy these notes to the model folder.

**B. Log entry** — append one JSON line to `experiments/log.jsonl` in the **main** worktree (not the experiment worktree):

```json
{"name": "<slug>", "branch": "exp/<slug>", "date": "<YYYY-MM-DD>", "hypothesis": "<one sentence>", "metrics": {"precision_at_80pct_sensitivity": 0.0, "sensitivity_at_90pct_precision": 0.0}, "baseline": {"model": "<name>", "precision_at_80pct_sensitivity": 0.0}, "conclusion": "<one sentence>", "reproduction": "<URL or 'no external artifacts'>", "merge_recommendation": "merge|cherry-pick|abandon"}
```

Create `experiments/` if it doesn't exist. Create `log.jsonl` if it doesn't exist. Commit the updated log to main after writing it.

### 6. Commit

Commit all code changes in the worktree:

```bash
git add -A
git commit -m "exp/<slug>: <what was tried and brief outcome>"
```

### 7. Any friction?
If you hit any frustrations that imply that the LOOP structure should be updated, log them in log.jsonl.
For example, timeouts because the 

Then stop. Do not merge into main. Do not delete the worktree or branch. The human will review the log and decide what to merge.

## Reading prior experiments

Before proposing a hypothesis, read `.local/experiments/log.jsonl` to see what has already been tried. Avoid re-running experiments with the same hypothesis unless you have a specific reason to believe the prior run was flawed.

## What counts as a useful negative result

A failed experiment is still useful if the notes explain *why* it failed — not just "metrics were worse" but what mechanism might have caused it. Record this. It saves future agents from retreading ground.

## Restoring a worktree

If a worktree was removed but the branch still exists:

```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

The branch preserves all committed code. External artifacts (embedder binaries, downloaded weights) must be re-fetched per the reproduction steps in `experiments/<slug>/notes.md`.

## Prohibited
You may not touch eval.py.
You may not touch 04_test/ or any of its contents.
You may not touch 01_annotate/ or any of its contents