# Autoresearch Loop

> **Methodology changed in 2026-08.** Every entry in `log.jsonl` predates the
> leave-one-fold-out rework: it was measured by training 5 models on a fixed
> train/validate split and scoring them against the hand-curated `04_test`
> corpus. That corpus, that split, and the `<name>_v1…_vN` layout are all gone.
> **Old numbers are not comparable to new ones** — read them for direction
> (which ideas are dead ends) and never as a baseline to beat.
>
> The first experiment after the rework must re-establish a baseline. See
> [Baseline](#baseline).

Read `README.md` first — [Reading the results](README.md#reading-the-results)
is where the metric choices below come from.

## Goal

Improve `ins_buzz` sensitivity at a fixed false-positive rate on **held-out
deployments**.

Primary number: `models/<name>/folds_pooled_sx.csv`, the `sensitivity` column at
`fpr` 0.001 / 0.005 / 0.01 — every rotating fold's held-out predictions pooled
into one ROC, threshold picked once. A training run prints it.

Secondary: the unweighted mean across folds, also printed, from
`folds_summary.csv`. If the two diverge, one high-volume deployment is carrying
the result and that belongs in the conclusion.

Sensitivity at fixed *precision* — the old goal — is still recoverable from
`folds_pooled_metrics.csv` via `04_test/metrics.py::metrics_at_precision`, but
precision mixes in each deployment's base rate, so prefer FPR when comparing.
The old log's "28% production standard" was measured on a different corpus with
a different base rate; it is not a target on this metric.

## Baseline

Main's default config is a linear probe on frozen YAMNet with `Dropout(0.2)` and
`BinaryCrossentropy(label_smoothing=0.2)`, Adam at 0.002 — the old `with-dropout`
config, which the log established as the best of the pre-CV era.

There is no CV-era baseline yet. Establish one before testing anything against
it: train main's default config unchanged, on the set you intend to use, and log
it as `cv-baseline`. Every later experiment compares against that.

## Constraints

- Embedder: YAMNet only. Others have proven wasteful.
- Set: `lite`. (`medium` is a work in progress and its `folds.csv` predates
  roles, so every fold there is silently treated as `rotate`.)
- Augmentation: has hurt training so far — avoid without strong reason.
- Err against hyperparameter tuning, unless you have a strong reason. We're
  looking for structural gains; hyperparameters can be tuned in one large sweep
  once a good structure is identified.
- Change one thing. Layering a change on top of another risks interaction
  effects that obscure whether the change itself helped.

## Experiment lifecycle

### 0. Orient and hypothesize

```bash
cat log.jsonl
cat IDEAS.md
```

Propose a hypothesis. Check both files first — most obvious ideas have been
tried, and the pre-CV conclusions about *dead ends* (bandpass, mel masking,
backbone fine-tuning, MLP heads, L2, handcrafted frequency features) still
stand; only the numbers are incomparable.

Entries with a `null` commit predate commit tracking — directional only. Read
`notes.md` on the relevant `exp/<slug>` branch for details on any entry.

If you use an IDEA, remove it from `IDEAS.md` after testing.

### 1. Create worktree

```bash
bash setup_worktree.sh <slug>
```

Confirm the sanity check passes. Uncommitted changes to tracked files won't
appear in the worktree — commit or stash first if they matter.

The script symlinks `embedders/*` and each set's `audio/`+`embeddings/` back to
main, so extraction is shared. To modify an embedder you must first break its
symlink, or you'll be editing main's copy:

```bash
cp -rL embedders/yamnet embedders/yamnet_real && rm embedders/yamnet && mv embedders/yamnet_real embedders/yamnet
```

Do the same for a set's `embeddings/` before re-extracting under a changed
embedder or changed extraction code — otherwise you overwrite main's cache.

All code changes go in the worktree. Do not touch main's tracked files.

### 2. Make changes

| What | Where |
|------|-------|
| YAMNet embedder | `embedders/yamnet/embedder.py` |
| Extraction | `02_set/extract.py` |
| Training | `03_train/train.py`, `03_train/dataset.py`, `03_train/train_utils.py` |
| Config/paths | `config.py` |

### 3. Run the pipeline

From the worktree, using the path `setup_worktree.sh` printed:

```bash
WT=/Users/luke/Documents/bioacoustics/buzzdetect-training/.local/worktrees/<slug>
cd $WT

# Stage 2 — only if the embedder or extraction changed
conda run -n buzzdetect-train python 02_set/main.py --set lite --embedder yamnet --workers 2

# Stage 3 — the whole CV: one model per rotating fold, then the shipped model
conda run -n buzzdetect-train python 03_train/main.py \
  --name <modelname> --set lite --embedder yamnet --translation general -y
```

There is no `--runs` and no stage 4. One training call *is* the experiment: it
trains one model per rotating fold and the shipped model, and prints the pooled
and unweighted numbers when it finishes.

**Reruns resume, silently.** Any fold directory already holding a
`config_model.json` is skipped, so re-running after a code change reuses the old
models and reports a mix of both. Always use a fresh `--name`, or pass
`--clear` via the root `main.py`, or delete `models/<name>/` by hand.

### 4. Read the results

```
models/<modelname>/
├── folds_pooled_sx.csv     ← primary: sensitivity at each target FPR
├── folds_summary.csv       ← per fold: epochs, val loss, frame counts, sens at each FPR
└── folds/<fold>/sx.csv     ← the same read, one fold at a time
```

`summarize_metrics.py`, `compare_metrics.py`, `evaluate_set.py`,
`compare_sets.py` and `04_test/` all read `models/<model>/tests/metrics.csv`,
which a CV run does not produce. **They do not work here.** Read the CSVs above
directly.

**Compare paired, per fold.** Both configs ran on the same folds, so the useful
comparison is the per-fold difference — join the two `folds_summary.csv` files
on `fold` and look at the deltas and how many folds moved which way. Comparing
two pooled scalars throws that pairing away.

Two cautions from the README, both of which apply to every conclusion you write:

- **Fold-to-fold spread is not a confidence interval.** Training pools overlap
  ~90% across rotations, so fold models are correlated and the spread
  understates uncertainty about a genuinely new deployment. Don't compute a CI
  across folds and treat it the way the old log treated its across-runs CI.
- **Per-fold sensitivity is unreliable where there's little buzz.** In the
  quietest deployments it's ±0.25 or worse. A fold that swings hard may just be
  a fold with forty seconds of buzz in it. Check `frames_val` in
  `folds_summary.csv` before believing a per-fold delta.

There is no seed control anywhere in the pipeline, so run-to-run variation from
TF's nondeterministic init and shuffling is unmeasured. A true noise floor needs
the whole CV repeated under a fresh `--name`, which is expensive. If a result
lands close to baseline, say so and say it's within unquantified run noise
rather than reaching for a significance claim.

### 5. Record results

Create `notes.md` in the worktree root. Write the hypothesis section *before*
touching code; fill in the rest after. Commit it.

```
# <slug>
## Hypothesis
## Changes
## Results
| fold | baseline sens@fpr0.005 | this exp | delta | val frames |
|---|---|---|---|---|
- Pooled (frame-weighted): baseline <val> → this <val>
- Unweighted mean across folds: baseline <val> → this <val>
<interpretation: how many folds moved which way? do the pooled and unweighted
numbers agree? are the movers folds with enough buzz to trust?>
## Conclusion
```

Append one line to `log.jsonl` in **main** and commit it. Be very brief; the log
only guides later agents toward where to dig. The `"method": "cv"` field is what
separates these entries from the pre-rework ones — always include it.

```json
{"name": "<slug>", "branch": "exp/<slug>", "date": "<YYYY-MM-DD>", "main_commit": "<git rev-parse --short HEAD>", "method": "cv", "set": "lite", "hypothesis": "...", "metrics": {"sens_at_fpr0.005_pooled": 0.0, "sens_at_fpr0.005_mean": 0.0, "n_folds": 0}, "baseline": {"model": "cv-baseline", "sens_at_fpr0.005_pooled": 0.0, "sens_at_fpr0.005_mean": 0.0}, "conclusion": "..."}
```

### 6. Commit worktree

```bash
git add -A && git commit -m "exp/<slug>: <what was tried and outcome>"
```

Then stop. Do not merge into main. Do not delete the worktree or branch. Do not
proceed to another experiment. You're done! Thank you!

## Restoring a worktree

```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

## Prohibited

Do not modify `01_annotate/`, `translations/`, or any set's `build.R` — changing
the data under an experiment makes it incomparable to everything else in the log.

Do not modify `04_test/metrics.py`. It is no longer only test code: `03_train`
imports `metrics_by_group` and `metrics_at_fpr` from it, so editing it changes
the metric itself. `04_test/`, `summarize_metrics.py` and `compare_metrics.py`
are stale but stay untouched until someone reworks them deliberately.
