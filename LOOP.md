# Autoresearch Loop

> **The slate was cleared in 2026-08.** `log.jsonl` holds CV-era runs only —
> leave-one-fold-out on day-long annotated recordings, scored on `sens_persite`.
> Those are comparable to each other and are what you beat.
>
> The 29 experiments before that ran on a fixed train/validate split against a
> retired corpus, with a training set of mixed provenance. They are archived at
> `.local/archive/log_precv.jsonl` and `.local/worktrees-fixed-test/`, and
> distilled into `IDEAS.md`. **Their verdicts are leads, not settled answers** —
> `temporal-context` was logged there as a clear negative and, rerun on the
> current set as `exp/context-stack`, is the largest gain yet. Read `IDEAS.md`
> rather than the archive; rerun rather than defer.

Read `README.md` first — [Reading the results](README.md#reading-the-results)
is where the metric choices below come from.

## Goal

Improve `ins_buzz` sensitivity at a fixed false-positive rate on **held-out
deployments**, with each deployment's threshold set on its own audio.

**The number: `sens_persite` at `fpr` 0.005, from `models/<name>/folds_sx.csv`.**
Every rotating fold is tuned to 0.5% FPR on its own held-out audio; the primary
figure is the plain mean of those sensitivities, each deployment counted once. A
training run prints it.

Each deployment counts once because the goal is a new deployment, and a new
deployment is one site — not a weighted blend of eleven. A buzz-weighted mean
would let a full-bloom mustard field speak over a quiet soybean patch, and its
weights mix real ecology with how many hours happened to get annotated.

The threshold is set per site because buzzdetect ships none: operators are told
to find their own. A metric that forces one global threshold measures a
constraint the product doesn't have, and confounds detection with how portable a
model's score scale is — it has made a strictly better model look half as good.
`folds_pooled_metrics.csv` still holds the pooled sweep for ROC plots; it is not
an endpoint.

`sens_persite` is an oracle: placing a fold at exactly 0.5% FPR uses that fold's
labels, which an operator doesn't have. Read it as the ceiling on operator
tuning. It is a fair ceiling to compare two models by, since both get it.

Why 0.005: 0.01 is too loose to be operationally useful, and 0.001 is not
measurable on the sets we have — it rests on ~4 negative frames per fold, and
some folds can't reach it at all. `folds_sx.csv` carries
`neg_frames_persite_median` and `folds_scored` so you can see what a row rests
on.

Sensitivity at fixed *precision* — the old goal — is still recoverable from
`folds_pooled_metrics.csv` via `03_train/metrics.py::metrics_at_precision`, but
precision mixes in each deployment's base rate, so prefer FPR when comparing.
The old log's "28% production standard" was measured on a different corpus with
a different base rate; it is not a target on this metric.

## Baseline

`cv-baseline` in `log.jsonl`: main's default config — a linear probe on frozen
YAMNet with `Dropout(0.2)`, `BinaryCrossentropy(label_smoothing=0.2)`, Adam at
0.002, under the `general` translation on the `medium` set.

**sens_persite @ fpr0.005 = 0.206**, over 11 rotating folds. The model is kept
at `models/yamnet_medium_general/`; its per-fold numbers are in
`folds_summary.csv` there, which is what you join against for a paired
comparison.

`models/yamnet_medium_binary/` is the same config under the `binary`
translation, at 0.203 — logged as `binary-translation-cv`. Keep it: the pair is
why the endpoint is per-deployment rather than pooled.

## Constraints

- Embedder: YAMNet only, unless the experiment *is* the embedder — see
  `context-embedder` and `yamnet-combined` in `IDEAS.md`, both of which need a
  re-extraction and should be run deliberately, not incidentally.
- Set: `medium` — day-long annotated recordings across a diversity of
  environments, 11 rotating folds. `lite` is kept for troubleshooting and
  `tiny` for smoke-testing the pipeline; neither is a place to draw conclusions.
  Annotation is still in progress, so folds will gain data over time and old
  numbers will drift; say which commit of the set a run used if it matters.
  Every entry in `log.jsonl` is implicitly on `medium` at 11 folds — that's
  fixed for as long as the log exists, so neither is logged per entry. If the
  fold roster itself ever changes (a deployment added or dropped), archive
  `log.jsonl` the way the pre-CV log was archived and rerun whichever
  experiments are worth keeping under the new set, rather than mixing fold
  counts silently in one log.
- Augmentation: has hurt training so far — avoid without strong reason.
- Err against hyperparameter tuning, unless you have a strong reason. We're
  looking for structural gains; hyperparameters can be tuned in one large sweep
  once a good structure is identified.
- Change one thing. Layering a change on top of another risks interaction
  effects that obscure whether the change itself helped.
- **Look for clear signals; don't try to measure the noise floor.** There is no
  seed control in the pipeline and a full CV is expensive, so repeating runs to
  bootstrap a confidence interval is not how this loop spends its compute. A
  result worth acting on shows up as most folds moving the same way, not as a
  small mean shift with folds scattered either side. Report the direction and
  the count; never compute a CI across folds and treat it as one (the training
  pools overlap ~90%, so folds are not independent). If a result is small and
  the folds are split, say it is inconclusive and move on rather than spending
  another CV on it.

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

Every entry in `log.jsonl` carries a `main_commit` and a `branch`; read
`notes.md` on that `exp/<slug>` branch for the detail behind any of them.

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

If the change touches `model.compile()` — a custom loss, a different output
layer — smoke-test it before spending a CV run on it:

```bash
conda run -n buzzdetect-train python tools/smoke_model.py  # self-test, no changes needed
```

`tools/smoke_model.py::smoke_test_loss()` builds the same architecture as
`03_train/train.py`, fits one step on dummy data, and runs it through the
exact `save(include_optimizer=True)` → `load(compile=False)` round trip
`write_model_py.py`'s inference path uses — a custom loss/metric that isn't
serializable the way that path expects fails in seconds instead of after a
full CV.

### 3. Run the pipeline

From the worktree, using the path `setup_worktree.sh` printed:

```bash
WT=/Users/luke/Documents/bioacoustics/buzzdetect-training/.local/worktrees/<slug>
cd $WT

# Stage 2 — only if the embedder or extraction changed
conda run -n buzzdetect-train python 02_set/main.py --set medium --embedder yamnet --workers 2

# Stage 3 — the whole CV: one model per rotating fold, then the shipped model
conda run -n buzzdetect-train python 03_train/main.py \
  --name <modelname> --set medium --embedder yamnet --translation general -y
```

Both stages run far longer than a foreground command should block for. Launch
each with `run_in_background: true` and then stop — do not sleep-loop, do not
`cat` the task's output file, do not spin up a `Monitor`. A backgrounded
command already delivers a completion notification on its own; that
notification *is* the wait. `Monitor` is for streaming an ongoing process's
events, not for a one-shot "tell me when this exits."

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
├── folds_sx.csv            ← the number: sens_persite, and what it rests on
├── folds_summary.csv       ← per fold: epochs, val loss, frame counts, sens@fpr
└── folds/<fold>/sx.csv     ← the same read, one fold at a time
```

`folds_sx.csv` is written by a training run, and can be rebuilt for a model
trained before it existed — no retraining, no TensorFlow:

```bash
conda run -n buzzdetect-train python 03_train/resummarize.py <modelname>
```

**Compare paired, per fold.** Both configs ran on the same folds, so the useful
comparison is the per-fold difference — join the two `folds_summary.csv` files
on `fold` and look at the deltas and how many folds moved which way. Comparing
two scalars throws that pairing away, and a headline gap that vanishes under
pairing was never a capability gap.

```bash
python tools/compare_folds.py <baseline model dir> <exp model dir>
```

Does the join above and prints the per-fold delta table, the up/down count,
and the two headline `sens_persite` numbers from `folds_sx.csv`. Either model
argument can be a bare name under `models/` or a path — an experiment's model
usually lives in its worktree's own (unsymlinked) `models/` dir, so pass
`.local/worktrees/<slug>/models/<modelname>` directly for that side.

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
- sens_persite @ fpr0.005: baseline <val> → this <val>
<interpretation: how many folds moved which way? are the movers folds with
enough buzz to trust? did any fold fail to reach the target FPR?>
## Conclusion
```

Append one line to `log.jsonl` in **main** and commit it. Be very brief; the log
only guides later agents toward where to dig. The `"method": "cv"` field is what
separates these entries from the pre-rework ones — always include it.

Every entry also needs a `trust` judgment, kept separate from the delta itself.
This is not a statistical significance test — there's no seed control to build
one from (see the noise-floor caution above) — just whether a later reader
should act on the number as measured, or dig first:

- `"clean"` — nothing about how it was measured should make you discount it.
- `"caveated"` — a specific factor (a stopping rule tuned for a different input
  scale, a fold too small to read, a metric that doesn't transfer across
  deployments) means the direction is probably right but the size shouldn't be
  trusted as stated.
- `"artifact"` — something about the *setup*, not the effect, inflated or
  deflated the number enough that comparing it to other entries at face value
  would mislead.

Say *why* in `conclusion`; `trust` is just the flag a later agent scans for
before building on your number. `context-stack` (`+0.050`, `artifact`) is why
this field exists: the eval's same-label-neighbour bug materially inflated
that number, and only `context-embedder`'s honest rerun (`+0.022`, `clean`)
revealed the size of the gap. Without a separate flag, a later agent skimming
deltas would have picked `context-stack` as the biggest win in the log and
built on the wrong number.

```json
{"name": "<slug>", "branch": "exp/<slug>", "date": "<YYYY-MM-DD>", "main_commit": "<git rev-parse --short HEAD>", "method": "cv", "hypothesis": "...", "metrics": {"sens_at_fpr0.005_persite": 0.0}, "baseline": {"model": "cv-baseline", "sens_at_fpr0.005_persite": 0.0}, "trust": "clean", "conclusion": "..."}
```

`main_commit` means **main's HEAD at the moment you log**, not the commit the
experiment's worktree branched from — those two diverge whenever main moves
while a long CV run is in flight (it did for `class-weight-fix`). Run
`git rev-parse --short HEAD` in the main checkout right before you commit the
log entry, not in the worktree.

`tools/log_entry.py` builds this line for you — it reads `sens_persite`
straight from each model's `folds_sx.csv` (no copying numbers by hand) and
fills in `branch`/`date`/`main_commit` by the convention above:

```bash
python tools/log_entry.py \
  --name <slug> \
  --model <experiment model dir, e.g. .local/worktrees/<slug>/models/<modelname>> \
  --baseline-model models/yamnet_medium_general \
  --hypothesis "..." --trust clean --conclusion "..." \
  --write   # omit to preview without appending
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

Do not modify `03_train/metrics.py`. `03_train`
imports `metrics_by_group` and `metrics_at_fpr` from it, so editing it changes
the metric itself.
