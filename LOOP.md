# Autoresearch Loop

> **The slate was cleared on 2026-09-08, when the training data was revised.**
> `log.jsonl` is empty. The 59 runs before it are in `archive/`, one directory
> per era, each with a README stating what made its numbers comparable and what
> ended that. Nothing in there is a number you can beat — a data change moves
> every float in both logs.
>
> **There is no baseline right now. Establishing one is the next run** — see
> [Baseline](#baseline).
>
> **Old verdicts are leads, not settled answers.** `temporal-context` was logged
> as a clear negative in the first era and, rerun as `context-stack` in the
> second, was the largest gain in the log. A verdict inverted on an eval change
> alone; this cutover changes the data as well. Read `IDEAS.md` for the distilled
> version, and rerun rather than defer.

Read `README.md` first — [Reading the results](README.md#reading-the-results)
is where the metric choices below come from.

## Goal

Improve `ins_buzz` sensitivity at a fixed false-positive rate on **held-out
deployments**, with each deployment's threshold set on its own audio.

**The number: the `total` row's `sensitivity` at `fpr` 0.005, from
`models/<name>/folds_sx.csv`.**
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
The pooled sweep is no longer written out; `read_fold_predictions()` plus
`metrics_by_group()` rebuilds it. It is not an endpoint.

That mean is an oracle: placing a fold at exactly 0.5% FPR uses that fold's
labels, which an operator doesn't have. Read it as the ceiling on operator
tuning. It is a fair ceiling to compare two models by, since both get it.

Why 0.005: 0.01 is too loose to be operationally useful, and 0.001 is not
measurable on the sets we have — it rests on ~4 negative frames per fold, and
some folds can't reach it at all. Every fold keeps its own row in
`folds_sx.csv` with its own `neg_frames`, and a fold that couldn't reach the
target is blank rather than averaged in, so what the total rests on is visible
in the file.

Sensitivity at fixed *precision* — the old goal — is a `precision` column of
`folds_sx.csv` at the same operating point, or a full sweep away via
`03_train/metrics.py::metrics_at_precision`, but precision mixes in each
deployment's base rate, so prefer FPR when comparing.
The old log's "28% production standard" was measured on a different corpus with
a different base rate; it is not a target on this metric.

## Baseline

`cv-baseline` in `log.jsonl`, model at `models/cv_baseline/`:
**0.218 mean sens@fpr0.005** over 5 rotating folds. Join against its
`folds_sx.csv` for a paired per-fold comparison.

Config: a linear probe on frozen YAMNet, `Dropout(0.2)`, label smoothing 0.2,
Adam 0.002, `medium`, `general` — plus the two correctness fixes carried over
from the archive: per-class weights inside the loss (`weighted_bce_loss`, since
Keras' `fit(class_weight=)` collapses a multi-hot target to `argmax`) and
restoring the true `val_loss` argmin (`RestoreTrueBest`).

**Those two fixes measured -0.006 here, folds 1 up / 2 down / 2 flat** — inside
the noise floor, and opposite in sign to the +0.013 and +0.008 the archive
recorded for them on the pre-revision 11-fold set. They are kept because both
are correctness fixes independent of the metric, not because they helped. The
pre-fix control is `models/yamnet_medium_general_v2/` (0.224, same data and
folds) if anyone wants to revisit that call.

`models/yamnet_medium_general/` is the **previous** era's baseline, 0.206 under
`general_v1` on the pre-revision annotations. It is an artifact now: keep it for
reference, never as a comparator. A paired per-fold join against it is invalid —
the folds no longer hold the same audio.

Worth rerunning early alongside it: the `binary` translation control. The
`general` / `binary` pair at 0.206 / 0.203 is why the endpoint is per-deployment
rather than pooled, and that argument should be re-established on live numbers
rather than inherited from the archive.

## Constraints

- Embedder: YAMNet only, unless the experiment *is* the embedder — see
  `context-embedder` and `yamnet-combined` in `IDEAS.md`, both of which need a
  re-extraction and should be run deliberately, not incidentally.
- Set: `medium` — day-long annotated recordings across a diversity of
  environments, 11 rotating folds. `lite` is kept for troubleshooting and
  `tiny` for smoke-testing the pipeline; neither is a place to draw conclusions.
  Annotation is ongoing, so folds gain data and old numbers drift; say which
  commit of the set a run used if it matters. Every entry in `log.jsonl` is
  implicitly on `medium` at its current fold roster, so neither is logged per
  entry — which is exactly why a roster change (a deployment added or dropped)
  or a data revision **ends the era**. When that happens, run
  `python tools/archive_era.py --slug <name>` (preflight; `--write` to do it)
  and start a fresh log, rather than mixing incomparable numbers in one file.
  The script refuses while any branch is unpushed or any worktree is dirty, and
  leaves the era README's prose to you.
  **Training or CV runs on `large` are forbidden as a loop experiment,
  regardless of how ready it looks (extraction finished, a promising diff
  pending).** `large` is not an alternative set to iterate structure on — same
  annotations and folds as `medium` at 5x frame density, kept only for a single
  final-confirmation pass *after* the structural search on `medium` concludes,
  and only when Luke asks for it. An agent must never launch it on its own
  initiative.
- **Translation: pass `--translation general`.** This flipped at the 2026-09-08
  cutover. `general_v1` was a frozen copy of the pre-2026-09-03 table (17
  classes, before `mech_plane` folded into `mech_auto`), kept only so runs
  during the last era stayed comparable to comparators trained before the table
  moved. That era is closed, so the freeze has no one left to protect.
  `general_v1.csv` stays on disk — the archived log was trained under it and
  `archive/2026-08_cv-medium-v1/set/translations/` holds a copy — but a new
  experiment that passes it is comparing against nothing.
- Augmentation: has hurt training so far — avoid without strong reason.
- **What survives a data change, and what doesn't.** Annotation is ongoing, so
  the folds and the training pool move under every result. Hyperparameter
  results do not survive that — don't bank them. Large one-directional
  structural results usually do. Diagnostics and instrumentation survive
  absolutely: they cost one cheap run, recompute for free whenever the data
  moves, and they tell Luke what to annotate next, which is the standing
  bottleneck. Weight the rotation accordingly.
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
cat log.jsonl          # this era — empty until the baseline lands
cat IDEAS.md           # candidates, with prior verdicts marked as leads
ls archive/            # the 59 runs before this era
```

Propose a hypothesis. Check all three — most obvious ideas have been tried, and
the conclusions about *dead ends* (bandpass, mel masking, MLP heads, L2,
handcrafted frequency features) are the part likeliest to still hold; it is the
positive results that a data change invalidates.

For detail behind any archived run: `archive/<era>/notes/<slug>.md`, which is
complete for both eras. The code is on `exp/<slug>` where the branch survives,
and at `refs/archive/<slug>` where it doesn't — see `archive/README.md`.

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
WT=/home/luke/projects/buzzdetect-training/.local/worktrees/<slug>
cd $WT

# Stage 2 — only if the embedder or extraction changed
# Stage 3 — the whole CV: one model per rotating fold, then the shipped model
#   02_set/main.py  --set medium --embedder yamnet --workers 2 --verbose
#   03_train/main.py --name <modelname> --set medium --embedder yamnet --translation general -y
# Launch each DETACHED, not in the foreground and not via run_in_background —
# see "Running long jobs" in CLAUDE.md for the exact nohup recipe.
```

Stage 2 always runs far longer than a foreground command should block for.
Stage 3's cost depends entirely on what is being trained, and the two ends are
orders of magnitude apart — **measure before assuming**:

| config | per epoch | one CV |
|---|---|---|
| frozen probe, 1024-d YAMNet (the baseline) | ~1 s | **~9 min** for 5 folds, 651 epochs (measured 2026-09-08) |
| trunk fine-tune, 12288-d + unfrozen layers | ~80 s | **~24 h** for 11 folds (measured 2026-09-05, `unfreeze-one`) |

A frozen-probe CV is a foreground-scale job; only a trunk fine-tune needs the
detached-and-hand-off treatment. Claude Code's `run_in_background` gets
SIGKILLed running either stage (and also killed *waiting* on them). **See "Running long jobs" in `CLAUDE.md` for the
pattern that works** — launch the job detached with `nohup … & disown` (direct
env-python, not `conda run`; capture `$!`).

Then **hand off rather than wait** — see "If the run will outlast you" below.
Don't sleep-loop in Bash, don't `cat` a running log on a timer, and don't hold a `Monitor` open across a
multi-hour run: waking to watch a job you can't hurry costs uncached tokens for
no information.

There is no `--runs` and no stage 4. One training call *is* the experiment: it
trains one model per rotating fold and the shipped model, and prints the pooled
and unweighted numbers when it finishes.

**Reruns resume, silently.** Any fold directory already holding a
`config_model.json` is skipped, so re-running after a code change reuses the old
models and reports a mix of both. Always use a fresh `--name`, or pass
`--clear` via the root `main.py`, or delete `models/<name>/` by hand.

**If the run will outlast you, write a handoff doc.** Any job you expect to take
more than about an hour will outlive your context — waiting on it burns uncached
tokens for nothing. Launch it detached, commit a `HANDOFF_<slug>.md` in the
worktree, and end your turn. Keep it short; it needs four things:

1. **The one-command progress check** — is it running, and how many of 11 folds
   are done (`pgrep -f 03_train/main.py`, `find models/<name>/folds -name
   summary.json | wc -l`, `tail -3 <log>`).
2. **"If it's still running, STOP."** Say it explicitly: report `fold N/11` and
   quit. No tailing the log on a timer, no Monitor, no reading the rest of the
   repo — everything read while waiting is paid for twice.
3. **What to do when it finishes** — which model is the comparator (rarely
   `cv-baseline`; usually the matched control), which folds are too thin to
   trust, then notes.md → `log.jsonl` → commit.
4. **What to do if it died** — how to tell a self-healing restart from a real
   crash, and the exact relaunch command.

Also note anything a fresh agent would get wrong: symlinked caches it must not
delete, why the branch is based where it is. Don't name it plain `HANDOFF.md` —
that's taken by the machine-setup doc.

### 4. Read the results

```
models/<modelname>/
├── folds_sx.csv                 ← everything: a row per fold, then a `total` row
└── folds/<fold>/
    ├── predictions.csv          ← every held-out frame; all of the above derives from it
    ├── summary.json             ← epochs, val_loss, frame counts, sens-monitor peaks
    └── sens_curves.svg          ← sens@fpr per epoch vs the epoch that got restored
```

`folds_sx.csv` is written by a training run, and can be rebuilt for a model
trained before it existed — no retraining, no TensorFlow:

```bash
conda run -n buzzdetect-train python 03_train/resummarize.py <modelname>
```

**Compare paired, per fold.** Both configs ran on the same folds, so the useful
comparison is the per-fold difference — join the two `folds_sx.csv` files on
`fold` and look at the deltas and how many folds moved which way. Comparing
two scalars throws that pairing away, and a headline gap that vanishes under
pairing was never a capability gap.

```bash
python tools/compare_folds.py <baseline model dir> <exp model dir>
```

Does the join above and prints the per-fold delta table, the up/down count,
and the two headline mean sensitivities from `folds_sx.csv`'s `total` rows. Either model
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
  a fold with forty seconds of buzz in it. Check `buzz_frames` in
  `folds_sx.csv` before believing a per-fold delta.

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
- mean sens@fpr0.005: baseline <val> → this <val>
<interpretation: how many folds moved which way? are the movers folds with
enough buzz to trust? did any fold fail to reach the target FPR?>
## Conclusion
```

Append one line to `log.jsonl` in **main** and commit it. Be very brief; the log
only guides later agents toward where to dig. Keep `"method": "cv"` — it is what
distinguishes an entry from the first era's fixed-split ones if the logs are ever
read together.

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

`tools/log_entry.py` builds this line for you — it reads the `total` row's
sensitivity straight from each model's `folds_sx.csv` (no copying numbers by hand) and
fills in `branch`/`date`/`main_commit` by the convention above:

```bash
python tools/log_entry.py \
  --name <slug> \
  --model <experiment model dir, e.g. .local/worktrees/<slug>/models/<modelname>> \
  --baseline-model models/<this era's cv-baseline> \
  --hypothesis "..." --trust clean --conclusion "..." \
  --write   # omit to preview without appending
```

### 6. Commit worktree

```bash
git add -A && git commit -m "exp/<slug>: <what was tried and outcome>"
git push origin exp/<slug>
```

**Push it.** The branch is the only durable copy of your `notes.md` — the
worktree is disposable and this machine's disk is not backed up. Thirty-one
branches from the last two eras were deleted with their notes still only local,
and were recovered from dangling commits by luck.

Then prune the worktree — the branch keeps everything that matters, and a
worktree that re-extracted embeddings can be holding several GB:

```bash
git worktree remove --force .local/worktrees/<slug>
```

Keep it only if the run is unfinished and resumable. **Check
`git status` before pruning** — uncommitted work in a worktree dies with it.

Then stop. Do not merge into main. Do not proceed to another experiment.
You're done! Thank you!

## Restoring a worktree

```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

Its symlinked caches are not restored with it — rerun `setup_worktree.sh`'s
linking, or the run will re-extract from scratch.

## Prohibited

Do not modify `01_annotate/`, `translations/`, or any set's `build.R` — changing
the data under an experiment makes it incomparable to everything else in the log.

Do not modify `03_train/metrics.py`. `03_train`
imports `metrics_by_group` and `metrics_at_fpr` from it, so editing it changes
the metric itself.
