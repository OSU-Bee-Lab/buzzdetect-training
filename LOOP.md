# autoresearch loop
This document describes the workflow for iteratively improving buzzdetect models in a series of experiments.

The loop has two arms, and **either is a legitimate target for an experiment**:

| arm | stage | what improves | log |
|---|---|---|---|
| **training** | `03_train` | `ins_buzz` sensitivity of the probe models (the *standard* tier) | `03_train/log.jsonl` |
| **distillation** | `05_distill` | the speed / sensitivity frontier of students distilled from a deployed teacher (the *lite* tier) | `05_distill/log.jsonl` |

One experiment is one hypothesis in one arm, one worktree, one run against a
matched control, and its record in that arm's log. The lifecycle below is shared;
where the arms differ (goal, baseline, constraints, how to run, read and record)
each has its own section or sub-step. Pick the arm the hypothesis belongs to; a
change that touches both (e.g. a better teacher) is a training experiment first,
and its distillation is a follow-up experiment.

Results from different eras may be directionally informative within an era, but are not numerically comparable between eras.

**This era (`cv-medium-v4`) started 2026-09-25.
One thing moved at the cutover: the Hard Negatives annotation effort (non-buzz
audio that drew high `ins_buzz` activation) joined the training pool. Rotating
folds and metric are unchanged from `cv-medium-v3` (archived), but its numbers
are not comparable. `moderate` (framehop 0.2) replaces `large` as the big-data set.
Old verdicts are leads, not answers: `temporal-context` was a clear negative in
era 1 and, rerun, the largest gain in era 2. Rerun rather than defer.

# Training arm (`03_train`)

## Goal

Improve `ins_buzz` sensitivity at a fixed false-positive rate on the deployments that are held-out during the rotating CV training.
Each fold is one deployment, a single site, date, and recording target (soybean, apple, yellow jacket nest, etc.).

**The number is the `total` row's `sensitivity_exclquiet` at `fpr` 0.005 in
`models/<name>/folds_sx.csv`**: the plain mean over the rotating folds, each
tuned to 0.5% FPR on its own held-out audio. README's
[Reading the results](README.md#reading-the-results) explains every choice in
it: per-deployment thresholds, why quiet buzz leaves the score but not
training, the loudness tiers, and why 0.005.

**Say which loudness tiers moved; they are part of every result.** A lever that
lifts `loud` or `untagged` is a detection gain, even if other categories aren't lifted.

Two facts about this era's tiers:

- **Tagging is in progress.** 1375 of 1842 buzz annotations have no loudness tag. These will be added in future eras. Training data is still a work in progress;
these experiments aren't intended to produce shipped models, but to identify the best configuration for the final, large training run.
- **`_background` buzz is scored** (Luke's call, 2026-09-11) and is 51% of all
  scored buzz seconds, entirely in `1_29` and `53` (median duration 254 s,
  against ~1 s elsewhere). These buzzes are qualitatively different than isolated ones,
  but they are legitimate targets and appear routinely in high-activity areas.
  Optimizing for them at the cost of isolated buzzes is a mistake, but so is the converse.
  It's a gain to improve in either direction without losing in the other.

## Baseline

The baseline model for any era is the simplest possible model, a logistic regression with a linear probe
against YAMNet's embedding layer.
If you're building off of the baseline, any improvement is a real win,
even if a prior experiment has shown a greater improvement using a different method.
If you're building off of a prior experiment, that experiment serves as your baseline and you must show a
meaningful improvement in metrics against that experiment's model.

This era's baseline is `cv-baseline-v4` = **0.324** (0.262 incl. quiet; frozen `yamnet` linear probe, 400 epochs; run
2026-09-25 from HANDOFF.md's queue).

The baseline model is **not** the last era's best.
Its purpose is not to stack the gains of a prior era.
The early experiments in a new era will often be checking to see that a prior era's
wins still hold in the current era.


## Constraints
- **Training set: `medium`**, always. The set has 8 rotating folds for evaluation.
  You may use the `lite` set for troubleshooting and `tiny` for smoke tests; draw no conclusions from either.
- **Never train or run a CV on `large`.** This set is reserved for production-ready training runs.
- **Every rotation trains the fixed `--epochs` budget.** There is no
  early-stopping option; it was removed outright (2026-09-17) after
  measuring it undertrained hard folds unevenly. The shipped model's epoch
  count is always read off the rotations' pooled val_loss curves instead
  (`train._consensus_epoch`) — no flag needed.
- **Prefer structure to hyperparameters.** Annotation keeps moving the data;
  hyperparameter results don't survive that, large one-directional structural
  results usually do.
- **A new embedder doesn't inherit the probe config.** The head was tuned on
  YAMNet, whose embedding is 89.6% exact zeros, non-negative and nearly linearly
  separable; on a dense, signed code the same head can do the damage. Before
  logging an embedder negative, sweep the readout offline against
  `03_train/metrics.py:sens_at_fpr` on the cached embeddings (seconds per config,
  no Keras) and say which part of the loss you measured. `aves-probe` logged
  0.218 → 0.074 as an embedder verdict, and an offline L2 sweep then recovered
  most of it. This is one diagnostic sweep, not licence to tune.
- **Don't measure the noise floor.** There is no seed control, and repeat runs to
  build a CI are not how this loop spends compute. Read the headline delta
  against its eval SD, and confirm a large gain with one repeat run.
- **Never build an artificial frame/embedding cache.** Every embedding row used
  for training or CV must come from a real embedder's `embed()` running on real
  audio in a real extraction step (`02_set/main.py`, or an equivalent full
  extraction pass for a new embedder). Do not synthesize, broadcast, nearest-
  neighbour-join, zero-fill, interpolate, or otherwise construct a cache from
  other caches, however the join is justified (shared frame grid, "no
  re-extraction needed," matching another embedder's labels). A join across two
  embedders' frame grids duplicates one embedder's row across several of the
  other's frames that don't share its audio window, which hands a "negative"
  frame information about audio outside its own span — the same inflation
  mechanism as `context-stack`'s cached-row leak, and unlike it, it can silently
  fabricate rows outright (e.g. zero-filling wherever no real frame is close
  enough). We have the compute budget for a real extraction; take it. If a new
  embedder needs a different frame grid or context window than an existing one,
  write it as a real embedder and extract it for real — do not assemble it from
  pieces of other embedders' caches. (This closed `perch-broadcast-join`
  unfinished, mid-2026-09; do not resume it as written — Perch context belongs
  in a real embedder, e.g. a per-frame-centred extraction, item 2b.)
- Do not modify `01_annotate/`, any set's `build.R`, or
`03_train/metrics.py`.

# Distillation arm (`05_distill`)

`05_distill/README.md` is the operator's guide and `DESIGN.md` the contract.
Before proposing, read IDEAS.md's distillation standing facts and run
`python 05_distill/ladder_record.py frontier` (the current frontier).
`FRONTENDS.md` and `LADDER.md` are closed records of the work before the loop:
read them for the reasoning behind a fact, never append to them. This section is
only what the loop needs on top of these.

## Goal

**Push out the speed / sensitivity frontier of students of the era's teacher**
(`v4-ft-ps-e60-moderate`, two YAMNet trunk passes). There is no target (Luke,
2026-09-29): a student is a gain if it sits above the current non-dominated line,
i.e. a higher headline than every student at its speed or faster, or the same
headline faster. The two tiers it feeds are a *standard* (teacher-like) and a
*lite* (faster, less sensitive) model.

**The number is the same headline as the training arm** (`sensitivity_exclquiet`
at fpr 0.005 over the teacher's set's rotating folds, `eval_folds.py`), against
**speed**: `x YAMNet` at 200 s of audio on the GPU (`speed_200.json`). Both are in
the student's `05_distill/log.jsonl` row. Floors: headline >= 50% of the era
baseline's (0.207 against `cv-baseline-v4-moderate`'s 0.414), and a lite student
must beat YAMNet's speed (>= 1.5x) to be worth having. Read loudness tiers as in
the training arm.

## Baseline

The comparator is the **matched student**: same teacher, rung, schedule and step
budget, seed, front end and width, differing only in your change. Look for it in
`05_distill/log.jsonl` (`python 05_distill/ladder_record.py frontier` lists the
comparable rows); if it isn't there, put it in the same `main.py --runs` call.
Also say where the student lands against the frontier, which is the real goal.

Noise: headline seed noise on a student is ~0.01-0.02; the repeat rule
(a gain > 0.02 over everything at its speed or faster gets one repeat at the next
seed before it counts) is built in: `ladder_record.py repeats` prints the commands.

## Constraints

- **Students never see labels or the eval deployments.** Any new data path goes
  through the plan's blacklist; 59 rows were quarantined on 2026-10-04 for
  training on test-set audio (FRONTENDS.md, "Update 2026-10-04").
- **Only comparable rows count.** Rows whose `key` (teacher build, V pool, eval
  roster) differs from the current one are left out of every table automatically;
  don't compare against them by hand.
- **Rung C is the working rung; rung D does not run** (LADDER.md's rule; LADDER is
  closed). Prefer WSD step-budget curves to single cosine runs: `--wsd-max 56000
  --wsd-stop 1.3` (README, "Step budget"): the trunk stops after two doublings
  in a row each gain < 1.3 points of hit@K.
- **Speed is GPU only.** CPU int8 timings on this machine are meaningless (no AVX2).
- **A front end is edited by renaming it** (`05_distill/CLAUDE.md`); its
  spectrogram cache is stamped with its definition.
- `05_distill/CLAUDE.md`'s invariants hold: no teacher names, class indices or
  widths as literals; paths only from `dpaths.py`.

# Both arms

## Building off of prior work
Check the arm's log (`03_train/log.jsonl` or `05_distill/log.jsonl`) to see the results of previous experiments.
If you are building off of a prior experiment, you may branch from that experiment's worktree,
or you can manually copy the code you need.
If you are building off of a previous experiment, you should compare your results against that experiment's
in addition to the baseline.


## Experiment lifecycle

### 0. Orient

Read both logs (`03_train/log.jsonl`, `05_distill/log.jsonl`) and `IDEAS.md`, and pick the arm. If and only if
you want to examine prior eras, see `archive/` (each era keeps both logs under `<era>/<stage>/`).
Detail on an archived run is in `archive/<era>/notes/<slug>.md`, and its code on
`exp/<slug>` or `refs/archive/<slug>`.

Some of the models in the models/ dir are unrelated to this autoresearch loop or are old artifacts.
Make sure you're only comparing against models from this era.

### 1. Create a worktree

```bash
bash tools/setup_worktree.sh <slug>
```

Commit anything the experiment needs first; uncommitted changes don't reach the
worktree. All code changes go in the worktree, never in main's tracked files.

The script symlinks `embedders/*` and each set's `audio/` and `embeddings/` back
to main, so extraction is shared. Keep it that way: a new embedder directory or
`--embedder` name goes straight into the shared tree so future runs can use it.
A worktree only gets links for what existed when it was made: after creating an
embedder in main, run `bash tools/setup_worktree.sh --relink <slug>`.
The cache is keyed by `(set, embedder name)`, and its fingerprint covers annotations, not code. So
break a symlink only when your run would overwrite something under an existing
name. Git sees those symlinks as typechanges, so any commit from the worktree
(including `finish_experiment.sh`'s) shows shared dirs flipping to mode 120000.
That noise is expected; don't revert it.

### 2. Make the change

Write the Hypothesis section of `notes.md` (template in step 5) before touching code.
Then, implement changes.
Before stages 2 and 3, smoke test on the lite set, through `tools/launch_job.sh` if it
extracts (a new embedder over lite runs for minutes on CPU). Note: tiny and lite are small and may
not be perfectly representative of medium (e.g., missing a class).

### 3. Run it

From the worktree. **Training arm:**

```bash
# stage 2, only if extraction changed
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder <emb>
# stage 3
tools/launch_job.sh train.log -- 03_train/main.py --name <name> --set medium --embedder yamnet --translation general -y
```

**Distillation arm:** one `main.py` call trains, exports, evaluates, times and
records every student in `--runs`, and resumes on rerun. Give it a `--prefix`
or run-spec suffix that names your experiment, so its students are easy to find:

```bash
tools/launch_job.sh distill.log -- 05_distill/main.py --rung C --runs "<frontend>:<arch>[:...] <matched control>" \
    --wsd-max 56000 --wsd-stop 1.3
```

It writes data under the main checkout's `05_distill/data/` and its rows into
the main checkout's `05_distill/log.jsonl` (`dpaths.MAIN`), even from a worktree.
A WSD trunk reaches its later budgets in hours: the park rule applies.

### Phase 2 (training arm)
Extractions re-launch safely, picking up where the last left off.
Idents have wildly varying sample sizes, so it's difficult to extrapolate ETA from per-ident times.

### Phase 3 (training arm)
Per-fold training durations are even, so you can derive a fairly accurate ETA after the first.
Leave the shipped model untrained. `folds_sx.csv` comes entirely from the rotations, so there's no research benefit to training the production model.

### 4. Read the results

**Distillation arm:** `python 05_distill/ladder_record.py frontier` (where each
student sits, repeat rule), `ladder_record.py wsd --name <trunk>` (a step-budget
curve), and `tools/results.py <control> <student>` works on the students' eval
dirs (`05_distill/data/<teacher>/eval/<name>`) for the per-fold and tier table.
On eval dirs it can run past 2 min, so launch it with `tools/launch_job.sh`
(CLAUDE.md "Running long jobs") rather than in the foreground.
Report headline and speed together; a student slower than its control needs a
headline gain to count. The rest of this step is the training arm.

```bash
python tools/results.py <matched control model> <experiment model>
```

Bare model names work from main or the worktree: a name is looked up in this
checkout's `models/`, then main's, then the worktrees'. A path works too.

It prints `notes.md`'s Results section: the per-fold table with each delta's
eval-sampling SD, the headline ± SD, the inclusive figure, and the tier deltas.
`tools/compare_folds.py` and `tools/eval_sampling_sd.py` are its parts, if you
need more detail. A blank ± SD means the two models' frame grids differ (for
example, an AVES-hop embedder against YAMNet), and no paired SD exists.

Always read `docs/judging-results.md` for guidance before interpretation.

### 5. Record

Create `notes.md` in the worktree root:

```
# <slug>
## Hypothesis
## Changes
## Results
<tools/results.py output>
<interpretation: the headline against its SD; each fold that moved, against its
own SD, saying "unsure" where it is within it; which tiers moved>
## Conclusion
```

If you used an idea, delete its section from main's `IDEAS.md` (not the
worktree's copy, which `--commit-also` never sees). Then, from main:

```bash
# training arm
tools/finish_experiment.sh <slug> --summary "<what was tried, and the outcome>" \
  --model .local/worktrees/<slug>/models/<name> --baseline-model <matched control> \
  --hypothesis "..." --trust <trust> --conclusion "..." [--commit-also IDEAS.md]
# distillation arm: the rows are already written by main.py's record stage
tools/finish_experiment.sh <slug> --arm distill --summary "<what was tried, and the outcome>" \
  --runs "<student name> [<student name> ...]" [--commit-also IDEAS.md]
```

It commits and pushes `exp/<slug>` (the only durable copy of `notes.md`), then
records the experiment in main and commits it: the training arm appends the
`03_train/log.jsonl` line; the distillation arm stamps `"exp": "<slug>"` on its
students' rows in `05_distill/log.jsonl` (`05_distill/log_exp.py`). For the
distillation arm, the hypothesis, verdict and tier reading live in `notes.md`
only, so write its Conclusion as you would the training arm's `conclusion`. Keep `conclusion` brief,
since it only points later agents at where to dig, and put the tier reading in
it. Pass `--commit-also IDEAS.md` only if `git diff IDEAS.md` shows nothing but
your deletion.

To amend an earlier training entry, don't rewrite its conclusion or metrics.
Do a new experiment with the amendment you'd like to make, then set `trust`
of the prior experiment and add a dated `amended` field saying what changed and which run supersedes it.

### 6. One loop completed!

Don't merge into main, and don't prune the worktree. Pruning has repeatedly
destroyed the only copy of something: an untracked `embedder.py` from a `clean`
result, and 31 branches' notes. `models/.gitignore` re-includes each model's
`folds_sx.csv` and per-fold `summary.json`, so paired comparisons survive; use
`git add -f` on predictions only if they are the point. If a worktree holds a
multi-GB unshared cache, remove just that directory.

If instructed to run more than one loop, go back to the top and do it all again.
Run experiments sequentially, not in parallel, in either arm. CPU and GPU are likely fully saturated by a
single experiment (the 4 GB card fits one job), so you'll see no gain from concurrency.


## Restoring a worktree

```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

Then `bash tools/setup_worktree.sh --relink <slug>`, or the run re-extracts from scratch.

## Best practices
Keep the main thread context minimal - don't give long summaries or interpretations in the main thread as you work, unless the user asks for them.
Even when finishing a loop, give only a brief couple-sentence summary of the results.

# Special instructions
These bind both arms (a teacher or a student front end included).

## INJUNCTION: aves models are forbidden
The aves embedder has shown to be intolerably slow. Future loops may not utilize aves unless they are able to bring it to within an order of magnitude of the YAMNet inference rate.

## INJUNCTION: perch models are forbidden
Perch's CPU inference rate is roughly 0.005x YAMNet's (`perch-centred`'s medium
extraction, 2026-09-22/23: ~22.6 h wall-clock for one embedder over one set,
against YAMNet's single-digit minutes) -- not feasible for `medium`-or-larger
extractions as a routine lever, whatever its accuracy. `exp/perch-pitchshift-concat`
(launched 2026-09-23, still running at the time of this injunction) is exempted
as an already-in-flight run; let it finish and log its result. Future loops may
not start any new Perch extraction (`perch`, `perch_centred`, or a derivative)
unless its inference rate is brought within an order of magnitude of YAMNet's,
the same bar the aves injunction sets.
