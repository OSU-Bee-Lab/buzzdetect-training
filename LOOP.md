# Autoresearch Loop

> **The slate was cleared on 2026-09-08, when the training data was revised.**
> The 59 runs before that are in `archive/`, one directory per era, each with a
> README stating what made its numbers comparable and what ended that. Nothing
> in there is a number you can beat — a data change moves every float.
>
> **Old verdicts are leads, not settled answers.** `temporal-context` was logged
> as a clear negative in the first era and, rerun as `context-stack` in the
> second, was the largest gain in the log. A verdict inverted on an eval change
> alone; this cutover changed the data as well. Read `IDEAS.md` for the
> distilled version, and rerun rather than defer.

Read `README.md` first — [Reading the results](README.md#reading-the-results)
is where the metric choices below come from.

## Goal

Improve `ins_buzz` sensitivity at a fixed false-positive rate on **held-out
deployments**, with each deployment's threshold set on its own audio.

**The number: the `total` row's `sensitivity` at `fpr` 0.005, from
`models/<name>/folds_sx.csv`.** Every rotating fold is tuned to 0.5% FPR on its
own held-out audio; the primary figure is the plain mean of those
sensitivities, each deployment counted once. A training run prints it.

Each deployment counts once because the goal is a new deployment, and a new
deployment is one site — not a weighted blend. A buzz-weighted mean would let a
full-bloom mustard field speak over a quiet soybean patch, and its weights mix
real ecology with how many hours happened to get annotated.

The threshold is set per site because buzzdetect ships none: operators are told
to find their own. A metric that forces one global threshold measures a
constraint the product doesn't have, and confounds detection with how portable a
model's score scale is — it has made a strictly better model look half as good.
The pooled sweep is not written out; `read_fold_predictions()` plus
`metrics_by_group()` rebuilds it. It is not an endpoint.

That mean is an oracle: placing a fold at exactly 0.5% FPR uses that fold's
labels, which an operator doesn't have. Read it as the ceiling on operator
tuning — a fair ceiling to compare two models by, since both get it.

Why 0.005: 0.01 is too loose to be operationally useful, and 0.001 is not
measurable on the sets we have — it rests on ~4 negative frames per fold, and
some folds can't reach it at all. Every fold keeps its own row in
`folds_sx.csv` with its own `neg_frames`, and a fold that couldn't reach the
target is blank rather than averaged in, so what the total rests on is visible.

Sensitivity at fixed *precision* — the old goal — is a `precision` column at the
same operating point, or a full sweep away via `metrics_at_precision`, but
precision mixes in each deployment's base rate, so prefer FPR when comparing.
The old log's "28% production standard" was measured on a different corpus with
a different base rate; it is not a target on this metric.

## Baseline

`cv-baseline` in `log.jsonl`, model at `models/cv_baseline/`:
**0.218 mean sens@fpr0.005** over 5 rotating folds. Join against its
`folds_sx.csv` for a paired per-fold comparison.

Config: a linear probe on frozen YAMNet, `Dropout(0.2)`, label smoothing 0.2,
Adam 0.002, `medium`, `general` — plus two correctness fixes carried over from
the archive: per-class weights inside the loss (`weighted_bce_loss`, since
Keras' `fit(class_weight=)` collapses a multi-hot target to `argmax`) and
restoring the true `val_loss` argmin (`RestoreTrueBest`).

**Those two fixes measured -0.006 here, folds 1 up / 2 down / 2 flat** — inside
the noise floor, and opposite in sign to the +0.013 and +0.008 the archive
recorded on the pre-revision 11-fold set. They are kept because both are
correctness fixes independent of the metric, not because they helped. The
pre-fix control is `models/yamnet_medium_general_v2/` (0.224, same data and
folds) if anyone wants to revisit that call.

Worth rerunning early: the `binary` translation control. The `general`/`binary`
pair at 0.206/0.203 is why the endpoint is per-deployment rather than pooled,
and that argument should be re-established on live numbers rather than
inherited from the archive.

## Constraints

- **Embedder: YAMNet only, unless the experiment *is* the embedder.** Those need
  a re-extraction and should be run deliberately, not incidentally.
- **Set: `medium`** — day-long annotated recordings across a diversity of
  environments, 5 rotating folds. `lite` is for troubleshooting and `tiny` for
  smoke-testing the pipeline; neither is a place to draw conclusions.
  Annotation is ongoing, so folds gain data and old numbers drift; say which
  commit of the set a run used if it matters. Every `log.jsonl` entry is
  implicitly on `medium` at its current fold roster, which is why a roster
  change or a data revision **ends the era**: run
  `python tools/archive_era.py --slug <name>` (preflight; `--write` to do it)
  and start a fresh log rather than mixing incomparable numbers in one file. It
  refuses while any branch is unpushed or any worktree is dirty, and leaves the
  era README's prose to you.
- **Training or CV runs on `large` are forbidden as a loop experiment**,
  regardless of how ready it looks. It is the same annotations and folds as
  `medium` at 5x frame density, kept for a single final-confirmation pass
  *after* the structural search concludes, and only when Luke asks. An agent
  must never launch it on its own initiative.
- **Translation: pass `--translation general`.** This flipped at the 2026-09-08
  cutover; `general_v1` protected a log that is now archived, so an experiment
  passing it is comparing against nothing.
- **Augmentation has hurt training so far** — avoid without strong reason, and
  read the cross-fold rule in `03_train/CLAUDE.md` before building any.
- **What survives a data change, and what doesn't.** Annotation is ongoing, so
  the folds and the training pool move under every result. Hyperparameter
  results do not survive that — don't bank them. Large one-directional
  structural results usually do. Diagnostics and instrumentation survive
  absolutely: they cost one cheap run, recompute for free whenever the data
  moves, and they tell Luke what to annotate next, which is the standing
  bottleneck. Weight the rotation accordingly.
- **Err against hyperparameter tuning.** We're looking for structural gains;
  hyperparameters can be tuned in one sweep once a good structure is found.
  **Exception: a new embedder does not inherit the probe config's validity.**
  See below.
- **The probe config is YAMNet-tuned, and it is hardcoded.**
  `03_train/train.py` builds `Dropout(0.2) -> Dense(n_classes)` with
  `label_smoothing=0.2`, Adam at 0.002 and no weight decay, with no CLI flag to
  vary any of it. Those values were tuned on YAMNet, whose embedding is 89.6%
  exact zeros, non-negative, and already linearly separable — it is the
  penultimate layer of a supervised classifier whose 521 AudioSet classes
  include `Buzz` and `Bee, wasp, etc.`. Input dropout on a code like that is
  mild; on a dense, signed, zero-centred code it is heavy multiplicative noise,
  and with no weight decay a 768-dimension distributed representation overfits
  the training sites instead of transferring.

  **So when an embedder swap comes back negative, that number is
  `embedder + this head`, and the head may be doing the damage.** Before logging
  an embedder verdict, sweep the readout offline against
  `03_train/metrics.py:sens_at_fpr` on the cached embeddings — no Keras, no
  extraction, seconds per configuration — and say which part of the loss you
  measured. `aves-probe` (2026-09-09) did not do this, logged
  0.218 -> 0.074 as an embedder verdict, and an offline L2 sweep on the *same*
  embeddings then recovered most of it. The follow-up is `exp/aves-readout`;
  if that entry has not been amended, the amendment is still owed.

  This is not licence to tune. It is one diagnostic sweep to attribute a
  negative before it becomes a verdict, and it stays offline unless it changes
  the reading.
- **Change one thing.** Layering a change on another risks interaction effects
  that obscure whether the change itself helped.
- **Look for clear signals; don't try to measure the noise floor.** There is no
  seed control and a full CV is expensive, so repeating runs to bootstrap a
  confidence interval is not how this loop spends its compute. A result worth
  acting on shows up as most folds moving the same way, not as a small mean
  shift with folds scattered either side. Report the direction and the count;
  never compute a CI across folds and treat it as one (training pools overlap
  ~90%, so folds are not independent). If a result is small and the folds are
  split, say it is inconclusive and move on.

Do not modify `01_annotate/`, `translations/`, any set's `build.R`, or
`03_train/metrics.py` — see the stage CLAUDE.md files for why.

## Experiment lifecycle

### 0. Orient and hypothesize

Read `log.jsonl` (this era), `IDEAS.md` (candidates, prior verdicts marked as
leads), and `ls archive/` (the 59 runs before this era). Most obvious ideas have
been tried, and the conclusions about *dead ends* — bandpass, mel masking, MLP
heads, L2, handcrafted frequency features — are the part likeliest to still
hold; it is the positive results a data change invalidates.

**That last sentence has one known exception, and it is easy to walk into: every
one of those dead ends was measured on YAMNet embeddings.** Head and
regularisation verdicts are statements about a *representation*, not about the
pipeline. If your experiment changes the embedder, they do not carry — see
"the probe config is YAMNet-tuned" under Constraints before treating any of them
as settled.

For detail behind an archived run: `archive/<era>/notes/<slug>.md`, complete for
both eras. Code is on `exp/<slug>`, or `refs/archive/<slug>` where the branch
didn't survive (`archive/README.md`).

If you use an IDEA, remove it from `IDEAS.md` after testing.

### 1. Create worktree

```bash
bash setup_worktree.sh <slug>
```

Confirm the sanity check passes. Uncommitted changes to tracked files won't
appear in the worktree — commit or stash first if they matter.

The script symlinks `embedders/*` and each set's `audio/`+`embeddings/` back to
main, so extraction is shared. **Keep it that way.** A cache written through the
symlink survives the worktree pruning at step 6 and is there for the next
experiment; a worktree-local copy dies with the worktree and costs the next
agent a re-extraction. Adding a *new* embedder directory, and extracting under a
*new* `--embedder` name, both write somewhere nothing else claims — do them
straight into the shared tree.

Break a symlink only when your run would write over something already there
under the same name. The cache is keyed by `(set, embeddername)` and the
staleness fingerprint covers **annotations, not code**, so nothing detects a
directory rebuilt by unmerged experiment code — it just becomes what every later
run in main reads. Concretely, break it when you are:

- **modifying an existing embedder in place** (`embedders/yamnet/embedder.py`) —
  otherwise you are editing main's copy, not a worktree copy:

  ```bash
  cp -rL embedders/yamnet embedders/yamnet_real && rm embedders/yamnet && mv embedders/yamnet_real embedders/yamnet
  ```

- **changing what extraction produces for an embedder name already on disk** —
  then break that set's `embeddings/`, or better, give the experiment a new
  embedder name so the question doesn't arise. Breaking `embeddings/` need not
  copy anything: replace the symlink with a real directory and symlink the
  subdirectories you are *not* rebuilding back to main.

A change gated so existing embedders take a byte-identical path (a new
`context_frames`-style flag defaulting to 0) is not a reason to break anything.

An extraction killed part-way is safe to relaunch: the ident directory it was
writing carries an `extraction.incomplete` marker until its fingerprint is
stamped, so the next run rebuilds it rather than reading a truncated product.

All code changes go in the worktree. Do not touch main's tracked files.

### 2. Make changes

Stage-local invariants live in `02_set/CLAUDE.md` and `03_train/CLAUDE.md`; read
the one you're editing. If the change touches `model.compile()`, smoke-test it
with `tools/smoke_model.py` before spending a CV on it.

### 3. Run the pipeline

From the worktree, using the path `setup_worktree.sh` printed:

```bash
cd .local/worktrees/<slug>
# Stage 2 — only if the embedder or extraction changed
02_set/main.py   --set medium --embedder yamnet --workers 2 --verbose
# Stage 3 — the whole CV, one model per rotating fold
03_train/main.py --name <modelname> --set medium --embedder yamnet --translation general -y
# or root main.py --model <name> ... to chain both in one process
```

**Launch detached — not foreground, not `run_in_background`.** The exact recipe
and why it is the only thing that works are in CLAUDE.md's "Running long jobs".

### The first fold is a gate. Do not step over it.

**After launching, you do exactly one thing: wait for the first fold's
`summary.json` to appear, and compute the ETA from its wall-clock time.** Not
the epoch rate, not the table below, not the kind of job. One finished fold x
the number of folds still to run x the number of chained configs. Until that
file exists you do not know how long the run takes, and **you may not say that
you do** — no ETA, no duration, no "this finishes inside my context", no
handoff-vs-Monitor decision, in your own reasoning or in anything you tell
Luke. An unmeasured ETA is not a rough answer, it is a fabricated one.

```bash
find models/<name>/folds -name summary.json -printf '%T@ %TH:%TM:%TS %p\n' | sort -n
```

Arming a Monitor is *not* the gate. A Monitor tells you when the run ends; the
gate is about deciding, before then, whether you should still be here at all.
Arm one only after the measured ETA says to.

**Every agent that has skipped this skipped it the same way**, so recognise the
move in yourself:

> "It's a frozen probe, the table says ~9 min, that's well inside an hour —
> I'll set a Monitor and keep working."

That is the failure, verbatim, and it is *reasoning from the table below.* The
table's rows are **configs, and your experiment is a new config** — that is
what makes it an experiment. It cannot be in the table.

The worked example is `shared-trunk-head` (2026-09-09), and note which way it
went wrong. The agent took an epoch-rate reading 40 s after launch with zero
folds finished, misread its own arithmetic into "~3x baseline per epoch", and
projected a 3-CV ladder at ~60 min — landing it exactly on the handoff line and
nearly buying a `HANDOFF.md` for a run that did not need one. The first fold
then measured 156 s, putting the ladder at **~30 min**: the hidden layer cost
almost nothing, and the pre-fold estimate was off by ~2x *upward*. It was
reported to Luke as if measured. **A guess that errs long is not the safe
direction** — it spends the turn on handoff machinery instead of results. The
table is for budgeting a run you have not launched; it is never evidence about
a run you have, and neither is anything you can compute before the first fold
lands.

| config | per epoch | one CV |
|---|---|---|
| frozen probe, 1024-d YAMNet (the baseline) | ~1 s | **~9 min** for 5 folds (2026-09-08) |
| trunk fine-tune, 12288-d + unfrozen layers | ~80 s | **~24 h** for 11 folds (2026-09-05) |

Two ends orders of magnitude apart, and **your run's place between them is
measured, never assumed.** "A frozen-probe CV is foreground-scale" is a
*conclusion the first fold licenses*, not a premise you may enter with — and
it is not a reason to skip the gate, because the gate is what tells you the run
is foreground-scale in the first place. Only a trunk fine-tune or an extraction
needs the full detached-and-hand-off treatment, but **every** run needs the
first-fold measurement, including the ones that turn out to take nine minutes.
It costs one `find`.

There is no `--runs` and no stage 4. One training call *is* the experiment.

**The shipped model is not trained by default**, and an experiment should leave
it that way. `folds_sx.csv` is built entirely from the rotations; the shipped
model contributes nothing to it. It is a deliverable, and belongs at the end of
the search. Its epoch count derives from the fold curves left on disk, so
training it later with the same `--name` gives an identical model:

```bash
03_train/main.py --name <exp> ... --skip-cv    # later, once <exp> wins
```

**Reruns resume silently** — always use a fresh `--name`, or pass `--clear`, or
delete `models/<name>/` by hand.

**If the run will outlast you, write a handoff doc — and decide that from the
first-fold measurement above, never from a guess at launch.** The mechanics of
detaching, and the Monitor script that covers the failure states, are in
CLAUDE.md's "Running long jobs"; the decision rule is here, and here only.
Under ~1 h left, set a completion `Monitor` and keep working — **do not write a `HANDOFF.md` for a run that finishes inside your own
context**, it is written, committed and never opened. That Monitor fires
**once, on a terminal state** (done / crashed / vanished), never per fold: you
armed it in order to stop watching, and every progress line it echoes is
context spent re-learning that. See CLAUDE.md for the loop that covers all
three states. Over ~1 h, commit a
**`HANDOFF.md`** in the worktree (that exact name, so a fresh agent opens the
one place it always is) and end your turn. Keep it short; it needs four things:

1. **The one-command progress check** — is it running, and how many of the 5
   folds are done (`pgrep -af <entry point>`, `find models/<name>/folds -name
   summary.json | wc -l`, `tail -3 <log>`).
2. **"If it's still running, STOP."** Say it explicitly: report `fold N/5` and
   quit. No tailing the log on a timer, no Monitor, no reading the rest of the
   repo — everything read while waiting is paid for twice.
3. **What to do when it finishes** — which model is the comparator (rarely
   `cv-baseline`; usually the matched control), which folds are too thin to
   trust, then notes.md → `log.jsonl` → commit.
4. **What to do if it died** — how to tell a self-healing restart from a real
   crash, and the exact relaunch command.

Also note anything a fresh agent would get wrong: symlinked caches it must not
delete, why the branch is based where it is.

### 4. Read the results

```
models/<modelname>/
├── folds_sx.csv                 ← everything: a row per fold, then a `total` row
└── folds/<fold>/
    ├── predictions.csv          ← every held-out frame; all of the above derives from it
    ├── summary.json             ← epochs, val_loss, frame counts, sens-monitor peaks
    └── sens_curves.svg          ← sens@fpr per epoch vs the epoch that got restored
```

`03_train/resummarize.py <modelname>` rebuilds `folds_sx.csv` for a model
trained before it existed — no retraining, no TensorFlow.

**Compare paired, per fold.** Both configs ran on the same folds, so the useful
comparison is the per-fold difference. Comparing two scalars throws that pairing
away, and a headline gap that vanishes under pairing was never a capability gap.

```bash
python tools/compare_folds.py <baseline model dir> <exp model dir>
```

Prints the per-fold delta table, the up/down count, and the two headline means.
Either argument can be a bare name under `models/` or a path — an experiment's
model usually lives in its worktree's own (unsymlinked) `models/` dir.

Two cautions that apply to every conclusion you write:

- **Fold-to-fold spread is not a confidence interval.** Training pools overlap
  ~90% across rotations, so fold models are correlated and the spread
  understates uncertainty about a genuinely new deployment.
- **Per-fold sensitivity is unreliable where there's little buzz.** In the
  quietest deployments it's ±0.25 or worse. A fold that swings hard may just be
  a fold with forty seconds of buzz in it. Check `buzz_frames` in
  `folds_sx.csv` before believing a per-fold delta.

There is no seed control anywhere in the pipeline, so run-to-run variation from
TF's nondeterministic init and shuffling is unmeasured. If a result lands close
to baseline, say it's within unquantified run noise rather than reaching for a
significance claim.

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
distinguishes an entry from the first era's fixed-split ones.

**Amending an entry.** Entries are not immutable — a later run can show that an
earlier conclusion, though correctly measured, claimed more than it measured.
Do not rewrite `conclusion` or the metrics: they are the record of what was
believed and what was observed. Instead set `trust` to what the entry now
deserves and add an `amended` field, dated, saying what changed and which run
supersedes it. `aves-probe` is the worked example. An entry nobody can trust
and nobody has marked is worse than a wrong one.

Every entry also needs a `trust` judgment, kept separate from the delta. This is
not a significance test — there's no seed control to build one from — just
whether a later reader should act on the number as measured, or dig first:

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
this field exists: the eval's same-label-neighbour bug materially inflated that
number, and only `context-embedder`'s honest rerun (`+0.022`, `clean`) revealed
the gap. Without a separate flag, a later agent skimming deltas would have built
on the wrong number.

`tools/log_entry.py` builds the line for you — it reads the `total` row's
sensitivity straight from each model's `folds_sx.csv` (no copying numbers by
hand) and fills in `branch`/`date`/`main_commit`:

```bash
python tools/log_entry.py \
  --name <slug> \
  --model <experiment model dir, e.g. .local/worktrees/<slug>/models/<modelname>> \
  --baseline-model models/cv_baseline \
  --hypothesis "..." --trust clean --conclusion "..." \
  --write   # omit to preview without appending
```

`main_commit` means **main's HEAD at the moment you log**, not the commit the
worktree branched from — those diverge whenever main moves while a long CV is in
flight (it did for `class-weight-fix`). Run `git rev-parse --short HEAD` in the
main checkout, not the worktree.

### 6. Commit worktree

```bash
git add -A && git commit -m "exp/<slug>: <what was tried and outcome>"
git push origin exp/<slug>
```

**Push it.** The branch is the only durable copy of your `notes.md` — the
worktree is disposable and this machine's disk is not backed up. Thirty-one
branches from the last two eras were deleted with their notes still only local,
and were recovered from dangling commits by luck.

`git add -A` picks up **every model's `folds_sx.csv` and per-fold
`summary.json`** — `models/.gitignore` re-includes those two, and only those two
(~4 KB per model). That is deliberate: `models/` is otherwise gitignored, so
pruning used to destroy the only copy of an experiment's per-fold numbers, which
is exactly what a later paired comparison needs. `standardize-blocks`' 0.261 —
cited in `IDEAS.md` as the ceiling — cannot be joined against today for that
reason. Weights, `predictions.csv` and plots stay ignored; if your experiment's
predictions are the point (surprisal, annotation triage), `git add -f` them
deliberately.

Then prune. **Check `git status` first**; uncommitted work in a worktree dies
with it.

```bash
git worktree remove --force .local/worktrees/<slug>
```

**Prune by size, not by reflex.** A worktree that only trained is ~50 MB and
costs nothing to keep; the "several GB" case needs a *broken* embeddings
symlink, and step 1 already routes a new embedder straight into the shared tree
so that rarely happens. For scale on `medium`: yamnet 351 MB, the largest single
cache (`yamnet_context`) 982 MB, all six embedders 2.6 GB, `large` 8.4 GB. Keep
the worktree if the run is unfinished and resumable, or if you want its
`predictions.csv` around; otherwise prune and rely on the branch.

Then stop. Do not merge into main. Do not proceed to another experiment.
You're done! Thank you!

## Restoring a worktree

```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

Its symlinked caches are not restored with it — rerun `setup_worktree.sh`'s
linking, or the run will re-extract from scratch.
