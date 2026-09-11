# Autoresearch Loop

> **The slate was cleared on 2026-09-11.** The 24 runs of the previous era are
> in `archive/2026-09-08_cv-medium-v2/`, with a README stating what made them
> comparable and what ended that; 59 more are in the two eras before it.
> **Nothing in there is a number you can beat** — three things moved at once at
> this cutover, and any one of them moves every float:
>
> 1. **The annotations were revised again**, including a new `_quiet` tag.
> 2. **Scoring split in two.** `sensitivity` counts every buzz frame;
>    `sensitivity_exclquiet` drops the frames whose buzz is only `_quiet`-tagged.
> 3. **The stopping rule changed.** `--fixed-epochs` is now the default and
>    early stopping is opt-in — worth +0.031 to +0.040 on its own, i.e. more
>    than most levers being tested.
>
> **Old verdicts are leads, not settled answers.** `temporal-context` was logged
> as a clear negative in era 1 and, rerun as `context-stack`, was the largest
> gain in era 2. A verdict inverted on an eval change alone. Read `IDEAS.md` for
> the distilled version and the archived era's "Reading these forward", and
> **rerun rather than defer**.

Read `README.md` first — [Reading the results](README.md#reading-the-results)
is where the metric choices below come from.

## Goal

Improve `ins_buzz` sensitivity at a fixed false-positive rate on **held-out
deployments**, with each deployment's threshold set on its own audio.

**The number: the `total` row's `sensitivity_exclquiet` at `fpr` 0.005, from
`models/<name>/folds_sx.csv`.** Every rotating fold is tuned to 0.5% FPR on its
own held-out audio; the primary figure is the plain mean of those
sensitivities, each deployment counted once. A training run prints it, and
`tools/log_entry.py` reads it straight off the file.

**Quiet buzz is out of the equation.** A buzz whose annotation carries a
`_quiet` tag is really there, but it needs audio filtering and an expert ear to
perceive — well below what an operator could reasonably ask buzzdetect for. So
missing one is not a false negative and catching one is not a credit: those
frames leave the *scored* set entirely.

They do **not** leave training. They are ordinary `ins_buzz` positives in the
training pool, because calling faint buzz a negative would teach the model that
faint buzz is background — worse than either scoring choice.

### Read the loudness tiers. They are part of every result, not an add-on.

`folds_sx.csv` carries the whole family at **one identical threshold per
fold** — restricting which positives count leaves the negative pool and the FPR
sweep untouched, so every column below is directly comparable to every other:

| column | what it counts |
|---|---|
| `sensitivity_exclquiet` | **the headline** — every buzz frame but the quiet-only ones |
| `sensitivity` | the inclusive companion — every annotated buzz frame |
| `sensitivity_loud` / `_normal` / `_untagged` / `_background` / `_quiet` / `_faint` | one tier each, with `<tier>_frames` counts |

Every training run prints the tier block, and `tools/log_entry.py` records the
headline plus the inclusive figure (`sens_at_fpr0.005_persite_inclquiet`).

**What the tiers are for, and why they are not optional.** The standing
question about the near-chance folds is whether they are hard *because* their
buzz is faint, or hard on audible buzz too. Those are different problems with
different fixes, and no headline can tell them apart. The tier row answers it
directly, for free, on every run. So when you write up a result, say **which
tiers moved**: a lever that lifts `loud` and `untagged` is a detection gain;
one that only lifts `quiet`/`faint` is a gain on buzzes nobody was promised.

**Tier semantics.** `train_utils.buzz_tier` takes the **maximum** over a
frame's buzz labels under `faint < quiet < background < untagged < normal <
loud` — a frame is only as hard as its most audible buzz. `untagged` sits above
the faint end so an unknown is never mistaken for a faint one, and below
`normal`/`loud` so an unknown never promotes a frame past a known one.

**The vocabulary is what the annotations actually use**, surveyed 2026-09-11
over 1842 buzz annotations: `_quiet` (406), `_loud` (39), `_background` (14),
`_faint` (8), and **1375 with no loudness suffix at all**. There is no
`_normal`. Note `_high`/`_medium`/`_low` are **pitch**, not loudness — so
`ins_buzz_medium_quiet` is a medium-pitch buzz that is quiet.

**`train_utils.TIERS_EXCLUDED_FROM_HEADLINE` is the one line that decides which
tiers leave the score**, and it is revisable for free: `predictions.csv` stores
the observed tier rather than a scored/not-scored boolean, so changing that line
and rerunning `03_train/resummarize.py` re-scores every model on disk with no
retraining. Currently `faint` and `quiet`.

**`_background` is scored as buzz, and you should know what that costs.** Luke's
call, 2026-09-11. Those 14 annotations have a **254 s median duration** against
~1 s for every other tier, and they are **51% of all scored buzz seconds** —
concentrated entirely in `1_29` (11) and `53` (3). So those two folds'
sensitivity is substantially a measure of *continuous background buzz*, not of
discrete pollination events. Read their per-fold numbers with that in mind, and
check `sensitivity_background` against `sensitivity_untagged` before concluding
anything about either fold.

**Interim state (as of 2026-09-11):** tagging is in progress — 1375 of 1842
buzz annotations carry no loudness suffix yet. The end state is a tag on every
buzz, across all 24 subsamples of every ident in `01_annotate/Even Sample`.
Two consequences. The headline's denominator **keeps shrinking as tagging
proceeds**, so an early-era run is not cleanly comparable to a later one on that
column — say which commit of the set a run used. And `untagged` shrinking
toward zero is the progress bar: when it is empty, the tier split is the real
decomposition.

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

**`cv-baseline-v3` in `log.jsonl`, model at `models/cv_baseline_v3/`:
0.330 mean sens@fpr0.005 excl. quiet (0.269 inclusive), 8/8 folds.**
Join against its `folds_sx.csv` for a paired per-fold comparison.

The anchor is the simplest thing that could work: **a bare linear probe on
frozen YAMNet** — one `Dense(15)` straight off the 1024-d embedding, *no
dropout*, no hidden layer, label smoothing 0.2, Adam 0.002, `medium`,
`general`, `--fixed-epochs`. Everything else in the pipeline is an addition to
be justified against it.

```bash
03_train/main.py --name cv_baseline_v3 --set medium --embedder yamnet \
                 --translation general --verbose -y
```

**Its tier breakdown is the most useful thing in it**, and it is monotonic in
audibility at one shared threshold: `loud` 0.778 (122 frames, 5/8 folds) >
`untagged` 0.350 > `background` 0.322 > `quiet` 0.070 > `faint` 0.000. Two
readings follow. The scoring split is vindicated — quiet buzz is caught 7% of
the time and faint never, so counting them as misses measured a target the tool
was never promised to hit. And the model is not broadly weak: on buzz an
operator would call audible it is already at 0.78. **A lever that does not move
`untagged` or `loud` is not a detection gain**, whatever it does to the
headline.

`1_95` is still the known failure at **0.052**, with a threshold of -0.137
against -1.4 to -1.9 everywhere else — the jet-flyover pathology, carried
intact across the cutover.

**On the budget**: fold peaks scatter 116-398 and
`mean(last 21 epochs) - mean(e300-320)` is -0.003 to +0.013, so 400 is not
obviously short. This is *not* last era, where every fixed-budget run was still
climbing at its cap. The ladder is still worth one run to settle the era's
budget, but it is no longer urgent — which moves it below the win
re-verifications in priority.

That anchor is deliberately *not* where the last era finished (0.321, on
`yamnet_aves` + `yamnet_context` concatenated with a 1024-wide hidden head).
Building the new era on that config would carry three unverified assumptions
into every number. The era's first few loops should instead **re-verify the
last era's wins against this anchor, one at a time** — the concatenated
embedder, the wide hidden head, and dropout itself are now three separate
experiments, not premises.

Two things to know before quoting the old numbers at all:

- **The archived `cv-baseline` 0.218 is not comparable to this anchor** — it
  early-stopped, it had `Dropout(0.2)`, and it ran on the pre-revision
  annotations. Expect the fixed-budget change alone to be worth +0.02 to +0.04,
  chiefly at `1_150`.
- **The flags the era's leads need are not all in `main`.** `--hidden` lives on
  `exp/yamnet-aves-head-fixed`, and `--context-frames` plus the fused
  `yamnet_context_aves` extraction on `exp/yamnet-aves-context`. A
  re-verification experiment should branch from the one it needs, or port the
  flag deliberately, rather than rediscovering that main can't express the
  config.

Worth running early alongside the anchor: the **`binary` translation control**.
It is why the endpoint is per-deployment rather than pooled, and it has never
been rerun cleanly. Two confounds unrelated to the taxonomy: `ins_buzz`'s
positive weight moves ~5.4x (`build_weights` puts the class *count* in every
denominator), and `val_loss`'s composition goes from ~1/15 buzz to ~1/2 — the
second disappears under a fixed budget, which is a reason to run it now.

## Constraints

- **Embedder: YAMNet only, unless the experiment *is* the embedder.** Those need
  a re-extraction and should be run deliberately, not incidentally.
- **Set: `medium`** — day-long annotated recordings across a diversity of
  environments, **8 rotating folds** as of the 2026-09-11 revision (up from 5;
  new are `wooster/2024-07-26/1_143`, `2025-07-03/1_37`, `2025-08-12/1_114`).
  `lite` is for troubleshooting and `tiny` for
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
- **The stopping rule is `--fixed-epochs`, and every arm of a comparison must
  share it.** It is the default: train exactly N epochs, no early stopping, no
  restore-best, no epoch selection of any kind. `--early-stop` restores the
  pre-2026-09-11 rule and exists only so the archived era reproduces. The rule
  is worth **+0.031 to +0.040** on its own — more than most levers being tested
  — so a run under one rule tells you nothing about a run under the other, and
  `config_model.json`'s `epoch_rule` records which each used.

  **The budget itself is unsettled.** 400 is the default because every
  fixed-budget run from the last era was still rising at its cap, and the one
  400-epoch run cost 0.003 against its own peak at e234. Whether a wider head
  keeps gaining past 250 was never tested. **Run the budget ladder early** — it
  is cheap, and it sets the number every later run is scored at.
- **The head is a bare linear probe, and that is now the anchor, not a
  hardcoded premise.** `Dropout -> Dense(n_classes)` with `label_smoothing=0.2`
  and Adam 0.002 was the fixed config through 2026-09-11; dropout is now
  `--dropout`, defaulting to **0.0**. Those values were tuned on YAMNet, whose
  embedding is 89.6% exact zeros, non-negative, and already linearly separable
  — it is the penultimate layer of a supervised classifier whose 521 AudioSet
  classes include `Buzz` and `Bee, wasp, etc.`. Input dropout on a code like
  that is mild; on a dense, signed, zero-centred code it is heavy
  multiplicative noise, and with no weight decay a wide distributed
  representation overfits the training sites instead of transferring.

  **So when an embedder swap comes back negative, that number is
  `embedder + this head`, and the head may be doing the damage.** Before logging
  an embedder verdict, sweep the readout offline against
  `03_train/metrics.py:sens_at_fpr` on the cached embeddings — no Keras, no
  extraction, seconds per configuration — and say which part of the loss you
  measured. `aves-probe` (2026-09-09) did not do this, logged
  0.218 -> 0.074 as an embedder verdict, and an offline L2 sweep on the *same*
  embeddings then recovered most of it.

  This is not licence to tune. It is one diagnostic sweep to attribute a
  negative before it becomes a verdict, and it stays offline unless it changes
  the reading.
- **Err against hyperparameter tuning.** We're looking for structural gains;
  hyperparameters can be tuned in one sweep once a good structure is found.
  **Exception: a new embedder does not inherit the probe config's validity** —
  see the bullet above.
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

If you use an IDEA, **delete its section** from `IDEAS.md` after testing —
delete it, do not convert it into a `## Closed:` writeup. The verdict, the
mechanism and the "don't rerun this" belong in your `log.jsonl` entry and your
`notes.md` on the branch, which is where a later agent looks for them. `IDEAS.md`
reached 1047 lines, 46% of it closed material, before this was enforced; it is a
queue, and every line of it is re-read uncached by every agent.

### 1. Create worktree

```bash
bash setup_worktree.sh <slug>
```

Confirm the sanity check passes. Uncommitted changes to tracked files won't
appear in the worktree — commit or stash first if they matter.

The script symlinks `embedders/*` and each set's `audio/`+`embeddings/` back to
main, so extraction is shared. **Keep it that way.** A cache written through the
symlink lands in the shared tree and is there for the next experiment; a
worktree-local copy is only ever seen by this run. Adding a *new* embedder
directory, and extracting under a
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
03_train/main.py --name <modelname> --set medium --embedder yamnet --translation general -y --verbose
#   the stopping rule is --fixed-epochs 400 by default; match your control's budget exactly
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

Arming a Monitor is *not* the gate. A Monitor tells you how the run is going;
the gate is about deciding, before then, whether you should still be here at
all. Arm one only after the measured ETA says to.

**What the ETA is for — read the gap, not the total.** The prompt cache's TTL
(~1 h) refreshes every time it is read, so each fold event renews it and a
chain of them keeps one agent warm indefinitely. What forces a handoff is a
single silence longer than the TTL. So the number you compare against is the
**per-fold time**, not the length of the run: under ~50 min per fold, a
per-fold Monitor carries the experiment however many hours it takes; over, no
Monitor can help and you hand off. A 4 h CV landing a fold every 20 min is a
run you stay with.

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
| frozen probe, 1024-d YAMNet, early-stopped | ~1 s | **~9 min** for 5 folds (2026-09-08) |
| frozen probe, 1792-d, `--fixed-epochs 150` | ~1 s | **~13 min** for 5 folds (2026-09-10) |
| 1792-d + 1024-wide hidden, `--fixed-epochs 150` | ~7 s | **~92 min** for 5 folds (2026-09-10) |
| trunk fine-tune, 12288-d + unfrozen layers | ~80 s | **~24 h** for 11 folds (2026-09-05) |

**Note the top two rows and the era's new default.** A fixed budget runs every
epoch it is given — there is no early exit — so a CV at `--fixed-epochs 400` is
roughly 2.7x the 150-epoch rows above, and the budget ladder is the single
biggest lever on how long your run takes. That is arithmetic for *budgeting*,
not a measurement of your run.

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

**A handoff is a last resort, not a precaution — and the trigger is measured,
never guessed at launch.** The mechanics of detaching, and the Monitor script,
are in CLAUDE.md's "Running long jobs"; the decision rule is here, and here
only. **Do not write a `HANDOFF.md` for a run you can stay with**: it is
written, committed and never opened, and the agent that opens one pays full
uncached price to re-read this file, CLAUDE.md, `log.jsonl` and `IDEAS.md`, then
reorients into an experiment it did not design. That is the expensive path, not
the wakeups it saves.

So: **arm a Monitor that fires once per fold plus on the terminal states, with
a ~50-minute timeout built in.** The fold events double as progress and as the
cache keepalive; answer each in one line and go back to waiting. If the timeout
fires, the cache is lapsing whatever you do — *that* is the moment to commit a
**`HANDOFF.md`** in the worktree (that exact name, so a fresh agent opens the
one place it always is) and end your turn. Keep it short; it needs four things:

1. **The one-command progress check** — is it running, and how many of the 5
   folds are done (`pgrep -af <entry point>`, `find models/<name>/folds -name
   summary.json | wc -l`, `tail -3 <log>`).
2. **"If it's still running, STOP."** Say it explicitly: report `fold N/5` and
   quit. No tailing the log on a timer, no Monitor, no reading the rest of the
   repo — everything read while waiting is paid for twice.
3. **What to do when it finishes** — which model is the comparator (rarely
   the era anchor; usually the matched control), which folds are too thin to
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

Three cautions that apply to every conclusion you write:

- **Your comparator must have run under the same rule and the same budget.**
  This era starts with one rule — `--fixed-epochs`, no selection at all — so
  the trap is narrower than last era's, but it is still there: **a different
  `--fixed-epochs` N is a different rule.** Every fixed-budget run on disk from
  last era was still rising at its cap, so N moves the score. Run a **matched
  control** — same rule, same N, your one variable — and compare to that.
  A run under `--early-stop` is not a comparator for anything in this era; it
  is worth +0.031 to +0.040 less on its own. Background:
  `exp/pairwise-rank:notes/new-era-audit.md` and the archived era's README.
- **Say which loudness tiers moved.** The headline is
  `sensitivity_exclquiet`, but `folds_sx.csv` carries the inclusive figure and
  a per-tier breakdown at the same threshold, and the decomposition is usually
  the more informative half of a result. A lever that lifts `loud`/`normal` is
  a detection gain; one that only lifts `quiet` is a gain on buzzes no operator
  was promised; one that only lifts `untagged` may be about annotation
  coverage. `tools/compare_folds.py` and `tools/log_entry.py` read the headline
  column; put the tier reading in `conclusion`, where a later agent will find
  it.
- **Fold-to-fold spread is not a confidence interval.** Training pools overlap
  ~90% across rotations, so fold models are correlated and the spread
  understates uncertainty about a genuinely new deployment.
- **Per-fold sensitivity is less certain where there's little buzz — and every
  figure in this bullet is from the 5-fold roster, so re-measure before quoting
  any of it.** Folds no longer differ in *audio*:
  snip equalization has `frames_val` within ~1.5x (4947-7617) and `neg_frames`,
  the negative sample the threshold rests on, at 24-35 across all five. What
  cannot be equalized is buzz *density*, which is ecology — `buzz_frames` still
  spans 146 to 2144 (14.7x), because annotating more of a quiet site buys mostly
  negatives.

  **The "±0.25 or worse" figure this bullet used to carry was ~7x too large.**
  It was an 11-fold-roster number from before equalization. Re-measured
  2026-09-09 (`tools/eval_sampling_sd.py`, seconds, no training): per-fold
  bootstrap SD is **0.010-0.037**, worst case `1_150` at 0.037, headline SD from
  eval sampling 0.007 to 0.012 on last era's runs. Run the tool
  rather than quoting any figure from these docs — it is cheap and the roster
  moves.

  Two limits on that number. The bootstrap resamples frames independently while
  frames inside one buzz event are correlated, so it is a **floor**. And it
  measures eval sampling only: `1_150` moved **0.055** between two identical runs
  against a 0.012 eval-sampling SD, so its run-to-run noise is dominated by
  **training stochasticity**, not by its buzz count. More annotation will not fix
  that; a seed control or seed averaging would.

### The thin-fold caution is about magnitude. It is not permission to discount a hard-fold gain.

**The hard folds are the target.** The endpoint is a *new* deployment, and a new
deployment may well be a quiet one. `1_29` sits at ~0.43 whatever we do; a lever
that moved only the rich folds would be close to worthless, and one that lifts a
near-chance deployment is the thing this project is for. So when a gain is
concentrated in `1_150`, `1_95` or `willard`, that concentration is the
**result** — do not write it up as "the headline rests on one thin fold" and
discount it. Read a delta table as: which way did the hard folds move, and are
the losses elsewhere material? A -0.005 does not cancel a +0.068, and calling
that "3 up, 2 down, inconclusive" is bad accounting.

**The specific error to avoid — a repeat spread bounds noise WITHIN a treatment,
not a difference BETWEEN two.** `1_150` moved 0.007 → 0.062 across two identical
baseline runs, and that number is quoted all over this repo. It is the right
caution for reading two `val_loss` runs against each other. It says nothing
about a difference produced by a change in configuration, and using it that way
will talk you out of every hard-fold result the loop is capable of finding.

**The worked example below has since been withdrawn — the lesson stands, its
example does not.** `monitor-leakage` (2026-09-09) showed `context-monitor`'s
`1_150` gain was selection leakage: the held-out fold is its own early-stopping
monitor, so a `val_sens` run reports `max` over epochs of the statistic it is
scored on, and the non-overlapping group split below is a comparison of *a
point on a curve* against *the maximum of that same curve*. Re-scored at a
cross-fold epoch the two groups overlap completely. Read what follows as the
method for *how to check* a hard-fold gain — grouping the era's runs by the
variable you changed is exactly what settled it, in both directions — and note
the check it does not perform: **ask whether the treatment could have selected
on the reported metric.** Keep the accounting lesson; it is right, and the
hard folds are still the target.

`context-monitor` (2026-09-09) is the worked example, and it was logged
`caveated` before being amended to `clean` the same day. The agent saw +0.049
carried entirely by `1_150`, reached for the 0.007–0.062 spread, and flagged the
size as untrustworthy. Splitting all 17 runs of the era by which monitor they
used takes about two minutes and settles it:

| monitor | `1_150` | n |
|---|---|---|
| `val_loss` | 0.007, 0.014 x5, 0.021 x4, 0.027, 0.034, 0.041, 0.062 | 13 |
| `val_sens` | 0.089, 0.158, 0.171, 0.219 | 4 |

**The groups do not overlap.** Every `val_sens` run beats every `val_loss` run
on that fold. The within-`val_loss` spread was never evidence about the gap
between the groups.

**What to do instead of discounting.** Confirm a large hard-fold gain with a
**repeat run** — there is no seed control, so a rerun is an independent draw and
it costs one CV. Frame it as confirmation, not as a check on a suspect number.
And before you reach for a repeat spread at all, ask whether the runs you are
comparing sat in the same treatment group; if they did not, go and group the
era's runs by the variable you changed and look, as above.

**`trust` is what a later agent scans before building on your result.** Flagging
a hard-fold gain `caveated` on magnitude grounds tells the next agent to
discount exactly the movement the loop exists to produce. Reserve `caveated` for
a defect in how the number was *measured*, and put "this rests on one fold, here
is the group split" in `conclusion`, where it belongs.

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
  --baseline-model models/<your matched control> \
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
is exactly what a later paired comparison needs. The archived `standardize-blocks`' 0.261 —
cited in `IDEAS.md` as the ceiling — cannot be joined against today for that
reason. Weights, `predictions.csv` and plots stay ignored; if your experiment's
predictions are the point (surprisal, annotation triage), `git add -f` them
deliberately.

**Do not prune the worktree.** It was pruned here by reflex for a long time and
it kept costing real things: `yamnet_aves`'s `embedder.py` (a logged `clean`
result, a live IDEAS lead) was lost with its worktree because a gitignore gap
meant `git add -A` never captured it, and thirty-one branches' `notes.md` went
with theirs before the push step was enforced. A worktree that only trained is
~50 MB. Leaving it costs nothing you will notice; removing it has repeatedly
destroyed the only copy of something. The branch is the durable record, but the
worktree is a free second copy and a resumable checkout — keep it.

The one real disk case is a *broken* embeddings symlink (several GB), and step 1
routes a new embedder into the shared tree specifically so that does not happen.
If a worktree ever does hold a multi-GB unshared cache and you need the space,
remove just that directory, not the worktree:

```bash
du -sh .local/worktrees/<slug>/*        # confirm what is actually large
rm -rf .local/worktrees/<slug>/02_set/sets/<set>/embeddings/<embedder>
```

Then stop. Do not merge into main. Do not proceed to another experiment.
You're done! Thank you!

## Restoring a worktree

```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

Its symlinked caches are not restored with it — rerun `setup_worktree.sh`'s
linking, or the run will re-extract from scratch.
