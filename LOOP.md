# Autoresearch Loop

One experiment is one hypothesis, one worktree, one CV against a matched control,
and one line in `log.jsonl`.

**This era (`cv-medium-v3`) started 2026-09-11, and nothing archived is a number
to beat.** Three things moved at the cutover: the annotations were revised (adding
a `_quiet` tag), scoring split into `sensitivity` and `sensitivity_exclquiet`,
and `--fixed-epochs` became the stopping rule, worth +0.031 to +0.040 by itself.
Old verdicts are leads, not answers: `temporal-context` was a clear negative in
era 1 and, rerun, the largest gain in era 2. Rerun rather than defer.

## Goal

Improve `ins_buzz` sensitivity at a fixed false-positive rate on held-out
deployments, with each deployment's threshold set on its own audio.

**The number is the `total` row's `sensitivity_exclquiet` at `fpr` 0.005 in
`models/<name>/folds_sx.csv`**: the plain mean over the rotating folds, each
tuned to 0.5% FPR on its own held-out audio. README's
[Reading the results](README.md#reading-the-results) explains every choice in
it: per-deployment thresholds, why quiet buzz leaves the score but not
training, the loudness tiers, and why 0.005.

**Say which loudness tiers moved; they are part of every result.** A lever that
lifts `loud` or `untagged` is a detection gain. One that only lifts `quiet` or
`faint` is a gain on buzz nobody was promised.

Two facts about this era's tiers:

- **Tagging is in progress.** On 2026-09-11, 1375 of 1842 buzz annotations had no
  loudness tag. The headline's denominator shrinks as tagging proceeds, so say
  which commit of the set a run used when it matters.
- **`_background` buzz is scored** (Luke's call, 2026-09-11) and is 51% of all
  scored buzz seconds, entirely in `1_29` and `53` (median duration 254 s,
  against ~1 s elsewhere). Those two folds' headlines largely measure continuous
  background buzz; read their `untagged` column before concluding anything
  about either.

## Baseline

**`cv-baseline-v3`, model `models/cv_baseline_v3/`: 0.330 excl. quiet (0.269
inclusive), 8/8 folds.** A bare linear probe on frozen YAMNet: one `Dense(15)`,
no dropout, no hidden layer, label smoothing 0.2, Adam 0.002, `medium`,
`general`, `--fixed-epochs` 400.

```bash
03_train/main.py --name cv_baseline_v3 --set medium --embedder yamnet --translation general -y
```

Its tiers at one shared threshold: `loud` 0.778 > `untagged` 0.350 >
`background` 0.322 > `quiet` 0.070 > `faint` 0.000. On audible buzz it is
already at 0.78, so **a lever that doesn't move `untagged` or `loud` is not a
detection gain**, whatever it does to the headline.

- **`1_95` is the known failure** at 0.052. Its threshold is -0.137, against -1.4
  to -1.9 elsewhere, set by jet flyovers.
- **`1_29` and `53` are the easiest folds on discrete buzz** (`untagged` 0.564 and
  0.651, against 0.250-0.402 elsewhere). Their middling headlines come from
  `background`.
- **The budget looks adequate.** Fold peaks scatter e116-398, and the last 21
  epochs sit -0.003 to +0.013 against e300-320.

The anchor is the simplest thing that could work, not where the last era
finished (0.321, `yamnet_aves` + `yamnet_context` with a 1024-wide head). Each of
that era's wins is its own experiment against it; check `log.jsonl` before
assuming one is untested. Some flags those leads need live off `main`:
`--hidden` on `exp/yamnet-aves-head-fixed`, and `--context-frames` on
`exp/yamnet-aves-context`. Branch from the one you need, or port the flag
deliberately.

## Constraints

- **Embedder: YAMNet, unless the experiment is the embedder.** Those need a
  re-extraction; run them deliberately.
- **Set: `medium`**, 8 rotating folds as of 2026-09-11. `lite` is for
  troubleshooting and `tiny` for smoke tests; draw no conclusions from either.
  A roster change or data revision ends the era: run
  `python tools/archive_era.py --slug <name>` (preflight; `--write` does it) and
  start a fresh log.
- **Never train or run a CV on `large`.** It is a one-time final confirmation
  after the structural search concludes, and only when Luke asks.
- **Pass `--translation general`.**
- **Every arm uses `--fixed-epochs` at the same N.** A different N is a different
  rule. `--early-stop` exists only so the archived era reproduces;
  `config_model.json`'s `epoch_rule` records which rule a run used.
- **Change one thing, against a matched control**: same rule, same budget, your
  one variable. The era anchor is rarely the right comparator.
- **Prefer structure to hyperparameters.** Annotation keeps moving the data;
  hyperparameter results don't survive that, large one-directional structural
  results usually do, and diagnostics survive absolutely. Diagnostics also tell
  Luke what to annotate next, which is the standing bottleneck.
- **Augmentation has hurt so far.** Read the cross-fold rule in
  `03_train/CLAUDE.md` before building any.
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
  against its eval SD, and confirm a large hard-fold gain with one repeat run.

Do not modify `01_annotate/`, `translations/`, any set's `build.R`, or
`03_train/metrics.py`.

## Experiment lifecycle

### 0. Orient

Read `log.jsonl` (this era) and `IDEAS.md`. `archive/` holds the 59 runs before
this era. Its dead ends (bandpass, mel masking, MLP heads, L2, handcrafted
frequency features) are likelier to hold than its wins, **but all were measured
on YAMNet**, so no head or regularisation verdict carries to a new embedder.
Detail on an archived run is in `archive/<era>/notes/<slug>.md`, and its code on
`exp/<slug>` or `refs/archive/<slug>`.

### 1. Create a worktree

```bash
bash setup_worktree.sh <slug>
```

Commit anything the experiment needs first; uncommitted changes don't reach the
worktree. All code changes go in the worktree, never in main's tracked files.

The script symlinks `embedders/*` and each set's `audio/` and `embeddings/` back
to main, so extraction is shared. Keep it that way: a new embedder directory or
`--embedder` name goes straight into the shared tree. The cache is keyed by
`(set, embedder name)`, and its fingerprint covers annotations, not code. So
break a symlink only when your run would overwrite something under an existing
name:

- **Modifying an existing embedder in place:**
  `cp -rL embedders/yamnet embedders/yamnet_real && rm embedders/yamnet && mv embedders/yamnet_real embedders/yamnet`
- **Changing what extraction produces under a name already on disk:** use a new
  embedder name instead.

A change gated so existing embedders take a byte-identical path (a flag that
defaults off) needs no break. A killed extraction is safe to relaunch.

### 2. Make the change

Write the Hypothesis section of `notes.md` (template in step 5) before touching
code. Read the stage CLAUDE.md for what you edit. If the change touches
`model.compile()`, smoke-test it with `tools/smoke_model.py` first.

### 3. Run it

From the worktree:

```bash
# stage 2, only if extraction changed
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder <emb> --workers 1
# stage 3
tools/launch_job.sh train.log -- 03_train/main.py --name <name> --set medium --embedder yamnet --translation general -y
```

Then wait: its notifier pings you on each fold, an error, the job's end, and
every 50 min otherwise. There is nothing to arm.

**You don't know how long a run takes until its first fold pings**, whose
timestamps give the measured time per fold. Don't state a duration before
them, from an epoch rate or from the table below: your experiment is a new config,
so it isn't in the table. The table is for budgeting a run you haven't launched.

| config | per epoch | one CV |
|---|---|---|
| frozen probe, 1024-d YAMNet, early-stopped | ~1 s | ~9 min, 5 folds |
| frozen probe, 1792-d, `--fixed-epochs 150` | ~1 s | ~13 min, 5 folds |
| 1792-d + 1024-wide hidden, `--fixed-epochs 150` | ~7 s | ~92 min, 5 folds |
| trunk fine-tune, 12288-d + unfrozen layers | ~80 s | ~24 h, 11 folds |

At the default 400 epochs, a CV costs roughly 2.7x the 150-epoch rows.

- **Reruns resume silently.** Use a fresh `--name`, pass `--clear`, or delete
  `models/<name>/`.
- **Leave the shipped model untrained.** `folds_sx.csv` comes entirely from the
  rotations. `--skip-cv` with the same `--name` later builds an identical one.
- **`HANDOFF.md` is only for a session that must end while its job runs.** The
  heartbeat pings keep a waiting session alive, so a slow run is no reason to
  write one. If you do, commit it in the worktree with four things: the
  one-command progress check, plus `tools/notify_job.sh <pid> --log <log>` to
  get the job's pings; "if it's still running, report progress and stop"; what to do
  when it finishes (the comparator, then steps 4-5); and what to do if it died,
  including the exact relaunch command.

### 4. Read the results

```bash
python tools/results.py <matched control model> <experiment model>
```

It prints `notes.md`'s Results section: the per-fold table with each delta's
eval-sampling SD, the headline ± SD, the inclusive figure, and the tier deltas.
`tools/compare_folds.py` and `tools/eval_sampling_sd.py` are its parts, if you
need more detail. A blank ± SD means the two models' frame grids differ (for
example, an AVES-hop embedder against YAMNet), and no paired SD exists.

**Read `docs/judging-results.md` before writing an interpretation.** It holds the
evidence behind these three rules:

- **The headline mean is the estimate** (delta SD ~0.012). A single fold is an
  investigation: report its delta with its SD, and "unsure" is a normal verdict.
- **The hard folds (`1_150`, `1_95`, `willard`) are the target.** A gain
  concentrated there is the result, not thin-fold noise. Confirm it with a
  repeat run rather than discounting it.
- **Ask whether the treatment could have selected on the reported metric.**

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

If you used an idea, delete its section from `IDEAS.md`. Then, from main:

```bash
tools/finish_experiment.sh <slug> --summary "<what was tried, and the outcome>" \
  --model .local/worktrees/<slug>/models/<name> --baseline-model <matched control> \
  --hypothesis "..." --trust clean --conclusion "..." [--commit-also IDEAS.md]
```

It commits and pushes `exp/<slug>` (the only durable copy of `notes.md`), then
appends the `log.jsonl` line in main and commits it. Keep `conclusion` brief,
since it only points later agents at where to dig, and put the tier reading in
it. Pass `--commit-also IDEAS.md` only if `git diff IDEAS.md` shows nothing but
your deletion.

`trust` judges the measurement, separately from the delta:

- **`clean`:** nothing about how it was measured should discount it.
- **`caveated`:** a specific measurement defect means the direction is probably
  right but the size isn't. "It rests on one fold" is not such a defect; that
  goes in `conclusion`.
- **`artifact`:** the setup, not the effect, moved the number enough that
  comparing it at face value would mislead.

To amend an earlier entry, don't rewrite its conclusion or metrics. Set `trust`
and add a dated `amended` field saying what changed and which run supersedes it
(`aves-probe` is the example).

### 6. Stop

Don't merge into main, and don't prune the worktree. Pruning has repeatedly
destroyed the only copy of something: an untracked `embedder.py` from a `clean`
result, and 31 branches' notes. `models/.gitignore` re-includes each model's
`folds_sx.csv` and per-fold `summary.json`, so paired comparisons survive; use
`git add -f` on predictions only if they are the point. If a worktree holds a
multi-GB unshared cache, remove just that directory.

Then stop: one experiment per session.

## Restoring a worktree

```bash
git worktree add .local/worktrees/<slug> exp/<slug>
```

Then redo `setup_worktree.sh`'s linking, or the run re-extracts from scratch.
