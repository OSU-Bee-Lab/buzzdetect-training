# buzzdetect-training

Trains Keras classifiers that detect insect buzzing in passive acoustic
recordings. Raw audio and annotations go in; a shallow probe (dropout + one
dense layer) over a frozen audio embedder — YAMNet by default — comes out.

This file is the operator's guide: how to run the pipeline and how to read what
it produces. `CLAUDE.md` is the orientation file for agents editing the code.

---

## Contents

- [Setup](#setup)
- [The pipeline](#the-pipeline)
- [Quick start](#quick-start)
- [Stage 1 — annotate](#stage-1--annotate)
- [Stage 2 — build a set and extract](#stage-2--build-a-set-and-extract)
- [Stage 3 — train](#stage-3--train)
- [Reading the results](#reading-the-results)
- [Tools](#tools)
- [Design: why folds are deployments](#design-why-folds-are-deployments)
- [Troubleshooting](#troubleshooting)

---

## Setup

```bash
conda env create -f environment.yml     # creates 'buzzdetect-train'
conda run -n buzzdetect-train python --version
```

Every Python command below assumes that prefix. R is run from RStudio (there's
an `.Rproj`) or via `Rscript`; the R side needs `dplyr`, `stringr`, `tidyr`,
`here`, and `withr`.

Data lives outside git (see `.gitignore`):

| Path | Holds |
|---|---|
| `audio/` | raw training audio, laid out so `audio/<ident>.<ext>` resolves |
| `01_annotate/<effort>/data/` | that effort's raw annotation files |
| `02_set/sets/<set>/audio/`, `.../embeddings/` | snips, framed-audio cache, embeddings |
| `models/<name>/` | model artifacts |

An **ident** is a single recording, named by its path under `audio/` without
extension (e.g. `JamesU - MustardBumbler/1_29/240903_1319`). It is the join key
across every stage.

A **fold** is a deployment: one recorder, one site, one period — typically the
ident's parent directory.

---

## The pipeline

```
audio/ + 01_annotate/<effort>/data/
        │
        │  1.  01_annotate/MAKE.R          → per-effort annotations_combined.csv + folds.csv
        ▼
02_set/sets/<set>/build.R                  → set-level annotations.csv + folds.csv
        │
        │  2.  02_set/main.py              → snips → framed audio → embeddings/<embedder>/raw/<fold>/<ident>/
        ▼
03_train/main.py                           → models/<name>/  (CV folds + shipped model)
        │
```

Stages are normally run one at a time. Root `main.py` chains 2→3 for
convenience.

---

## Quick start

Train a model on the checked-in `lite` set with YAMNet:

```bash
cd /path/to/buzzdetect-training

# extract (writes snips + embeddings under 02_set/sets/lite/)
conda run -n buzzdetect-train python 02_set/main.py \
  --set lite --embedder yamnet --workers 2

# train
conda run -n buzzdetect-train python 03_train/main.py \
  --name my_first_model --set lite --embedder yamnet --translation general
```

Or both at once, from the project root:

```bash
conda run -n buzzdetect-train python main.py --model my_first_model --set lite
```

Results land in `models/my_first_model/`. Everything numeric is in
`folds_sx.csv`: a row per held-out fold, then a `total` row.

---

## Stage 1 — annotate

Each **annotation effort** is a directory under `01_annotate/` with its own
`.Rproj` and a `combine.R`. `combine.R` reads that effort's raw annotation
files (Audacity label tracks, via `01_annotate/utils.R`), fills unannotated gaps
with `ambient_background`, assigns each ident to a fold, and writes two files
next to itself:

- `annotations_combined.csv` — `ident, start, end, label`
- `folds.csv` — `ident, fold, role`

Run every effort:

```bash
Rscript 01_annotate/MAKE.R
```

`MAKE.R` walks `01_annotate/`, treats every directory holding a `combine.R` as
an effort, and sources each one with the working directory set to that effort.
Errors in one effort are caught and reported, not fatal to the rest; at the end
it warns about any effort that didn't leave both CSVs behind.

To add an effort: create the directory, drop raw annotations in `annotations/`,
and write a `combine.R` that emits those two CSVs — fold assignment included, in
that same file. Copy the closest existing effort; `01_annotate/Even Sample/` is
the current model, and its `README.md` documents the sampling design.

Most efforts' raw annotation files are tracked in this repo under
`<effort>/annotations/`, so one `git pull` moves code and annotations together.
The exceptions — efforts with their own annotation repo, and the one effort not
reproducible from a fresh clone — are listed in `01_annotate/data_sources.json`;
clone those where it says before running `MAKE.R`. `MAKE.R` aborts an effort
with an empty `annotations/` and tells you its data is not synced.

`role` is per-fold and picks how the fold is used at training time:

| role | trained on | scored |
|---|---|---|
| `train` | always | never |
| `rotate` | every CV iteration but its own | on its own iteration |
| `holdout` | never | always |
| `exclude` | never | never |

Rationale in [Design](#design-why-folds-are-deployments). All rows sharing a
fold must agree on its role — `03_train` errors otherwise.

---

## Stage 2 — build a set and extract

### 2a. Build the set

A **set** is a named selection of annotations plus any relabelling on top of
them. Sets live in `02_set/sets/<name>/`, each defined by a `build.R`:

```bash
cd 02_set/sets/medium && Rscript build.R   # then large / tiny / lite as needed
```

`medium` is the base set and the only one built from `01_annotate`: its
`build.R` concatenates the chosen efforts' `annotations_combined.csv` and
`folds.csv`, applies label collapsing (e.g. every `ins_buzz*` variant →
`ins_buzz`), checks that no ident lands in two folds, and writes:

- `annotations.csv` — `source, ident, start, end, label, duration, item`
- `folds.csv` — `source, ident, fold, role`
- `summary_per_fold.csv`, `summary_per_role.csv`, `summary_per_class.csv`
- `translations/<name>.csv` — per-label relabel maps for train time

Every other set derives from an **already-built `medium`** and never touches
`01_annotate`. `large` copies medium's files unchanged; `tiny` and `lite` write
a subset of `medium/annotations.csv` and then `source('../_derive.R')`, which
rebuilds `folds.csv`, the summaries, and `translations/` for that subset. Only
the annotation-level files are derived — each set still extracts its own snips
and embeddings under `--set <name>`. Roles pass through unchanged.

So: rebuild `medium` first, then rebuild any derived set that needs to track it.

Checked-in sets:

| set | derivation | scale |
|---|---|---|
| `medium` | Even Sample + 2025-06-04 original annotations, from `01_annotate` | ~9.6k annotations, 79 folds |
| `large` | copy of `medium`, framehop 0.2 instead of 1 | same annotations, finer frames |
| `lite` | `medium` subset: Even Sample only, ≤10 annotations per ident per label | ~1.2k annotations, troubleshooting |
| `tiny` | `medium` subset: first two annotations per ident | smoke-test scale |

> `medium` is the set experiments run on: day-long annotated recordings from a
> diversity of environments. Its `folds.csv` assigns `rotate` to 11 deployment
> folds and `train` to the rest. Annotation is ongoing, so folds gain data over
> time and numbers drift — note the set's state when a run matters.
> `lite` is kept for troubleshooting, `tiny` for smoke-testing the pipeline.

### 2b. Extract embeddings

```bash
conda run -n buzzdetect-train python 02_set/main.py \
  --set <set> --embedder <embedder> --workers <n> [--verbose]
```

| flag | meaning |
|---|---|
| `--set` | set name under `02_set/sets/` |
| `--embedder` | directory name under `embedders/` — `yamnet`, `yamnet_bandpass`, `yamnet_doublerate`, `yamnet_combined`, `yamnet_copy` |
| `--workers` | processes for the framing + embedding phase; **`0` runs in-process** |
| `--snip-workers` | threads for the snip-sync phase (source-drive I/O, no GPU); default 4, `1` = serial |
| `--overlap-event-prop` | annotation overlap needed to label a frame, as a fraction of frame length |
| `--framehop-prop` | frame hop as a fraction of frame length; `1` = no overlap |
| `--verbose` | one line per ident |

Extraction runs in three layers, each cached and each skippable on re-run:

1. **Snips** — `audio/snips/<ident>/snip_<start>_<end>.flac`. One file per
   cluster of annotations, padded 30 s each side, at the source sample rate.
   Embedder-independent, so every embedder reuses them. Can be synced alone:
   `python 02_set/extract.py --set <set> [--workers <n>]`. Idents are synced
   concurrently (a thread pool — the read/encode is I/O-bound and releases the
   GIL); `--workers 1` forces serial.
2. **Framed-audio cache** — `audio/sr<rate>_fl<len>/raw/<fold>/<ident>/<labels>.pickle`.
   Frames grouped by their collapsed label string (`ins_buzz+ambient_noise`),
   resampled for the embedder. Keyed by sample rate and frame length, so
   embedders sharing those share this cache too.
3. **Embeddings** — `embeddings/<embedder>/raw/<fold>/<ident>/<labels>.pickle`.
   One pickled embedding per frame. Each ident directory (layers 2 and 3) also
   carries `frametimes.csv` — `label,row,start` — mapping every embedding row
   back to its start in source-file seconds. It is the only record of that
   mapping; stage 3's `surprisal/` output depends on it.

Re-running skips any ident that already has both audio and embeddings; an ident
with cached audio but no embeddings re-embeds without touching disk audio. The
run prints how many idents it will actually work on before starting.

**Editing annotations invalidates only what it touched.** Every extraction run
re-syncs the snips first. Each ident's snip directory carries a `manifest.json`
naming the snips its annotations imply and a fingerprint of those annotations;
an ident whose fingerprint still matches is skipped without opening its source
audio, so a re-run over unchanged annotations costs one small read per ident and
never touches the slow media. When the fingerprint has moved — an event edited,
two fragmented events merged into one continuous one, an event deleted — the
snips that are no longer implied are deleted and only the missing ranges are
read from the source file.

The framed-audio cache and the embeddings carry the same fingerprint (an
`annotations.fingerprint` file per ident directory). A mismatch means what's on
disk was built from annotations that have since changed, so that ident's cached
audio, embeddings, and any augmented derivatives are deleted and re-extracted —
everyone else's are left alone. An ident dropped from `annotations.csv`
altogether has its snips and outputs removed. There is no longer any reason to
delete a set's audio by hand after an annotation pass.

A directory extracted before fingerprinting existed carries no fingerprint and
can't be judged retroactively, so it is left alone and stamped with the current
one on the next run — the snip layer still catches it if the edit moved the
snips. The one case that slips through the transition is an annotation edited
before this existed in a way that changed labels but not snip bounds.

**Extraction parameters are frozen per set.** The first extraction writes
`config_extract.json`; every later run uses the saved values and warns if you
pass different ones. To change them, delete `config_extract.json` *and* the
embeddings, then re-extract — otherwise one set would mix framings.

Extraction is deliberately strict: a frame with no annotation overlap at all
raises rather than warns, since that means the framing and the annotations have
drifted apart.

### 2c. Augmentation (optional)

```bash
conda run -n buzzdetect-train python 02_set/augment.py \
  --set <set> --embedder <embedder> --fold <fold> \
  [--noise 0.05 0.2] [--volume 2.5 0.75] [--overwrite]
```

Writes sibling directories to `raw/` — `embeddings/<embedder>/augment_noise_0.05/`
etc. — that `03_train --augment` can pull in. With neither `--noise` nor
`--volume`, a default spec set runs (noise 0.05/0.075/0.2, volume 2.5/0.75).
`augment_specs.py` also defines a `CombineSpec` for mixing two classes.

Augmented embeddings go to **training only**, never to validation or scoring.
Historically augmentation has hurt here; treat it as an experiment, not a
default.

---

## Stage 3 — train

```bash
conda run -n buzzdetect-train python 03_train/main.py \
  --name <name> --set <set> --embedder <emb> --translation <t> \
  [--epochs 400] [--patience 50] [--stop-tol 0.01] [--skip-cv] [--augment <aug_dir> ...] [-y] [--verbose]
```

| flag | default | meaning |
|---|---|---|
| `--name` | required | output directory under `models/` |
| `--set` | `medium` | set to train on |
| `--embedder` | `yamnet` | must already be extracted for this set |
| `--translation` | `general` | CSV under the set's `translations/`, falling back to the project-wide `translations/` |
| `--epochs` | `400` | max epochs per fold model |
| `--patience` | `50` | `EarlyStopping` patience (`min_delta` 0.002) |
| `--stop-tol` | `0.01` | shipped-model epoch count: stop this fraction short of the consensus val_loss floor (larger = fewer epochs) |
| `--skip-cv` | — | train no rotations; build only the shipped model, epoch count from the fold results already on disk |
| `--augment` | — | augmentation subdirectory names to include |
| `-y` / `--yes` | — | accept untranslated labels without prompting |
| `--verbose` | — | per-epoch output and per-fold detail |

### Translations

`02_set/sets/<set>/translations/<name>.csv` maps raw labels (`from`) to model
classes (`to`). The classes are the sorted unique `to` values, minus the two
keywords:

- `ignore` — the label produces no target. A frame labelled *only* with ignored
  labels is dropped.
- `exclude` — any frame carrying this label is dropped outright, whatever else
  it carries.
- **no row at all** — left unchanged, so it matches no class and silently
  contributes nothing. This is almost always a mistake, so before the first
  epoch the run surveys the extracted labels, lists any with no row, and asks
  for confirmation. Non-interactive runs refuse unless given `-y`.

`general.csv` is the working translation; `binary.csv` is buzz vs. everything;
`blank.csv` is a scaffold of every observed label with empty `to` values.

Translations belong to the set, and are written by the set's own `build.R` from
exactly the labels that set's annotations carry — so every label in the set has a
row, and no row exists for a label the set never sees. The mapping rules live in
that `build.R`, not in the CSVs; hand-edits to the written files are overwritten.

```bash
cd 02_set/sets/<set> && Rscript build.R   # rebuilds the set, translations included
```

The rebuild is **wholesale**, not additive: a label the set no longer emits loses
its row entirely, even if embeddings named after it are still on disk. Such a
label then has no row at all, which the run reports before the first epoch.

The project-wide `translations/` directory is the legacy layout, and is still
used as a fallback by sets whose `build.R` predates the move — `lite` and `tiny`.
`medium` owns the rules; `large` is the same set at a finer framehop and sources
medium's build steps, so the two always agree.

Any new label that no rule matches falls through to itself, i.e. becomes its own
class. Check what the rebuild reports and add rules for anything that shouldn't
be one.

`build.R` also fills in identity rows (`mech_hum → mech_hum`) for every value it
produces as a `to`. Set builds collapse labels before extraction, so those
collapsed names land on disk as raw labels; without the identity row those
frames would translate to nothing and be dropped silently.

### What actually gets trained

Per rotation: hold out one `rotate` fold, train on every other `rotate` fold
plus all `train` folds, early-stop on the held-out fold, then score it. Fold
model binaries are not kept — only their scores and training artifacts.

Then the **shipped model** trains on `rotate` + `train` pooled. Nothing is held
out, so there's nothing clean to monitor: it runs for a fixed epoch count read
off the rotations' pooled `val_loss` curves (`train._consensus_epoch`) — each
fold's "best so far" trace, min-max normalised, averaged with weight by
validation-frame count, stopped `--stop-tol` short of the averaged floor. On a
frozen-embedding probe the per-fold `val_loss` argmins scatter by 100+ epochs
in a flat basin, so their `median` lurches with fold composition; the pooled
curve is steadier. Falls back to `median(best_epoch)` for a model resumed from
summaries written before the curves were stored. It's the only model saved with
a binary, and it gets scored on each `holdout` fold.

The CV loop resumes — folds with a `config_model.json` are skipped — so a
re-run after an interruption only trains what's missing, then the shipped
model. `--skip-cv` goes further: it trains *no* rotations and builds only the
shipped model, taking the epoch count from whatever fold results are already on
disk (`folds_sx.csv` is then a partial CV). Use it to get a deployable model
out of a CV you don't intend to finish; it errors if no fold has run yet.

Class weights are inverse-frequency over the training pool. Loss is
`BinaryCrossentropy(from_logits=True, label_smoothing=0.2)`; the architecture is
`Dropout(0.2)` → `Dense(n_classes)` on the frozen embedding.

Needs at least two `rotate` folds, or it errors out.

### Reruns resume

A model directory containing `config_model.json` is skipped. The CV summary is
reassembled by reading each fold's results off disk, so an interrupted run
picks up where it stopped and the skipped folds still appear in the summary. To
start clean, delete `models/<name>/` (or use `main.py --clear`).

### Output

```
models/<name>/
├── model.keras, model.py, config_model.json   the shipped model
├── annotations.csv, folds.csv                 copies of the set's, for provenance
├── weights.csv, translation.csv, history.pickle, loss_curves.svg
├── folds_sx.csv               the results: a row per fold, then a `total` row
├── folds/<fold>/              per-rotation archive (no model.keras)
│   ├── predictions.csv        every held-out frame's activation and label
│   ├── summary.json           epochs, val_loss, frame counts, sens-monitor peaks
│   ├── config_model.json, loss_curves.svg, sens_curves.svg
├── surprisal/<ident>_surprisal.csv   per-frame class activations + loss (see below)
└── holdout/<fold>/            shipped model scored on each holdout fold
```

`model.py` is generated so `models.load_model('<name>')` works for inference.

**`folds_sx.csv` is the whole metrics summary.** Columns: `fold`, `fpr`,
`threshold`, `sensitivity`, `precision`, `buzz_frames`, `neg_frames`,
`frames_val`, `best_epoch`. One row per (fold, FPR target), then a row with
`fold` = `total`, where the counts are summed and threshold/sensitivity/
precision are the plain mean over the folds that could reach the target. That
mean is the headline number. A fold that couldn't reach the target keeps its
row with blanks and is not averaged in, so counting the non-blank rows tells
you what the total rests on.

Three files used to carry this, disagreeing with each other about folds too
small to reach the target; see `03_train/sx.py::_fold_sens` for the policy that
survived. `predictions.csv` is the source everything is derived from —
`03_train/resummarize.py` rebuilds `folds_sx.csv` from it without TensorFlow,
and a pooled ROC or a `metrics_at_precision` read is `read_fold_predictions()`
plus `metrics_by_group()` away.

**`surprisal/` finds bad annotations and hard negatives.** Written by default
(one call per held-out fold, right after `predictions.csv`; `--no-surprisal`
turns it off), one CSV per source-audio ident, mirroring the ident's path under
`surprisal/` with `_surprisal.csv` appended to the leaf. One row per held-out
frame, ordered by `start`. Columns: `start` (seconds into the original audio
file), `label` (the classes the frame's annotation asserts, `;`-joined),
`activation_<class>` (the model's sigmoid activation, one column per class — the
class set varies between models, so the columns follow the model's own class
list), and `loss` (mean over all classes of the per-class binary cross-entropy
against hard 0/1 targets — not the label-smoothed targets training uses, since
this is a labeling audit). Sort by `loss` to rank frames; read the
`activation_*` columns to see which class drove it and which way — a high
activation on a class the frame doesn't assert points at a missing annotation
or a hard negative, a low activation on a class it does assert points at a
faint positive or an over-wide annotation span. Multi-label frames need no
special handling: every asserted class contributes a y=1 term to `loss` and
every other class a y=0 term. Frame timestamps come from the
`frametimes.csv` the extractor writes beside each ident's embedding pickles; a
set extracted before that file existed has its idents skipped with a warning
(re-extract to include them). The scoring model is each fold's own out-of-fold
submodel during CV, or the shipped model for a holdout fold — it never trained
on the frames it scores.

`sens_curves.svg` plots sens@fpr0.005 on the held-out fold per epoch, with
val_loss on a twin axis and a line at the epoch EarlyStopping restored to. It's
a diagnostic, not a control: stopping is still on val_loss, and nothing reads
these curves back. What they're for is the question of whether it *should* be —
if the sens@FPR peak sits far from the restored epoch, run after run, val_loss
is a poor proxy for the number the model is judged by. Three keys in that
fold's `summary.json` say the same thing numerically:
`val_sens_fpr0.005_at_best` (sensitivity at the restored epoch — it should
equal that fold's `sensitivity` in `folds_sx.csv`, since they score the same
weights on the same frames), `val_sens_fpr0.005_peak`, and
`val_sens_fpr0.005_peak_epoch`.

All of them are blank on a fold that never reaches 0.005 FPR at any epoch — too
few negative frames for the target to correspond to even one of them. Small
sets are mostly blank; see the caveat in `03_train/sx.py`.

### Running stages 2 and 3 together

```bash
conda run -n buzzdetect-train python main.py --model <name> [--set medium] \
  [--embedder yamnet] [--translation general] [--workers 2] [--clear] ...
```

Same flags as the stage scripts, except the model name is `--model`. **Must be
run from the project root** — it resolves stage paths relative to the cwd.
There is no stage 4; see [Tools](#tools).

---

## Reading the results

**Sensitivity at a fixed FPR depends on who draws the threshold.** It is not a
property of a model on its own — it is what the model catches once a line is
drawn, and where that line goes is a deployment decision. buzzdetect ships no
threshold: operators are told to find their own. So every fold is scored at a
threshold set on its own held-out audio, and the headline is the plain mean
across folds — the `total` row's `sensitivity` in `folds_sx.csv`, at fpr 0.005.
Each deployment counts once, because the question is what a new deployment
gets, and a new deployment is one fold. The same row's `threshold` is what a
typical fold's own audio set, i.e. what to actually try shipping, and the
per-fold rows above it show how much folds disagree rather than just the
average.

It is an oracle: putting a fold at exactly 0.5% FPR uses that fold's labels,
which an operator doesn't have. Read it as the ceiling on operator tuning. Both
models in a comparison get the same ceiling, so it is fair for ranking.

Two readings deliberately *not* reported, both of which have misled here before:

- A **buzz-weighted** mean — total buzzes found over total buzzes present. Its
  weights are annotated hours times buzz density, so a full-bloom mustard field
  speaks over a quiet soybean patch, and real ecology is mixed with how much of
  each site got worked up. It is the lower-variance estimator, which only helps
  if the folds share one true sensitivity — and the premise here is that they
  don't.
- A **pooled** read, one global threshold across every fold. It answers "one
  shipped threshold, everywhere", which is not how the tool is used, and it
  confounds detection with how portable a model's score scale is. It has made a
  strictly better model look half as good. It used to be written out anyway, as
  `folds_pooled_metrics.csv`; it no longer is. `read_fold_predictions()` plus
  `metrics_by_group()` rebuilds it if you want a ROC or a
  `metrics_at_precision` read.

**Check what a number rests on.** Count the per-fold rows in `folds_sx.csv`
with a `sensitivity`, and read their `neg_frames`: how many folds could reach
the target at all, and how many non-buzz frames sat above each fold's
threshold. At fpr 0.001 on a
set this size that is about four frames per fold, and some folds can't reach it
— which is why only 0.005 is reported. A fold too small for the target is
dropped from the mean rather than interpolated inside a single frame. No amount
of data elsewhere fixes a per-fold read on one small fold.

Don't reweight the training set to equalize folds. How many *hours* a fold
contributes is an artifact of annotation effort; how much *buzz* those hours
hold is largely real, and the two are tangled together in any fold's frame
count, so there's no correct weighting to recover — and the shipped model will
be biased toward high-volume folds either way. Better to know that than to
launder it.

**Don't read the fold-to-fold spread as a confidence interval.** Training pools
overlap by ~90% across rotations, so fold models are highly correlated and the
spread understates true uncertainty about a genuinely new deployment.

**Sensitivity and FPR need different amounts of data.** FPR is estimated from the
flood of non-buzz frames and is tight in every fold. Sensitivity is estimated
from buzz frames only, and in the quietest deployments (tens of seconds of buzz)
it's hopeless — ±0.25 or worse. So let every deployment contribute to pooled
FPR, and treat per-deployment sensitivity as meaningful only where there's buzz
to support it. The quiet folds aren't weak folds; they're the best
false-positive probes available, and nighttime false positives are the known
real-world failure mode (see `log.jsonl`, `yamnet-mask`).

Prefer sensitivity at fixed FPR over sensitivity at fixed precision when
comparing across folds — precision mixes in each deployment's base rate, FPR
doesn't. `folds_sx.csv` carries a `precision` column because it is what an
operator sees in the output, but read it against that fold's own buzz density,
which is a property of what got annotated.

The per-fold and pooled numbers are mildly optimistic: each fold also chose its
own stopping epoch on the fold it's scored against. See
[There is no `validate` role](#there-is-no-validate-role) for how much that's
worth and how to remove it if it ever matters.

---

## Tools

`tools/compare_folds.py` joins two models' `folds_sx.csv` on fold and prints
the per-fold deltas — LOOP.md step 4. `tools/check_sens_at_fpr.py` pins
`metrics.sens_at_fpr` (the per-epoch monitor's read) to the `metrics_by_group`
→ `metrics_at_fpr` pair it restates.

`tools/night_positives.py` is gone. It plotted a model's activations across an
all-night recording, taking its threshold off `folds_pooled_metrics.csv`, which
is no longer written. The recording is still at `tools/night-positives/`, and
nighttime false positives are still the known real-world failure mode (see
`log.jsonl`, `yamnet-mask`) — the tool is recoverable from git if that check is
wanted again.

There is no stage 4. It scored a fixed model against a hand-curated corpus and
expected the pre-CV repeated-run layout; a CV run scores every held-out fold
itself. The code is gone (recoverable from git); the corpus is at
`.local/archive/04_test-corpus/`.

---

`LOOP.md` is the experiment protocol — one worktree per hypothesis, results in
`log.jsonl`, candidate ideas in `IDEAS.md`. `log.jsonl` holds CV-era runs only;
the 29 experiments that predate the rework are archived under `.local/` and
distilled into `IDEAS.md`, where they are marked as leads rather than verdicts.

---

## Design: why folds are deployments

A fold is a **deployment**: one recorder at one site over one period. Folds are
assigned in `01_annotate/` (one per recorder path) and carried through set
builds unchanged.

Deployment is the right unit because it's the thing that varies in the field.
Two frames from the same recorder-day share a microphone, a background, a
species assemblage, and a weather pattern; splitting them across train and test
leaks all of that and flatters the model. Holding out a whole deployment asks
the question that actually matters: how will this do on a recorder we haven't
seen?

### Roles

Two independent questions — is it ever trained on, is it ever scored — give four
cells, so one enum covers the space.

- **`train`** is for data that's useful ballast but isn't a deployment. The
  2025-06-04 original annotations are the case in point: high volume (~46% of
  annotated audio), but a hodgepodge of annotation strategies with, e.g., no
  nighttime coverage. Training on them helps; scoring on them tells you about
  the annotation process rather than about a recorder.
- **`rotate`** is the leave-one-fold-out set — the diel-representative Even
  Sample deployments. Each takes a turn as the held-out fold.
- **`holdout`** is the old fixed-test semantics, kept for when a deployment
  should stay untouched for a headline number. Not currently used.
- **`exclude`** drops a fold from training and scoring without removing it from
  the set. Useful for quarantining a deployment whose annotations are suspect,
  or for ablations, without rebuilding and re-extracting.

Roles are a **training-time policy**, so `02_set` ignores them and embeds
everything including `exclude`. Flipping a role should never require
re-extraction.

### There is no `validate` role

Each rotation's held-out fold *is* its early-stopping monitor. A separate
`validate` role would cost a second deployment per rotation, and which
deployment drew the short straw would swing the stopping epoch — several folds
here hold only a few dozen seconds of buzz, where `val_loss` is mostly a measure
of ambient reconstruction.

**Never split within a fold to make a validation set.** Snips from one
deployment share a recorder, a site, a background, and a species assemblage, so
a within-fold split puts the same site on both sides. The stopping signal then
measures within-deployment generalization, which is easier than the
cross-deployment thing we care about, and `val_loss` keeps falling after
cross-site performance has started to degrade. The bias has a known direction: it
stops too late and leaves the model more site-overfit than intended.

The cost of monitoring on the held-out fold instead: each fold's reported score
was measured on the fold that chose its stopping epoch, so per-fold numbers and
the pooled ROC are mildly optimistic. Bounded, and small here — the stopping
epoch is a single coarse scalar (patience 50, `min_delta` 0.002), adjacent
epochs are highly correlated, and selection is on multi-class `val_loss` while
the reported metric is buzz sensitivity at fixed FPR. It also mostly cancels when
*comparing* configurations, since every config carries the same bias; it does
not cancel when quoting an absolute number.

Cheap to remove later if it ever matters: keep monitoring as-is but report each
fold at `median(best_epoch)` of the *other* folds, so the epoch that produced the
quoted number never saw the fold it's scored on. The probe is small enough
(~13k params) to snapshot every epoch and re-score without retraining.

Empirically this is close to moot — the loss curves are flat. On
`yamnet_medium_v2`, every held-out fold plateaus and none turn back up, with a
constant train/val gap of ~0.055. Early stopping is barely doing anything.

That same flatness is why the shipped model's epoch count is taken from the
*pooled* rotation curves rather than `median(best_epoch)`: in a basin this flat
the per-fold argmins are almost arbitrary (on medium they span 27–153), and
their median swings with which folds are in the pool, while the averaged curve
elbows in a stable place. `--stop-tol` sets how far short of that elbow to stop;
tune it against the per-fold `loss_curves.svg`.

---

## Troubleshooting

**Extraction hangs partway through.** Import order: `import tensorflow` must
come before anything that imports pandas, in every entry point. pandas eagerly
imports pyarrow, and whichever of pyarrow/TF loads first wins abseil's weak
synchronization symbols process-wide; if libarrow wins, TF's in-graph FFT
threadpool deadlocks. The comment at the top of each entry point has the long
version.

**`no embeddings for fold(s) [...]`** — `folds.csv` lists a fold that has no
directory under `embeddings/<embedder>/raw/`. Re-run stage 2 for that embedder,
or check that the fold's audio actually resolves under `audio/`.

**`... holds embeddings for fold(s) [...] that are absent from folds.csv`** —
leftovers from an earlier build of the set. Harmless; they're ignored. Delete
them if you want a clean tree.

**`folds.csv has no role column`** — an old set build. Every fold is treated as
`rotate`. Re-run the set's `build.R`.

**Untranslated labels reported before training.** Add a `from` row to the
translation mapping each label to a class, to `ignore`, or to `exclude`.
Otherwise those frames silently never train. `-y` accepts them as-is.

**Extraction params ignored with a warning.** The set already has a
`config_extract.json` and it wins. Delete it *and* the embeddings to change
them.

**Deadlock or crash with multiprocessing on macOS.** Some embedders load TF
SavedModels, which don't survive `fork`. Pass `--workers 0` to extract
in-process.

**`no trainable frames across N fold(s)`** — every sample was ignored or
excluded by the translation. Check the translation's `to` column against the
labels actually present in the set's `annotations.csv`.
