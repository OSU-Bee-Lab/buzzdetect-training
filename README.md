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
- [Stage 4 — test (stale)](#stage-4--test-stale)
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
| `04_test/audio/` | inference test audio |

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
        │  3.  (optional) 04_test/main.py  → scores a fixed model on a separate corpus
        ▼
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

Results land in `models/my_first_model/`. Start with `folds_summary.csv` and
`folds_pooled_sx.csv`.

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

To add an effort: create the directory, drop raw annotations in `data/`, and
write a `combine.R` that emits those two CSVs — fold assignment included, in
that same file. Copy the closest existing effort; `01_annotate/Even Sample/` is
the current model, and its `README.md` documents the sampling design.

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

A **set** is a named selection of annotation efforts plus any relabelling on top
of them. Sets live in `02_set/sets/<name>/`, each defined by a `build.R`:

```bash
cd 02_set/sets/lite && Rscript build.R
```

`build.R` concatenates the chosen efforts' `annotations_combined.csv` and
`folds.csv`, applies set-specific label collapsing (e.g. every `ins_buzz*`
variant → `ins_buzz`), checks that no ident lands in two folds, and writes:

- `annotations.csv` — `source, ident, start, end, label, duration, item`
- `folds.csv` — `source, ident, fold, role`
- `summary_per_fold.csv`, `summary_per_class.csv` — annotated seconds by fold, and by fold × class

Roles pass through unchanged, so a set picks roles by choosing its sources.
Overriding one means editing that set's `build.R`.

Checked-in sets:

| set | sources | scale |
|---|---|---|
| `medium` | Even Sample + 2025-06-04 original annotations | ~9.6k annotations, 79 folds |
| `lite` | same, Even Sample only in practice | ~325 annotations, 11 rotate folds, ~2.8 h |
| `tiny` | same, two annotations per ident | smoke-test scale |

> `medium` is a work in progress. Its `folds.csv` predates the `role` column, so
> training against it warns and treats every fold as `rotate`.

### 2b. Extract embeddings

```bash
conda run -n buzzdetect-train python 02_set/main.py \
  --set <set> --embedder <embedder> --workers <n> [--verbose]
```

| flag | meaning |
|---|---|
| `--set` | set name under `02_set/sets/` |
| `--embedder` | directory name under `embedders/` — `yamnet`, `yamnet_bandpass`, `yamnet_doublerate`, `yamnet_combined`, `yamnet_copy` |
| `--workers` | parallel extraction processes; **`0` runs in-process** |
| `--overlap-event-prop` | annotation overlap needed to label a frame, as a fraction of frame length |
| `--framehop-prop` | frame hop as a fraction of frame length; `1` = no overlap |
| `--verbose` | one line per ident |

Extraction runs in three layers, each cached and each skippable on re-run:

1. **Snips** — `audio/snips/<ident>/snip_<start>_<end>.flac`. One file per
   cluster of annotations, padded 30 s each side, at the source sample rate.
   Embedder-independent, so every embedder reuses them. Can be synced alone:
   `python 02_set/extract.py --set <set>`.
2. **Framed-audio cache** — `audio/sr<rate>_fl<len>/raw/<fold>/<ident>/<labels>.pickle`.
   Frames grouped by their collapsed label string (`ins_buzz+ambient_noise`),
   resampled for the embedder. Keyed by sample rate and frame length, so
   embedders sharing those share this cache too.
3. **Embeddings** — `embeddings/<embedder>/raw/<fold>/<ident>/<labels>.pickle`.
   One pickled embedding per frame.

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
  [--epochs 400] [--patience 50] [--augment <aug_dir> ...] [-y] [--verbose]
```

| flag | default | meaning |
|---|---|---|
| `--name` | required | output directory under `models/` |
| `--set` | `medium` | set to train on |
| `--embedder` | `yamnet` | must already be extracted for this set |
| `--translation` | `general` | CSV under `translations/` |
| `--epochs` | `400` | max epochs per fold model |
| `--patience` | `50` | `EarlyStopping` patience (`min_delta` 0.002) |
| `--augment` | — | augmentation subdirectory names to include |
| `-y` / `--yes` | — | accept untranslated labels without prompting |
| `--verbose` | — | per-epoch output and per-fold detail |

### Translations

`translations/<name>.csv` maps raw labels (`from`) to model classes (`to`). The
classes are the sorted unique `to` values, minus the two keywords:

- `ignore` — the label produces no target. A frame labelled *only* with ignored
  labels is dropped.
- `exclude` — any frame carrying this label is dropped outright, whatever else
  it carries.
- **no row at all** — left unchanged, so it matches no class and silently
  contributes nothing. This is almost always a mistake, so before the first
  epoch the run surveys the extracted labels, lists any with no row, and asks
  for confirmation. Non-interactive runs refuse unless given `-y`.

`general.csv` is the working translation; `specific.csv` keeps finer classes;
`binary.csv` is buzz vs. everything; `blank.csv` is a scaffold of every observed
label with empty `to` values.

```bash
Rscript translations/build.R     # regenerates blank.csv and general.csv
```

The rebuild is **additive**: it surveys every effort's raw labels, keeps every
existing row as-is, and appends any label it hasn't seen before, printing what
it added. It will never drop a curated row — a label can vanish from the current
annotations while embeddings named after it are still on disk in some set.
Retiring a mapping means editing the rules in `build.R` *and* deleting the stale
row by hand.

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
out, so there's nothing clean to monitor: it runs for a fixed
`median(best_epoch)` across the rotations. It's the only model saved with a
binary, and it gets scored on each `holdout` fold.

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
├── folds_summary.csv          one row per rotating fold
├── folds_pooled_metrics.csv   every fold's held-out predictions pooled into one ROC
├── folds_pooled_sx.csv        that ROC read at target FPRs
├── folds/<fold>/              per-rotation archive (no model.keras)
│   ├── metrics.csv, sx.csv, predictions.csv, summary.json
│   ├── config_model.json, history.pickle, loss_curves.svg
│   └── weights.csv, translation.csv
└── holdout/<fold>/            shipped model scored on each holdout fold
```

`metrics.csv` is the full threshold sweep; `sx.csv` reads it at FPR targets
0.1% / 0.5% / 1%. `model.py` is generated so `models.load_model('<name>')` works
for inference.

### Running stages 2 and 3 together

```bash
conda run -n buzzdetect-train python main.py --model <name> [--set medium] \
  [--embedder yamnet] [--translation general] [--workers 2] [--clear] ...
```

Same flags as the stage scripts, except the model name is `--model`. **Must be
run from the project root** — it resolves stage paths relative to the cwd.
Stage 4 is not chained, by design.

---

## Reading the results

**Report both pooled and per-deployment.** A run prints both and writes the
pooled ROC to `folds_pooled_metrics.csv` / `folds_pooled_sx.csv`: every fold's
held-out predictions pooled, threshold picked once. That's frame-weighted by
construction, which matches how the shipped model is actually biased. The
unweighted mean across deployments comes from `folds_summary.csv`. The gap
between the two *is* the answer to "should I weight for volume?": if they agree,
fold identity doesn't matter; if they diverge, one high-volume deployment is
carrying the result and the headline needs that caveat attached.

Don't reweight the training set to equalize folds. Fold volumes here are an
artifact of annotation effort, not of nature, so there's no correct weighting to
recover — and the shipped model will be biased toward high-volume folds either
way. Better to know that than to launder it.

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

Prefer sensitivity at fixed FPR (`metrics_at_fpr`, the `sx.csv` files) over
sensitivity at fixed precision when comparing across folds — precision mixes in
each deployment's base rate, FPR doesn't.

The per-fold and pooled numbers are mildly optimistic: each fold also chose its
own stopping epoch on the fold it's scored against. See
[There is no `validate` role](#there-is-no-validate-role) for how much that's
worth and how to remove it if it ever matters.

---

## Stage 4 — test (stale)

`04_test/` scores a fixed model against a separate hand-curated corpus
(`04_test/audio/` + `04_test/annotations.csv`), writing to
`models/<model>/tests/`. It expects the older `<name>_v1 … _vN` repeated-run
layout and so does not fit CV-trained models:

```bash
conda run -n buzzdetect-train python 04_test/main.py --name <base> --runs 5
```

The same applies to the top-level analysis scripts, which all read
`models/<model>/tests/metrics.csv`:

```bash
conda run -n buzzdetect-train python summarize_metrics.py <model> [<model> ...]
conda run -n buzzdetect-train python compare_metrics.py [<model>] [--top N]
conda run -n buzzdetect-train python evaluate_set.py <set_base> [<set_base> ...]
conda run -n buzzdetect-train python compare_sets.py [<set_base> ...] [--top N]
```

Until these are reworked to aggregate across folds, read `folds_summary.csv`,
`folds_pooled_sx.csv`, or `folds/<fold>/sx.csv` directly.

`04_test/night_positives.py` is standalone and still useful: it plots a model's
activations across a night recording, the known failure mode.

`LOOP.md` is the experiment protocol — one worktree per hypothesis, results in
`log.jsonl`, candidate ideas in `IDEAS.md`. Every logged number predates the CV
rework and was measured against the fixed `04_test` corpus, so treat the log as
a record of which ideas are dead ends rather than as scores to beat.

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
constant train/val gap of ~0.055. Early stopping is barely doing anything, which
is also why the shipped model's fixed epoch count is safe.

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
