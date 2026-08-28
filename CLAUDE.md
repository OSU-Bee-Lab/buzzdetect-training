# buzzdetect-training

Raw audio + annotations → a Keras probe (dropout + one dense layer) over frozen audio embeddings, for insect buzz detection.

`README.md` is the operator's guide: how to run each stage, what every flag does, what the outputs mean, and the design rationale for folds and roles. Read it before changing pipeline behaviour. This file holds only what isn't obvious from the code itself.

## Layout

| Stage | Entry point | Language |
|------------------|-------------------------------|------------------------|
| 1\. Combine annotations per effort | `01_annotate/MAKE.R` | R |
| 2\. Build a set, extract embeddings | `02_set/sets/<set>/build.R`, `02_set/main.py` | R, Python |
| 3\. Train (leave-one-fold-out CV) | `03_train/main.py` | Python |

Root `main.py` chains 2→3. It resolves stage paths relative to the cwd, so it only works from the project root. There is no stage 4 — it was retired with the CV rework.

Environment: `conda run -n buzzdetect-train python <script>`.

## Invariants

- **`import tensorflow` must come first** in any entry point that later imports pandas. Every entry point has the full explanation in a header comment; keep it there and keep the import at the top.
- **`config.py` anchors `ROOT` to its own directory**, so stage scripts can be invoked from any cwd. Add paths there, not as literals in stage code.
- **A set's `config_extract.json` always wins** over CLI extraction params — embeddings on disk were built under it. Changing them means deleting the file and re-extracting (`02_set/extract.py::extract_set`).
- **Annotations are fingerprinted, and the fingerprint is what makes a rerun incremental.** Each ident's snip dir holds a `manifest.json` (fingerprint + the snips its annotations imply); each framed-audio and embedding dir holds an `annotations.fingerprint`. Matching means the product is current and the source file is never opened; differing means only that ident is deleted and rebuilt. Anything that changes what extraction produces from a given annotation set must be reflected in the fingerprint, or stale output will be kept (`02_set/extract.py::_fingerprint_annotations`, `AssignIdent`).
- **Idents and fold names are both path-like**, of no fixed depth. Never glob a fixed number of `*` components to find an ident's directory, and never read the top level of `audio/snips/` as a list of idents — walk (`_find_ident_dirs`, `_find_snip_dirs`). Getting this wrong deletes data.
- **Roles are training-time policy only.** `02_set` embeds every fold regardless, including `exclude`, so flipping a role never costs a re-extraction (`03_train/dataset.py::read_fold_roles`).
- **Validation is always a whole fold, never a split within one.** A within-fold split leaks site identity into the early-stopping signal. There is no snip-level splitter and there should not be one; README explains why.
- **An annotation effort is a directory under `01_annotate/` holding a `combine.R`** — that file is the whole contract, and it must write both `annotations_combined.csv` and `folds.csv` (fold assignment included). `MAKE.R` discovers efforts by that file, not by `.Rproj`, which is gitignored.
- **`translations/build.R` regenerates the tables wholesale** from the labels currently observed across every effort's `annotations_combined.csv`; the mapping rules live in `build.R` itself, not in the CSVs. It is *not* additive — a label no effort emits any more loses its row, even though embeddings named after it may still be on disk. Such a label then has no row at all, which `translate_labels` leaves unchanged and `survey_untranslated` reports at train time. Edit the rules in `build.R` and rerun; hand-edits to `general.csv` / `binary.csv` are overwritten.
- **Reruns resume.** `train_utils.can_write()` skips any model directory that already holds a `config_model.json`, and the CV summary is reassembled from disk so skipped folds still contribute.

## Where to look

- Fold roles, CV loop, shipped-model epoch count — `03_train/train.py`, `03_train/dataset.py`
- Three-layer extraction (snips → framed-audio cache → embeddings), worker fan-out — `02_set/extract.py`
- Embedder interface — `embedders/embedding.py`; model loader — `models/models.py`
- Threshold sweeps (`metrics_by_group`, `metrics_at_fpr`, `metrics_at_precision`) — `03_train/metrics.py`
- What the headline sens@FPR number means, and the two readings deliberately not reported — `03_train/sx.py`; its module docstring is the argument, README's "Reading the results" is the operator-facing version. `03_train/resummarize.py` rebuilds `folds_sx.csv` for an already-trained model, no TensorFlow
- **`folds_sx.csv` is the only metrics summary**, one row per (fold, FPR target) plus a `total` row, written by `sx.py::summarize_folds`. It replaced three files that reported the same per-fold sensitivity under three different NaN policies; `_fold_sens` holds the one that survived. Everything else is derived from `folds/<fold>/predictions.csv` on demand — don't add a fourth summary file, add a column or a function
- Per-epoch sens@FPR monitor (reporting only; stopping is still on `val_loss`) — `03_train/callbacks.py`, plotted by `plot_history.py::plot_sens_history`
- Label translation semantics (`ignore` / `exclude` / missing row) — `03_train/dataset.py::translate_labels`

## Known stale

- `models/model_general_v3/` predates the CV rework and is kept only as an artifact.

## Testing
For testing code, debugging, etc., you may do the following:

- **01_annotate.** You may create a dummy annotation project under 01_annotate called "test"

- **02_set.** You may create a dummy training set under 02_set/sets/

- **03_train.** You may create dummy models prepended with "test\_". E.g., "test_new_metrics"

You have permission to create, modify, or delete any of these test locations. You must ask the user before creating, editing, or modifying existing annotation projects, training sets, or models unless you have been directed to do so.