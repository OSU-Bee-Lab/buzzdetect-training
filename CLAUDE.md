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
- **No augmentation may cross a fold boundary.** Every augmented frame must be derivable from the source fold's audio alone: noise/volume transforms of one fold's frames, or mixes of two frames *from the same fold*. A `CombineSpec` / mixup that pairs a source frame with a background or partner frame drawn from another fold — or from a whole-pool shuffle — is forbidden, however tempting the "buzz over a different site's background" framing is. `load_augmented` loads augment dirs per training fold, so a cross-fold-paired frame carries an out-of-fold recording's embedding signature into the training pool, and when that other fold rotates in as validation the early-stopping signal and the held-out metrics are both contaminated. This inflates the reported number purely through leakage (the models here are extremely sensitive to it) and is not a real gain. The same rule bars any augmentation whose parameters (noise floor, SNR target, mix ratio, class balance) are fit on statistics pooled across folds rather than computed within the fold being augmented. If you catch a leak like this in existing code, stop and flag it rather than "fixing" it into a better score.
- **An annotation effort is a directory under `01_annotate/` holding a `combine.R`** — that file is the whole contract, and it must write both `annotations_combined.csv` and `folds.csv` (fold assignment included). `MAKE.R` discovers efforts by that file, not by `.Rproj`, which is gitignored.
- **`medium` is the only set built from `01_annotate`.** Its `build.R` drives the four steps next to it — `combine.R` → `annotations.csv`, `folds.R` → `folds.csv`, `summarize.R` → the `summary_*.csv`, `translate.R` → `translations/` (mapping rules live in `translate.R`, not the CSVs). Every other set derives its annotation-level files from the **already-built** `medium` and never opens `01_annotate`: `large` copies medium's files verbatim (it is medium at a finer framehop, only `config_extract.json` differs); `tiny`/`lite` write a subset of `medium/annotations.csv` and then `source('../_derive.R')`, which regenerates `folds.csv` (medium's rows for the kept idents), the summaries, and `translations/` (medium's CSVs with rows trimmed to labels the subset still carries — nothing is remapped). Snips and embeddings are never derived — each set extracts its own under `--set <name>`. Changing a mapping or fold rule means editing `medium/` and rebuilding it, then rebuilding the derived sets.
- **Translations are per-set**, at `02_set/sets/<set>/translations/<name>.csv`. Regeneration is **wholesale, not additive** — a label the set no longer emits loses its row, even though embeddings named after it may still be on disk. Such a label then has no row at all, which `translate_labels` leaves unchanged and `survey_untranslated` reports at train time. `config.py::path_translation` resolves a name against the set first, with a project-wide `translations/` fallback. Hand-edits to the CSVs are overwritten by `build.R`. The one exception is `medium/translations/general_v1.csv`: a frozen byte-copy of the pre-2026-09-03 `general` table, written by hand rather than by `translate.R`, so a rebuild never touches it. It is **historical** since the 2026-09-08 cutover — the log it protected is in `archive/`, and current experiments pass `--translation general`. Don't hand-edit or regenerate it.
- **Reruns resume.** `train_utils.can_write()` skips any model directory that already holds a `config_model.json`, and the CV summary is reassembled from disk so skipped folds still contribute.

## Where to look

- Fold roles, CV loop, shipped-model epoch count — `03_train/train.py`, `03_train/dataset.py`
- Three-layer extraction (snips → framed-audio cache → embeddings), worker fan-out — `02_set/extract.py`
- Frame → source-timestamp map (`frametimes.csv`, written per ident in layers 2+3; embedding row i ↔ frame i) — `02_set/extract.py::extract_ident_both`
- Per-frame class activations + multi-label loss `<model>/surprisal/<ident>_surprisal.csv` for finding bad annotations and hard negatives, on by default (`--no-surprisal`) — `03_train/surprisal.py`
- Embedder interface — `embedders/embedding.py`; model loader — `models/models.py`
- Threshold sweeps (`metrics_by_group`, `metrics_at_fpr`, `metrics_at_precision`) — `03_train/metrics.py`
- What the headline sens@FPR number means, and the two readings deliberately not reported — `03_train/sx.py`; its module docstring is the argument, README's "Reading the results" is the operator-facing version. `03_train/resummarize.py` rebuilds `folds_sx.csv` for an already-trained model, no TensorFlow
- **`folds_sx.csv` is the only metrics summary**, one row per (fold, FPR target) plus a `total` row, written by `sx.py::summarize_folds`. It replaced three files that reported the same per-fold sensitivity under three different NaN policies; `_fold_sens` holds the one that survived. Everything else is derived from `folds/<fold>/predictions.csv` on demand — don't add a fourth summary file, add a column or a function
- Per-epoch sens@FPR monitor (reporting only; stopping is still on `val_loss`) — `03_train/callbacks.py`, plotted by `plot_history.py::plot_sens_history`
- Label translation semantics (`ignore` / `exclude` / missing row) — `03_train/dataset.py::translate_labels`

## Known stale

Kept as artifacts, never as comparators:

- `models/model_general_v3/` — predates the CV rework.
- `models/yamnet_medium_general/` — the 2026-08 era's `cv-baseline` (0.206). Trained under `general_v1` on the pre-2026-09-08 annotations, so a paired per-fold join against it is invalid: its folds no longer hold the same audio.

## Experiment history

`log.jsonl` holds the current era only. Closed eras live in `archive/<first-date>_<slug>/` — log, README, `notes/`, and a snapshot of the set that produced the numbers. `archive/README.md` covers the layout, how to add an era at the next cutover, and how to reach experiments whose branches were deleted (`refs/archive/<slug>`).

Do not archive to `.local/` or any gitignored path. That was tried in 2026-08; the files evaporated and three docs went on citing them for a month.

## Running long jobs

Stage 2 (`02_set/main.py`) and stage 3 (`03_train/main.py`) both run far longer
than a foreground command should block, and **Claude Code's `run_in_background`
does not work for either of them.** This has cost every LOOP agent a rediscovery
— follow the recipe below and don't re-derive it.

- A `run_in_background` job running the pipeline itself gets **SIGKILLed** within
  ~15–60 s — no OOM, no traceback (stage 2: observed ~4/4 across 2026-09-01/02;
  stage 3 survived it once, not reliably).
- A `run_in_background` job that only *waits* on a detached run (an
  `until grep …; do sleep …; done` loop) is **also killed**, within a minute or
  two (observed 2026-09-02).
- A leading `sleep` in a Bash call is blocked by the harness outright.

### What works

**1. Launch the job itself detached from the shell** — direct env-python (no
`conda run` wrapper process), unbuffered, logging to a file in the worktree.
**Capture its PID** (`echo $!`) — you need it for the status check below,
and a `pkill -f` on the job name has previously killed the shell running it,
so prefer `pgrep -af` (read-only) over anything that kills by pattern.

```
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  BUZZDETECT_CHUNK_FRAMES=48 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  02_set/main.py --set <set> --embedder <emb> --workers <n> --verbose \
  > extract_<set>.log 2>&1 &
disown; echo "pid $!"
```

For stage 3 drop `BUZZDETECT_CHUNK_FRAMES`; keep `CUDA_VISIBLE_DEVICES=""` (the
12288-d trunk embedders OOM the 4 GB GPU, and CPU ≈ GPU on this box for the
1024-d probe anyway). Pass `--verbose` to stage 3 or it trains silently with no
per-epoch line in the log.

**2. Launch it, then end the turn.** Do not wait in-session, and do not
`Monitor` a job whose next real event is more than ~1 h away — persistent or
not. A persistent Monitor looks like the fix, since it only wakes on a real
state change, but the prompt cache has its own ~1 h TTL independent of
Monitor's: when events are sparser than that (any stage-3 CV, most stage-2
extractions) the cache goes cold in the gap either way, and every wake pays
full uncached-context price. For a ~40 h CV that is several cold wakes.

Come back on your own schedule instead and re-check — cheap, and it doesn't
hold the session's cache hostage to the job's pace:

```
pgrep -af 03_train/main.py   # or 02_set/main.py
find models/<name>/folds -name summary.json | wc -l   # of 11, stage 3
tail -5 <log>
```

To be *notified* rather than check back, it must happen outside this session —
a `nohup`-detached shell loop (launched like the job itself, not via
`run_in_background`) that polls for the completion marker and then invokes a
fresh `claude` CLI call.

- `--workers` only parallelises the framing+embedding phase. `extract_snips`
  (reading source audio off the slow HDD) is always serial — a large set's snip
  sync is a fixed up-front cost no worker count changes.

## Testing
For testing code, debugging, etc., you may do the following:

- **01_annotate.** You may create a dummy annotation project under 01_annotate called "test"

- **02_set.** You may create a dummy training set under 02_set/sets/

- **03_train.** You may create dummy models prepended with "test\_". E.g., "test_new_metrics"

You have permission to create, modify, or delete any of these test locations. You must ask the user before creating, editing, or modifying existing annotation projects, training sets, or models unless you have been directed to do so.