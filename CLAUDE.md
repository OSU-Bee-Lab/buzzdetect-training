# buzzdetect-training

Pipeline: raw audio + annotations → trained Keras classifiers for insect buzz detection.

## Pipeline

| Stage | Dir | Entry point | Language |
|-------|-----|-------------|----------|
| 1. Build annotation sets | `01_annotate/` | `MAKE.R` | R |
| 2–4. Extract, train, test | root | `main.py` | Python |

Stage scripts in `02_set/`, `03_train/`, `04_test/` can also be run independently.

## Key files

- `config.py` — all paths/constants; anchors ROOT to the project directory
- `embedders/embedding.py` — embedder interface
- `models/models.py` — model loader
- `translations/` — label translation CSVs

## Data layout (mostly gitignored)

- `audio/` — raw training audio
- `02_set/sets/<setname>/` — snips and embeddings
- `models/<modelname>/` — model artifacts
- `04_test/audio/` — inference test audio

## Environment

`conda run -n buzzdetect-train python <script>`

## Train (leave-one-fold-out CV)

A fold is one deployment (one recorder, one site, one period), assigned upstream
in `01_annotate/`. `README.md` has the design rationale; the mechanics:

`03_train` reads `02_set/sets/<set>/folds.csv` for each fold's **role** — `train`
(always trains, never scored), `rotate` (the leave-one-fold-out set), `holdout`
(only ever scored), `exclude` (neither) — and cross-checks it against the folds
actually extracted under `embeddings/<embedder>/raw/`. Roles are training-time
policy only; `02_set` embeds every fold regardless, so flipping a role never
costs a re-extraction.

Each `rotate` fold takes a turn held out: train on the other `rotate` folds plus
all `train` folds, early-stop on the held-out fold, score the held-out fold.
Validation is always a whole deployment — never a split within one, which would
leak site identity into the stopping signal. Fold model binaries are not kept.

The shipped model trains on `rotate` + `train` pooled. Nothing is held out, so
there is nothing clean to monitor: it trains for a fixed
`median(best_epoch)` across the rotations. It's the only model saved with a
binary.

```bash
conda run -n buzzdetect-train python 03_train/main.py \
  --name <name> --set <set> --embedder <emb> --translation <t> [--epochs E]
```

One model per fold — no repeat runs nested in the CV. (The old `_v1…_vN`
repeated-run pattern was for a different purpose, hypothesis-testing noise
floors per `LOOP.md`, not for CV; it doesn't apply here.)

Output under `models/<name>/`:
- `model.keras` etc. — the shipped model
- `folds_summary.csv` — one row per rotating fold: epochs, best val loss/accuracy, frame counts, sensitivity at each target FPR on the held-out fold
- `folds_pooled_metrics.csv` / `folds_pooled_sx.csv` — every fold's held-out predictions pooled into one ROC (frame-weighted). Compare against the unweighted mean across folds that the run prints; a gap means one high-volume fold is carrying the result. Mildly optimistic, since each fold also chose its own stopping epoch — see README.
- `folds/<fold>/` — archive for that held-out fold: `metrics.csv`, `sx.csv`, `predictions.csv`, `summary.json`, `config_model.json`, `history.pickle`, `loss_curves.svg`, `weights.csv`, `translation.csv`. No `model.keras`.
- `holdout/<fold>/` — the shipped model scored on each `holdout` fold, same files as above.

Reruns skip any model whose directory already holds a `config_model.json`, and
the CV summary is reassembled from disk, so a partial rerun keeps the folds it
skipped.

## Evaluation tools (stale — predate the CV rework)

`04_test` and the scripts below assume a fixed model + a separate hand-curated
test corpus (`models/<model>/tests/metrics.csv`), which CV-trained models
don't produce. Read `folds_summary.csv`, `folds_pooled_sx.csv`, or
`folds/<fold>/sx.csv` directly until these are reworked to aggregate across
folds.

```bash
conda run -n buzzdetect-train python summarize_metrics.py <model> [<model> ...]
conda run -n buzzdetect-train python compare_metrics.py [<model>] [--top N]
conda run -n buzzdetect-train python evaluate_set.py <set_base> [<set_base> ...]
conda run -n buzzdetect-train python compare_sets.py [<set_base> ...] [--top N]
```
