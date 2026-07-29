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

`03_train` discovers every fold under the set's embeddings (`02_set/sets/<set>/embeddings/<embedder>/raw/<fold>/`,
folds assigned upstream in `01_annotate/`, one fold per deployment) and rotates
each one out: train on the rest, evaluate on the held-out fold. Held-out-fold
model binaries are not kept — only training/evaluation artifacts. The shipped
model trains on every fold pooled and is the only one saved with a binary.

```bash
conda run -n buzzdetect-train python 03_train/main.py \
  --name <name> --set <set> --embedder <emb> --translation <t> \
  [--epochs E] [--val-prop P] [--seed S]
```

One model per fold — no repeat runs nested in the CV. (The old `_v1…_vN`
repeated-run pattern was for a different purpose, hypothesis-testing noise
floors per `LOOP.md`, not for CV; it doesn't apply here.)

Output under `models/<name>/`:
- `model.keras` etc. — the shipped model, trained on all folds pooled
- `folds_summary.csv` — one row per fold: epochs, best val loss/accuracy, held-out loss/accuracy
- `folds/<fold>/` — archive for that held-out fold: `config_model.json`, `history.pickle`, `loss_curves.svg`, `weights.csv`, `translation.csv`, `holdout_metrics.csv` (per-class sensitivity/FPR/precision on the held-out fold). No `model.keras`.

## Evaluation tools (stale — predate the CV rework)

`04_test` and the scripts below assume a fixed model + a separate hand-curated
test corpus (`models/<model>/tests/metrics.csv`), which CV-trained models
don't produce. Read `folds_summary.csv` / `folds/<fold>/holdout_metrics.csv`
directly until these are reworked to aggregate across folds.

```bash
conda run -n buzzdetect-train python summarize_metrics.py <model> [<model> ...]
conda run -n buzzdetect-train python compare_metrics.py [<model>] [--top N]
conda run -n buzzdetect-train python evaluate_set.py <set_base> [<set_base> ...]
conda run -n buzzdetect-train python compare_sets.py [<set_base> ...] [--top N]
```
