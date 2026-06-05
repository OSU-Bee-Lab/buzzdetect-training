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

## Tools

```bash
conda run -n buzzdetect-train python summarize_metrics.py <model> [<model> ...]
```
Sensitivity at 90%, 95%, 99% precision. Prefer over reading raw metrics.csv.

```bash
conda run -n buzzdetect-train python compare_metrics.py [<model>] [--top N]
```
Ranks all models by sensitivity at 95% precision. Top 5 by default; a named model always appears.
