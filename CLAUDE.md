# buzzdetect-training

Pipeline for training audio classifiers that detect insect buzz sounds, going from raw audio + human annotations to trained Keras models.

## Full pipeline
To run the full pipeline, execute main.py in the project root.

## Pipeline by stage

| Stage | Dir | Entry point | Language |
|-------|-----|-------------|----------|
| 1. Build annotation sets | `01_annotate/` | `MAKE.R` | R |
| 2–4. Extract, train, test | root | `main.py` | Python |

Stages 2–4 are run together via root `main.py`. Per-module `main.py` files in `02_set/`, `03_train/`, `04_test/` are loaded as submodules by the root entrypoint.

## Key files

- `config.py` / `config.R` — shared paths and constants (start here for directory layout)
- `utils.py` — shared Python utilities
- `embedders/embedding.py` — audio embedder wrappers (e.g. YAMNet)
- `models/models.py` — model loader/wrapper
- `translations/` — label name translation CSVs; `build.R` compiles them

## Data layout (mostly gitignored)

- `audio/` — raw training audio
- `02_set/sets/<setname>/` — extracted snips and embeddings per set
- `models/<modelname>/` — trained model artifacts
- `04_test/audio/` — audio used for inference testing
- `.local/` — local-only data and notes; `README_annotations.md` documents the annotation set conventions

## Environment

Python deps in `environment.yml` (conda); run Python with `conda run -n buzzdetect-train`. R project: `buzzdetect-training.Rproj`.
