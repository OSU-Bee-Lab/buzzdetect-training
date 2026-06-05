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

## Tools

### summarize_metrics.py

Prints sensitivity at 90%, 95%, and 99% precision for one or more models in a table. Use this whenever investigating or comparing model performance — prefer it over reading raw metrics.csv files.

```
conda run -n buzzdetect-train python summarize_metrics.py <model> [<model> ...]
```

Example:

```
conda run -n buzzdetect-train python summarize_metrics.py yamnet_medium yamnet_bandpass_medium
```

```
                       sens@90prec sens@95prec sens@99prec
model
yamnet_medium                28.0%       25.5%       18.1%
yamnet_bandpass_medium       13.6%       11.4%        9.1%
```

Run without arguments to see available models.

### compare_metrics.py

Scans all models with a `metrics.csv` and ranks them by sensitivity at 95% precision. Use this to find the best-performing models and to see where a model of interest stands relative to them.

```
conda run -n buzzdetect-train python compare_metrics.py [<model>]
```

Without an argument, shows the top 5 models. With a model name, guarantees that model appears in the table even if it falls outside the top 5.

Example:

```
conda run -n buzzdetect-train python compare_metrics.py yamnet_lite
```

```
Model                     Sensitivity @ 95% Precision
-----------------------------------------------------
yamnet_medium             0.2544
yamnet_bandpass_medium    0.1156
yamnet_bandpass_lite      0.0753
yamnet_lite               N/A  [current]
```
