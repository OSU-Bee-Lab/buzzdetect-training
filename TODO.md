# TODO

## Skills

### High

- **Step 0 briefing skill** — synthesizes log.jsonl, eval.py output, and available embeddings into a structured "what's been tried / what's available / what looks promising" briefing. Runs at the start of every loop iteration.

- **`finish_experiment.sh <modelname>`** — shell script that runs stage 4 (via spec-loader), copies the model to main, and generates a log.jsonl entry from the model's artifacts. Also handles the `models/models.py` symlink that stage 4 currently needs manually.

### Medium

- **Log entry generator** — reads `config_model.json` + `tests/metrics.csv` and writes the JSON line. Probably folds into `finish_experiment.sh` above.

- **Richer metrics view** — shows sensitivity at 80%, 90%, 95% precision for two models side-by-side. May just be a small addition to eval.py rather than a separate tool.
