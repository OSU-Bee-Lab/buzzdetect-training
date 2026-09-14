# buzzdetect-training

Raw audio + annotations → a Keras linear probe over frozen audio embeddings, for
insect buzz detection.

- `README.md` is the operator's guide: running each stage, every flag, what the
  outputs mean, why folds are deployments. Read the relevant section before
  changing pipeline behaviour.
- `LOOP.md` is the experiment protocol.
- `02_set/CLAUDE.md` and `03_train/CLAUDE.md` hold stage-local gotchas; read the
  one for the stage you edit.

This file holds only what isn't obvious from the code.

## Layout

| Stage | Entry point | Language |
|---|---|---|
| 1\. Combine annotations per effort | `01_annotate/MAKE.R` | R |
| 2\. Build a set, extract embeddings | `02_set/sets/<set>/build.R`, `02_set/main.py` | R, Python |
| 3\. Train (leave-one-fold-out CV) | `03_train/main.py` | Python |

Root `main.py` chains 2→3 and only works from the project root. There is no
stage 4.

Environment: `conda run -n buzzdetect-train python <script>` for anything short;
long jobs are below. Embedder interface: `embedders/embedding.py`. Model loader:
`models/models.py`.

`DEPLOYMENTS.md` maps each fold to its crop, site and date. Nothing reads it, but
every "hard fold" conclusion is really a conclusion about a crop.

## Invariants

- **`import tensorflow` must come first** in any entry point that later imports
  pandas. Each entry point's header comment explains why; keep it there.
- **`config.py` anchors `ROOT` to its own directory**, so stage scripts run from
  any cwd. Add paths there, not as literals in stage code.
- **An annotation effort is a directory under `01_annotate/` holding a
  `combine.R`**, which must write both `annotations_combined.csv` and
  `folds.csv` (fold assignment included). `MAKE.R` discovers efforts by that
  file, not by `.Rproj`, which is gitignored. Do not modify `01_annotate/`:
  changing the data under an experiment makes it incomparable to the log.
- **`IDEAS.md` is a queue of untried ideas, nothing else.** When an idea has been
  tested, delete its whole section. The verdict goes in `log.jsonl` and the
  branch's `notes.md`, never back into `IDEAS.md` as a "DONE" or "Closed"
  writeup. If only some legs ran, trim it to the untested leg. Agents have
  repeatedly done the opposite, and every line of the file is re-read uncached
  by every agent.

## Known stale

Kept as artifacts, never as comparators:

- `models/model_general_v3/` predates the CV rework.
- `models/cv_baseline/` and every other 2026-09-era model predate the
  2026-09-11 annotation revision, the `_quiet` scoring split and
  `--fixed-epochs`: no `sensitivity_exclquiet` column, and their folds no
  longer hold the same audio. `tools/compare_folds.py` warns rather than
  pairing on the wrong column. Per-fold numbers are in
  `archive/2026-09-08_cv-medium-v2/`.
- `models/yamnet_medium_general/` is the 2026-08 era's `cv-baseline` (0.206),
  on pre-2026-09-08 annotations under `general_v1`; a paired join against it is
  invalid.

## Experiment history

`log.jsonl` holds the current era only. Closed eras live in
`archive/<first-date>_<slug>/`; `archive/README.md` covers the layout, how to
close an era, and how to reach deleted branches (`refs/archive/<slug>`).

Never archive to `.local/` or any gitignored path. That was tried in 2026-08; the
files evaporated and three docs kept citing them for a month.

## Running long jobs

Anything longer than a foreground command (stage 2, stage 3, a weights
download, a slow diagnostic) goes through two tools:

```bash
tools/launch_job.sh <log> -- <command...>   # prints the pid and the watch command
tools/watch_job.sh <pid> --log <log>         # run it as a Monitor, persistent: true
```

Their headers document the options. What they encode, so nobody retries the
alternatives:

- **`run_in_background` doesn't work.** A pipeline job launched with it is
  SIGKILLed within ~15–60 s, and a waiter loop within a minute or two. A leading
  `sleep` in a Bash call is blocked outright.
- **`pgrep -f` and `pkill -f` match their own shell.** Claude Code runs each call
  as `bash -c '<command>'`, so an unbracketed pattern always finds a "running"
  job, and `pkill -f` has killed its own shell. `watch_job.sh` follows the PID.
  If you must match a pattern, bracket its first letter:
  `pgrep -af "[0]3_train/main.py"`.
- **The heartbeat keeps the prompt cache warm.** The cache's ~1 h TTL refreshes
  on every read, so wakes under an hour apart carry one agent through a run of
  any length; a slow job is no reason to hand off. Answer a progress or
  heartbeat line in one line: anything read while waiting is paid for twice.
- **The GPU is hidden by default.** The 12288-d trunk embedders OOM the 4 GB
  card, and CPU ≈ GPU for the probe. `--gpu` opts out.

## Testing

You may create and freely modify:

- a dummy annotation project `01_annotate/test/`
- a dummy set under `02_set/sets/`
- dummy models prefixed `test_` (e.g. `test_new_metrics`)

Ask before creating, editing, or deleting any *existing* annotation project,
training set, or model unless you have been directed to do so.
