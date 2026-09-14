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
download, a slow diagnostic) goes through one tool:

```bash
tools/launch_job.sh <log> -- <command...>   # detaches the job and starts its notifier
```

Its header documents the options. What it encodes, so nobody retries the
alternatives:

- **Launch, then wait: the job pings you.** A notifier messages your session on
  each stage-3 fold, on an error, when the job ends, and every 50 min
  otherwise. Pings arrive as "Another Claude session sent a message: [job …]".
  Don't arm a Monitor, poll or sleep. To get pings for a job another session
  launched (a HANDOFF.md resume): `tools/notify_job.sh <pid> --log <log>`.
- **A ping gets one line of text and nothing else**: "Fold 3/5 done.", "Still
  extracting, 1h32m in." Each carries the time and the job's elapsed time,
  because you can't see when a message arrived. Trust the notifier: read the
  log only when a ping looks wrong, such as a job running well past what you
  expected. DONE, FAILED and error pings need real attention. Stage 2 pings
  only on errors, its end and the 50-min heartbeat, on purpose.
- **The heartbeat keeps the prompt cache warm.** Its ~1 h TTL refreshes on every
  read, so pings under an hour apart carry one agent through a run of any
  length; a slow job is no reason to hand off.
- **Tools always run from the main checkout.** Each `tools/*.sh` re-runs the main
  checkout's copy, because a worktree's `tools/` is frozen at its branch point.
  A branch from before 2026-09-14 lacks that redirect: merge main into it first.
- **`run_in_background` doesn't work.** A pipeline job launched with it is
  SIGKILLed within ~15–60 s, and a waiter loop within a minute or two. A leading
  `sleep` in a Bash call is blocked outright.
- **`pgrep -f` and `pkill -f` match their own shell.** Claude Code runs each call
  as `bash -c '<command>'`, so an unbracketed pattern always finds a "running"
  job, and `pkill -f` has killed its own shell. The notifier follows the PID.
  If you must match a pattern, bracket its first letter:
  `pgrep -af "[0]3_train/main.py"`.
- **The GPU is hidden by default.** The 12288-d trunk embedders OOM the 4 GB
  card, and CPU ≈ GPU for the probe. `--gpu` opts out.

## Testing

You may create and freely modify:

- a dummy annotation project `01_annotate/test/`
- a dummy set under `02_set/sets/`
- dummy models prefixed `test_` (e.g. `test_new_metrics`)

Ask before creating, editing, or deleting any *existing* annotation project,
training set, or model unless you have been directed to do so.
