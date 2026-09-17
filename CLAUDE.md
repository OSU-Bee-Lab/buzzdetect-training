# buzzdetect-training

This is the project for training new passive acoustic monitoring pollinator detection models for buzzdetect.

- `README.md` is the operator's guide: running each stage, every flag, what the
  outputs mean, why folds are deployments. Read the relevant section before
  changing pipeline behaviour.
- `LOOP.md` is the experiment protocol for autoresearch.
- `02_set/CLAUDE.md` and `03_train/CLAUDE.md` hold stage-local gotchas.


## Layout

| Stage | Entry point | Language |
|---|---|---|
| 1\. Combine annotations per effort | `01_annotate/MAKE.R` | R |
| 2\. Build a set, extract embeddings | `02_set/sets/<set>/build.R`, `02_set/main.py` | R, Python |
| 3\. Train (leave-one-fold-out CV) | `03_train/main.py` | Python |
| 4\. Deploy (export to buzzdetect) | `04_deploy/main.py` | Python |

Root `main.py` chains 2→3 and only works from the project root. Stage 4 is
separate and manual.

Environment: `conda run -n buzzdetect-train python <script>` for anything short;
long jobs are below. Embedder interface: `embedders/embedding.py`. Model loader:
`models/models.py`.

`DEPLOYMENTS.md` maps each fold to its crop, site and date.
Check if you need further interpretation on folds.

## Invariants

- **`import tensorflow` must come first** in any entry point that later imports
  pandas. Each entry point's header comment explains why; keep it there.
- **`config.py` anchors `ROOT` to its own directory**, so stage scripts run from
  any cwd. Add paths there, not as literals in stage code.
- Do not modify `01_annotate/` unless directly instructed; these projects are essential raw data source. 

## Experiment history

`log.jsonl` holds the current era only. Closed eras live in
`archive/<first-date>_<slug>/`; `archive/README.md` covers the layout, how to
close an era, and how to reach deleted branches (`refs/archive/<slug>`).

Never archive to `.local/` or any gitignored path. That was tried in 2026-08; the
files evaporated and three docs kept citing them for a month.

## Running long jobs

Anything longer than a foreground command (stage 2, stage 3, a weights
download, a slow diagnostic, a smoke test that extracts) goes through one tool.
If you can't be sure it finishes in a minute, launch it: a new embedder over
the lite set runs for minutes on CPU, and piping a raw run through `tail`
hides all output until it exits.

```bash
tools/launch_job.sh <log> -- <command...>   # detaches the job and starts its notifier
```

Then wait. launch_job runs a notifier that pings you on milestones, errors, completions, or every 50 min if the job is still running.
There is nothing else to arm or poll (ScheduleWakeup is for `/loop`, not jobs) and there is no need to sleep.
Pings arrive as "Another Claude session sent a message: [job …]".

Summarize each ping minimally in the main session context, a few words. They aren't for the user, they're for you.

Trust the notifier: read the log only when a ping looks wrong, such as a job running well past what you expected.
DONE, FAILED and error pings need real attention. 

To get pings for a job another session launched (a HANDOFF.md resume): `tools/notify_job.sh <pid> --log <log>`.

**`run_in_background` doesn't work.** A pipeline job launched with it is SIGKILLed within ~15–60 s, and a waiter loop within a minute or two.
A leading `sleep` in a Bash call is blocked outright. launch_job is designed to solve all of these problems!

- **`pgrep -f` and `pkill -f` match their own shell.** Claude Code runs each call
  as `bash -c '<command>'`, so an unbracketed pattern always finds a "running"
  job, and `pkill -f` has killed its own shell. The notifier follows the PID.
  If you must match a pattern, bracket its first letter:
  `pgrep -af "[0]3_train/main.py"`.
  

## Gotchas

- **Tools always run from the main checkout.** Each `tools/*.sh` re-runs the main
  checkout's copy, because a worktree's `tools/` is frozen at its branch point.
  A branch from before 2026-09-14 lacks that redirect: merge main into it first.
- **`launch_job.sh` uses the GPU by default.** `--cpu` hides it. If a GPU job dies out of memory, deciding whether to re-run it with `--cpu` is up to you; a re-run resumes where it stopped. The 4 GB card fits one GPU job at a time: don't launch a second alongside it.

## Hardware settings (benchmarked on `bench/gpu-vs-cpu` branch)

- **Training: prefer the GPU; fall back to CPU only on OOM.** The gain scales with how much there is to fit. A bare linear probe is overhead-bound, so GPU is only ~1.15x (a 400-epoch fold is ~3 min either way). A `--hidden 1024` head on YAMNet is ~8x (~4 min vs ~27 min per fold). A fold that crashes can be resumed: re-running the same command skips finished folds.
- **YAMNet extraction:** GPU is ~1.6x faster than CPU (the card sits ~10% busy, held back by the CPU). With a GPU visible, stage 2 caps `--workers` (embedding processes) at 1, since forked workers collide on the card; `--snip-workers` is unaffected. On CPU (`--cpu`), `--workers 4` gives ~1.6x over 1.
- **AVES extraction: use the GPU** (`--workers 0`, `BUZZDETECT_NO_GPU=1`). It is ~10x faster: ~19 min for medium vs ~3 h on CPU.
  If a YAMNet+AVES embedder OOMs anyway, add `BUZZDETECT_AVES_BATCH=16` (default 64) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `yamnet_aves_mid_avesshift` needed both on lite.
- **Keep `BUZZDETECT_CHUNK_FRAMES=48`.** It changes YAMNet's output at chunk edges (~1 frame in 48), so existing embeddings only match at 48. Larger values barely speed things up and use more VRAM (300 ≈ 2.3 GB). AVES ignores it.

## Testing

You may create and freely modify:

- a dummy annotation project `01_annotate/test/`
- a dummy set under `02_set/sets/`
- dummy models prefixed `test_` (e.g. `test_new_metrics`)

Ask before creating, editing, or deleting any *existing* annotation project,
training set, or model unless you have been directed to do so.


## Starting a new era
If the user request a new era, read `python tools/archive_era.py`.
Run the script and start a fresh log.

