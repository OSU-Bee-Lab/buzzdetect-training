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
tools/launch_job.sh <log> -- <command...>   # detaches the job; prints its pid
```

Then watch it with Claude Code's built-in **Monitor** tool, `timeout_ms: 1800000`
(the 30-min maximum). **One Monitor per session, not per job:**

```bash
tools/watch_job.sh            # every job this session launched, and any it launches later
```

It follows every job launch_job started from this session, including ones
launched after it was armed, so starting a second job (extraction, then
training) needs no new Monitor, and N jobs still mean one expiry per 30 min.
Arming a second Monitor, or one per job, multiplies your turns: expiries at
different offsets land every ~30/N min. Check with TaskList before arming. A
job another session launched (a HANDOFF.md resume) joins with
`tools/watch_job.sh --adopt <pid>`; `tools/watch_job.sh <pid> <log>` still
watches one job alone.

Its first lines are each job's state, timestamped: running or not, folds done,
error lines so far, the last log line (a job that ended while no Monitor was
armed is reported once). Re-arming therefore also catches anything logged in
between. After that it stays silent until something completes: stage 3's CV
headline, the shipped model, and each job's closing `[launch_job] exit N`,
prefixed with the job's log name. It exits when the last job does. The same
watch fits any command (a one-off diagnostic too): the state line and exit line
are generic, only the headline and shipped events are pipeline-specific. No
per-fold or per-error events: a crash ends the job and arrives as its exit
line, and an error that leaves a job hanging shows in the next re-arm's error
count. **Re-arm the Monitor at every expiry, every time, until the jobs are
done.** Expiries are expected and cheap; re-arming keeps the session alive and
its cache warm. A lapsed Monitor means nobody is watching.

**Between Monitor events, do not look at the job at all.** The re-arm's first
line is your check. No `tail`/`grep` of the log, no `nvidia-smi`, no
`ReadNotifications` unless a notice says some are pending, no `sleep`, no
ScheduleWakeup (that's for `/loop`). The known failure is a check-in loop:
ReadNotifications (empty), then `tail -3 train.log`, then again, once per
epoch (~40 s). Each check is a full turn, and one batch-12 session burned its
budget on it. If you want to know sooner, the Monitor is the thing to change:
add a pattern to watch_job's `EVENTS` in main, don't hand-poll around it. If
you have nothing to do until an event, end your turn: the event wakes you.

`exit 0` is done; any other exit, or a rising error count, needs real attention. Read
the log when an event looks wrong or a job runs well past what you expected.
Don't PushNotification about jobs, not even on failure or completion: Luke
checks in when he wants. Push only when you can't go forward without his input.

**`run_in_background` doesn't work.** A pipeline job launched with it is SIGKILLed within ~15–60 s.
A leading `sleep` in a Bash call is blocked outright. launch_job exists for exactly this.

- **`pgrep -f` and `pkill -f` match their own shell.** Claude Code runs each call
  as `bash -c '<command>'`, so an unbracketed pattern always finds a "running"
  job, and `pkill -f` has killed its own shell. watch_job follows the PID.
  If you must match a pattern, bracket its first letter:
  `pgrep -af "[0]3_train/main.py"`.
- **Watch launch_job's own pid, not a `retry()`-wrapped chain's inner subprocess.**
  `chain*.sh` scripts wrap each step in `retry()`; a kill mid-fold respawns a new
  python pid every attempt, but the outer `launch_job.sh` pid (what it printed,
  and what the chain resumes under) lives for the whole chain. Watching an inner
  pid means watch_job exits (job "failed") on every retry, not just a real end.
  

## Gotchas

- **Tools always run from the main checkout.** Each `tools/*.sh` re-runs the main
  checkout's copy, because a worktree's `tools/` is frozen at its branch point.
  A branch from before 2026-09-14 lacks that redirect: merge main into it first.
- **Tools find Python via `paths.local.json`'s `"python"`** (`tools/python_path.sh`),
  falling back to `<conda base>/envs/buzzdetect-train`. They run on Linux and macOS.
- **`launch_job.sh` uses the GPU by default.** `--cpu` hides it. If a GPU job dies out of memory, deciding whether to re-run it with `--cpu` is up to you; a re-run resumes where it stopped. First check where it died: an OOM after a fold's training finished (scoring, surprisal, export) is an unbatched `model()` call, not a fold too big for the card; batch that call like `_score_fold` does instead of falling back to `--cpu`, which can be ~100x slower on trunk embedders. The 4 GB card fits one GPU job at a time: don't launch a second alongside it.

## Hardware settings (benchmarked on `bench/gpu-vs-cpu` branch)

- **Training: prefer the GPU; fall back to CPU only on OOM.** The gain scales with how much there is to fit. A bare linear probe is overhead-bound, so GPU is only ~1.15x (a 400-epoch fold is ~3 min either way). A `--hidden 1024` head on YAMNet is ~8x (~4 min vs ~27 min per fold). A fold that crashes can be resumed: re-running the same command skips finished folds.
- **YAMNet extraction:** GPU is ~1.6x faster than CPU (the card sits ~10% busy, held back by the CPU). With a GPU visible, stage 2 caps `--workers` (embedding processes) at 1, since forked workers collide on the card; `--snip-workers` is unaffected. On CPU (`--cpu`), `--workers 4` gives ~1.6x over 1.
- **AVES extraction: use the GPU** (`--workers 0`, `BUZZDETECT_NO_GPU=1`). It is ~10x faster: ~19 min for medium vs ~3 h on CPU.
  If a YAMNet+AVES embedder OOMs anyway, add `BUZZDETECT_AVES_BATCH=16` (default 64) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; `yamnet_aves_mid_avesshift` needed both on lite.
- **Perch extraction is CPU-only.** It runs in `.local/venv-perch-extract` (TF 2.21), which sees no GPU: its CUDA libraries don't load. perch-probe extracted medium on CPU on purpose (`CUDA_VISIBLE_DEVICES=""`, 32 frames/pass, 2 workers; `run_perch.sh` in that worktree). Nobody has benchmarked Perch on the GPU; making the venv see it is unsolved.
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

