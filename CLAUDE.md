# buzzdetect-training

Raw audio + annotations → a Keras probe (dropout + one dense layer) over frozen
audio embeddings, for insect buzz detection.

`README.md` is the operator's guide: how to run each stage, what every flag
does, what the outputs mean, and the design rationale for folds and roles. Read
it before changing pipeline behaviour. `LOOP.md` is the autoresearch loop.

This file holds only what isn't obvious from the code itself. Stage-local
gotchas live next to their stage — `02_set/CLAUDE.md`, `03_train/CLAUDE.md`.

## Layout

| Stage | Entry point | Language |
|------------------|-------------------------------|------------------------|
| 1\. Combine annotations per effort | `01_annotate/MAKE.R` | R |
| 2\. Build a set, extract embeddings | `02_set/sets/<set>/build.R`, `02_set/main.py` | R, Python |
| 3\. Train (leave-one-fold-out CV) | `03_train/main.py` | Python |

Root `main.py` chains 2→3. It resolves stage paths relative to the cwd, so it
only works from the project root. There is no stage 4 — it was retired with the
CV rework.

Environment: `conda run -n buzzdetect-train python <script>`.

## Invariants

- **`import tensorflow` must come first** in any entry point that later imports
  pandas. Every entry point has the full explanation in a header comment; keep
  it there and keep the import at the top.
- **`config.py` anchors `ROOT` to its own directory**, so stage scripts can be
  invoked from any cwd. Add paths there, not as literals in stage code.
- **An annotation effort is a directory under `01_annotate/` holding a
  `combine.R`** — that file is the whole contract, and it must write both
  `annotations_combined.csv` and `folds.csv` (fold assignment included).
  `MAKE.R` discovers efforts by that file, not by `.Rproj`, which is gitignored.
  Do not modify `01_annotate/` — changing the data under an experiment makes it
  incomparable to everything else in the log.

Embedder interface — `embedders/embedding.py`; model loader — `models/models.py`.

## Known stale

Kept as artifacts, never as comparators:

- `models/model_general_v3/` — predates the CV rework.
- `models/yamnet_medium_general/` — the 2026-08 era's `cv-baseline` (0.206).
  Trained under `general_v1` on the pre-2026-09-08 annotations, so a paired
  per-fold join against it is invalid: its folds no longer hold the same audio.

## Experiment history

`log.jsonl` holds the current era only. Closed eras live in
`archive/<first-date>_<slug>/` — log, README, `notes/`, and a snapshot of the
set that produced the numbers. `archive/README.md` covers the layout, how to add
an era at the next cutover, and how to reach experiments whose branches were
deleted (`refs/archive/<slug>`).

Do not archive to `.local/` or any gitignored path. That was tried in 2026-08;
the files evaporated and three docs went on citing them for a month.

## Running long jobs

Stage 2 (`02_set/main.py`) and stage 3 (`03_train/main.py`) can both run far
longer than a foreground command should block, and **Claude Code's
`run_in_background` does not work for either of them.** This has cost every LOOP
agent a rediscovery — follow the recipe below and don't re-derive it.

- A `run_in_background` job running the pipeline is **SIGKILLed** within
  ~15–60 s — no OOM, no traceback (stage 2: ~4/4 across 2026-09-01/02; stage 3
  survived it once, not reliably).
- A `run_in_background` job that only *waits* on a detached run (an
  `until grep …; do sleep …; done` loop) is **also killed**, within a minute or
  two (2026-09-02).
- A leading `sleep` in a Bash call is blocked by the harness outright.

### What works

**1. Launch the job itself detached from the shell** — direct env-python (no
`conda run` wrapper process), unbuffered, logging to a file in the worktree.
**Capture its PID** (`echo $!`); a `pkill -f` on the job name has previously
killed the shell running it, so prefer read-only `pgrep`.

**Always bracket the first letter of a `pgrep -f` / `pkill -f` pattern.** Claude
Code runs every Bash call as `bash -c '<your command>'`, so your pattern text is
itself in a live process's command line and **`pgrep -f foo` matches the shell
that is asking**. It reports a match for a job that is not running, and
`pkill -f` kills its own shell (observed 2026-09-09: exit 144).

```
pgrep -af "[0]3_train/main.py"    # right
pgrep -af 03_train/main.py        # WRONG: always matches, even with nothing running
```

The bracket works because the regex `[0]3_train` does not match the literal text
`[0]3_train` in the wrapper's own command line. It fails if the unbracketed
string also appears elsewhere in the same Bash call, so don't echo the plain
name alongside the check.

This cost a whole grid launch on 2026-09-09: a detached
`while pgrep -f <job>; do sleep; done` waiter could never exit, because the
pgrep matched the waiter itself. It was diagnosed at first as the harness
killing the waiter (the failure below) — it was not.

```
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  BUZZDETECT_CHUNK_FRAMES=48 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  02_set/main.py --set <set> --embedder <emb> --workers <n> --verbose \
  > extract_<set>.log 2>&1 &
disown; echo "pid $!"
```

For stage 3 drop `BUZZDETECT_CHUNK_FRAMES`; keep `CUDA_VISIBLE_DEVICES=""` (the
12288-d trunk embedders OOM the 4 GB GPU, and CPU ≈ GPU on this box for the
1024-d probe anyway). Pass `--verbose` to stage 3 or it trains silently.

**2. Launch it, then end the turn.** Do not wait in-session, and do not
`Monitor` a job whose next real event is more than ~1 h away — persistent or
not. A persistent Monitor looks like the fix, since it only wakes on a real
state change, but the prompt cache has its own ~1 h TTL independent of
Monitor's: when events are sparser than that the cache goes cold in the gap
either way, and every wake pays full uncached-context price.

Come back on your own schedule instead — cheap, and it doesn't hold the
session's cache hostage to the job's pace:

```
pgrep -af "[0]3_train/main.py"   # or "[0]2_set/main.py" — bracket it, see above
find models/<name>/folds -name summary.json | wc -l
tail -5 <log>
```

To be *notified* rather than check back, it must happen outside this session — a
`nohup`-detached shell loop (launched like the job itself, not via
`run_in_background`) that polls for the completion marker and then invokes a
fresh `claude` CLI call.

Cost is entirely config-dependent and the two ends are orders of magnitude
apart, so **measure before assuming**: a frozen-probe CV is ~9 min and needs
none of this machinery, while a trunk fine-tune is ~24 h and needs all of it.

## Testing

You may create and freely modify:

- a dummy annotation project `01_annotate/test/`
- a dummy set under `02_set/sets/`
- dummy models prefixed `test_` (e.g. `test_new_metrics`)

Ask before creating, editing, or deleting any *existing* annotation project,
training set, or model unless you have been directed to do so.
