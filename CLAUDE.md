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

`DEPLOYMENTS.md` maps each fold to its crop, site and date. Nothing reads it, but
every "hard fold" conclusion is really a conclusion about a crop.

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

**2. Launch it, then measure before deciding how to wait.**

> **The decision rule lives in LOOP.md, under "The first fold is a gate."**
> Read it there; it is not repeated here, so that there is one copy to keep
> right. In one line: launch detached, note the wall clock, **wait for the
> first fold's `summary.json`**, and compute the ETA from that file's mtime —
> never from the epoch rate, the kind of job, or the cost table. Then decide
> from the **gap between folds**, not the total: under ~50 min per fold, a
> per-fold `Monitor` carries the run however long it is; over, `HANDOFF.md`.
>
> ```
> find models/<name>/folds -name summary.json -printf '%T@ %p\n' | sort -n
> ```
>
> For stage 2, which has no folds, the per-ident progress lines in the log are
> the equivalent first unit — same rule, same prohibition on guessing.

The rest of this section is the *mechanics* the rule depends on: what a Monitor
has to cover, and how to check by hand if you are handing off instead.

**Why ~50 min, and why it is a gap and not a duration.** The prompt cache's TTL
is ~1 h, and **it refreshes on every read** — it measures time since the cache
was last *used*, not since it was created. So any wake inside the hour renews
it, and a chain of them holds one warm indefinitely. What can kill it is a
single silent stretch longer than the TTL, which means the quantity that
decides a handoff is the **longest gap between events**, never the run's total
length. 50 min is the TTL with a margin for a fold running late.

The consequence is the important part: **a 4 h CV whose folds land every
~20 min needs no handoff at all.** One agent carries it start to finish, each
fold event renewing the cache, and the experiment keeps the hypothesis and the
launch decisions that were never written down. A handoff is not the cheap
option it looks like — the incoming agent pays *full uncached* price to re-read
LOOP.md, CLAUDE.md, `log.jsonl` and `IDEAS.md`, and reorients into work it did
not design. That is worth far more than a dozen cached re-reads at ~10% of
input price. **Hand off when the cache cannot be kept alive, not when the run
is long.**

A completion Monitor must also **cover the failure states**, or a crash is
indistinguishable from a long fold — poll for the completion marker, for
tracebacks in the log, *and* for the trainer having vanished without either.

**Fire one event per fold, plus the terminal states.** Each fold event is doing
two jobs at once: it is the progress line, and it is the cache keepalive that
makes a long run survivable without a handoff. This was got backwards once —
an earlier revision of this file argued for terminal-state-only firing on the
grounds that the TTL is absolute and there is nothing to ping. **That premise
was false** (see above), and with it the conclusion: silence is not free, it is
what eventually forces the handoff.

Keep the events cheap rather than rare. A fold event should be one short line,
and **the right response to one is one short line back** — acknowledge and stop.
Do not re-read the log, re-check the diff, or start a side task on a fold
event; everything read while waiting is paid for twice, and that cost, not the
wake itself, is what makes a progress meter expensive.

**The 50-minute timeout is the handoff trigger, and it belongs in the monitor.**
If no fold lands within it, the cache is about to lapse whatever you do, so the
monitor says so explicitly and the agent writes `HANDOFF.md` *then* — at the
moment it is actually needed, on measured evidence, rather than being
guessed at launch. This is also the hang detector: a wedged trainer and a fold
too slow to keep the cache alive call for the same action.

```
stall_after=3000   # ~50 min: the cache TTL with margin
newest() { find models/<name>/folds -name summary.json -printf '%T@\n' 2>/dev/null | sort -n | tail -1; }
last=$(newest); last_change=$(date +%s); n=0
while true; do
  now=$(newest)
  if [ "$now" != "$last" ]; then
    last=$now; last_change=$(date +%s)
    echo "fold $(find models/<name>/folds -name summary.json | wc -l)/<total> done"
  fi
  [ -f <marker> ] && { echo "DONE"; break; }
  grep -qE "Traceback|MemoryError|Killed" <log> 2>/dev/null && { echo "CRASH"; break; }
  pgrep -f "[0]3_train/main.py" >/dev/null || { echo "STALLED: trainer gone"; break; }
  [ $(( $(date +%s) - last_change )) -gt $stall_after ] \
    && { echo "WRITE HANDOFF: no fold in ${stall_after}s, cache will lapse"; break; }
  sleep 30
done
```

The three terminal checks above are not optional — without them a crash is
indistinguishable from a long fold. The loop below is the shape for the three terminal states; keep it, and add the
stall check to it when a hang would cost you the turn:

```
while true; do
  [ -f <marker> ] && { echo "DONE"; break; }
  grep -lE "Traceback|MemoryError|Killed" <log> 2>/dev/null | grep -q . && { echo "CRASH"; break; }
  pgrep -f "[0]3_train/main.py" >/dev/null || { echo "STALLED"; break; }
  sleep 30
done
```

If you are handing off instead, the manual check is:

```
pgrep -af "[0]3_train/main.py"   # or "[0]2_set/main.py" — bracket it, see above
find models/<name>/folds -name summary.json | wc -l
tail -5 <log>
```

To be *notified* after your context is gone, it must happen outside this
session — a `nohup`-detached shell loop (launched like the job itself, not via
`run_in_background`) that polls for the completion marker and then invokes a
fresh `claude` CLI call.

## Testing

You may create and freely modify:

- a dummy annotation project `01_annotate/test/`
- a dummy set under `02_set/sets/`
- dummy models prefixed `test_` (e.g. `test_new_metrics`)

Ask before creating, editing, or deleting any *existing* annotation project,
training set, or model unless you have been directed to do so.
