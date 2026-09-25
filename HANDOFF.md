# Handoff: era cv-medium-v4 opening queue

Launched 2026-09-25 ~16:52 (relaunched after a first attempt died on missing PyAV; log `queue_v4_grid.failed-noav.log`), expected to run through the weekend (~20-30 h).

| | |
|---|---|
| Script | `tools/queue_v4_grid.sh` (read its header; it's the plan) |
| PID | `2753184` (the outer `launch_job` pid, alive for the whole queue) |
| Log | `queue_v4_grid.log` (project root) |
| Launched as | `tools/launch_job.sh queue_v4_grid.log -- tools/queue_v4_grid.sh` |
| Commit | see `git log -1 -- HANDOFF.md` |

## Checking on it

```bash
tools/watch_job.sh --adopt 2753184          # arm as a Monitor, timeout_ms 1800000; re-arm at every expiry
grep -E '\[queue\]' queue_v4_grid.log       # which step it's on, retries, and the final summary
```

- Last line `[launch_job] exit 0` plus `[queue] ... QUEUE-DONE failed: none` means everything finished.
- `QUEUE-DONE failed: <steps>` means those steps failed 6 retries each. The queue
  still moved on, so later steps may have run. Read the log around each failed step.
- **Dead without an exit line** (killed, reboot): re-run the launch command above.
  Extraction is incremental and CV skips finished folds, so it resumes where it
  stopped. Update the PID here.
- Don't hand-poll (see CLAUDE.md "Running long jobs"). The Monitor surfaces each
  `[queue]` step, each CV headline, and the exit.

## What it runs (in order)

All on `medium` (now including Hard Negatives), `--translation general`.

| Model | Embedder | Head | Epochs |
|---|---|---|---|
| `cv-baseline-v4` | `yamnet` (frozen) | linear | 400 |
| `v4-ft` | `yamnet_trunk_depth12` | linear | 30 |
| `v4-ft-dense` | `yamnet_trunk_depth12` | Dense(1024) ReLU | 30 |
| `v4-ft-ps` | `yamnet_trunk_pitchshift_depth12` | linear | 30 |
| `v4-ft-ps-dense` | `yamnet_trunk_pitchshift_depth12` | Dense(1024) ReLU | 30 |
| `v4-ft-ctx` | `yamnet_trunk_context_depth12` | linear | 30 |
| `v4-ft-ctx-dense` | `yamnet_trunk_context_depth12` | Dense(1024) ReLU | 30 |
| `v4-ft-psctx` | `yamnet_trunk_pitchshift_context_depth12` | linear | 30 |
| `v4-ft-psctx-dense` | `yamnet_trunk_pitchshift_context_depth12` | Dense(1024) ReLU | 30 |

All `v4-ft*` runs retrain YAMNet layers 12-14 with the previous era's best settings
(`TRUNK_LR_BACKBONE=1e-5`, `TRUNK_FP16=1`, head LR 2e-4), at 30 epochs rather than
last era's 60: the old best's cross-fold mean val sens plateaus from epoch 20 to 40 (0.378),
then dips slightly at 60 (0.374). `v4-ft-ps` is the direct
rerun of last era's best (`trunk-pitchshift-depth12`, 0.472 / 0.454 on two runs),
now on the new data. After training, it extracts `yamnet` and
`yamnet_trunk_pitchshift_depth12` on `moderate` (the era's big-data set), but
trains nothing there.

## Known risks

- **`v4-ft-psctx*` host RAM.** Its embedding is 73,728-d float16 (~12 GB for medium)
  on a 23 GB machine. It's ordered last so it can't block anything else. If it
  keeps failing, the fallback is the known leak fix noted in the archived
  `trunk-ft-pitchshift` entry: explicitly `del` each fold's training data.
- A deterministic crash burns 6 retries before moving on. That's expected, not a hang.

## After it finishes (not done yet)

1. Log each model with `tools/log_entry.py`. `cv-baseline-v4` is the era anchor;
   read the `v4-ft*` grid against it and against `v4-ft`.
2. `cv-baseline-v4` finished: 0.324 (0.262 incl. quiet), recorded in LOOP.md; still needs its log entry.
3. **Epoch check:** `v4-ft` was still rising slowly at epoch 30 (mean val sens incl. quiet
   0.297 at epoch 20, 0.305 at 25, 0.307 at 30; 2/8 folds peaked at 30). All grid runs share
   the 30-epoch budget, so comparisons within the grid are matched. Re-run the grid's winner
   at 60 epochs before treating its number as final.
4. Delete this file when the queue is logged.
