# Handoff: era cv-medium-v4, top configurations on `moderate`

Launched 2026-09-27 ~15:55 at Luke's request ("launch the best three models on
moderate"). `v4-ft-pshud` was dropped (Luke: tie at 0.455, twice the cost), so
two CVs, both 60 epochs:

| Model | Embedder | Medium result |
|---|---|---|
| `v4-ft-ps-e60-moderate` | `yamnet_trunk_pitchshift_depth12` | `v4-ft-ps-e60` 0.468 |
| `v4-ft-vu-e60-moderate` | `yamnet_trunk_vocoder_up_depth12` (extracted on moderate in the queue) | `v4-ft-vu` 0.455 (30 epochs) |

| | |
|---|---|
| Script | `tools/queue_v4_moderate.sh` (its header is the plan) |
| PID | `3103244` (the outer `launch_job` pid) |
| Log | `queue_v4_moderate.log` (project root) |
| Launched as | `tools/launch_job.sh queue_v4_moderate.log -- tools/queue_v4_moderate.sh` |

Watch with `tools/watch_job.sh --adopt 3103244`. If it died without an exit
line, re-run the launch command: finished extractions and folds are skipped.

The trunk pool on moderate (~18 GB float16) doesn't fit in host RAM, so the queue
sets `TRUNK_STREAM=1` (new, `03_train/CLAUDE.md`). Expect ~5x medium's per-fold
time (5 rotating folds, ~5x frames, 60 epochs): very roughly 1.5-2 h per fold,
~9 h per model, plus the vu extraction (~1 h).

These are not loop experiments (moderate is final confirmation), so there is no
log.jsonl entry and no matched moderate control. Compare the two against each
other with `tools/results.py`, and note moderate has 5 rotating folds, not medium's 8.
Shipped models are left untrained.

## Shipped models (queued 2026-09-27 ~18:00)

`tools/queue_v4_moderate_ship.sh` (launch_job pid `3110322`, log
`queue_v4_moderate_ship.log`) waits for the CV queue to exit, then trains both
shipped models with `--skip-cv`. It skips any model whose CV has fewer than 5
folds. Re-launch with the same command (pass no pid if the CV queue is gone).
vu has no ONNX export, so its shipped model can't go through stage 4 yet.
