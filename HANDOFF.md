# Handoff: era cv-medium-v4 follow-up queue

The opening grid (`queue_v4_grid.sh`) finished 2026-09-26 02:23 and is logged in
`log.jsonl`. Winner: `v4-ft-ps` = 0.452. Both `v4-ft-psctx*` runs were killed
for host RAM; per Luke, psctx will not be fixed.

Launched 2026-09-26 ~08:50, expected ~4-5 h.

| | |
|---|---|
| Script | `tools/queue_v4_psud.sh` (read its header; it's the plan) |
| PID | `2953300` (the outer `launch_job` pid) |
| Log | `queue_v4_psud.log` (project root) |
| Launched as | `tools/launch_job.sh queue_v4_psud.log -- tools/queue_v4_psud.sh` |

Watch it with `tools/watch_job.sh --adopt 2953300` as a Monitor. If it died
without an exit line, re-run the launch command: it resumes where it stopped.

## What it runs

1. `v4-ft-psud`: embedder `yamnet_trunk_pitchshift_updown_depth12`, views
   [plain, octave-up, octave-down], 30 epochs. Compare it to `v4-ft-ps`.
2. `v4-ft-ps-e60`: `v4-ft-ps` at 60 epochs. This is the epoch confirmation.

## Next (Luke's plan)

- Log both runs.
- Train the overall winner on `moderate`. Luke asked for this explicitly: it's
  his call, not a loop experiment. Use 60 epochs if `-e60` holds up.
- The updown embedder has no `to_onnx()`. It needs one before any deployment.
- Dense heads are overfitting, not under-trained. Their val loss is lowest at
  epoch 3-5 and their val sens is flat or falling after epoch 10. So 60 epochs
  would not rescue them.
