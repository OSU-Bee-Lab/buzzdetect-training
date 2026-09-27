# Handoff: era cv-medium-v4, pitch-shift method x direction grid

**Done 2026-09-27 03:53, all six clean (no retries, pshud not RAM-killed); logged in `log.jsonl`.** Verdict: up is the effect (0.452-0.455), down alone ~0.39-0.40 (noise-level over `v4-ft`), vocoder = resample, down added to up is flat. `v4-ft-ps` stays the pick. Next open items: steps 3-4 below (step 3 moot unless a vocoder variant is revisited).

Launched 2026-09-26 18:04. Luke is out of office and asked for all six runs at
once instead of gating the later ones on down-alone.

| | |
|---|---|
| Script | `tools/queue_v4_psd.sh` (its header is the plan) |
| PID | `2998219` (the outer `launch_job` pid) |
| Log | `queue_v4_psd.log` (project root) |
| Launched as | `tools/launch_job.sh queue_v4_psd.log -- tools/queue_v4_psd.sh` |

Watch it with `tools/watch_job.sh --adopt 2998219` as a Monitor. If it died
without an exit line, re-run the launch command: it resumes where it stopped
(finished extractions and folds are skipped).

Expected time: about 12-14 h. Each CV is ~1.5 h of training (8 folds, ~11 min
each at `v4-ft-psud`'s pace; four views will be slower) plus extraction. Vocoder
extraction is slower than resample: ~0.9 s per 48 frames for up+down, against
~0.3 s for resample.

Checked at 18:35: `psd` extraction took 9 min, and two folds finished cleanly
(~8 min each, host RAM ~8 GB, no errors). Fold sens: 1_29 0.506, 53 0.514.

## Where the grid came from

Already logged (`log.jsonl`):

- `v4-ft-psud` [plain, up, centre-down] = 0.452, the same as `v4-ft-ps`
  [plain, up] = 0.452. Adding down to up was flat.
- `v4-ft-ps-e60` = 0.468, +0.016 over 30 epochs. That is about one delta SD
  (~0.014) and under the ~0.027 MDE, so it's weak evidence. 60 epochs is fine
  for the moderate run.

Luke's mechanism for down: halving every frequency drops truck and prop-plane
rumble below YAMNet's 125 Hz mel floor, while buzz harmonics stay in range. So
the down direction gets its own tests, across three methods.

## The six runs (in queue order, all 30 epochs, matched to `v4-ft-ps`)

| Model | Views after plain | Embedder |
|---|---|---|
| `v4-ft-psd` | resample down, centre half | `yamnet_trunk_pitchshift_down_depth12` |
| `v4-ft-pshd` | resample down, first half + second half | `yamnet_trunk_pitchshift_halves_down_depth12` |
| `v4-ft-vd` | vocoder down | `yamnet_trunk_vocoder_down_depth12` |
| `v4-ft-vu` | vocoder up | `yamnet_trunk_vocoder_up_depth12` |
| `v4-ft-vud` | vocoder up, vocoder down | `yamnet_trunk_vocoder_updown_depth12` |
| `v4-ft-pshud` | resample up, down first half, down second half | `yamnet_trunk_pitchshift_halves_updown_depth12` |

The full grid, with existing results:

| | down | up | down+up |
|---|---|---|---|
| Resample (centre) | `v4-ft-psd` | `v4-ft-ps` 0.452 | `v4-ft-psud` 0.452 |
| Resample (both halves) | `v4-ft-pshd` | = resample up | `v4-ft-pshud` |
| Phase vocoder | `v4-ft-vd` | `v4-ft-vu` | `v4-ft-vud` |

The no-shift baseline is `v4-ft` = 0.375 (same fine-tuned depth12 trunk, same
LRs and epochs).

The new embedders share `embedders/trunk_views.py`: a list of views, each
through the one shared tail. A smoke test confirmed the following before launch:

- The plain view is bit-identical across all six.
- `psd`'s down view matches `psud`'s exactly.
- The base's up view reproduces `v4-ft-ps`'s exactly.
- The vocoder views really differ from the resample views.

None of the embedders has `to_onnx()`, so they are CV only.

## Risks

- **`v4-ft-pshud` is four views**, the biggest host-RAM load yet on this trunk.
  `v4-ft-psud` (three views) used 10-12 GB of host RAM; the `psctx` runs were
  killed at a larger footprint. It runs last on purpose. If it dies from RAM,
  the other five are done anyway; report it rather than fixing it (Luke's
  standing call on psctx).
- Disk: about 30 GB of new embeddings; 143 GB was free at launch.

## Next

1. Log each run with `tools/log_entry.py --baseline-model models/v4-ft-ps
   --baseline-name v4-ft-ps --branch main`, and state the delta against
   `v4-ft` (0.375) in the conclusion too for the down-only runs.
2. Read the grid:
   - Direction: are the down-only runs above `v4-ft`?
   - Method: vocoder vs resample, per direction.
   - Coverage: both-halves vs centre.
   Differences under ~0.027 are noise-level. Repeat any winner before trusting
   it, and check the hard folds (1_150, 1_95, willard) per fold with ± SD.
3. A vocoder winner has a deployment cost: phase vocoder per frame in
   buzzdetect. Measure it before recommending it.
4. Still pending from before: train the overall winner on `moderate` at 60
   epochs. That's Luke's call, not a loop experiment.
