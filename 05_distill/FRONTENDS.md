# Front-end frontier (05_distill), started 2026-09-29

Handoff-grade record: enough for a fresh agent to resume. Read `DESIGN.md` (contract),
`LADDER.md` (data-size ladder) and `README.md` (scripts) first; this file covers what came after.

## Why this exists

Luke wants **tiers of models**: a *standard* (the teacher, or the a0.50 student, per later metrics) and a
*lite* (faster, less sensitive). Goal stated 2026-09-29: **explore the speed / sensitivity frontier, no fixed
target**. "4x faster for half the sensitivity" was only an illustration; losing up to ~50% of the baseline's
sensitivity is fine *if speed improves*. Floors from LADDER.md still print, but the sensitivity floor
(headline >= 0.207 = 50% of the moderate baseline's 0.414) is the only one that matters now; the
"1.5x YAMNet at 200 s" floor is superseded by the frontier itself.

## State of the data-size ladder (closed, do not resume)

`.local/distill/ladder.jsonl`, rung B seed 1 (headline = `sensitivity_exclquiet` @ fpr 0.005, 5 rotating folds):

| run | headline | x YAMNet 20 s / 200 s |
|---|---|---|
| a0.50 (YAMNet front end, layer-wise refit init) | 0.625 | 1.47 (bench) / - |
| a0.50_d12 (layers 13-14 removed) | 0.623 | 1.61 / 1.42 |
| a0.375 | 0.519 | 1.67 / 1.48 |

A->B advanced (0.51 -> 0.625), B->C did not (0.608), so by the LADDER rule rung D never runs. `chain_ladder2.sh`
was **killed by hand at its streaming-loader sanity check** (that check exists only to validate the loader
rung D needs). `lad_B_s1_stream` in `.local/distill/runs/` is an unfinished leftover. Comparison points: baseline
0.414, teacher honest rotation 0.574, teacher ONNX via harness 0.692 (inflated: trained on those folds).
Students never see labels or eval deployments, but inherit fold knowledge through the teacher: 0.62 is not
"better than the teacher".

## Finding 1: speed is set by the front end, and by FFT length

`chain_frontier_speed.sh` (random weights, engine venv, GPU, x YAMNet; results
`.local/distill/arch_fe/` and `arch_fe2/` (`_shared/arch_fe*` after `migrate_layout.py`), `time_20.txt` / `time_200.txt`). YAMNet's own front end alone is 1.65x at 200 s, so no
YAMNet-front-end student can beat that. Front-end cost tracks **FFT length** (bands barely matter):
fft 256 ~2.7-3.1x, fft 512 (YAMNet) 1.65-2.0x, fft 1024 ~0.95x, fft 2048 ~0.45x. Any front end with a window
long enough to resolve a ~200 Hz fundamental (fft >= 1024) is *slower than YAMNet*. Lowering the frame rate
(hop 16 / 32 ms, 60 / 30 frames per patch) also helps. At 200 s (a0.50 / a0.375 / a0.25 trunk):
`fast32h16` 2.50 / 2.61 / 2.75, `fast32h32` 2.33 / 2.45 / 2.59, `fast32` 1.53 / 1.73 / 1.99,
`twofast32` 1.76 / 1.82 / 1.87, YAMNet front end 1.37 / 1.48 / -. `fast16` (16 bands) is *slower* than `fast32` with a trunk
(odd small-tensor kernel behaviour; unexplained). Timings taken under some CPU contention: repeat before quoting.

## Finding 2: band profile (`band_profile.py`)

Teacher-called buzz frames vs clean-negative frames, mean log-mel per YAMNet band: buzz frames are *quieter*
in every band and the standardised difference is nearly flat across frequency (d ~ -0.6..-1.2; ~66% of |d|
mass below 2.5 kHz, ~83% below 4 kHz). No narrow buzz band. Crude (mean level only, teacher-defined labels);
it does not rule out band-limited front ends, but gives no reason to expect a big win from them.

## The code

- `frontends.py`: `FRONTENDS` registry. A `Frontend` = channels (each: window, bands, fmin, fmax, fft) + `hop`
  (must divide 15360; frames per patch = 15360/hop). Channels stack as conv input channels and share band count.
  Numpy `mel_patches` (cache builder) and Keras `make_features_layer` (deployed graph), both continuous STFT then
  cut into patches, mel matrix = `tf.signal.linear_to_mel_weight_matrix`. `python 05_distill/frontends.py`
  describes every spec. `test_frontends.py` checks numpy vs Keras vs YAMNet's own layer (all < 3e-5).
- `student.py`: `build_student(..., frontend=name)`; `'yamnet'` is the untouched original path.
- `cache_fe.py`: decodes each slice that already has a targets npz and writes `mel` only to
  `<distill_cache>/_mel/<spec>/...` (float16 (n, frames, bands, channels); before 2026-09-29 the chains wrote
  `<teacher cache>/_fe/<spec>/`, which readers still find and `migrate_layout.py` moves). It stamps each spec dir with
  a fingerprint of the spec, so editing a spec without renaming it now stops the next run. Teacher `code`/`logits` are reused (they do not
  depend on the student front end). `check_cache_align.py`: numpy 'yamnet' spec vs stored mel agrees to ~2e-3
  except a few first-window elements of some mp3 slices (decode start effect, negligible).
- `distill_train.py --frontend X`: packs shards to `.local/distill/shards/<rung>__<X>` (mel from `_mel`, targets
  from the main cache). In-memory loader only. `--arch` gained `a0.25`.
  **Init:** a non-YAMNet front end cannot use the layer-wise refit (it pairs activations position by position,
  and the frequencies at those positions differ), so `--init yamnet` silently becomes `select` (channel
  selection + BN recalibration). Layer 1 gets YAMNet's kernel split across input channels. The **control run**
  (YAMNet front end, `--init select`) measures what that init change alone costs.
- `export_student.py export` reads the run's `frontend` from `curve.json`; `samples_min` = 15360 - hop + max window.
  ONNX vs Keras parity checked at 4e-6 (two-channel spec).
- `ladder_record.py`: rows now carry `frontend` and `init`; ladder-rule maths ignores non-YAMNet-front-end and
  init-control rows. `ladder_record.py frontier` prints every rung-B run by 200 s speed with headline and % of baseline.
- `bench_arch.py` names: `a0.50@two32` (trunk on front end), `fe@two32` (front end alone), `a0.25` etc; `--out DIR`.

## Running job (as of 2026-09-29 13:30; the handoff: enough to resume or take over)

One job runs everything, `main.py`, launched from the `distill-lite` worktree (branch `worktree-distill-lite`, which now
holds all of `main` plus this work; **do not edit `.py`/`.sh` under `05_distill/` there while it runs**, docs are safe):

```bash
tools/launch_job.sh .local/distill/main_stage5.log -- /home/luke/anaconda3/envs/buzzdetect-train/bin/python \
  05_distill/main.py --rung B --runs "fast32h16:a0.25 fast32h16:a0.25:lam=0 \
  fast32h16:a0.25:classes=ins_buzz+ambient_rain+human fast32h16:a0.25:classes=ins_buzz+ambient_rain+human:lam=0 \
  twofast32:a0.50 fast32h16:a0.375 fast32h32:a0.25 two32:a0.50 lo32:a0.50 twofast32:a0.375 fast32lo:a0.50 fast32h16lo:a0.50"
```
(launch_job pid 2020483 at 13:28; log `.local/distill/main_stage5.log`, stage lines `[chain] <label>: start/done/FAILED`.)
Order: the class-subset experiment first (below), then the rest of the frontier list. 12 runs, ~40-60 min each plus ~25 min
of shard packing for each front end not packed yet (two32, lo32, fast32lo, fast32h16lo; fast32h32 a0.25 reuses
fast32h32's): **ETA about 10-12 h, i.e. ~00:00-02:00 on 2026-09-30**. One GPU job at a time (4 GB card): do not start another.

- **Resume / restart:** rerun the same command (or a shorter `--runs`): finished stages skip, an interrupted training resumes from
  its last checkpoint (every 2000 steps). A failed stage stops the job with `[chain] <label>: FAILED exit N` and a nonzero
  `[launch_job] exit`; read the log above it. `main.py --dry-run` with the same `--runs` shows done/pending per stage.
- **Cancel:** `kill` the launch_job pid and its `main.py` / `distill_train.py` children by PID (never `pkill -f`, see CLAUDE.md).
- The chains that ran before (`chain_frontends*.sh`, pre-generalization) were killed at 13:27 by hand, with `twofast32`
  packed and just starting to train; their runs' data were migrated to `.local/distill/<teacher>/` and are all recognised
  by `main.py`. They are kept as the record; do not relaunch them (they hard-code the old layout).
- Data layout was migrated on 2026-09-29 13:28 (`migrate_layout.py`): `.local/distill/v4-ft-ps-e60-moderate/{runs,models,eval,shards,ladder.jsonl}`,
  `.local/distill/_shared/arch*`, and `<distill_cache>/_mel/<spec>` (was `<teacher>/_fe/<spec>`). paths.local.json's
  `distill_cache` is the parent `/media/server storage/distill-cache`.
- Branch state: `main` is an ancestor of `worktree-distill-lite`, so a fast-forward merge of that branch in the main
  checkout (`git merge --ff-only worktree-distill-lite`) finishes the merge. It was not done automatically because the
  main checkout has uncommitted changes (`diagnostics/2026-09-28_int8-ptq/README.md`, `tools/human/log_viewer.html`,
  `tools/log_entry.py`; none overlap the merge). The `distill-generic` worktree and branch were fully merged and removed (2026-09-29), so `distill-lite` is the only
place the code lives; the models and data are all under `.local/distill/` in the main checkout, never in a worktree.

## Results so far (append as runs land; `ladder_record.py frontier` is the source of truth)

- 2026-09-29 (rung B, seed 1, 7000 steps, a0.50 trunk, init select, headline `sensitivity_exclquiet` @ fpr 0.005; speed x YAMNet at 200 s):
  `fast32h32` **0.572** at 2.56x, `fast32h16` 0.562 at 2.51x, `fast32` 0.606 at 2.17x, against the YAMNet-front-end control 0.694 at
  1.39x. So the faster front ends cost ~0.09-0.13 headline for ~1.6-1.8x more speed, all still ~136-146% of the 0.414 baseline (floor
  0.207). Band-placement twins, `twofast32`, the smaller trunks and the class-subset runs are still to come.
- 2026-09-29 09:48 control `fe_B_yamnet_a0.50_s1_select` (YAMNet front end, `--init select`): headline **0.694**
  (incl. quiet 0.580) vs 0.625 for the same model with the layer-wise refit init. So the simpler init did not hurt
  (it scored higher; one seed, a 0.07 gap is well above the ~0.02 noise, so the refit is not helping at 7000 steps).
  Compare new front ends against **0.694**, not 0.625.

## Class-subset experiment (2026-09-29, queued in the relaunched main.py job)

Question: does distilling only `ins_buzz`, `ambient_rain`, `human` (and/or dropping the code-regression loss)
buy buzz sensitivity at a small width, where the frontier's sensitivity is lost (a0.375 = 0.519 vs a0.50 = 0.625 on
the YAMNet front end)? Not a speed lever by itself (the 15-way head is ~8 kFLOPs). Runs on the cheapest front end
and width, `fast32h16:a0.25` (init `select`, rung B, seed 1, 7000 steps), so they compare directly with that run's
full-class result (`fe_B_fast32h16_a0.25_s1`, in the standard list): `lam=0` (`..._lam0`), classes only
(`..._c-buzz-rain-human`), and both (`..._c-buzz-rain-human_lam0`). Read gaps below ~0.03 as ties (seed noise ~0.02,
one seed each). Caveat the other way: the other classes are supervision too (jets and machinery teach the trunk
what confounders look like, and jet false positives at 1_95 are a known problem), so a buzz-only student might gain
sensitivity and lose false-positive behaviour; the headline holds fpr fixed, so it would show up as a lower
headline, not a separate number. If the subset wins, the next step is the same subset at `fast32h16:a0.375`/`a0.50`
and a check of what buzzdetect does with a 3-column model.

## How to read the result / what to decide

Run `conda run -n buzzdetect-train python 05_distill/ladder_record.py frontier`. Questions, in order:
1. Control vs `ladder` a0.50 B (0.625): how much did the init change alone cost? Compare every new front end
   against the *control*, not against 0.625.
2. `fast32` vs `fast32h16` vs `fast32h32`: what does cheaper front end / lower frame rate cost in sensitivity.
3. `fast32lo` vs `fast32`, `fast32h16lo` vs `fast32h16`: pure band-placement effect (same speed).
4. Trunk shrink (a0.375, a0.25) on the best fast front end: where does sensitivity fall under 0.207.
5. Frontier = non-dominated (speed, headline) points; the lite tier is a choice along it (Luke's call).
Noise: A-rung repeat spread on `lost_pct` was 1.31 points; headline seed noise on the 5 folds is ~0.01-0.02 (see
memory noise-floor-cv). One seed per run, so read gaps below ~0.03 as ties.

## Caveats

- Eval scores each frame alone with zero padding (`eval_folds.py`), while training patches use real audio after the
  frame: slightly pessimistic for long windows (two32/lo32, `twofast32` ch0).
- Headline is `sensitivity_exclquiet` at fpr 0.005 of the student's own ONNX, not of the full pipeline.
- Speed here is random-weight ONNX on the GTX 1650 via the buzzdetect engine session (bench_arch harness). CPU numbers are
  informational (i7-2600, no AVX2).
- Not done / not planned: pruning always-zero output channels, non-uniform widths, a learned (conv) front end,
  int8, dropping more layers beyond d12, seeds beyond 1, rung C+ data for a front end. `IDEAS.md` has none of these yet.
- Nothing here is shipped to buzzdetect. Export writes only under `.local/distill/models/`. Shipping is Luke's call.

## Where things live

Checkpoints and curves `.local/distill/runs/<name>/`; ONNX `.local/distill/models/<name>/`; eval
`.local/distill/eval/<name>/folds_sx.csv`; table `.local/distill/ladder.jsonl`; caches under the `distill_cache` path
(`_mel/<spec>/`, formerly `_fe/<spec>/`); shards `.local/distill/shards/`. Per-teacher locations are `.local/distill/<teacher>/...` once migrated (README.md). All of `.local/` is gitignored and lives in the main checkout.
Code is in the `worktree-distill-lite` worktree, uncommitted as of this writing.
