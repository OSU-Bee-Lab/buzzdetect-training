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

## Running jobs (as of 2026-09-29 ~09:30; check `tools/watch_job.sh` or the logs)

**Since 2026-09-29 afternoon:** `main.py` (README.md) replaces these chains for new work: `main.py --runs "yamnet:a0.50:select fast32:a0.50 ..."`
is the same list, resumable per stage and mid-training, and works for any teacher. The chains below ran from the
`distill-lite` worktree against the pre-teacher layout; the generalization lives on branch `distill-generic` and
must not be merged into `distill-lite` while a chain runs from it (bash reads a running script incrementally, and
python scripts are re-read at every stage). After the chains finish: merge, `python 05_distill/migrate_layout.py`
(dry run, then `--apply`), point paths.local.json's `distill_cache` at the parent `.../distill-cache/` directory, and
`main.py --dry-run` should then show every finished run as done.

- `chain_frontends.sh` (launch_job pid 1161007, log `.local/distill/chain_frontends.log`): caches `_fe` (now `_mel`) inputs for
  rung B + V, then per run: train -> export -> eval -> speed -> record -> print frontier table. Resumable
  (re-running skips finished stages). Run list (`RUNS` env overrides): control `yamnet:a0.50:select`, then
  fast32, fast32h16, fast32h32, twofast32 (a0.50); fast32h16 a0.375 / a0.25, fast32h32 a0.25; two32, lo32
  (sensitivity-only: slower than YAMNet); twofast32 a0.375. Run names `fe_B_<frontend>_<arch>_s1[_select]`.
- `chain_frontends2.sh` (pid 1270986, log `chain_frontends2.log`): waits for chain 1, then caches and runs the
  band-placement twins `fast32lo` (100-2500 Hz) and `fast32h16lo`, same window/hop/bands as `fast32` / `fast32h16`
  so speed is identical and only band placement differs.
- ETA (estimate): chain 1 ~21:00, chain 2 ~00:00 on 2026-09-30. One GPU job at a time (4 GB card).
- Per CLAUDE.md, one Monitor on `tools/watch_job.sh`, re-armed at each expiry.
- If a chain died: relaunch the same command; finished stages skip. Chain 1 stages log `[chain] <name> <stage>`.

## Results so far (append as runs land; `ladder_record.py frontier` is the source of truth)

- 2026-09-29 09:48 control `fe_B_yamnet_a0.50_s1_select` (YAMNet front end, `--init select`): headline **0.694**
  (incl. quiet 0.580) vs 0.625 for the same model with the layer-wise refit init. So the simpler init did not hurt
  (it scored higher; one seed, a 0.07 gap is well above the ~0.02 noise, so the refit is not helping at 7000 steps).
  Compare new front ends against **0.694**, not 0.625.

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
