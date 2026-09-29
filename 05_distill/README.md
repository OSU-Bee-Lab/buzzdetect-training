# 05_distill

Distilling a teacher (a deployed buzzdetect model, e.g. `v4-ft-ps-e60-moderate`: two YAMNet trunk
passes) into a single-pass MobileNetV1-style student: YAMNet's architecture with channel widths scaled
by alpha, and optionally a different, cheaper spectrogram front end. The student never sees labels or
the evaluation deployments; it learns the teacher's logits and head-input code on a large pool of
unlabelled deployment audio, and is judged on the same rotating CV folds as every other model.

Read `DESIGN.md` for the contract (teacher targets, sampling, loss, judging), `LADDER.md` for the
data-size ladder (closed), and `FRONTENDS.md` for the front-end frontier (the live experiment). This file
is the operator's guide.

## One command

```bash
tools/launch_job.sh <log> -- conda run -n buzzdetect-train python 05_distill/main.py \
    --teacher v4-ft-ps-e60-moderate --rung B --runs "yamnet:a0.50:select fast32:a0.50 fast32h16:a0.375"
python 05_distill/main.py --runs fast32:a0.50 --dry-run     # what is done, what would run
```

`main.py` runs the teacher stages once, then every student in `--runs` (`<frontend>:<arch>[:<init>]`)
through train, export, eval, speed and record, and with `--deploy` copies each into buzzdetect.
**Re-running the same command after a crash or a quit picks up where it stopped**: every stage is
skipped when its artifact exists, and training resumes from its last checkpoint. A failed stage stops the
whole run (`[chain] <label>: FAILED exit N`); fix it and rerun. Stage lines are `[chain] ...`, and the
eval prints the `[<name>] sens@fpr...` headline, so `tools/watch_job.sh` follows it like any other job.
Options: `--rung A-D` (D trains from streamed shards), `--steps`, `--seed`, `--prefix`, `--until <stage>`.
Stage list and the stage-to-script map are in `main.py`'s docstring.

## A new teacher

1. Train and deploy the teacher the usual way (`03_train`, `04_deploy`): it needs `models/<name>/`
   (config_model.json with `classes`, `activation_centers`, `set`, `folds_train`; folds.csv) and
   `<buzzdetect_dest>/<name>/model.onnx`. It must be probe-headed (a MatMul head on a code tensor) and
   16 kHz / 0.96 s framed; `teacher_onnx.py` checks both and stops otherwise.
2. `main.py --teacher <name> --rung B --runs ...`. Its cache lives in `<distill_cache>/<name>/`, its runs
   under `.local/distill/<name>/`. The class list, the dead classes (no activation center: zero loss, no
   readout), the code width, the eval set and the deployments to blacklist all come from the teacher's own
   files; nothing in this directory names a teacher except `dpaths.py`'s default.
3. Optional: `DISTILL_BASELINE` (or `distill_baseline` in paths.local.json) names the CV baseline model the
   headline is compared to (default `cv-baseline-v4-moderate`). The comparison rows in the ladder table
   only print for reference files that exist.

## Layout and what is shared

`dpaths.py` is the only place paths and teacher constants live. Data is shared at the level of what it
depends on, and stamped so a stale product is caught instead of silently mixed:

| product | where | depends on | shared across |
|---|---|---|---|
| ffprobe durations | `<distill_cache>/_shared/durations.csv` | the file (relpath, size) | all teachers |
| spectrograms | `<distill_cache>/_mel/<spec>/<relpath>/h<hour>.npz` + `fingerprint.json` | audio + front-end definition | all teachers, all architectures |
| plan (slices, rungs, blacklist) | `<distill_cache>/<teacher>/_manifest/` | the teacher's training deployments | that teacher's students |
| teacher targets | `<distill_cache>/<teacher>/<relpath>/h<hour>.npz` (`code`, `logits`) | audio + teacher ONNX | that teacher's students |
| packed rungs | `.local/distill/<teacher>/shards/<rung>[__<frontend>]` | rung + front end + teacher + plan | all architectures of a (rung, front end) |
| runs, models, eval, `ladder.jsonl` | `.local/distill/<teacher>/` | one student | - |
| random-weight speed timings, `frontend_only.onnx` | `.local/distill/_shared/arch/` | the architecture | all teachers |

Pre-split caches embedded the YAMNet `mel` in every targets npz; readers still accept that
(`store.mel_path`), so an old teacher cache works unchanged and a new teacher's targets are small.

How interrupted work is picked up, as in `02_set` (see `store.py`):

- Every npz is written to a temp name and renamed: presence means done, a killed job leaves no half-file.
- `plan.csv` is kept once written (like a set's `config_extract.json`); `plan.py --replan` rewrites it.
  Cached slices are keyed by (relpath, hour) and ranked by a hash of that key, so a replan reuses them.
- `teacher_onnx.py` is a no-op when `teacher.json` records the ONNX's sha256; `cache.py` refuses to run if
  the ONNX changed since (a re-shipped teacher would mix old and new targets): move the cache aside.
- Each `_mel/<spec>` and each packed shard set carries a fingerprint of its inputs (front-end definition,
  teacher sha, plan, slices present). A stale pack is deleted and rebuilt; a stale `_mel/<spec>` stops the
  run. Editing a spec in `frontends.py` means renaming it (or deleting its `_mel` dir) on purpose.
- Training checkpoints (weights, BN statistics, optimizer, curve) at every `--eval-every` steps into
  `runs/<name>/ckpt.npz`; a rerun resumes there with a fresh batch order (statistically the same, not
  bit-identical). Same name with other settings stops (`other settings`); a finished run is a no-op.

## Scripts

Long ones go through `tools/launch_job.sh`; everything reads the teacher from `DISTILL_TEACHER` (main.py sets it).

| script | env | what |
|---|---|---|
| `main.py` | train | the chain above |
| `plan.py`, `teacher_onnx.py`, `cache.py`, `cache_fe.py` | onnx venv | slices and rungs; teacher graph; teacher targets (+ YAMNet mel); other front ends' mel |
| `shards.py pack --rung D` | train | streaming shards for rung D |
| `distill_train.py` | train | trains one student; `--rung --steps --name --arch --frontend --init --loader`; resumable |
| `student.py`, `student_init.py`, `frontends.py` | train | builder, YAMNet channel-selection init (+ layer-wise refit), front-end registry |
| `export_student.py export` / `time` | train / engine venv | ONNX with the teacher's centers folded in, parity vs Keras 1e-4; speed vs YAMNet |
| `eval_folds.py run` | train (+ engine venv) | headline `sensitivity_exclquiet` at fpr 0.005, mean over the 5 rotating folds (`03_train/sx.py`) |
| `ladder_record.py record/table/frontier` | train | one jsonl row per run; tables |
| `deploy_student.py <run>` | train | copy into buzzdetect: model.onnx, config_model.json, folds_sx.csv, README card |
| `migrate_layout.py` | any | one-time move from the single-teacher layout (dry run by default) |
| `bench_arch.py`, `band_profile.py`, `check_cache_align.py` | | speed benchmark, band statistics, mel alignment check |
| `test_distill.py [--train]`, `test_frontends.py`, `test_synth.py` | train | tests (scratch dirs, GPU hidden) |

`chain_*.sh` are the experiment chains that ran before `main.py`; they hard-code the pre-teacher layout
and are kept as the record of those runs (`chain_frontends*.sh` map 1:1 onto `main.py --runs`).

## Judging

`eval_folds.py` scores each student's ONNX over the teacher's set's rotating folds exactly as
`03_train` scores a probe (frames alone, translation `general`, `--check-labels` against the baseline's
predictions). Reference rows: the baseline CV model, the teacher's honest rotation, and the teacher's ONNX
through the same harness (trained on those folds, so inflated: 0.692 vs 0.574 honest for
`v4-ft-ps-e60-moderate`). Detections are logit > 0. The student is compared with the teacher on its
own (flips vs cached teacher logits on the held-out V pool) and against labels (the headline).

## Step 0: speed of the architecture alone (2026-09-28, GTX 1650, audio s per wall s)

`bench_arch.py export` (train env) then `time` (engine venv) time random-weight candidates; outputs in
`.local/distill/_shared/arch/`. Method matches `buzzdetect/benchmarks/model-speed`: 20 s audio, 2 warmup +
15 timed runs. CPU rates are informational only (i7-2600, no AVX2).

| model | GPU | x YAMNet | x teacher | CPU | params | MMACs/frame |
|---|---|---|---|---|---|---|
| yamnet_large_general | 2970 | 1.00 | 2.44 | 293 | | |
| teacher v4-ft-ps-e60-moderate | 1216 | 0.41 | 1.00 | 152 | | |
| a1.00 | 2723 | 0.92 | 2.24 | 365 | 3,232,719 | 68.6 |
| a0.75 | 3342 | 1.13 | 2.75 | 422 | 1,835,871 | 39.1 |
| a0.50 | 4363 | 1.47 | 3.59 | 681 | 831,471 | 17.8 |
| a0.375 | 4903 | 1.65 | 4.03 | 847 | 476,439 | 10.3 |
| a0.25 | 5457 | 1.84 | 4.49 | 1006 | 219,519 | 4.8 |
| a0.50, layers 13-14 removed | 4712 | 1.59 | 3.88 | 638 | 422,127 | 15.4 |
| front end only (STFT/mel) | 6280 | 2.11 | 5.17 | 1168 | | |

The front end alone runs at 6280 s/s, so it caps any student with YAMNet's front end at about 2.1x
YAMNet on GPU. Speedup saturates well before the MAC count does: 68.6 to 4.8 MMACs (14x) buys only 2.0x.
That is why the live work is the front end: **FRONTENDS.md**.

## Testing

```bash
conda run -n buzzdetect-train python 05_distill/test_distill.py            # layout, fingerprints, shared lookups, migration, main.py stages
conda run -n buzzdetect-train python 05_distill/test_distill.py --train    # + checkpoint/resume on CPU (~2 min)
```

They run against a scratch tree through `DISTILL_CACHE_ROOT`, `DISTILL_LOCAL_ROOT`, `DISTILL_MODELS_DIR`
and never touch real data or the GPU. Not covered by any test: a full `main.py` run on a new teacher (it
needs audio, ONNX runtime and hours); the first real run of one is the integration test.
