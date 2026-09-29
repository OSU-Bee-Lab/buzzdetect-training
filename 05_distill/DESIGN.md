# Distillation design (contract for the 05_distill scripts)

Decided 2026-09-28 with Luke; generalized to any teacher 2026-09-29 (`dpaths.py`, `main.py`; this file
describes `v4-ft-ps-e60-moderate`, the first teacher, wherever it gives a number, and the teacher-independent
rule otherwise). Target: a single-pass width-0.5 YAMNet-shaped student
(`student.py`, alpha 0.5, YAMNet's own log-mel front end; later also cheaper front ends, FRONTENDS.md)
distilled from a teacher. Step 0 (README.md) measured it at 1.47x regular YAMNet on GPU;
Luke: fine. Floors: >= ~1.5x normal on GPU inference, and >= 50% of the era baseline's
sensitivity (`cv-baseline-v4`, 0.324) at the same FPR. "Just try and see": no kill rule.

## Teacher

Any deployed probe-headed model: `<buzzdetect_dest>/<name>/model.onnx` plus `models/<name>/`
(`classes`, `activation_centers`, `set`, `folds_train`), selected by `--teacher` / `DISTILL_TEACHER`. The first one
is `v4-ft-ps-e60-moderate`: the shipped ONNX (waveform -> 15 predictions with `activation_centers` already
subtracted, so detection is logit > 0). Targets per 0.96 s frame (non-overlapping frames, exactly the deployed framing,
frame k = samples [k*15360, (k+1)*15360) of a 16 kHz mono slice):
- `logits`: the graph's 15 outputs, float32 (the deployed detection scale).
- `code`: the 2048-d tensor feeding the head (two 1024-d views concatenated), float16.
  Obtained by adding a graph output to the ONNX at the head's input (surgery, check the
  extra-output graph reproduces `logits` to 1e-5 on real audio).
Classes without an `activation_center` (for v4: 0 `aambient_scraping` and 14 `mech_quadcopter`) sit above 0
everywhere: give them zero loss weight and ignore them in every parity/flip readout. The code width (2048 for
v4) is read off the graph and recorded in `teacher.json`; class count and buzz index come from the config.
The slice geometry (62 frames of 0.96 s at 16 kHz) is fixed: a teacher framed differently is refused.

## Audio: what is used and what is off limits

Source root `/media/server storage/experiments` (spinning 18 TB HDD, ~160 MB/s: read
sequentially, seek sparingly). Long mp3 (~25-50 h per file, 48 kbps mono 44.1 kHz),
so slices are windowed reads (ffmpeg -ss/-t, resample to 16 kHz mono float32).

Excluded: `[trash]`, `Luke - External Data Sources` (ESC-50, InsectSound1000: public datasets,
not deployments), weather data, and the **blacklist** (`blacklist.py`): every deployment
sharing a grandparent dir with any annotation fold in the teacher's `folds_train`/`folds.csv`, since the teacher trained on
all of them (this covers the 5 rotating eval deployments). The deployment of a file is
`dirname(dirname(file))`, a file directly under a project counts as its own project; use the
stricter reading for `Reed - Illinois Soybean` (blacklist the date dir, e.g.
`.../2026-07-17`, not just `.../6_76`). Result to expect: ~473 deployments, ~155k h left.

## Sampling (unfiltered, Even-Sample style)

The unit is a **slice**: 62 whole frames (59.52 s) starting at `h * 3600` s of a file, for every
h with a full slice available (so a file gives one slice per hour of audio; files shorter than
an hour give one slice at 0). No filtering by teacher output at all: the raw distribution is
the point. Deterministic rank `u = sha1(relpath + ':' + h)` mapped to [0,1).

Rungs are **nested by rank** (rung k is a subset of rung k+1) with a per-deployment floor so
that every deployment appears even at the first rung:
- take a slice if `u < f_rung` OR it is among that deployment's `min_per_deploy` lowest-rank slices.
- A ~50 h (~3k slices), B ~200 h, C ~800 h, D = every 1-minute slice (~2.6k h),
  E (later) 5-minute slices, if D still improves.
`plan.py` writes `<cache>/_manifest/plan.csv` (columns: relpath, hour, start_s, deployment,
rank, first_rung) and prints rung sizes in hours and deployments. Hold out 10% of
**deployments** (hash of the deployment) as a validation pool that no rung trains on.

## Cache (mirrors the input tree)

Root: `<distill_cache>` (paths.local.json; `/media/server storage/distill-cache/`), one level per kind of product,
each shared as widely as what it depends on (README.md has the table). The audio drive keeps these outside the repo.

```
<distill_cache>/_shared/durations.csv                    ffprobe durations, keyed (relpath, size): every teacher
<distill_cache>/_mel/<spec>/<relpath, no ext>/h<hour:06d>.npz   student-input spectrograms per front end (+ fingerprint.json)
<distill_cache>/<teacher>/README.md                      what this is, how it was made, how to regenerate
<distill_cache>/<teacher>/_manifest/{plan.csv,teacher.json,teacher_ext.onnx,blacklist.txt}
<distill_cache>/<teacher>/<relpath, no ext>/h<hour:06d>.npz     the teacher's targets
```
Targets npz (uncompressed): `code` float16 (62,D) = the head input, `logits` float32 (62,C), `start_s` scalar.
`mel` float16 (62,96,64) is YAMNet's log-mel of the plain frame, `_mel/yamnet/...`: the mel depends on the
audio only, so a second teacher does not store another copy (about 80% of the bytes). Caches made before the
split hold `mel` inside the targets npz and `_fe/<spec>` under the teacher; readers accept both
(`store.mel_path`) and `migrate_layout.py` moves `_fe` up to `_mel`. About 16 KB of mel plus 4 KB of code
per frame. `teacher.json` records the onnx path + sha256 (`cache.py` refuses a changed ONNX), code tensor and
width, opset, git commit. The student trains from `mel` (its front end is not learned, so decode and STFT
never run during training). `cache.py` is resumable (skips existing npz, the targets file is the done marker),
runs `--rung`, and is one GPU process fed by N decode workers (default 16, like buzzdetect's streamers).
`plan.csv` is kept once written; `plan.py --replan` rewrites it.

## Student and loss

Student = `student.build_student(widths_for(0.5, 14))` on `mel` input (add an input_type to skip the
front end during training; the exported deploy graph keeps it). Initialised from YAMNet's
weights by output-channel selection per layer (top-k by L1 norm of the pointwise filter after BN
fold scaling; a depthwise layer's channels follow the layer before it), not random. Heads:
`Dense(15)` on the GAP code gives `logits`; an auxiliary linear map 512 -> 2048 regresses the teacher `code`.
Loss: Huber on logits (classes 0 and 14 weight 0) + lambda * MSE on standardised code (lambda 0.1
to start). Fixed step budget chosen before the run, Adam with cosine decay, batch 512 frames,
fp32 on the 1650 (4 GB). No label term at first (no annotation ever reaches the student, so the
CV numbers are honest w.r.t. the student's own training data).

## Judging

1. **Flips** (fast, every checkpoint that matters): on the held-out validation deployments,
   detections gained/lost at logit > 0 vs the teacher, `ins_buzz` first, other classes after, plus
   mean absolute logit error. Report per rung; the metric is `detection-parity` from the int8 work.
2. **Sensitivity at the same FPR** (the real test): export the student ONNX (waveform -> logits,
   `centers` from the teacher's config subtracted), run it over the rotating folds' annotated audio,
   compute the headline (`sensitivity_exclquiet` at fpr 0.005, plain mean of the 5 folds, each fold's
   threshold set on its own negatives) with `03_train/metrics.py`, and per fold. Compare against
   `cv-baseline-v4-moderate`, `v4-ft-ps` (CV-honest pitch-shift, 0.452) and, for context only, the teacher
   (`v4-ft-ps-e60-moderate`: trained on these folds, so it is not a fair CV number).
3. **Speed** with real weights: same `bench_arch.py time` harness, 20 s and 200 s chunks.

Stopping rule for the ladder and for each run is written before the run (README of the run's
worktree): continue to the next rung while the held-out flip count on `ins_buzz` improves by more than
the repeat-run spread at the previous rung; stop at the first rung that doesn't.

## Process rules

Everything longer than a minute goes through `tools/launch_job.sh`, watched with ONE Monitor
(`tools/watch_job.sh`, 30 min, re-armed at every expiry). One GPU job at a time. No polling.
Commit as you go on the distill branch; never push to main. `main.py` is the chained, resumable form of all of this.
