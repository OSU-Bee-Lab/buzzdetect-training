# HANDOFF — int8 PTQ: sensitivity at fpr 0.005 (written 2026-09-28, session paused mid-run)

Luke asked: "what's the sensitivity at 0.005 for the recent benchmark quantized models?" The first-pass README
only measured detection flips at logit > 0 (README "Not done" list); nothing had scored sensitivity at a fixed
FPR. This picks that up. **Two of the graphs are scored; the rest are not.** Nothing is committed from this
session's work. Luke stopped the job on purpose (another agent needed the machine), so resume only when the CPU
and GPU are free.

## Result so far

Sensitivity at fpr 0.005 on the 5 rotating folds of `moderate` (mean of per-fold values, the headline
convention), graph = `v4-ft-ps-e60-moderate`:

| scored by | 1_29 | Marysville/53 | willard/1_11 | wooster/1_143 | 1_37 | mean |
|---|---|---|---|---|---|---|
| CV held-out heads (`folds/*/predictions.csv`), for reference | 0.543 | 0.497 | 0.419 | 0.469 | 0.422 | **0.470** |
| shipped head on stored embeddings | 0.793 | 0.634 | 0.458 | 0.470 | 0.544 | 0.580 |
| **fp32 `model.onnx`** | 0.794 | 0.632 | 0.471 | 0.467 | 0.554 | **0.584** |
| **weights-only int8** (`q_wonly.onnx`) | 0.792 | 0.636 | 0.469 | 0.472 | 0.552 | **0.584** |

Same, `sensitivity_exclquiet` (the LOOP.md headline): CV 0.574; fp32 0.696 (0.818 0.657 0.654 0.654 0.695);
weights-only 0.696 (0.815 0.661 0.655 0.656 0.692).

**Reading: weights-only int8 costs nothing measurable at fpr 0.005** (per-fold moves are ≤ 0.005, mean identical
to 3 dp), even though README's flip count had it losing 32 of 1,124 `ins_buzz` detections at logit > 0. Those
flips sit away from the FPR-0.005 operating point, or were mostly lost false positives; this run does not say
which. One measurement deep. It says nothing about the activation-quantized graphs, which are the ones that
flipped 75 and would be the speed-relevant ones.

**Not measured:** entropy int8 (all convs) has fold 0 only (`act_entropy_u8_0.npy`); min/max, the two
12-pointwise-conv entropy variants, and `q_entropy_s8_pw` have nothing. Per-loudness-tier sensitivity is not
printed by `score.py` yet (see Next).

## Caveats — read before quoting any number above

1. **These are in-sample.** The exported graph is the *shipped* model, and its `config_model.json` lists all 5
   rotating folds in `folds_train`. Its absolute sensitivity (0.58) is inflated versus the held-out CV number
   (0.47). Only the **fp32-vs-int8 difference on identical frames** is informative; never quote 0.584 as the
   model's sensitivity. There is no fold the shipped model did not train on in `moderate`, and the per-fold CV
   heads are not saved (no weights under `models/<name>/folds/`), so a held-out quantized score would need
   re-training heads on quantized-trunk embeddings.
2. **Alignment is good but not exact.** The shipped head run on the stored embeddings, minus the exported
   `activation_centers["ins_buzz"]` (-1.076), differs from the fp32 graph on the reconstructed audio by mean |d|
   0.05-0.08, max ~0.9 (`logs/score.log`). Most likely the embedding path and the exported graph differ at
   YAMNet chunk edges / the pitch-shift stage; not investigated. Sensitivities from the two agree within ~0.01
   (0.580 vs 0.584), so it is fine for a fp32-vs-int8 comparison but is a second-order difference.
3. **Frames are the stored training frames**, not a fresh sample: every frame of every rotating-fold snip that
   survived the `general` translation, with the same `correct`/`loudness` labels as `predictions.csv`
   (row-for-row, asserted). The FPR threshold is per fold over that fold's real negative pool, computed by
   `03_train/sx._fold_sens`, the same code as the headline table. The CV-vs-shipped gap (0.47 vs 0.58) is
   what training on these folds buys; it is not evidence about quantization.
4. The int8 graphs were built with README's caveats intact: the two `skip*` variants are mislabeled (they quantize
   12 pointwise convs, not 18), calibration is 40 random moderate snips x 10 s, one seed.

## How the evaluation works (why it looks odd)

The graph takes a waveform and emits one output per **0.96 s, non-overlapping** frame, but the stored training
frames are 0.96 s windows at hop **0.192 s** (`framehop_prop` 0.2). So:

- `prep.py` (buzzdetect-train env, TF first): for each rotating fold, rebuilds the sample order exactly as stage 3
  does (`build_fold_dataset` with the model's `translation.csv`; asserts frame counts equal `predictions.csv`'s
  per-`sample` counts), reads the cached framed audio (`02_set/sets/moderate/audio/sr16000_fl0.96/raw/...pickle`,
  one 15,360-sample float32 frame per pickle record), and stitches consecutive frames (start diff 0.192 s, and the
  overlap is bit-identical) back into contiguous runs. Saves `audio_<fold>.npy` (all runs concatenated),
  `runs_<fold>.npy` (frames per run) and `headlogit_<fold>.npy` (shipped head on stored embeddings).
- `run.py` (`.local/venv-quant`, onnxruntime 1.30, CPU 4 threads): per run of n frames, calls the graph 5 times at
  sample offsets 0, 3072, ..., 12288 and interleaves the outputs (`out[k::5]`) so every stored frame gets a score.
  Asserts the output length. Saves `act_<variant>_<fold>.npy` (ins_buzz logit, centre already subtracted; the
  shift does not affect sensitivity at fixed FPR). It skips any output file that exists, so it resumes per fold.
- `score.py` (buzzdetect-train env): builds each fold's frame table (variant activation + `correct`/`loudness`
  from `predictions.csv`), calls `sx._fold_sens(df, [0.005])`, prints `sensitivity` and `sensitivity_exclquiet`
  per fold and the mean.

Cost: this is 5x the audio a plain pass would process. Measured on this i7-2600: fp32 ~150 s per 37.6k-frame fold
(~11 min for all 5), weights-only ~100 s per fold, **entropy int8 ~343 s per fold (~26 min for all 5)**; the int8
graphs are slower than fp32 here because the CPU has no AVX2/VNNI (README). Expect min/max about like entropy.

## Where everything is

Committed-able (small, in this directory, `diagnostics/**` is gitignored so `git add -f` as the last commit did):

- `sens/prep.py`, `sens/run.py`, `sens/score.py`
- `sens/results/act_*.npy` (fp32 x5, wonly x5, entropy_u8 fold 0), `sens/results/headlogit_*.npy`
- `sens/logs/{prep,run,score}.log` (prep log has TF noise filtered)

Work dir, **gitignored and re-creatable**, in `.local/int8-ptq/`:

- `graphs/q_*.onnx` — the 8 quantized graphs (88 MB), copied out of the previous session's `/tmp` scratchpad,
  which will disappear. Rebuild with `quant.py` / `wonly.py` per README "Reproducing" if lost.
- `sens/audio_*.npy`, `runs_*.npy` (2.1 GB), `headlogit_*.npy`, `act_*.npy` — the intermediates.
  If `audio_*` is gone, `prep.py` recreates it in ~2 min.

The fp32 reference graph is read from `/home/luke/projects/buzzdetect/engine/models/v4-ft-ps-e60-moderate/model.onnx`
(path in `run.py`).

## Resume

```bash
cd /home/luke/projects/buzzdetect-training
D=$PWD/diagnostics/2026-09-28_int8-ptq; W=$PWD/.local/int8-ptq/sens
# (if .local/int8-ptq/sens was wiped: mkdir -p $W; cp $D/sens/results/*.npy $W/; then rebuild audio with prep, below)
# tools/launch_job.sh $W/prep.log -- $D/sens/prep.py $W        # only if audio_*.npy is missing; wants the GPU (or --cpu)

# remaining graphs; entropy_u8 resumes at fold 1
tools/launch_job.sh $W/run.log -- .local/venv-quant/bin/python $D/sens/run.py $W entropy_u8 minmax_u8
# then: Monitor on `tools/watch_job.sh <pid> $W/run.log` (pass the pid AND log; bare watch_job.sh exited 2
# with "no $CLAUDE_JOB_DIR" in this session)

CUDA_VISIBLE_DEVICES="" conda run -n buzzdetect-train python $D/sens/score.py $W fp32 wonly entropy_u8 minmax_u8
```

Do not run the graph job alongside anything CPU-heavy: it uses 4 threads and the timings above assume the box
is otherwise idle.

## Next

1. Finish `entropy_u8` (the activation-quantized one flips 75 of 1,124; its sensitivity is the actual question),
   then `minmax_u8`; optionally the pointwise variants (`entropy_u8_pw`, `_skiplast4`, `_skipfirst4`; add names to
   the `P` dict in `run.py`, they are already there).
2. Append the results to README's table (a "sens@fpr0.005" column) and drop the "Not done" line about it. Keep the
   in-sample caveat next to every number.
3. Print the per-loudness-tier columns too (`_fold_sens` already returns `sensitivity_quiet/faint/.../loud`);
   Luke wants tier sensitivity in every run.
4. If activation quantization does cost sensitivity, the README's suggested follow-up still applies: re-derive
   the centre on the quantized outputs. For a *held-out* number, retrain per-fold heads on quantized-trunk
   embeddings (not done; needs an embedder built from the quantized graph).
5. Speed is still unmeasured on AVX2/VNNI/ARM (memory `int8-timing-needs-modern-cpu`); only worth doing for a
   variant that holds sensitivity.
