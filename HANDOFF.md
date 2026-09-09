# HANDOFF — aves-probe

Relaunched **2026-09-09 ~01:25 on the GPU**. Root `main.py`, stage 2 + stage 3
in one detached process. Worktree `.local/worktrees/aves-probe`, branch
`exp/aves-probe`. Read `notes.md` for the hypothesis.

**Expected finish: ~2 h from launch (~03:30).** Stage 2 ~1.0 h of GPU embedding
plus ~1 h of serial CPU framing; stage 3 is a frozen probe, ~10 min for 5 folds.

```
WT=/home/luke/projects/buzzdetect-training/.local/worktrees/aves-probe
```

## 1. Progress check — one command

```bash
cd $WT
pgrep -af "main.py --model aves_probe"                      # alive?
grep -v NNPACK run_aves_probe.log | tail -3
find /home/luke/projects/buzzdetect-training/02_set/sets/medium/embeddings/aves \
     -name annotations.fingerprint | wc -l                  # of 83, stage 2
find models/aves_probe/folds -name summary.json | wc -l      # of 5, stage 3
```

Once the log says `=== 03 train ===`, stage 2 is done and it is ~10 min from
finishing.

## 2. If it's still running, STOP.

Report `stage 2, N/83 idents` or `stage 3, fold N/5` and quit. No tailing the
log on a timer, no Monitor, no reading the rest of the repo — everything read
while waiting is paid for twice.

## 3. When it finishes

Comparator is **`models/cv_baseline`** (0.218). AVES's 1.0 s frame is
near-matched to YAMNet's 0.96 s, so the join is direct — this is the reason AVES
was worth running at all where Perch was not. One caveat to state, not to panic
about: `overlap_event_s = framelength_s * 0.2` moves 0.192 -> 0.200, a 4% shift
in the labelling rule, against the 5.2x shift that made `perch-probe`
uninterpretable. If the result is close, check `buzz_frames` per fold in both
`folds_sx.csv` files before reading anything into it.

```bash
cd /home/luke/projects/buzzdetect-training
python tools/compare_folds.py models/cv_baseline \
  .local/worktrees/aves-probe/models/aves_probe
```

Fold trust, from the baseline's own `folds_sx.csv`:

| fold | buzz_frames | baseline sens | read it? |
|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 2144 | 0.426 | yes — the only rich fold |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 1031 | 0.425 | yes |
| Lily Adam - One Hive/.../1_11 | 305 | 0.180 | weak |
| Luke - Diel Drivers/2026-04-08/1_150 | 146 | 0.021 | **no** — near chance; moved 0.007->0.062 between two identical runs |
| Luke - Diel Drivers/2026-05-06/1_95 | 433 | 0.037 | **no** — near chance |

Noise floor on this roster is ~0.016 headline, concentrated in exactly those
thin folds. A result worth acting on is most folds moving the same way.

Then: fill in `notes.md` Results + Conclusion -> `tools/log_entry.py` (run
`git rev-parse --short HEAD` in the **main** checkout for `main_commit`) ->
commit -> `git push origin exp/aves-probe` -> prune the worktree.

**Also update `IDEAS.md` in main.** E1 (AVES intermediate layers) predicted a
loss and asked for the embedding-geometry diagnostic as a cheap pre-filter.
That filter is now free — the `medium` AVES embeddings exist on disk.
Recompute the dimension-wise buzz/non-buzz separation (2026-06 measured 0.13 for
AVES vs 0.23 for YAMNet, on a different set) and write the current number into
E1 whatever the CV says. If the CV is a clear loss AND the separation is still
about half YAMNet's, close E1 — a middle-layer sweep costs one re-extraction per
layer and the geometry argues against all of them. Remove the idea from
`IDEAS.md` once tested either way.

## 4. If it died

`grep -v NNPACK run_aves_probe.log | tail -40`. Distinguish:

- **Stage 2 partial** — each ident dir carries an `extraction.incomplete` marker
  until its fingerprint is stamped, so a relaunch rebuilds it rather than
  reading a truncated product. Safe to relaunch the identical command; it
  resumes at ident granularity.
- **Stage 3 partial** — folds holding a `config_model.json` are **skipped
  silently**. If the crash was mid-fold that is what you want; if you changed
  anything, delete `models/aves_probe/` first or use a fresh `--model`.
- **CUDA OOM** — drop `BUZZDETECT_AVES_BATCH` to 32 (0.76 GB) or 16 (0.56 GB).
  Observed steady-state use in this run is ~2.0 GB of the card's 4 GB at batch
  64 (model + activations + framing buffers), so there is real headroom, but
  nothing else should be put on the GPU while it runs.
  Throughput is flat from batch 16, so this costs essentially nothing. Do **not**
  raise workers above 1 to compensate; see below.

Relaunch, from `$WT`:

```bash
nohup env PYTHONUNBUFFERED=1 BUZZDETECT_NO_GPU=1 MALLOC_ARENA_MAX=2 \
  BUZZDETECT_AVES_DEVICE=cuda BUZZDETECT_AVES_BATCH=64 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  main.py --model aves_probe --set medium --embedder aves \
  --translation general --workers 0 --verbose -y \
  >> run_aves_probe.log 2>&1 &
disown; echo "pid $!"
```

## Things a fresh agent would get wrong

- **`BUZZDETECT_NO_GPU=1` is not a mistake, and `CUDA_VISIBLE_DEVICES=""` must
  NOT be set.** They do different jobs here. `BUZZDETECT_NO_GPU` is read by
  `main.py` and hides the GPU **from TensorFlow only**, so stage 3 trains on the
  CPU (CPU ~= GPU for this probe) and TF never pre-allocates VRAM that torch
  needs during stage 2. Torch still sees the GPU. Setting
  `CUDA_VISIBLE_DEVICES=""` — which CLAUDE.md's generic recipe does — would hide
  it from torch too and put you back on the 14 h CPU path.
- **`--workers 0` is mandatory, not a tuning choice.** 0 means in-process; any
  value >= 1 forks a worker and **the run dies instantly** with
  `RuntimeError: CUDA driver initialization failed`. `main.py` imports
  TensorFlow before anything else, and TF touches the CUDA driver during import
  — before `BUZZDETECT_NO_GPU` gets to hide the GPU from it — so the parent
  already holds a CUDA context and `fork` poisons it for the child. This cost
  one crashed relaunch (`run_aves_probe_fork_crash.log`). Losing the workers
  costs nothing here anyway: throughput is flat from batch 16, so one stream
  already saturates the GTX 1650.
- **`embedders/aves/embedder.py` was rewritten and committed to main**
  (`8fe1336`), not to this branch. It is a pure speedup — batching + GPU,
  14.3 h -> 1.0 h — and `embedders/*` is symlinked to main from every worktree,
  so it belongs there. Output is unchanged against the old one-frame-at-a-time
  path: max abs diff 6.7e-6 on a scale of 0.85, per-frame cosine >= 0.9999998.
  No aves embeddings existed on disk to be made inconsistent by it. **This
  branch's own diff is still only `notes.md` and this file.**
- **cuDNN is disabled inside the embedder on purpose.** The pinned env raises
  `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` finalizing this model's conv1d
  descriptors. Torch's native kernels are the same speed (0.0139 vs 0.0140
  s/frame). Do **not** "fix" it by installing or repinning anything — that drags
  TF 2.16.2 with it and breaks era comparability.
- **`02_set/sets/*/embeddings` and `audio` are symlinks into main**, as is every
  `embedders/*` dir. Never `rm -rf` through them. The AVES embeddings land in
  main's shared cache on purpose — they survive the worktree pruning and save
  the next experiment the extraction.
- **`run_aves_probe_cpu_aborted.log`** is the first, CPU-only attempt, killed at
  13 min once it was clear it would take ~13 h. Ignore it.
- **`-y` was passed**, so untranslated labels were accepted without a prompt.
  Check the `survey_untranslated` block in the stage-3 section of the log before
  trusting the numbers.
- The shipped model was **not** trained, by design. `folds_sx.csv` comes
  entirely from the rotations.
