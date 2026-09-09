# HANDOFF — aves-probe

Launched 2026-09-09 ~01:02. Root `main.py` doing stage 2 + stage 3 in one
process, detached. Worktree `.local/worktrees/aves-probe`, branch
`exp/aves-probe`. Read `notes.md` for the hypothesis.

```
WT=/home/luke/projects/buzzdetect-training/.local/worktrees/aves-probe
```

## 1. Progress check — one command

```bash
cd $WT
pgrep -af "main.py --model aves_probe"          # alive?
tail -3 run_aves_probe.log
find models/aves_probe/folds -name summary.json | wc -l   # of 5, stage 3 only
```

Stage 2 is the long part: 83 idents, ~39,800 frames at 0.199 s/frame on
2 workers — **estimate ~1.5–2.5 h**. Progress there is
`ls /home/luke/projects/buzzdetect-training/02_set/sets/medium/embeddings/aves/raw`
plus the `[N]` ident counter in the log. Stage 3 is a frozen probe: **~10 min
for all 5 folds**, so once the log says `=== 03 train ===` it is nearly done.

## 2. If it's still running, STOP.

Report `stage 2, N/83 idents` or `stage 3, fold N/5` and quit. Do not tail the
log on a timer, do not open a Monitor, do not read the rest of the repo while
waiting — everything read while waiting is paid for twice. Come back later.

## 3. When it finishes

Comparator is **`models/cv_baseline`** (0.218), not a matched control — this is
an embedder swap, and AVES's 1.0 s frame is near-matched to YAMNet's 0.96 s, so
the join is direct. One caveat to state, not to panic about:
`overlap_event_s = framelength_s * 0.2` moves 0.192 -> 0.200, a 4% shift in the
labelling rule. That is the same confound that made `perch-probe`
uninterpretable, but at 4% instead of 5.2x. If the result is close, check
`buzz_frames` per fold in the two `folds_sx.csv` files before reading anything
into it.

```bash
cd /home/luke/projects/buzzdetect-training
python tools/compare_folds.py models/cv_baseline \
  .local/worktrees/aves-probe/models/aves_probe
```

Fold trust, from the baseline's own `folds_sx.csv`:

| fold | buzz_frames | baseline sens | read it? |
|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 2144 | 0.426 | yes — the only rich fold |
| Lily - Fit+Fast/.../53 | 1031 | 0.425 | yes |
| Lily Adam - One Hive/.../1_11 | 305 | 0.180 | weak |
| Luke - Diel Drivers/2026-04-08/1_150 | 146 | 0.021 | **no** — near chance, moved 0.007->0.062 between two identical runs |
| Luke - Diel Drivers/2026-05-06/1_95 | 433 | 0.037 | **no** — near chance |

Noise floor on this roster is ~0.016 headline, concentrated in exactly those
thin folds. A result worth acting on is most folds moving the same way.

Then: fill in `notes.md` Results + Conclusion -> `tools/log_entry.py` (run
`git rev-parse --short HEAD` in the **main** checkout for `main_commit`) ->
commit -> `git push origin exp/aves-probe` -> prune the worktree.

**Also update `IDEAS.md` in main.** E1 (AVES intermediate layers) predicted a
loss and asked for the embedding-geometry diagnostic as a cheap pre-filter.
That filter is now free — the `medium` AVES embeddings exist on disk. Recompute
the dimension-wise buzz/non-buzz separation (2026-06 measured 0.13 for AVES vs
0.23 for YAMNet, on a different set) and write the current number into E1
whatever the CV says. If the CV is a clear loss AND the separation is still
about half YAMNet's, close E1 — a middle-layer sweep costs one re-extraction per
layer and the geometry argues against all of them. Remove the idea from
`IDEAS.md` once it is tested either way.

## 4. If it died

`tail -40 run_aves_probe.log`. Distinguish:

- **Stage 2 partial** — an ident dir carries an `extraction.incomplete` marker
  until its fingerprint is stamped, so a relaunch rebuilds it rather than
  reading a truncated product. Safe to relaunch the identical command; it
  resumes at ident granularity.
- **Stage 3 partial** — folds holding a `config_model.json` are **skipped
  silently**. If the crash was mid-fold that is what you want (it resumes); if
  you changed anything, delete `models/aves_probe/` first or use a fresh
  `--model`.

Relaunch, from `$WT`:

```bash
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  BUZZDETECT_CHUNK_FRAMES=48 OMP_NUM_THREADS=4 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  main.py --model aves_probe --set medium --embedder aves \
  --translation general --workers 2 --verbose -y \
  >> run_aves_probe.log 2>&1 &
disown; echo "pid $!"
```

## Things a fresh agent would get wrong

- **`02_set/sets/*/embeddings` and `audio` are symlinks into main, and so is
  every `embedders/*` dir.** Never `rm -rf` through them. The AVES embeddings
  this run builds land in main's shared cache **on purpose** — they survive the
  worktree pruning and save the next experiment a 2 h re-extraction. Extracting
  under a new `--embedder` name is exactly the case LOOP.md says needs no
  symlink broken; nothing else claims `aves`.
- **No tracked code changed.** `embedders/aves/embedder.py` was already on main
  and is unmodified. The only thing that was missing was the gitignored 377 MB
  `aves-base-bio.pt` checkpoint, fetched with `embedders/aves/BUILD.py` into
  main's `embedders/aves/`. The branch's whole diff is `notes.md` and this file.
- **`-y` was passed**, so untranslated labels were accepted without a prompt.
  Check the `survey_untranslated` block near the top of the stage-3 section of
  the log before trusting the numbers.
- The shipped model was **not** trained, by design. `folds_sx.csv` comes
  entirely from the rotations.
