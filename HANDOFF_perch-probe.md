# HANDOFF — perch-probe

A full stage-2 extraction **plus** stage-3 CV is running detached as one job:
`run_perch.sh`, launched 2026-09-08 21:21 EDT from the worktree.

Read `notes.md` first — especially **"The caveat this run carries"**, which
says in advance how this result may and may not be read. Perch's frame is 5.0 s
against YAMNet's 0.96 s, so the headline is **not** directly comparable to
`cv_baseline`. A flat result is inconclusive, not negative.

## 1. Progress check (one command)

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/perch-probe
pgrep -af "02_set/main.py|03_train/main.py" | grep -v "bash -c"
tail -3 perch_medium.log
du -sh /home/luke/projects/buzzdetect-training/02_set/sets/medium/embeddings/perch
find models/perch_probe/folds -name summary.json | wc -l   # of 5, once stage 3 starts
```

Rough budget: extraction ~1.5-3 h (measured ~0.44 s per 5 s frame per worker on
CPU, 2 workers), then a frozen-probe CV of ~10 min. Expect it done well before
morning. `=== 03 train ===` in the log is the boundary.

## 2. If it's still running, STOP.

Report which stage and quit. Don't tail on a timer, don't open a Monitor.

## 3. When it finishes

```bash
python /home/luke/projects/buzzdetect-training/tools/compare_folds.py \
  models/cv_baseline models/perch_probe
```

Comparator is `models/cv_baseline` (0.218) — **with the frame-length caveat**.
Report the per-fold table before the headline: `1_150` (146 buzz frames) and
`1_95` (433) are noise and have swung by more than any real effect this era;
the two rich folds (`1_29` 2144, `Fit+Fast` 1031) are where a real effect must
show. That is exactly how `standardize-blocks` produced a fake +0.031.

Also check what the coarser frame did to the labels, since it is half the
story:

```bash
grep -c "rescued" perch_medium.log
python - <<'PY'
import json, glob
for f in sorted(glob.glob('models/perch_probe/folds/**/summary.json', recursive=True)):
    d = json.load(open(f)); print(f.split('folds/')[1].rsplit('/summary',1)[0][:48], d['n_epochs'], 'best', d['best_epoch'])
PY
```

Then: notes.md Results/Conclusion -> `log.jsonl` in **main** (via
`tools/log_entry.py`, `--baseline-model models/cv_baseline`) -> commit -> push
`exp/perch-probe` -> prune the worktree.

## 4. If it died

`tail -40 perch_medium.log`. **Extraction resumes** — an ident killed part-way
carries `extraction.incomplete` and is rebuilt, so just relaunch
`nohup ./run_perch.sh > perch_medium.log 2>&1 & disown`. Stage 3 resumes too
(any fold dir holding `config_model.json` is skipped).

If stage 2 finished and only stage 3 failed, don't re-extract — run stage 3
alone with the pinned conda python:

```bash
CUDA_VISIBLE_DEVICES="" /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  03_train/main.py --name perch_probe --set medium --embedder perch \
  --translation general --verbose -y
```

## Things a fresh agent would get wrong

- **The two stages run under different interpreters, deliberately.** Perch v2
  needs TF >= 2.21; the pipeline is pinned to 2.16.2 and every model in
  `log.jsonl` was trained under it. Stage 2 uses
  `.local/venv-perch-extract` (TF 2.21), stage 3 the pinned conda env. This is
  safe *only* because stage 3 never runs the embedder (`initialize=False`, it
  reads `n_embeddings` and loads pickles). **Do not "simplify" this by
  upgrading the conda env** — that changes the training runtime mid-era and
  silently breaks comparability with every logged number. `notes.md` has the
  full reasoning.
- **`embedders/*` and the sets' `audio/`+`embeddings/` are symlinks into
  main.** Perch's embeddings and the new `sr32000_fl5` framed-audio cache are
  being written into the shared tree on purpose (a new embedder name claims a
  path nothing else uses). Never delete through the symlinks.
- `embedders/perch/embedder.py` is committed **in main** (2b99ce4), not on this
  branch, because the worktree sees it through a symlink.
- A partial `lite` perch extraction exists from a rehearsal that was stopped on
  purpose; it is marked incomplete and will rebuild if anyone ever wants it.
  Not a failure, and not something to clean up.
- Don't train the shipped model.
