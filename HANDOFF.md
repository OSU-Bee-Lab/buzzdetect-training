# HANDOFF — exp/mech-margin

A 4-point dose-response, run sequentially in two detached runner shells.
~10 min per CV, ~40 min total, started 2026-09-09 13:26.

| model | lambda | margin m |
|---|---|---|
| `yamnet_mechmargin_l0125` | 0.125 | 2.0 |
| `yamnet_mechmargin_l05`   | 0.5   | 2.0 |
| `yamnet_mechmargin_l20`   | 2.0   | 2.0 |
| `yamnet_mechmargin_l80`   | 8.0   | 2.0 |

All are `--set medium --embedder yamnet --translation general`, frozen probe,
`val_loss` stopping — i.e. everything except the loss term matches `cv_baseline`.

## 1. Progress check (one command)

```bash
cd .local/worktrees/mech-margin
ls ALL_DONE BOTH_DONE 2>/dev/null; pgrep -af "[0]3_train/main.py"
for m in l0125 l05 l20 l80; do
  echo "$m: $(find models/yamnet_mechmargin_$m/folds -name summary.json 2>/dev/null | wc -l)/5"
done
tail -3 cv_*.log
```

`BOTH_DONE` appears after `l05`+`l20`; `ALL_DONE` after all four. Runner 2 waits
on `BOTH_DONE` in an `until` loop, so it looks idle until then — that is normal.

## 2. If it is still running, STOP

Report `<model> fold N/5` and end the turn. No log tailing on a timer, no
Monitor, no reading the rest of the repo. Everything read while waiting is paid
for twice.

## 3. When it finishes

Comparator is **`models/cv_baseline`** (0.218) — this is a flat experiment, not
a stack.

```bash
cd /home/luke/projects/buzzdetect-training
for m in l0125 l05 l20 l80; do
  python tools/compare_folds.py models/cv_baseline \
    .local/worktrees/mech-margin/models/yamnet_mechmargin_$m
done
```

Read the **dose curve and the per-fold table**, not any single headline. A
single-run comparison has an MDE of ~0.027 (`probe-grid`: baseline SD 0.0095,
n=3), so no one point here is readable alone — the evidence is whether the four
points move monotonically with lambda.

Folds to distrust: `1_150` (146 buzz frames; moved 0.055 between two *identical*
runs) and any fold with a blank sensitivity in `folds_sx.csv`. The named target
is **`Luke - Diel Drivers/2026-05-06/1_95`**, which was byte-stable across two
nondeterministic runs of `harmonic-comb`, so movement *there* is the signal that
is not noise.

Mechanism check, if the metric moved: recount the raw labels of negatives above
each fold's own fpr0.005 threshold from `models/<m>/surprisal/**/*.csv`
(`start,label,activation_*,loss`) and compare the `mech_auto` share against
`cv_baseline`'s 75 `mech_auto` / 46 `ambient_background` / 43 `ins_trill`. The
term is supposed to cut the first without inflating the other two.

Then: `notes.md` → `python tools/log_entry.py … --write` in **main** →
`git add -A && git commit && git push origin exp/mech-margin`.

## 4. If it died

A traceback in `cv_<dose>.log` is a real crash. Reruns **resume silently**
(`can_write()` skips a populated model dir), so relaunch with a **fresh
`--name`** or delete the partial `models/yamnet_mechmargin_<dose>/` first.
Relaunch command: the block at the top of `notes.md` §Changes, or copy the
`nohup env … 03_train/main.py --margin-lambda <λ> …` line out of this repo's
shell history pattern in CLAUDE.md "Running long jobs".

## Do not

- Do not delete the symlinked `02_set/sets/*/embeddings` or `audio` — they point
  at main's shared caches and a `rm -rf` through them costs a re-extraction.
- Do not merge into main. Only `log.jsonl` is committed there.
