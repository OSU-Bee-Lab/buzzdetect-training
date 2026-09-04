# HANDOFF — exp/trunk-ft-restore-sens

**Status at handoff: CV launched 2026-09-04 ~17:31, running detached. Nothing
analysed yet.** Written so the next agent can decide in ~2 tool calls whether
there is anything to do, and quit cheaply if there isn't.

Worktree: `.local/worktrees/trunk-ft-restore-sens` (branch `exp/trunk-ft-restore-sens`).
Run every command below **from that directory**.

---

## FIRST: is it still running?

```bash
pgrep -f 03_train/main.py >/dev/null && echo RUNNING || echo NOT-RUNNING
tail -3 cv.log
find models/trunk_ft_restore_sens/folds -name summary.json | wc -l   # n/11 folds done
```

- **`RUNNING` and fold count < 11 → STOP. Quit and check back later.** Do not
  tail the log on a timer, do not start a Monitor, do not read anything else in
  this repo — a CV is ~1.5–2 h and everything you read now is context you pay
  for again later. Expect roughly 5–15 min per fold. Just report "still on fold
  N/11" and end the turn.
- `NOT-RUNNING` and `models/trunk_ft_restore_sens/folds_sx.csv` exists → **done,
  go to "When it finishes"**.
- `NOT-RUNNING` and no `folds_sx.csv` → **it died**, go to "If it died".

The wrapper is `run_cv.sh`: it relaunches `03_train/main.py` (which resumes from
disk — `can_write()` skips any fold dir already holding a `config_model.json`)
until `folds_sx.csv` appears, up to 40 attempts. Per-fold OOM between rotations
was routine on this 4 GB GPU during the original trunk-ft runs, so **an
`attempt N` line and a CUDA OOM traceback in `cv.log` are expected and
self-healing, not a failure.** The wrapper's PID at launch was 1321060 (stale
after any restart — use `pgrep` above, not this number).

---

## What this experiment is

`trunk_ft_1e5`'s config with **one change**: stop on `val_loss` + patience as
before, but restore the weights of the epoch that maximised a smoothed
(trailing rolling mean, w=5) `val_sens_fpr0.005` curve instead of the `val_loss`
argmin. Ported from `exp/restore-on-sens`. Full reasoning in `notes.md`;
the short version is that `restore-on-sens` measured only ~+0.006 for the
sens/loss divergence on a *frozen* probe, but the divergence it was built for
was observed on a *fine-tuned trunk* (`unfreeze_more_1e5`), where
label-smoothed BCE penalises a specialising backbone's growing confidence while
the rank-based sens@fpr does not. This is that test.

**Prediction: clearly more than +0.006, with most folds moving the same way.
Flat means the rule is cosmetic here and gets logged as a non-lever.**

---

## When it finishes

The comparator is `trunk_ft_1e5`, **not** `cv-baseline` and not
`yamnet_medium_general` — same pipeline, batch, LRs, float16 trunk cache, so the
paired delta isolates the restore rule.

```bash
python tools/compare_folds.py \
  ../trunk-ft/models/trunk_ft_1e5 \
  models/trunk_ft_restore_sens
```

Read it against LOOP.md's bar: **direction (how many of 11 folds moved which
way), not the scalar.** Two folds' 0.5% FPR threshold rests on 1 and 2 negative
frames (`1_23` and `2025-08-05/31`) — check `neg_frames` and `buzz_frames` in
`folds_sx.csv` before believing any single fold's delta, and say so if those two
carry the result (they distorted `unfreeze-more`). The measured noise floor is
~0.017 median per-fold / ~0.014 headline, and there is no seed control.

There is a within-run comparison available here that needs no extra training,
and it is the most informative thing in the run — `summary.json` per fold now
carries `best_epoch` (what shipped), `loss_argmin_epoch` (what the old rule
would have shipped), `restored_on`, and both full curves:

```bash
find models/trunk_ft_restore_sens/folds -name summary.json -print0 \
  | xargs -0 -I{} python -c "
import json,sys; d=json.load(open('{}'))
print(d['best_epoch'], d['loss_argmin_epoch'], d.get('restored_on'))"
```

If `best_epoch` ≈ `loss_argmin_epoch` on most folds, the two curves never
diverged and a null result is *explained*, not just observed — that is worth
more in the log than the delta itself.

Then: fill in `notes.md` Results + Conclusion, `git add -A && git commit`, and
append one line to `log.jsonl` **in main** via `tools/log_entry.py` (see
LOOP.md §5 — `--main-commit` convention is main's HEAD at logging time, and
`trust` should be `caveated` at best: no seed control, and the two thin folds).
Do **not** merge to main, delete the worktree, or start another experiment.

---

## If it died

```bash
grep -aE "Traceback|Error|Killed|MemoryError|ResourceExhausted" cv.log | tail -20
grep -c "=== attempt" cv.log
```

Genuine restart-loop failure (40 attempts, no progress) most likely means the
GPU is occupied by something else — `nvidia-smi`. To resume after fixing,
re-run `./run_cv.sh` detached; completed folds are kept:

```bash
nohup env PYTHONUNBUFFERED=1 MALLOC_ARENA_MAX=2 ./run_cv.sh > cv.log 2>&1 & disown
```

Note `cv.log` is overwritten by that redirect — `cp cv.log cv.log.1` first if
the old traceback still matters.

## Gotchas

- The `yamnet_trunk` embedder and its 2.8 GB layer-12 cache are **symlinked into
  the `trunk-ft` worktree**, not copied. Don't delete or re-extract that
  worktree's `02_set/sets/medium/embeddings/yamnet_trunk`, and don't run stage 2
  here — nothing in this experiment needs it.
- Branched from `exp/trunk-ft`, not main, so main's later commits
  (`shuffle=False`, the `_consensus_epoch` rewrite, surprisal) are absent by
  design. Don't "update" the branch from main; that would break the pairing.
- `git status` shows a wall of deleted `embedders/*` files. That is normal for
  these worktrees — `setup_worktree.sh` replaces the checked-out dirs with
  symlinks. Leave it.
