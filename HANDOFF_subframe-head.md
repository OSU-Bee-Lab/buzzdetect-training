# HANDOFF — exp/subframe-head

A full 11-fold CV was launched **2026-09-06 03:29** in this worktree
(`/home/luke/projects/buzzdetect-training/.local/worktrees/subframe-head`),
backbone frozen (`--lr-backbone 0`, matching `trunk_frozen`'s regime, not the
fine-tune). Expect **~5.5-6h on CPU**: measured ~20s/epoch on fold 1, and
`trunk_frozen` (same regime) ran 993 total fold-epochs across its 11 folds
before stopping. Read `notes.md` for the hypothesis, why the backbone is kept
frozen (fast iteration across pooling variants before spending a slow
fine-tuned run on the winner), and the smoke-test bug it caught (Lambda-layer
deserialization) before touching the embedder again.

(An earlier launch at 03:24 used `--lr-backbone 1e-5`, stacking this on the
fine-tune by mistake — killed before any fold finished, no output from it on
disk. The 03:29 launch above is the one that matters.)

## 1. One-command progress check

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/subframe-head
pgrep -af 03_train/main.py
find models/subframe_head/folds -name summary.json | wc -l   # of 11
tail -3 train_subframe_head.log
```

(Launch PID was 1440511, but trust `pgrep` — PIDs get reused.)

## 2. If it's still running, STOP

Report `fold N/11 done` and end your turn. Do not tail the log on a timer, do
not open a Monitor, do not read the rest of the repo while waiting.

## 3. When it finishes (`models/subframe_head/folds_sx.csv` exists)

The comparator is **`trunk_frozen`, not `trunk_ft_1e5` or `cv-baseline`** (backbone is frozen here -- see notes.md):

```bash
cd /home/luke/projects/buzzdetect-training
python tools/compare_folds.py \
  .local/worktrees/trunk-ft/models/trunk_frozen \
  .local/worktrees/subframe-head/models/subframe_head
```

Reading it:
- Noise floor is ~0.017 median per-fold, ~0.014 headline (`noise-floor-cv`).
  A split-fold result under that is inconclusive — say so and move on.
- Check `buzz_frames`/`neg_frames` in `folds_sx.csv` before trusting any
  per-fold delta; the usual thin folds (`Various Opportunistic/48`,
  `Diel/2026-05-06`, `1_23`) swing hard on their own.
- **`willard`** (`Lily Adam - One Hive/.../willard/2024-08-07/1_11`) is the
  named test case — highest fraction of short/isolated buzz events, the fold
  `willard-regression` in IDEAS.md points at this exact mechanism from the
  context-stack side. Check it specifically, not just the headline.
- Compare `best_epoch` (per fold's `summary.json`) against `trunk_frozen`'s
  (median 36, range 10-88) — a much earlier or later stop would suggest the
  new pooling changes the loss landscape, not just the endpoint.

Then: fill in `notes.md` Results + Conclusion -> `tools/log_entry.py --write`
(run `git rev-parse --short HEAD` in **main**, not here) -> commit both. If
positive, update IDEAS.md's `subframe-head` entry with the result and note
options 2/3 (log-sum-exp, keep-frequency) as the next step; if negative or
inconclusive, note that clearly so nobody reruns it.

**Do not merge into main; do not delete this worktree.**

## 4. If it died

```bash
grep -aE "Traceback|Error|Killed|MemoryError|Exception" train_subframe_head.log | tail -20
```

Reruns resume: any fold with `config_model.json` is skipped, so the same
launch command is safe to re-issue as-is:

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/subframe-head
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  03_train/main.py --name subframe_head --set medium --embedder yamnet_trunk \
  --translation general --lr-backbone 0 --lr-head 2e-4 --batch 1024 --verbose -y \
  >> train_subframe_head.log 2>&1 &
disown; echo "pid $!"
```

Notes: `embedders/yamnet_trunk` here is a real (git-tracked) directory, not a
symlink — it carries the pooling change and must not be deleted or replaced.
`02_set/sets/medium/embeddings/yamnet_trunk` IS a symlink, into
`.local/worktrees/trunk-ft/02_set/sets/medium/embeddings/yamnet_trunk` (that's
where the cache was actually built) — don't delete that worktree either.
