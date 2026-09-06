# HANDOFF — exp/lab-positives-ablation

A full 11-fold CV was launched **2026-09-06 09:58** in this worktree
(`/home/luke/projects/buzzdetect-training/.local/worktrees/lab-positives-ablation`).
Expect similar order-of-magnitude runtime to `trunk_ft_1e5`/`unfreeze_one`
(~20-25 h on CPU), possibly a bit faster since the excluded InsectSound1000
idents shrink the per-epoch training pool. Read `notes.md` for the hypothesis.

## 1. One-command progress check

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/lab-positives-ablation
pgrep -af 03_train/main.py
find models/lab_positives_ablation/folds -name summary.json | wc -l   # of 11
tail -3 train_lab_positives_ablation.log
```

(Launch PID was 1507084, but trust `pgrep` — PIDs get reused.)

## 2. If it's still running, STOP

Report `fold N/11 done` and end your turn. Do not tail the log on a timer, do
not open a Monitor, do not read the rest of the repo while waiting.

## 3. When it finishes (`models/lab_positives_ablation/folds_sx.csv` exists)

The comparator is **`trunk_ft_1e5`, not `cv-baseline`**:

```bash
cd /home/luke/projects/buzzdetect-training
python tools/compare_folds.py \
  .local/worktrees/trunk-ft/models/trunk_ft_1e5 \
  .local/worktrees/lab-positives-ablation/models/lab_positives_ablation
```

Reading it:
- Noise floor is ~0.017 median per-fold, ~0.014 headline (`noise-floor-cv`).
  A split-fold result under that is inconclusive — say so and move on.
- Check `buzz_frames`/`neg_frames` in `folds_sx.csv` before trusting any
  per-fold delta; thin folds swing hard on their own.
- IDEAS.md's caveat: dropping ~33% of positive frames also shrinks the
  training pool, so a decline is confounded with plain data volume, not
  necessarily domain mismatch. If this comes back negative, the honest
  conclusion is "removing this data hurt", not "lab data helps the
  fine-tune" — a downweighting or field-positive-subsampling follow-up
  would be needed to separate volume from domain, per IDEAS.md's caveat.
  Don't overclaim a domain-transfer story from this run alone.

Then: fill in `notes.md` Results + Conclusion → `tools/log_entry.py --write`
(run `git rev-parse --short HEAD` in **main**, not here) → commit both.
Remove the `lab-positives-ablation` entry from `IDEAS.md` on main.
**Do not merge into main; do not delete this worktree.**

## 4. If it died

```bash
grep -aE "Traceback|Error|Killed|MemoryError|Exception" train_lab_positives_ablation.log | tail -20
```

Reruns resume: any fold with a `config_model.json` is skipped, so relaunch
with the identical command:

```bash
cd /home/luke/projects/buzzdetect-training/.local/worktrees/lab-positives-ablation
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" MALLOC_ARENA_MAX=2 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  03_train/main.py --name lab_positives_ablation --set medium --embedder yamnet_trunk \
  --translation general --lr-backbone 1e-5 --lr-head 2e-4 --batch 1024 --verbose -y \
  >> train_lab_positives_ablation.log 2>&1 &
disown; echo "pid $!"
```

## Things a fresh agent would get wrong

- **This branch is based on `exp/trunk-ft` @ 38ee546, not on main.** main has
  no `yamnet_trunk` embedder and no `--lr-backbone` plumbing. `--translation
  general` here resolves to this branch's own frozen pre-split
  `translations/general.csv` (17 classes) via the project-wide fallback —
  main's per-set `general_v1.csv` doesn't exist on this branch and isn't
  needed; passing `--translation general_v1` here would error.
- **`02_set/sets/medium/embeddings/yamnet_trunk` is a symlink into
  `.local/worktrees/trunk-ft/`** (shared cache, unmodified). Never delete or
  re-extract through it.
- **`02_set/sets/medium/folds.csv` is a real file in this worktree, not the
  symlink `setup_worktree.sh` would normally create** — it's the one thing
  deliberately edited (605 InsectSound1000 rows set to `role=exclude`).
  Don't "fix" it back into a symlink; don't copy this pattern into another
  worktree without checking whether that experiment wants the shared file.
- `embedders/yamnet` (and the other stock embedders) are symlinks into
  main's `embedders/` for the real `.keras` weight files, same as
  `setup_worktree.sh` does; only `embedders/yamnet_trunk` is a real,
  independently-committed directory (this experiment's branch inherited it
  unmodified from `exp/trunk-ft`).
