# lab-positives-ablation

## Hypothesis

The `train`-role pool includes 605 InsectSound1000 idents (lab recordings of
insects) supplying ~2,700 of the buzz frames every rotating fold trains on,
against ~4,700 field frames from the other ten folds. On a *frozen* probe
this costs little — the backbone is fixed and extra positives mostly
regularise a linear head. On the fine-tuned trunk (`trunk_ft_1e5`, layers
13-14 unfrozen at lr 1e-5), gradient flows into the backbone itself, so a
third of the positive signal is pulling those filters toward lab acoustics
instead of field acoustics. Nobody has measured whether this helps or hurts
once the trunk is trainable.

Prediction: if the lab positives are a drag on the fine-tune, excluding them
should raise sens@fpr0.005 on most folds, especially ones with thin field
buzz coverage. If they're a crutch (extra positive volume the fine-tune
needs to avoid overfitting on ~11 field folds' worth of buzz), excluding
them should hurt.

## Changes

- `02_set/sets/medium/folds.csv` (real file in this worktree, not the
  symlink to main): `role` set to `exclude` for all 605 rows whose `source`
  is `2025-06-24 InsectSound1000`. Confirmed this drops the train-only pool
  from 311 distinct folds (675 train rows) to 31 distinct folds (70 train
  rows) — i.e. only the InsectSound1000 rows moved, the other 70 train-only
  rows (from the original annotations) are untouched.
- No code changes. Roles are training-time policy only (`02_set` embeds
  every fold regardless), so this needed no re-extraction — `embeddings/`
  is symlinked straight into `exp/trunk-ft`'s cache.
- This is free per `LOOP.md`'s "Change one thing": same trunk-FT config as
  `trunk_ft_1e5` (embedder `yamnet_trunk`, `--lr-backbone 1e-5 --lr-head 2e-4
  --batch 1024`, translation `general` resolving to this branch's frozen
  pre-split table since it forks from `exp/trunk-ft`), differing only in the
  excluded source.

Comparator: **`trunk_ft_1e5`** (`.local/worktrees/trunk-ft/models/trunk_ft_1e5`),
not `cv-baseline` — same trunk pipeline, same LRs/batch, differing only in
whether InsectSound1000 trains the backbone.

## Results

(fill in after the CV finishes — see HANDOFF_lab-positives-ablation.md)

## Conclusion

(fill in after the CV finishes)
