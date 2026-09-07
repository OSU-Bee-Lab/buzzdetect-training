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

CV finished 2026-09-07. `tools/compare_folds.py` against `trunk_ft_1e5`:

```
                                                    fold  sensitivity_base  frames_val  sensitivity_exp  delta
Luke - Various Opportunistic Recordings/2025-08-12/1_114             0.316        3768            0.247 -0.069
   Luke - Various Opportunistic Recordings/2025-08-05/31              0.185         942            0.132 -0.053
 Lily Adam - One Hive/recorders/wooster/2024-07-26/1_143             0.337        4708            0.305 -0.032
                    Luke - Diel Drivers/2026-04-08/1_150             0.068        4947            0.041 -0.027
                   Lily - Fit+Fast/2023_R3_Marysville/53             0.432        4712            0.411 -0.021
 Luke - Various Opportunistic Recordings/2025-06-23/1_23              0.400         315            0.383 -0.017
                            JamesU - MustardBumbler/1_29              0.469        6984            0.457 -0.012
                     Luke - Diel Drivers/2026-05-06/1_95              0.048        6628            0.048  0.000
  Lily Adam - One Hive/recorders/willard/2024-08-07/1_11              0.318        4730            0.325  0.007
   Luke - Various Opportunistic Recordings/2025-08-27/48              0.020        1571            0.043  0.023
 Luke - Various Opportunistic Recordings/2025-07-03/1_37              0.293        4715            0.333  0.040

3 folds up, 7 down, 1 flat (mean delta -0.0146)
sensitivity @ fpr0.005: trunk_ft_1e5 0.262 -> lab_positives_ablation 0.248 (-0.014)
```

`buzz_frames`/`neg_frames` from `folds_sx.csv`: no fold looks thinner than
the ones `trunk_ft_1e5` was already evaluated on, so this isn't a
thin-fold artifact.

## Conclusion

Headline delta (-0.014) sits right at the measured noise floor
(~0.014 headline, ~0.017 median per-fold — `noise-floor-cv`), and the split
(3 up / 7 down / 1 flat) is not a clean sweep. **Inconclusive** — this run
does not support "lab positives drag the fine-tune down" or the reverse.

Per IDEAS.md's caveat, even if this delta were real, dropping ~33% of
positive frames also shrinks the training pool, so a decline would be
confounded with data volume, not necessarily domain mismatch. No follow-up
run needed to draw that distinction here, since the headline result doesn't
clear the noise floor either way.

Net: keep InsectSound1000 in the training pool for future trunk-FT runs;
this ablation found no evidence it should be excluded.
