# ps-depth8-redraw
## Hypothesis
`pseudo-label` (2026-10-10) left an open thread: its teacher, meant to be a third draw of the `ps-depth8` recipe, scored 0.466 where the two earlier draws scored 0.515 (2026-10-01) and 0.509. If nothing in `03_train` or the medium set drifted since 2026-10-01, a plain `ps-depth8` CV from current main lands with the earlier draws (within ~2 SD of a headline delta, ~0.03, of 0.512) and the 0.466 belongs to the `PSEUDO_SET` code path or to an unlucky draw. Falsified if this run lands near 0.466: then main's recipe has drifted, the era's best control is stale, and the cause needs finding before anything else is built on it.

Either way this run is the fresh matched control for the batch's next training experiments (controls go stale; `docs/training-proposer-facts.md`).

Control: `ps-depth8-repeat` (0.509), with `ps-depth8` (0.515) and `pseudo-label_teacher` (0.466) as the other draws.

## Changes
None. Main at e43b4c6, existing `yamnet_trunk_pitchshift_depth8` cache.
Run: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8 --epochs 30`, set medium, translation general.
