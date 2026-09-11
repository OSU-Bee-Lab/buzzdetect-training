# context-verify

## Hypothesis

Re-verify E3's `yamnet_context` result (IDEAS.md item 1a) against the
cv-medium-v3 era anchor. `context-embedder` (E3) measured +0.022 `clean` on
5-fold, dropout, val_loss-stopped data — an honest rebuild of `context-stack`'s
inflated +0.050. Under the new era (8 folds, no dropout, --fixed-epochs 400,
sensitivity_exclquiet headline) that number has never been re-measured.

`yamnet_context` (`embedders/yamnet_context/embedder.py`) is a standalone
embedder already on `main` — concat(frame t-1, t, t+1) built from real
neighbouring audio at extraction time, 3072-d. No train-time flags to port; no
code changes needed. Just extract it fresh for the current annotations/fold
roster and train under the anchor's exact config, embedder swapped.

**Read the tier row, not just the headline**: cv_baseline_v3 is already at
0.778 on `loud`. A context gain has to show up in `untagged` or `loud` to be a
detection gain rather than a shuffle among frames nobody was promised.

*Falsifier:* no movement in `untagged` or `loud`, whatever the headline does.

## Changes

None — no code changes. `--embedder yamnet_context` in place of `yamnet` on an
otherwise unchanged anchor invocation.

## Results

(filled in after the run)

## Conclusion

(filled in after the run)
