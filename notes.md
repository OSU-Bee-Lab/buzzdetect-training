# xfold-epoch

## Hypothesis

Every rotating fold is its own early-stopping monitor, so the epoch that
produced a fold's reported sensitivity was chosen while looking at that fold.
`monitor-leakage` (2026-09-09) showed the extreme case — under
`--monitor val_sens` the selection statistic and the reported statistic are the
same, and the fold reports `max` over its own curve. That flag is withdrawn.
But the milder version survives under `val_loss`, and it cuts both ways: on a
wide input the `val_loss` argmin arrives almost immediately, so
`context_embedder` shipped **best_epoch 5** on `1_150` and scored 0.014 on a
barely-trained probe whose own curve reaches 0.144.

**Cross-fold epoch selection is the rule that fixes both.** Score each fold at
an epoch derived from the *other* folds: the epoch never saw the fold it is
scored on, so there is no selection optimism to bound, and a fold that stops
absurdly early is rescued by the consensus rather than by its own labels. It is
also the only rule an operator could actually run, having no labels for a new
site.

`tools/honest_epoch.py` estimates it offline at **+0.019 on `context_embedder`
(0.258 -> 0.277)** and ~0 on `cv_baseline`. But that estimate is compromised:
early stopping truncates each fold's curve at its own patience tail, so the
pooled epoch is only testable up to the *shortest* other-fold curve. In every
run on disk, 4 of 5 folds land ON that cap (`*` in the tool's output) — the rule
wants a later epoch than the data can score. So the offline number is a lower
bound taken at a distorted epoch, not the rule's value.

**Prediction.** Run on a fixed epoch budget with no early stopping, so all five
curves span the same epochs and the cap never binds, the rule is worth *at least*
the offline +0.019 on `yamnet_context`, carried by `1_150` (0.014 -> ~0.14), and
is roughly neutral on the rich folds.

## Design: the comparison is paired within one run

The control is not a separate CV. A fixed-budget run records the whole
val_loss and val_sens curve for every fold, and a weight snapshot per epoch, so
**both epoch rules are read off the same training trajectory**:

- `earlystop` — the epoch `RestoreTrueBest(val_loss, patience=50, min_delta=0.002)`
  would have restored, replayed offline from the val_loss curve. This is what
  the current shipped rule does.
- `xfold` — argmax of the mean of the *other* four folds' val_sens curves.

Same weights, same init, same shuffle, two epoch choices. That removes
run-to-run noise from the comparison entirely, which matters because
`probe-grid` put the single-run MDE at ~0.027 and the predicted effect is
+0.019.

## Changes

## Results

## Conclusion
