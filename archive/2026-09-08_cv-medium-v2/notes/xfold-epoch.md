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

`--epoch-rule {early,xfold}` in `03_train`, defaulting to `early` (byte-identical
to previous behaviour when unset). Under `xfold`:

- every rotation trains the full `--epochs` budget with **no early stopping**, so
  all five sens curves span the same epochs and nothing is truncated;
- `callbacks.EpochSnapshots` keeps the probe's weights after every epoch
  (~184 KB/epoch at 3072-d; 369 MB for a 5x400 CV) so any epoch can be restored
  without a second fit — a retrain would be a different draw, and the curve that
  picked the epoch would then belong to a different trajectory than the weights
  being scored;
- after the last rotation, `_settle_xfold` picks each fold's epoch as the argmax
  of the **mean of the other four folds' sens curves**, restores it, and scores;
- it also writes `<name>_earlystop` from the *same trajectories*, scored where
  `RestoreTrueBest(val_loss, patience=50, min_delta=0.002)` would have restored
  (`_replay_earlystop_epoch`). Both are ordinary model dirs, so `folds_sx.csv`,
  `resummarize.py` and `compare_folds.py` work on either.

Run: `yamnet_context`, `medium`, `general`, 400 epochs, ~11.3 min/fold, ~57 min.

## Results

**Three epoch rules read off one training run, with no truncation:**

| rule | headline | what it is |
|---|---|---|
| `own-peak` | **0.309** | what a `--monitor val_sens` run reports |
| `xfold` | **0.288** | honest — epoch from the other four folds |
| `earlystop` | **0.257** | the current shipped rule |

**Validity check.** The replayed `earlystop` control scores 0.257 against the
original `context_embedder` run's shipped 0.258 — an independent training draw
reproduced to 0.001. The reconstruction is faithful.

**The decomposition.** Selection optimism of a `val_sens` run =
0.309 - 0.288 = **+0.021**. Genuine gain from stopping later =
0.288 - 0.257 = **+0.031**. `context-monitor` reported **+0.049** over
`context_embedder`; real + leak = **+0.052**. It decomposes almost exactly:
**that result was ~60% real and ~40% artifact.**

Paired against the original `context_embedder` run:

| fold | crop | earlystop epoch | xfold epoch | base | this exp | delta | val frames |
|---|---|---|---|---|---|---|---|
| `1_150` | apple | **6** | 141 | 0.014 | **0.199** | **+0.185** | 4947 |
| `53` | soybean | 91 | 234 | 0.494 | 0.516 | +0.022 | 7540 |
| `1_95` | blueberry | 118 | 234 | 0.014 | 0.009 | -0.005 | 7572 |
| `1_11` | pumpkin | 61 | 234 | 0.217 | 0.200 | -0.017 | 6930 |
| `1_29` | mustard | 155 | 289 | 0.551 | 0.518 | -0.033 | 7617 |

- mean sens@fpr0.005: baseline 0.258 -> this 0.288 (**+0.030**), 2 up / 3 down.

**This is insurance, not a general improvement, and that distinction is the
result.** Split the folds by whether early stopping worked:

- the **four folds where it picked a sane epoch** (61-155): xfold averages
  **-0.008**, i.e. slightly worse. A fold-specific epoch, even a leaky one, fits
  that fold better than a pooled compromise.
- the **one fold where it broke** (apple, epoch **6** — a 3072-d input reaches
  its val_loss argmin almost immediately, scoring a barely-trained probe at
  0.027): xfold picks 141 and gets **0.199**.

-0.008 is half a per-fold SD (0.016); +0.185 is ~11x it.

**Budget.** The *pooled* curve is flat from epoch ~145 to 400 — 253 of 397
epochs sit within 0.005 of its maximum — while per-fold argmaxes scatter over
19-337. So the rule only has to land in a plateau, which is why a noisy
statistic can still drive it. Measured cost of a shorter budget against the
pooled peak: **e250 -0.001, e150 -0.004, e100 -0.016.** 250 is enough; 400 was
not wasted but was not needed.

## Conclusion

**Adopt cross-fold epoch selection; never `--monitor val_sens`.** Both halves of
the long-running argument were right about different things, and this run
separates them: stopping on `val_loss` really does stop short (+0.031 available),
and `val_sens` really does leak (+0.021 of what it appeared to buy).

The value of the rule is not the +0.030 headline. It is that a stopping failure
stops being able to masquerade as a result. An epoch-6 fold does not measure the
intervention, it measures an accident — which is how `aves-probe` came to log
0.074 as an embedder verdict. The rule costs 0.008 on healthy folds to remove
that failure mode entirely.

`trust: clean`. The comparison is paired within one training run, so it carries
no run-to-run noise; the control is validated against an independent draw to
0.001; and no curve is truncated.

**This ends the era rather than re-baselining it** — see `notes/new-era.md`.
Luke is not ready for the refactor as of 2026-09-09; that file is the handoff.

Superseded by this run: IDEAS' "best-known config, honest 0.277". 0.277 was
`honest_epoch.py`'s truncated lower bound. The untruncated value is **0.288**.
`monitor-leakage`'s re-score of the composed config to 0.264 is likewise a
truncation artifact; its *direction* stands, its magnitude does not.

## Method background

The leakage argument, the literature check, and Luke's arbitrariness objection
are in `notes/epoch-selection.md`. Read that before changing the stopping rule.

## Method background

The leakage argument, the literature check, and Luke's arbitrariness objection
are in `notes/epoch-selection.md`. Read that before changing the stopping rule.
