# monitor-leakage

## Hypothesis

Each rotation's held-out fold **is** its early-stopping monitor
(`03_train/train.py:568`, `README.md:628`). README:642 argues the resulting
optimism is "bounded, and small", but every clause of that argument is a
statement about `val_loss`:

- "selection is on multi-class `val_loss` while the reported metric is buzz
  sensitivity at fixed FPR" — under `--monitor val_sens` the selection
  statistic *is* the reported statistic.
- "the stopping epoch is a single coarse scalar ... adjacent epochs are highly
  correlated" — true of a flat loss basin, not of a sens@fpr0.005 curve that on
  `1_150` moves in steps of 1/146.
- "it mostly cancels when comparing configurations, since every config carries
  the same bias" — it cannot cancel when the thing being compared *is* the
  monitor.

So the hypothesis: **`--monitor val_sens` reports `max` over epochs of the very
statistic it is scored on, and a material part of its +0.031 (`probe-grid`) and
of the composed 0.310 (`context-monitor`) is that maximisation, not capability.**

Free supporting observation before any code was written — `peak − at_best` of
`val_sens_fpr0.005_curve` over every run on disk is **exactly 0.000 on all five
folds for every `val_sens` run** (`L1_sens`, `L1_sens_r2`, `L1L6_combo`,
`context_monitor`), while `val_loss` runs carry a gap that is large on the thin
folds and ~zero on the rich ones — the same shape as the claimed `val_sens`
gain:

| run | monitor | reported | peak-substituted | `1_150` gap |
|---|---|---|---|---|
| `cv_baseline` | val_loss | 0.218 | 0.236 | +0.055 |
| `context_embedder` | val_loss | 0.258 | **0.290** | +0.130 |
| `context_monitor` | val_sens | 0.307 | 0.307 | +0.000 |

## Test

`README.md:651` prescribes the fix and it costs no training: *"report each fold
at `median(best_epoch)` of the other folds, so the epoch that produced the
quoted number never saw the fold it's scored on."* `val_sens_fpr0.005_curve` is
persisted per fold for every run, so that re-scoring is a lookup, not a retrain.

Cross-fold epoch selection, applied identically to every run in the era:
choose fold f's epoch from the *other* four folds' curves only, read
f's sensitivity off its own curve at that epoch. Report the honest headline
beside the logged one.

## Changes

`tools/honest_epoch.py` — no training, no TensorFlow. Reads every run's
persisted `val_sens_fpr0.005_curve` and re-scores each fold at an epoch chosen
four ways. All four score fold f off f's own curve; they differ only in how the
epoch is picked:

| rule | epoch for fold f | sees f? |
|---|---|---|
| `own-peak` | argmax of f's own curve | yes — this is what a `val_sens` run reports |
| `shipped` | f's own `best_epoch` | yes |
| `xfold-median` | median of the other four folds' peak epochs | no — README:651, literally |
| `xfold-pooled` | argmax of the *mean* of the other four folds' curves | no |

`xfold-pooled` is the one to read: it uses all of the other folds' signal rather
than just their argmaxes, it is how the shipped model's epoch is actually
chosen, and it is the only rule available to an operator who has no labels for
the new site. The other folds' curves end where their own early stopping ended,
so the latest testable shared epoch is capped; folds where the rule lands on
that cap are marked `*` and are, if anything, scored too early.

Swept across the whole era — 26 runs, pulling committed `summary.json` out of
every `exp/*` branch, not just what is checked out.

## Results

**The rule is neutral on `val_loss` runs and expensive for `val_sens` runs.**
That asymmetry is the finding; it is the exact signature predicted above.

| group | n | shipped | honest (`xfold-pooled`) | delta |
|---|---|---|---|---|
| yamnet + `val_loss` (`cv_baseline`, `base_r2`, `base_r3`) | 3 | 0.218 | 0.216 | **-0.002** |
| yamnet + `val_sens` (`L1_sens`, `L1_sens_r2`) | 2 | 0.249 | 0.223 | **-0.026** |
| context + `val_loss` (`context_embedder`) | 1 | 0.258 | **0.277** | **+0.019** |
| context + `val_sens` (`context_monitor`, `_r2`) | 2 | 0.310 | 0.263 | **-0.047** |

Re-reading the two headline claims of the era with the leakage removed:

| effect | as logged | honest |
|---|---|---|
| `--monitor val_sens` on yamnet (`probe-grid` L1) | **+0.031** | **+0.007** |
| `--monitor val_sens` on context (`context-monitor`) | **+0.052** | **-0.014** |
| the context embedder itself (`context-embedder`) | +0.040 | **+0.061** |

+0.007 is well inside the 0.0095 single-run SD `probe-grid` measured. The
context embedder is unaffected — it gets *larger*, because cross-fold epoch
selection repairs the `1_150` stopping failure that `context-monitor` was built
to fix.

**Paired, per fold — `context_embedder` vs `context_monitor`, both honest:**

| fold | context (honest) | context+monitor (honest) | delta |
|---|---|---|---|
| 1_11 | 0.223 | 0.230 | +0.007 |
| 1_29 | 0.530 | 0.547 | +0.017 |
| 1_95 | 0.025 | 0.022 | -0.003 |
| 53 | 0.461 | 0.447 | -0.014 |
| **1_150** | **0.144** | **0.075** | **-0.069** |
| mean | **0.277** | 0.264 | -0.013 |

2 up / 3 down, and the fold the entire `val_sens` case rests on flips hard
against it.

**`1_150` is the whole story, and it is a selection artifact.**

- `context_embedder` shipped 0.014 there at epoch 5 — a real stopping failure,
  correctly diagnosed in `IDEAS.md`. But its own curve *already reaches 0.144*,
  and the cross-fold rule finds it (epoch 66 of 88): **the stopping failure is
  fixable without touching the monitor.**
- `context_monitor` reported 0.219 at epoch 160; `_r2` reported 0.219 at epoch
  192. In both runs the other four folds put the shared optimum at epoch
  **19** and **27** respectively, and in neither run is that capped — the four
  folds that did not see `1_150` disagree with epoch 160 by an order of
  magnitude. Scored there, `1_150` is 0.075 and 0.068.
- The "reproduced exactly" reading of 0.219 twice inverts. `1_150` has 146 buzz
  frames, so its granularity is 1/146 and 0.219 = 32/146. Two runs maximising a
  1/146-granular statistic over ~200 epochs landing on the same value is what
  maximising noise looks like, not what reproducibility looks like.

**Why the era-wide monitor split in `LOOP.md` and `IDEAS.md` is not evidence
against this.** That table ("the two groups do not overlap on `1_150`") is real,
and it is fully explained by leakage: it compares *a point on a curve* against
*the maximum of that same curve*. Grouping all 26 runs of the era by monitor:

| `1_150` | `val_sens` (n=5) | `val_loss` (n=21) |
|---|---|---|
| **shipped** | 0.089 / 0.158 / 0.171 / 0.219 / 0.219 | 0.000 - 0.152, median 0.021 (18 of 21 at or below 0.041) |
| **honest** | 0.055 / 0.068 / 0.075 / 0.089 / 0.123 | 0.000 - 0.165, median 0.068 |

Shipped, the groups are disjoint. Honest, the `val_sens` group falls *entirely
inside* the `val_loss` range, the two medians are 0.075 and 0.068, and the six
highest values on this fold are all `val_loss` runs. The non-overlap was
produced by the selection rule, not by the models.

(The `val_loss` n=21 spans the whole era including the deliberately-crippled
`mech-margin` ladder and the other embedders, so its *range* is wide by
construction. The medians and the top of the distribution are the part to read.)

Side effects on other entries, same sweep: `perch_probe` 0.244 -> **0.169**
(the largest drop in the era; it strengthens that entry's "leaning negative"
reading), and `shared-trunk-head`'s hidden layers *gain* under the honest rule
(`hidden_1024` 0.226 -> 0.253), because they shipped early-stopped epochs.

## Conclusion

**`--monitor val_sens` does not survive removing the leakage, and the context
embedder does.** The held-out fold is its own early-stopping monitor, so a
`val_sens` run reports `max` over epochs of the statistic it is scored on —
confirmed mechanically: `peak - at_best` is exactly 0.000 on all five folds of
every `val_sens` run on disk, and non-zero on every `val_loss` run.
`README.md:642` argues this optimism is small, and it is right *for
`val_loss`* — every clause of that argument is about the selection statistic
differing from the reported one, which is precisely what `--monitor val_sens`
removes.

The era's best config should be read as **`yamnet_context`, honest headline
0.277**, not `yamnet_context + --monitor val_sens` at 0.310. `IDEAS.md`
currently instructs every future experiment to build on the latter.

Nothing here needs a confirming CV: re-scoring a fold at epoch e from its
persisted curve is the model's actual score at epoch e, so a rerun would only
add a fresh noise draw. What *does* need one is the follow-up — implementing
cross-fold epoch selection as the shipped rule. That is not a drop-in, because
early stopping truncates the curves (the `*` caps above): it needs a fixed
epoch budget, or a two-pass fit. Left as an IDEA rather than smuggled in here.

Recommend amending `probe-grid`, `context-monitor` and `context-monitor-r2`
per `LOOP.md` step 5.
