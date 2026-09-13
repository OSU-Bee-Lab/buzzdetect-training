# binary-control

## Hypothesis

IDEAS item 12: `LOOP.md` flags the `general`/`binary` translation pair as
worth rerunning early, but a naive rerun measures two confounds unrelated to
the taxonomy question: (1) `build_weights` puts the class *count* in every
denominator, so `ins_buzz`'s pos_weight moves ~5.4x between translations
(2 classes under `binary` vs 15 under `general`), and (2) under `--early-stop`
the `val_loss` stopping signal's composition would shift from ~1/15 buzz to
~1/2 (this confound disappears under `--fixed-epochs`, already the default).

Free prediction that tests the framing: the head is `Dropout -> Dense(15)`
into `weighted_cross_entropy_with_logits` (15 independent sigmoids, no
softmax), so `W[:, buzz]` only ever sees gradient from the `ins_buzz` term.
The 15-class head's buzz neuron *is* the binary probe up to label mapping,
so once weight is pinned, this control should come back **near null**. A
large effect either way would mean the decoupling reasoning is missing
something.

## Changes

Added `--pin-buzz-weight` to `03_train/main.py`/`train.py`: overrides
`ins_buzz`'s `build_weights` value with a fixed constant after the normal
per-fold computation, so a `general`/`binary` comparison isolates the label
structure rather than the weight. Recomputed the pin value fresh from the
current data pool (`build_weights` on `general`, all rotate+train folds) rather
than quoting IDEAS.md's stale 0.76 (that figure was from an earlier data
revision): **0.265**.

```
03_train/main.py --name binary_control --set medium --embedder yamnet \
                 --translation binary --pin-buzz-weight 0.265031 --fixed-epochs 400 -y --verbose
```

Matched control: `cv_baseline_v3` (plain YAMNet, `general`, same budget).

## Results

| fold | general (anchor) | binary, pinned weight | delta |
|---|---|---|---|
| 1_150 | 0.266 | 0.231 | -0.035 |
| 1_114 | 0.250 | 0.237 | -0.013 |
| 53 | 0.429 | 0.423 | -0.006 |
| 1_95 | 0.052 | 0.049 | -0.003 |
| willard/1_11 | 0.368 | 0.366 | -0.002 |
| wooster/1_143 | 0.461 | 0.459 | -0.002 |
| 1_37 | 0.369 | 0.369 | 0.000 |
| 1_29 | 0.441 | 0.457 | +0.016 |

- mean sens@fpr0.005 (excl. quiet): 0.330 -> 0.324 (-0.006)
- inclusive: 0.269 -> 0.265 (-0.004)
- 6/8 down, 1 up, 1 flat, but every delta is within 0.035 and most under 0.01

## Conclusion

Near null, as predicted. Once `ins_buzz`'s pos_weight is pinned to its
`general`-pool value, collapsing the other 14 auxiliary classes into one
`nonbuzz` bucket costs essentially nothing (-0.006, well inside the ~0.027
MDE) and no fold moves by more than 0.035. This confirms the decoupled-head
arithmetic: `W[:, buzz]`'s gradient comes only from the `ins_buzz` term
regardless of how many other classes share the head, so the auxiliary
25-class taxonomy is not doing detection work for `ins_buzz` through shared
representation (there isn't one — no hidden layer here). Consistent with
`shared-trunk-head`'s null result (giving classes a shared representation was
also null at every width). Settles item 12: the `general`/`binary` choice is
not a lever on this metric under the current decoupled head, and the
`~5.4x` weight swing was the entire naive-rerun confound this item warned
about.
