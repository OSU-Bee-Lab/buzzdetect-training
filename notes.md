# recorder-center

## Hypothesis

**Per-deployment input centering: subtract each fold's own median embedding
from every one of its frames, before the probe sees them.**

The argument is the endpoint's own. `buzzdetect` ships no threshold and the
metric sets one per deployment, because a model's score scale does not transfer
between sites — `README.md` makes that argument explicitly, and `1_95` is the
proof, carrying a threshold of **-0.401** while every other fold sits between
**-1.59 and -1.90** — displaced by ~1.2 against a spread of 0.3 among the rest.
(IDEAS.md carried +0.173 for this; that was measured on `cv_baseline`'s
predecessor and is corrected in main as of this experiment.) That non-transfer does not begin at the readout. It begins in the
input, where a recorder's gain, its placement, and its site's ambient floor
shift the whole embedding cloud before any weight is applied. Nothing tried in
this era has addressed it there.

Subtracting a deployment's own median is the standard domain-adaptation move for
exactly that, and here it is free: it uses **no labels**, so it needs no
training pool and no fold logic, and it cannot leak. Each fold — training,
validation and scored alike — is centered on statistics computed from its own
audio and nothing else. That is precisely the information an operator has when
they tune their own threshold, so the assumption is one the endpoint already
concedes.

**Why this is not `standardize-blocks`** (E3, null). That was global,
per-dimension, fitted on the pooled training folds, and equalised two *feature
blocks* against each other. This is per-*deployment*, fitted on nothing, and
equalises *sites* against each other. Different quantity, different target.

**Where it should bite**, and this is the pre-registered read:

- **`1_95`** — its threshold is set by ~10 minutes of vehicle noise in 2 of its
  24 snips, and its buzz/non-buzz mean-logit gap is 0.513 against 1.578 at
  `1_29`. A flyover is anomalous *for that recorder*; a per-deployment median is
  what exposes that, and a global statistic cannot see it.
- **`1_150`** — its buzz frames sit at 0.071 mean surprisal against a background
  of the same value, i.e. at its own ambient floor. Subtracting that floor is
  the most direct available attack on a fold whose problem is that its
  positives look like its own background.

Both are hard folds, so a gain concentrated there is the result, not a caveat.

**Falsifier.** A null or negative headline with the two hard folds unmoved says
the non-transfer is a readout/threshold phenomenon rather than an input one, and
closes per-deployment normalisation as an input-side lever. A gain carried
*only* by the rich folds would mean the median is doing something other than
what this hypothesis claims and needs explaining before it is banked.

**Prior.** Mixed. YAMNet's code is 89.6% exact zeros and non-negative; centering
makes it signed and roughly zero-centred, which is a different input regime for
`Dropout(0.2)` with no weight decay — the same regime shift that hurt AVES. So
if this comes back negative, check the stopping epoch before calling it a
representation verdict (the `aves-probe` trap). Median rather than mean
specifically for robustness to the flyover that motivates it.

## Changes

`03_train/dataset.py`: new `center_on_fold_median(samples)` — stacks a fold's
frames, takes the per-dimension median, subtracts it in place. `build_fold_dataset`
takes `center=False` and calls it just before returning. That function is called
once per fold everywhere it is used (each training fold separately, the
validation fold, the scored fold, the surprisal pass), so the median lands at
exactly the per-deployment grain with no fold bookkeeping of its own.

`03_train/train.py`, `03_train/surprisal.py`, `03_train/main.py`: `center`
threaded through `_load_data`, `_score_fold`, `_write_predictions`,
`write_fold_surprisal` and `train_set`, exposed as `--center`. Off by default;
with it off the input path is byte-identical to before the flag existed.
`--center` with `--augment` raises rather than mixing regimes — `load_augmented`
reads augment dirs, not fold dirs, so there is no per-deployment median to
center augmented frames on.

No change to the loss, the head, the optimiser, the stopping rule or the metric,
so `model.compile()` is untouched and `tools/smoke_model.py` does not apply.

### Verified before spending a CV (no training)

- Centered fold median is 0 to ~1e-7 on every fold; the result is exactly
  `A - median(A)` (`atol=1e-5`), i.e. a pure shift and nothing else.
- **The transform is not trivial.** Fold medians differ from each other by L2
  **2.33-4.58**, against per-fold median norms of 4.49-5.71 — the
  between-deployment offset is comparable in size to the vectors themselves.
- **And it is gentler than the prior feared.** Only **86 of 1024** dimensions
  have a nonzero median (8.4%), so sparsity moves only 0.894 -> 0.874 and the
  code stays overwhelmingly non-negative. The shift lands on exactly the
  persistently-active dimensions — which is what a site's ambient signature is —
  rather than remapping the whole basis. The `aves-probe` regime-shift worry in
  the hypothesis is therefore much weaker than written; noted before seeing
  results, not after.

| fold | median L2 |
|---|---|
| `1_29` (mustard) | 5.375 |
| `53` (Fit+Fast) | 5.670 |
| `1_11` (willard) | 4.493 |
| `1_150` | 5.089 |
| `1_95` | 5.705 |

## Results

## Conclusion
