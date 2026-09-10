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

**Headline: 0.218 -> 0.210 (-0.008), 1 fold up / 4 down.** Inside `probe-grid`'s
~0.027 MDE, so the headline alone is null. The shape is where the result is.

| fold | buzz frames | baseline | centered | delta | epoch base -> exp |
|---|---|---|---|---|---|
| `53` Fit+Fast | 1031 | 0.425 | 0.352 | **-0.073** | 174 -> 337 |
| `1_29` mustard | 2144 | 0.426 | 0.395 | -0.031 | 162 -> 243 |
| `1_11` willard | 305 | 0.180 | 0.167 | -0.013 | 65 -> 138 |
| `1_95` | 433 | 0.037 | 0.035 | -0.002 | 147 -> 264 |
| `1_150` | 146 | 0.021 | **0.103** | **+0.082** | **26 -> 213** |

### The +0.082 at 1_150 is a stopping artifact, not a centering gain

`best_epoch` rose on **every** fold, 1.4x to 8.2x. Centering changes where
`val_loss` bottoms out, so this run is not one variable — it is centering *plus*
a systematically later stopping epoch. And `1_150` is exactly the fold whose
early stopping was already known to be broken (epoch 26), the one `xfold-epoch`
measured the epoch-rule repair on at **+0.185**.

Disambiguated with `tools/honest_epoch.py`, which re-scores each fold at an
epoch chosen from the *other* folds' persisted `val_sens` curves — the same rule
applied to both arms, no training, no leak:

| rule | `cv_baseline` | `recorder_center` | delta |
|---|---|---|---|
| own-peak (uncapped, leak-favouring) | 0.236 | 0.222 | **-0.014** |
| shipped (what the CV reported) | 0.218 | 0.210 | -0.008 |
| xfold-median | 0.225 | 0.208 | **-0.017** |
| xfold-pooled (the operator-usable rule) | 0.219 | 0.203 | **-0.016** |

**Centering is negative under all four epoch rules**, and the two non-leaking
ones are the most negative. Per fold under `xfold-pooled`:

| fold | baseline | centered | delta |
|---|---|---|---|
| `53` | 0.416 | 0.371 | -0.045 |
| `1_11` | 0.184 | 0.167 | -0.017 |
| `1_95` | 0.044 | 0.030 | -0.014 |
| `1_150` | 0.075 | 0.068 | **-0.007** |
| `1_29` | 0.377 | 0.377 | 0.000 |

**4 down, 1 flat, 0 up.** `1_150`'s +0.082 becomes **-0.007** once both arms are
scored at a comparable epoch — the entire apparent hard-fold gain was the
stopping repair that centering caused incidentally. Even at its own uncapped
peak the fold gains only +0.028, not +0.082.

## Conclusion

**NEGATIVE, and the falsifier fired on both of its clauses.** Per-deployment
median centering does not help. The headline (-0.008) is inside the MDE, but the
direction is consistent across four independent epoch rules (-0.008 to -0.017),
4 folds down and 0 up under the non-leaking ones, and the losses are led by the
richest fold (`53`, 1031 buzz frames, -0.045 honest / -0.073 shipped — well
above the ~0.019 repeat movement for folds that size).

**Both pre-registered targets failed.** `1_95` — the fold the hypothesis was
built for, whose operating point is set by ~10 minutes of vehicle noise, and
where a per-recorder statistic should have exposed an excursion a global one
cannot — is **-0.002 shipped and -0.014 honest**. `1_150` moved, but for the
wrong reason. The hypothesis said the per-site score-scale non-transfer begins
in the input; the answer is that removing the input-side part of it does not
help, so it is a readout/threshold phenomenon, and the per-fold threshold the
metric already applies is evidently absorbing what centering removes. That is
consistent with `shared-trunk-head`'s finding via a different route: the
per-fold threshold makes a pure shift invisible, and centering a fold by its
median is close to a pure shift.

**Mechanism, and it is the transferable part.** Centering costs epochs on every
fold (1.4x-8.2x to the `val_loss` argmin). That is the third normalisation in
this project to do it — `input-standardization` (E2) and `standardize-blocks`
(E3, ~6x epochs) both broke the epoch budget the same way. **Treat "any input
normalisation changes the stopping epoch" as established**, and pair the next
one with a fixed epoch budget or a cross-fold epoch rule from the start, or its
number will be confounded exactly as this one was.

**The trap this run walked into, recorded because it nearly became the result.**
The shipped table reads as a +0.082 hard-fold gain on the fold the loop most
wants to move, against small losses elsewhere — which is precisely the shape
LOOP.md tells agents *not* to discount. It was still an artifact. The
distinguishing check is not magnitude and not a repeat spread: it is that the
treatment moved `best_epoch` by 8x on that fold, i.e. the treatment could have
selected on the reported metric. **Read `best_epoch` alongside every delta**;
when the treatment moves it, re-score both arms at a common epoch before
believing the delta. It costs one `honest_epoch.py` run and no training.

**Not adopted.** `--center` stays in the code, off by default, with the input
path byte-identical when unset. Do not rerun this on YAMNet. A per-ident
(file-level) variant is *not* an obvious follow-up — it is a strictly smaller
version of a transform that failed at the grain the hypothesis argued for, and
it risks centering out the buzz in a buzz-dense snip.
