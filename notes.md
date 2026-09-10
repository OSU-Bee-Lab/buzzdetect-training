# recorder-center

## Hypothesis

**Per-deployment input centering: subtract each fold's own median embedding
from every one of its frames, before the probe sees them.**

The argument is the endpoint's own. `buzzdetect` ships no threshold and the
metric sets one per deployment, because a model's score scale does not transfer
between sites — `README.md` makes that argument explicitly, and `1_95` is the
proof, carrying a threshold of **+0.173** while every other fold sits near
**-1.7**. That non-transfer does not begin at the readout. It begins in the
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

## Results

## Conclusion
