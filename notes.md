# ps-d8-ktails
## Hypothesis
Training arm (IDEAS 30). The logit mean of three separately trained `ps-depth8` draws is 0.558 against
members at 0.497-0.515 (`ps-d8-ensemble`), and the ten same-recipe threes average 0.545 over singles that
average 0.492 (`ps-d8-ens-same5`). Those draws already share everything frozen: the stem (layers 1-7) is the
cached embedding, and every tail starts from the same AudioSet weights. A draw differs from another only in
(i) the class Dense's random init and (ii) the batch order. So "K independent tails on the shared stem, each
with its own data order" is the logged ensemble by construction, and the open question is narrower: **does
the decorrelation need an independent data order, or is the head's init enough?**

This run trains K = 3 tails (YAMNet layers 8-14 + GAP + Dense, each a full `build_head()` model with its own
AdamMultiLR and its own gradient clip) inside one job, **all three on the same batches in the same order**,
and scores the mean of their logits. One job, one data pipeline, one model object (stem -> 3 tails -> mean),
which is the form a deployable graph would take.

Prediction: the tails diverge from the head init alone (each tail's gradient passes through its own Dense),
so the 3-tail mean lands with the same-recipe threes: within 0.02 of their mean (0.545; range 0.535-0.558),
with members at single-draw level (0.464-0.515) and member-to-member correlations near the separate draws'
(negatives 0.62, buzz 0.88).
Falsified if the 3-tail mean lands at a single draw's headline (<= ~0.515, the best single), with members
correlating well above the separate draws: then shared batch order co-adapts the tails and an ensemble needs
separately ordered runs.

Matched controls: `ps-d8-ens3` (0.558) and the ten same-recipe 3-subsets `ps-d8-same3-*` (mean 0.545) for
the ensemble; the five single draws (mean 0.492) for the members. Recipe unchanged:
`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`, `--embedder yamnet_trunk_pitchshift_depth8
--epochs 30`, set medium, translation general. Also timed against a single draw (~2 h 07 min per CV).

## Changes
- `03_train/train.py`: `TRUNK_KTAILS=K` builds K `build_head()` models (each its own Dense init, AdamMultiLR
  and clip) and wraps them in `KTails`: one train step calls each member's own compiled step in turn on the
  same batch, the wrapper's output is the mean of their logits. Each fold also writes every tail's own
  held-out predictions as a sibling model `<name>-t<k>`.
- `chain.sh`: the CV (`TRUNK_KTAILS=3`, recipe otherwise ps-depth8's), retried up to 4 times, then
  `resummarize.py` on the three tails.
- `tools_exp/member_corr.py`: pairwise correlation of the `ins_buzz` logit between members, split by
  `correct`, pooled over all held-out frames. Run on the three separate draws it gives 0.624 (negatives) /
  0.872 (buzz), the figures `ps-d8-ensemble` reported as 0.62 / 0.88, so the two are computed the same way.
- Models: `ps-d8-ktails3` (the mean), `ps-d8-ktails3-t0/t1/t2`.

## Results
Buzz logit SD of the mean, per fold: 1.20 / 1.05 / 0.65 / 0.71 / 0.48 / 0.65 / 0.71 / 0.80. `1_150`'s 0.48
is under the single draws' range (0.57-1.34) but nowhere near a collapse.

Against the three separately trained draws (`ps-d8-ens3`):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.619 | 0.528 | -0.091 | 0.023 | 32 |
| 53 | 0.581 | 0.605 | +0.024 | 0.018 | 28 |
| 1_11 | 0.632 | 0.611 | -0.021 | 0.021 | 26 |
| 1_143 | 0.640 | 0.631 | -0.009 | 0.034 | 22 |
| 1_150 | 0.470 | 0.389 | -0.081 | 0.039 | 21 |
| 1_95 | 0.312 | 0.264 | -0.048 | 0.021 | 46 |
| 1_37 | 0.701 | 0.670 | -0.031 | 0.032 | 14 |
| 1_114 | 0.509 | 0.497 | -0.012 | 0.038 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.558 → this 0.524 (-0.034 ± 0.010)
- inclusive (sensitivity), same thresholds: 0.467 → 0.437 (-0.030)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.861 | 0.820 | -0.041 | 122 |
| untagged | 0.580 | 0.552 | -0.028 | 2418 |
| background | 0.507 | 0.457 | -0.050 | 1874 |
| quiet | 0.162 | 0.146 | -0.016 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

Against the median single draw (`ps-depth8-r3`):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.486 | 0.528 | +0.042 | 0.022 | 32 |
| 53 | 0.550 | 0.605 | +0.055 | 0.024 | 28 |
| 1_11 | 0.573 | 0.611 | +0.038 | 0.022 | 26 |
| 1_143 | 0.586 | 0.631 | +0.045 | 0.034 | 22 |
| 1_150 | 0.435 | 0.389 | -0.046 | 0.048 | 21 |
| 1_95 | 0.284 | 0.264 | -0.020 | 0.025 | 46 |
| 1_37 | 0.648 | 0.670 | +0.022 | 0.043 | 14 |
| 1_114 | 0.414 | 0.497 | +0.083 | 0.042 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.497 → this 0.524 (+0.027 ± 0.012)
- inclusive (sensitivity), same thresholds: 0.411 → 0.437 (+0.026)

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.856 | 0.820 | -0.036 | 122 |
| untagged | 0.524 | 0.552 | +0.028 | 2418 |
| background | 0.406 | 0.457 | +0.051 | 1874 |
| quiet | 0.130 | 0.146 | +0.016 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

± SD is eval sampling only; training stochasticity is not in it.

Members and their correlation:

| | members | member mean | ensemble | ensemble - member mean | r negatives | r buzz |
|---|---|---|---|---|---|---|
| 3 tails, one job, shared batches | 0.469 / 0.470 / 0.466 | 0.468 | 0.524 | +0.056 | 0.613 (0.587-0.639) | 0.866 |
| 3 separate draws (`ps-d8-ens3`) | 0.515 / 0.509 / 0.497 | 0.507 | 0.558 | +0.051 | 0.624 (0.616-0.634) | 0.872 |
| ten same-recipe threes (`ps-d8-same3-*`) | singles 0.464-0.515 | 0.492 | 0.545 (0.535-0.558) | +0.053 | | |

Between-member SD 0.42 on negatives (0.40 for the separate draws) against an across-frame SD of 0.57; 0.50
against 1.29 on buzz.

Cost: 45.6-47.0 min per fold (mean 46.3), 6 h 11 min for the CV, against ~16 min per fold for one draw:
2.9x a single draw, so no training time is saved over three separate runs. One retry: a GPU OOM in fold 6's
first training step (06:30, straight after fold 5 finished); the rerun skipped the five finished folds and
lost about a minute.

Interpretation. The question asked was whether tails need their own data order to decorrelate. They do not:
on identical batches in identical order, with only the class Dense's init differing, the tails correlate
0.61 on negatives and 0.87 on buzz, the separate draws' 0.62 / 0.87, and their mean gains +0.056 over its
members, the same step the separate draws take (+0.051, +0.053). The falsifier (members correlating well
above the separate draws, mean at a single draw's level) did not happen.

The level did not match, though. The prediction was a headline within 0.02 of the same-recipe threes' mean
(0.545); 0.524 is 0.021 under it and below the lowest of the ten threes (0.535), and -0.034 ± 0.010 under
`ps-d8-ens3` (itself the best of the ten). All of that gap is in the members: 0.466-0.470, at the bottom of
the five single draws (0.464-0.515, mean 0.492). Three tails that shared one data order are not three
independent draws of the member level: the spread between them is 0.004 against the singles' 0.051, so the
order (or something else the tails share that separate runs do not) sets the level and this run drew a low
one. One run cannot tell a low draw of the shared order from a cost of the shared-batch training itself;
either way the result is one sample, 0.024 under the singles' mean with a single-draw SD near 0.02. Weak
evidence, at most, that training in one job costs member quality.

Folds against `ps-d8-ens3`: `1_29` -0.091 ± 0.023 and `1_95` -0.048 ± 0.021 are outside their eval SD,
`1_150` -0.081 ± 0.039 is about two SD, `53` +0.024 ± 0.018 unsure, the rest within their SD. Against the
median single draw six folds are up (`1_114` +0.083 ± 0.042, `53` +0.055 ± 0.024, `1_29`, `1_11` near two
SD) and the two hardest are not: `1_150` -0.046 ± 0.048 and `1_95` -0.020 ± 0.025, both within their SD, so
unsure, but the separate-draw ensemble lifted both (+0.090, +0.076 in `ps-d8-ensemble`). Tiers against the
single draw: background +0.051, untagged +0.028, quiet +0.016, loud -0.036 (122 frames); against `ens3`
every tier is down, background most (-0.050).

No selection on the scored metric: K, the recipe and the epoch budget were fixed before the run, nothing
was tuned on the held-out folds.

## Conclusion
Mixed. Mechanism confirmed: three tails trained in one job on the same batches decorrelate from the head
init alone (negatives 0.61, buzz 0.87, as separate draws) and their logit mean takes the full ensemble step
(+0.056 over its members). But the model is 0.524 (0.437 incl. quiet), -0.034 ± 0.010 under the three
separate draws and under all ten same-recipe threes, because its members sit at 0.466-0.470 against the
singles' mean of 0.492; `1_150` and `1_95` do not get the lift the separate-draw ensemble gave them. It
costs 2.9x a single draw's time, so it saves nothing over three separate runs. For the ensemble teacher
(IDEAS D1) train the members as separate runs: same compute, independent member levels, and no evidence
here that one job does as well. Open, if anyone wants it: whether the low member level is the shared order
(one more K-tails run, or K tails each with its own shuffle in one job, would say).
