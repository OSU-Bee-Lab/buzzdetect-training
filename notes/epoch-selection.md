# Epoch selection, leakage, and what the fold roster can and cannot tell us

Auxiliary to `notes.md`. Written 2026-09-09 during `exp/xfold-epoch`, from a long
exchange with Luke plus a literature check. It is here because the loop has
flip-flopped on this question repeatedly and the reasoning kept living only in
agent context. **Read this before changing the stopping rule or the monitor.**

---

## 1. The question

Every rotating fold is its own early-stopping monitor. So the epoch that produced
a fold's reported sensitivity was chosen while looking at that fold. Under
`--monitor val_sens` the selection statistic and the reported statistic are
literally the same, which `monitor-leakage` (2026-09-09) called leakage and used
to mark `probe-grid`, `context-monitor` and `context-monitor-r2` as `artifact`.

Luke's challenge, and it is the right one: **how is stopping on `val_loss` any
better? Both come from the same fold. Both are expressions of the same underlying
ability to discriminate.**

## 2. Two wrong answers this loop has given

**Wrong answer A: "val_loss is clean."** It is not. Choosing the epoch by
`val_loss` on the scored fold is still selection on that fold. README's "bounded,
and small" is an empirical observation about this data, not a theorem.

**Wrong answer B (mine, 2026-09-09): "the sens curves rise steadily, so there is
almost nothing to inflate."** This came from smoothing each sens curve across
epochs, finding 0.001–0.003 of wiggle, and concluding selection had no room.

**That measurement was invalid, and the way it was invalid is worth remembering.**
The validation fold is *fixed*: the same ~146 buzz frames and the same ~24–35
negative frames set the threshold at every epoch. So the error from *which frames
happen to be in this fold* is nearly identical at every epoch — it is a constant
offset, not a wiggle — and smoothing across epochs cannot see it at all. The
epoch-to-epoch jitter is the small noise; the eval-sample error is the big one,
and it is the one selection exploits.

The correct scale was already in the repo: `tools/eval_sampling_sd.py` puts
per-fold bootstrap SD at **0.010–0.037**. That is ~10x the jitter estimate, and
it is the same order as the 0.027 gap the grouped re-scoring found.

> **Rule of thumb:** to estimate how much an argmax can inflate a statistic, you
> need the sampling variability of that statistic under a *fresh draw of the
> evaluation data*. Never estimate it from the smoothness of the curve.

## 3. The actual reason to prefer `val_loss` as a *selector*

Not "it is unbiased" — it is a biased proxy, and Luke's objection is exactly
right that it is diluted across 15 classes when only `ins_buzz` matters. A
plateau in `val_loss` can mean the other 14 classes stopped improving while buzz
was still learning. `context_embedder` shipping epoch 5 on `1_150`, while that
fold's own buzz curve climbs to 0.144, is that failure in the wild.

The real argument is about **variance**, and it is the central thesis of Cawley &
Talbot (JMLR 2010), the standard citation here:

> "for the purpose of model selection, rather than performance evaluation,
> unbiasedness per se is relatively unimportant... the variance of the criterion
> is also vitally important"

Their Figure 3 contrasts an unbiased high-variance criterion with a biased
low-variance one and argues the **biased one is the better selector**, because
its minimum lands closer to the truth on any particular sample.

| criterion | right target? | stability |
|---|---|---|
| `val_loss` | no — diluted over 15 classes | averaged over ~7,500 frames → stable |
| `sens@fpr0.005` | yes | rests on ~24–35 negative frames → SD 0.010–0.037 |

So `sens@fpr0.005` is precisely the Figure 3(a) criterion: correct in
expectation, useless as a selector on one sample. Luke's own diagnosis was the
mechanism — at fpr 0.005 the threshold is set by roughly the single worst of ~30
negative frames.

The repo's own corroboration: on `Fit+Fast/53`, two **identical** runs shipped
epoch 17 and epoch 226 (0.465 vs 0.520). A steadily climbing curve does not move
its argmax 200 epochs between draws. That instability is the selector's variance,
visible directly.

## 4. Why the size of the leak matters so much here

Cawley & Talbot's headline finding, on their benchmarks:

> "the bias is larger than the typical difference in performance between
> classifiers evaluated using an unbiased protocol"

and they produced a classifier that was **statistically superior at 95%** purely
as an artifact — "entirely spurious... purely the result of reducing the effects
of over-fitting in model selection."

This log's effects are 0.02–0.04. The selection bias measured here is ~0.027.
**Same regime.** And the bias is not applied evenly: it only rewards configs that
stop late, which is itself a config choice. That is the mechanism by which it
manufactures a winner.

This is the answer to "I'm not taking the numbers as strictly true, I'm just
using them to get an idea." That is sound for an *absolute* number and dangerous
for a *comparison* — and comparison is all the loop does. Noise averages out
across many experiments; bias does not.

## 5. The grouped re-scoring, and why it is not conclusive on its own

Grouping the era's 18 scoreable runs by which statistic picked the epoch, then
re-scoring each at an epoch pooled from the other folds:

| selected on | n | mean change | direction |
|---|---|---|---|
| `sens@fpr` (`val_sens`) | 4 | −0.027 | 4/4 down |
| `val_loss` | 14 | +0.000 | 6 down, 8 up |

**The confound.** That re-scoring rule scores every run at ~75% of the way to its
own peak (0.75 for the val_sens group, 0.73 for val_loss — near identical). A run
that ships *at* its peak falls the whole distance; a run that ships *before* the
75% mark does not fall at all. So some of the −0.027 is the ruler, not the runs.
It arises because early stopping truncates each fold's curve at its own patience
tail, so the pooled epoch is only testable up to the shortest other-fold curve —
capped on 4 of 5 folds in every run on disk.

`exp/xfold-epoch` exists to remove that confound: a fixed 400-epoch budget with
no early stopping, so all five curves span the same epochs and the pooled argmax
is unconstrained.

**Also true, and independent of the confound:** `own-peak − shipped` averages
**+0.018** across the 14 `val_loss` runs (up to +0.044). Stopping on `val_loss`
really does stop short of the buzz peak. Both things hold at once — the leak is
real *and* the current stopping rule leaves sensitivity unclaimed. That is why
the fix is not switching monitors but **decoupling the epoch choice from the fold
being scored**.

## 6. Luke's arbitrariness objection, and how far it goes

> "That ordering is a choice. We're choosing for pumpkin to be the validation for
> mustard's test... There is simply no way past making a choice here."

**Correct, and it lands on the paper I cited.** Lostanlen et al. (2019) rotate
3 train / 2 validate / 1 test across 6 sensors, shifting boundaries cyclically —
but the cyclic *ordering* is arbitrary, and a different ordering gives different
validate/test pairings. Worse for the citation: their 6 sensors are identical
hardware, one 10-hour night, one ~1000 km² area. That is one deployment sampled
six ways. Ours is twelve crops across four sites and four years
(`DEPLOYMENTS.md`). The citation carries less weight here than it appears to.

**But the objection does not apply to the design in this experiment.** There is
no validation *role* to assign. For test fold `t`, all k−1 other folds are
simultaneously in the training pool *and* all k−1 contribute their curves, with
equal weight, to one pooled argmax. Equal weighting is not arbitrary either — it
matches the headline metric, which counts each deployment once. Luke's
pumpkin-validates-mustard-versus-blueberry scenario cannot arise.

### Two residuals, stated rather than waited to be caught on

1. **The design is not fully nested.** Fold `f`'s curve comes from a model
   trained on everything except `f` — which *includes* `t`. So `t`'s audio sits
   in the training pool of the models whose curves pick `t`'s epoch. This is not
   selection on `t`'s held-out labels, and it is heavily diluted (`t` is 1 of 5
   rotate folds among 43 training folds), but it is not zero. Full nesting —
   an inner CV excluding `t` entirely — is 20 models instead of 5, i.e. 4x.
   Untested; worth doing only if the effect under study is smaller than the
   suspected residual.

2. **Which folds exist at all is unfixable.** Cawley & Talbot's remedy for
   partition arbitrariness is "multiple randomised partitionings," but that
   averages over partitions of the folds you *have*; it cannot average over crops
   never recorded. And Bengio & Grandvalet (2004) proved there is no unbiased
   estimator of k-fold CV's variance, so a clean error bar is not available
   either.

**Therefore the standing posture:** these numbers **rank configs**; they do not
forecast performance on a new deployment. Kill bias that lands unequally on
configs, because it corrupts the ranking. Live with irreducible variance and
report it. Do not answer the objection by adding recorders inside an existing
deployment — those are not independent folds, which is the same point as
`03_train/CLAUDE.md`'s "validation is always a whole fold."

## 7. The open lead this produced

Combining Luke's objection (don't dilute across 15 classes) with Cawley &
Talbot's principle (the selector must be low-variance) points at a monitor that
is **neither** of the two argued about above: a **buzz-only but stable**
statistic — buzz-only cross-entropy, or buzz AUC — selected cross-fold. Luke has
green-lit trying it. See `IDEAS.md`; it needs a callback change to record those
curves, so it is a separate run, not a re-read of this one.

## Sources

- Cawley & Talbot, *On Over-fitting in Model Selection and Subsequent Selection
  Bias in Performance Evaluation*, JMLR 11:2079–2107, 2010.
  https://jmlr.org/papers/v11/cawley10a.html
- Varma & Simon, *Bias in error estimation when using cross-validation for model
  selection*, BMC Bioinformatics 7:91, 2006.
  https://pubmed.ncbi.nlm.nih.gov/16504092/
- Lostanlen et al., *Robust sound event detection in bioacoustic sensor
  networks*, PLOS ONE, 2019. https://arxiv.org/pdf/1905.08352
- Stowell, *Computational bioacoustics with deep learning: a review and
  roadmap*, PeerJ, 2022. https://arxiv.org/pdf/2112.06725
- Bengio & Grandvalet, *No unbiased estimator of the variance of k-fold
  cross-validation*, JMLR 5:1089–1105, 2004.
