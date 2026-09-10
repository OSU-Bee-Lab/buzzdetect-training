# Experiment Ideas

Candidate experiments, nothing else. Results go in `log.jsonl`, protocol in
`LOOP.md`, closed eras in `archive/` — each with a README digesting what that
era concluded. Check all three before proposing an experiment.

**State as of 2026-09-08.** `log.jsonl` holds the current era only; the training
data was revised and the era before it is archived. The baseline is
`cv_baseline` (frozen YAMNet probe, `general`, per-class weighting in the loss,
restore at the true val_loss argmin).

## Which era a number came from

Evidence in this file spans **three** eras, and which one a claim comes from
decides what it is worth. Every section below is tagged. An untagged claim is a
proposal, not a measurement.

| tag | era | what it was | how to treat it |
|---|---|---|---|
| **E1** | `archive/2026-06_fixed-test` — 2026-06-02..06-09, 29 runs | fixed test corpus, **precision against base rate**, the retired stage 4, often on `lite` | **Not a verdict.** Different metric, different data, two revisions ago. Re-establish from scratch or don't cite it. |
| **E2** | `archive/2026-08_cv-medium-v1` — 2026-08-19..09-08, 30 runs | the CV rework: sens@FPR, **11** rotating folds, pre-revision annotations, `general_v1` | Right metric, wrong data and roster. Directions survive; numbers don't. |
| **E3** | current `log.jsonl` — 2026-09-08 onward | **5** rotating folds, revised annotations, `general` | Directly comparable. The only numbers you can beat. |

The standing warning: `temporal-context` was a clear negative in E1 and, rerun
as `context-stack` in E2, became the largest gain in the log. A verdict
inverted on an eval change alone. **E1 and E2 negatives are leads, not
answers** — and an E1 negative on `lite` under the retired metric is barely
even a lead.

---

## Closed: standardize the input blocks

*Evidence: **E3** — ran and closed on the current roster. (The +0.014 it was promoted on was **E2**.)*

Ran 2026-09-08 as `standardize-blocks`, paired against a matched control
(`combined_control`, same embedder/data/code, flag off). **Negative.** Headline
0.230 -> 0.261, but the whole +0.031 is fold `1_150` (+0.151), the 146-buzz-frame
fold that has swung 0.007 -> 0.062 between two identical runs; both rich folds
are flat (`1_29` 0.000, `Fit+Fast` -0.003) and the other four average +0.001.

The epoch-cap question the section demanded be settled first was settled: with
the cap raised to 3000 and the stopping rule untouched, all folds converged at
502-1000 epochs, so this is a comparison of two converged models, not a
truncation artifact. Standardization simply costs ~6x the epochs for nothing.
Reading: the 521 sigmoid scores are a supervised readout of the block beside
them, so fixing their gradient share adds no information that was not already
there. `--standardize` is in `03_train`, off by default, fold-safe and
save/load-clean — available if an embedder ever mixes genuinely heterogeneous
blocks. Don't re-run this one on YAMNet.

## Closed: AVES as a drop-in embedder (and E1, its layer sweep)

*Evidence: **E3** — ran and closed on the current roster.*

Ran 2026-09-09 as `aves-probe`: `aves-base-bio`, final transformer layer, 768-d,
under `cv-baseline`'s exact probe config. **Clear negative — 0.218 -> 0.074
(-0.144), 5 folds down, 0 up**, and the two folds worth reading are the two
largest losses (`1_29` -0.410, `Fit+Fast` -0.220). That is 26x the ~0.016 noise
floor. The frame join is sound: AVES's 1.0 s against YAMNet's 0.96 s moved
`overlap_event_s` only 0.192 -> 0.200, and unlike `perch-probe` the dilution
check passes — the per-fold buzz *rate* is unchanged to three decimals
(`1_29` 0.2815 -> 0.2801).

This also retires the 2026-06 "performs at chance" note, which was worth no
weight on its own (`aves_lite`, retired stage-4 endpoint, never logged) but
turns out to have had the right sign.

**E1's middle-layer sweep ({5,7,9}) is closed with it — but not on the
pre-filter that was supposed to decide it.** The parked geometry diagnostic
(AVES at 0.13 dimension-wise buzz/non-buzz separation against YAMNet's 0.23,
i.e. half) was recomputed on the current `medium` embeddings, and it **inverts**:

| | mean \|Cohen d\| | mean \|AUC-0.5\| | top-10 dims \|AUC-0.5\| | frac exact zeros |
|---|---|---|---|---|
| YAMNet 1024-d | 0.155 | 0.022 | 0.260 | 0.896 |
| AVES 768-d | **0.242** | **0.074** | 0.226 | 0.000 |

**The replacement number, and the one worth quoting.** PCA-whitening to k
dimensions (fit on training folds only, converged `lbfgs` readout, same metric
and rotation) measures concentrated-vs-distributed directly:

| dims kept | YAMNet | (% of its best) | AVES | (% of its best) |
|---|---|---|---|---|
| 16 | 0.235 | 90% | 0.055 | 28% |
| 64 | 0.248 | 95% | 0.114 | 59% |
| 256 | 0.242 | 93% | 0.182 | 94% |
| all | 0.261 | 100% | 0.194 | 100% |

**YAMNet keeps 90% of its sensitivity in 16 principal directions; AVES keeps
28%, and needs ~256 to reach 94%.** This is independent of the
training-procedure confound that sank the rest of `aves-readout` — every row is
a converged fit — so it is the one result from that run that stands alone.
Full detail in `exp/aves-readout:notes.md`.

**Do not cite 0.13-vs-0.23 again.** It was a scale artifact: YAMNet's
activations are 89.6% exact zeros, so any mean across its basis is diluted by
~900 dead dimensions. What survives is the *shape* claim, not the magnitude —
YAMNet wins only where the signal is concentrated (top-10 dims 0.260 vs 0.226,
max \|d\| 1.91 vs 0.91), its few strong directions being sparse, non-negative and
linearly separable, while AVES spreads the same information thinly across a
dense zero-centred basis.

**E1 is downgraded to closed-on-cost, not refuted.** The original wording here
said a middle layer "would have to recover ~0.15 sensitivity the final layer has
nowhere." That overstates it, and the geometry table above is why: the
information is *not* nowhere. AVES's per-dimension separation is the better of
the two on average — what fails is the readout, not the representation.

Two things also argue that the last layer is a systematically bad place to have
tested. It is the layer specialised to the masked-prediction pretext task, and
last-layer-worst is the standard wav2vec2/HuBERT layer-wise probing result
rather than bad luck here. And `embedders/aves/embedder.py` mean-pools it
(`layer_outputs[-1].mean(dim=1)`, ~49 steps) — pooling a signed zero-centred
basis can cancel, where YAMNet's ReLU-non-negative features accumulate under the
same operation.

So: **E1 stays closed because a per-layer sweep costs a re-extraction each for
no specific reason to prefer one layer, not because AVES is empty.** Reopen it
if a cheaper readout fix lands first and works — that would establish the
representation is usable and make "which layer is most usable" worth paying for.

**The live lead is the readout, and it is cheaper than any layer.** Run as
`aves-readout` (2026-09-09) — see "Closed: probe-convergence levers" below. **The
mechanism guessed here was wrong** and is corrected there: the probe was *not*
overfitting. It was **undertrained**. `val_sens` rises monotonically to the
final epoch in all 5 folds, the training pool is 43 folds not 5
([[training-data-character]] — "5 rotating folds" is the eval split), and
training runs ~2 gradient steps per epoch before `EarlyStopping` halts it on a
`val_loss` plateau. A converged offline linear readout of the same embeddings
reaches 0.194 against the 0.074 logged, and YAMNet lifts too (0.218 -> 0.261).
If you find "val_sens peaks at 0.0020 in epoch 3 then decays" quoted anywhere
(it was in this section, and in `exp/aves-probe`'s `notes.md`), it is wrong —
the first ~17 epochs of a 153-epoch run misread as a trajectory.

**A non-linear head over frozen AVES is NOT blocked and is cheap — run it.**
An earlier version of this section said it was parked under the standing
1-layer injunction. That was a misreading: the injunction is about *runtime*
and bans unfreezing **backbone** layers, not head depth. An MLP over frozen
embeddings is ~1 s/epoch, the same order as the linear probe. It is also the
single most informative cheap experiment available on AVES, because it
separates "needs a better reader" from "needs the representation reshaped" —
see "What survives unfreezing" below. Caveat before running it:
`archive/2026-06_fixed-test/notes/mlp-head-repro.md` is a clear negative
(-0.028, non-overlapping CIs) but its own conclusion scopes itself — "buzz is
already linearly accessible in *this space*" — and that space is YAMNet's.
The PCA result is the reason to expect a different answer here.

Cheap now regardless of the verdict: AVES extraction went 14.3 h -> 1.0 h
(main, `8fe1336`, batched on the GPU) and `medium`'s AVES embeddings are in the
shared cache. Any future AVES question is one CV run, not a day.

## Closed: probe-convergence levers (L1-L9)

*Evidence: **E3** — `probe-grid` (2026-09-09), 13 CV runs. Closed.*

**L1 won and is the era's largest clean result: `--monitor val_sens` instead of
`val_loss`, +0.031** (0.241/0.257 over two draws vs a baseline mean of 0.2177
over three). No overlap between the two configs' draws. The gain is a
hard-deployment gain: `1_150` goes 0.021 (tight across three baseline draws) to
0.089/0.158. This revives E2's `restore-on-sens`, shelved on reasoning that does
not hold on this data.

**Everything else is dead**, and the grid's motivating hypothesis with it. The
probe was *not* step-starved: L3 (batch 4096, 9x the gradient steps) is +0.003,
L2 (stopping slack) -0.001, L4's epoch cap never binds. It converges fine — it
was being scored on the wrong curve while it did. L5 dropout (both directions),
L6 label smoothing (0.05 did not replicate: 0.233 then 0.197; 0 destabilises
`1_95` to 0.000) and L7 weight decay are all negative or inside noise. L1+L6 do
not stack (+0.005): label smoothing's overconfidence penalty is *what makes*
val_loss diverge from sens, so they are two fixes to one problem.

**The consequence for future work: the offline-readout gap is not a convergence
gap.** Do not spend another CV on optimiser levers. L8 (binary vs 15-class head)
and L9 (standardisation, already +0.043 as `standardize-blocks`) were not part
of the grid and remain open.

## WITHDRAWN: adopt `--monitor val_sens` as the default

> **Do not do this.** `monitor-leakage` (2026-09-09, **E3**) shows the lever is
> a measurement artifact. The held-out fold *is* the early-stopping monitor
> (`train.py:568`), so a `val_sens` run reports `max` over epochs of the exact
> statistic it is scored on — `peak - at_best` is exactly 0.000 on all five
> folds of every `val_sens` run on disk. Re-scored at an epoch chosen from the
> *other* folds, the +0.031 becomes **+0.007** and the composed gain becomes
> **-0.014**. `probe-grid`, `context-monitor` and `context-monitor-r2` are all
> marked `artifact` in `log.jsonl`. The section below is kept as the record of
> what was believed; the `1_150` stopping failure it describes is real, but see
> "cross-fold epoch selection" below for the fix that does not leak.

### The record of what was believed (superseded — read the box above first)

*Evidence: **E3** — `probe-grid`, strengthened by `context-monitor` (2026-09-09).
The flag exists on `exp/probe-grid` and `exp/context-monitor`, defaulting to the
old `val_loss` behaviour, so merging either branch changes nothing until the
default is flipped.*

The follow-up to `probe-grid`: make it the default and re-baseline the era on
it. Mechanical, but it moves every subsequent number, so it is Luke's call and
probably an era boundary.

**`context-monitor` raised the stakes: the flag is load-bearing for the best
config in the era, not a tidy-up.** `context_embedder` shipped **best_epoch 5**
on `1_150` — a 3072-d input reaches its `val_loss` argmin almost immediately, so
that fold was scored on a barely-trained probe (0.014, precision 0.077). Under
`val_sens` the same config ships epoch 160 and scores 0.219.

> **The hazard is real; the remedy in this paragraph was wrong.** The original
> wording here told any experiment on a wide input to pass
> `--monitor val_sens`. Do not — that is the leak (see the withdrawal box
> above), and it would replace a stopping failure with a selection artifact.
> **The non-leaking check is offline and costs nothing:** every fold persists
> `val_sens_fpr0.005_curve`, so run `tools/honest_epoch.py` on the run before
> logging an embedder verdict. A `shipped` column far below the run's
> `xfold-pooled` column is the stopping failure this paragraph is about — the
> same trap `aves-probe` fell into from the other direction — and the fix for
> it is the epoch rule, not the monitor.

## The best-known config

*Evidence: **E3** — `context-embedder`, re-scored by `monitor-leakage`
(2026-09-09).*

> **SUPERSEDED 2026-09-09 by `xfold-epoch`.** 0.277 was `honest_epoch.py`'s
> *truncated lower bound* — it re-scores at an epoch capped by the shortest
> other-fold curve. Measured on a fixed 400-epoch budget with nothing truncated,
> the honest best is **`yamnet_context` + cross-fold epoch selection at 0.288**,
> and the leak that section below attributes the whole monitor gain to measures
> **+0.021**, against a **real** +0.031 from stopping later. Read the paragraphs
> below for the reasoning, not the numbers. Cutover procedure:
> `exp/xfold-epoch:notes/new-era.md`.

**`yamnet_context` alone, honest headline 0.277.** Not `yamnet_context +
--monitor val_sens` at 0.310 — that number is `max`-over-epochs of its own
reported statistic (see the withdrawn section above). Under a cross-fold epoch
rule applied identically to both, the context embedder scores **0.277** and the
composed config **0.263**, paired 2 up / 3 down.

The context embedder is the era's real result and it got *larger* under the
honest rule: **+0.040 -> +0.061** over `cv_baseline`. Build on it.

Its shipped 0.258 understates it because `1_150` stopped at epoch 5 (a 3072-d
input reaches its `val_loss` argmin almost immediately). That failure is real —
but its own curve already reaches 0.144, so the fix is the epoch rule, not the
monitor.

## Cross-fold epoch selection as the shipped rule

*Evidence: **E3** — `monitor-leakage` (2026-09-09). Measured offline; the
implementation is unrun.*

`README.md:651` already proposes it: score each fold at an epoch derived from
the *other* folds, so the epoch that produced a fold's number never saw that
fold. `tools/honest_epoch.py` shows it is worth **+0.019** on
`context_embedder` (0.258 -> 0.277) and ~0 on `cv_baseline`, i.e. it is how you
repair a stopping failure without leaking.

**Not a drop-in.** Early stopping truncates each fold's curve at its own
patience tail, so a cross-fold epoch later than that is untestable — the tool
marks those folds `*` and most runs hit the cap. Making this the shipped rule
needs a **fixed epoch budget** (train every fold to a common cap, select
afterwards) or a two-pass fit. That is one cheap frozen-probe CV, and it is the
highest-value open item in the era.

> **DONE 2026-09-09 — `xfold-epoch`, `log.jsonl`.** Implemented as
> `--epoch-rule xfold`. Worth **+0.030** on `yamnet_context` (0.258 → 0.288),
> but the shape is the result: **−0.008** on the four folds whose early stopping
> was healthy, **+0.185** on `1_150`, which had stopped at epoch 6. It is
> insurance against stopping failures, not a general gain. **It ends the era**;
> Luke is not ready for the refactor as of 2026-09-09, so the handoff lives at
> `exp/xfold-epoch:notes/new-era.md`. Budget 250, not 400.

## Closed (negative): a buzz-only, low-variance selection statistic

*Evidence: **E3** — `buzz-selector-curves` (2026-09-09). The variance premise
failed the bootstrap when motivated; the across-epoch check it asked for is now
run and closes it.*

> **CLOSED. Do not swap the monitor for a buzz-only statistic.**
> `buzz-selector-curves` logged `val_auc_buzz` and `val_ce_buzz` per epoch
> (both now persisted as curves in `summary.json`, kept — cheap instrumentation)
> and compared all four candidates — `val_loss`, `sens@fpr0.005`, `val_auc_buzz`,
> `val_ce_buzz` — as epoch selectors under a non-leaking cross-fold rule. They
> return **identical per-fold sensitivities**, not just an identical mean:
> `[0.20, 0.41, 0.38, 0.04, 0.09]`, ~0.222 headline. The selector is not a lever
> while the pooled sens plateau is wide (e129-164), which is exactly
> `xfold-epoch`'s regime. The live epoch-selection question is `xfold-epoch`'s
> (land in the plateau, repair stopping failures), settled and deferred. The
> section below is the record of the reasoning.

Both statistics the loop has argued about are bad *selectors* for different

Both statistics the loop has argued about are bad *selectors* for different
reasons, and there is a third option neither side proposed.

- `val_loss` is averaged over all 15 classes, so it can flatten — and stop
  training — because the other 14 stopped improving while `ins_buzz` was still
  learning. `context_embedder` shipping **best_epoch 5** on `1_150`, whose own
  buzz curve climbs to 0.144, is that failure in the wild. Luke's objection:
  "I'm only interested in ins_buzz performance."
- `sens@fpr0.005` is the right target but is a tail statistic resting on ~24-35
  negative frames. Cawley & Talbot (JMLR 2010) argue that for *selection*,
  low variance matters as much as unbiasedness — an unbiased high-variance
  criterion picks a worse epoch than a biased low-variance one.

So: select on something **buzz-only but stable** — buzz-only cross-entropy, or
buzz AUC — while still reporting `sens@fpr0.005`.

**The measured caveat, and it is the reason this is a lead and not a
recommendation.** The premise is that AUC is much lower variance. Bootstrapped
over frames on `cv_baseline` (300 resamples, `models/cv_baseline` predictions),
**it is not**:

| fold | buzz frames | sens@fpr0.005 (SD) | AUC (SD) |
|---|---|---|---|
| `1_29` | 2144 | 0.426 (0.014) | 0.854 (0.0056) |
| `53` | 1031 | 0.425 (0.019) | 0.888 (0.0058) |
| `1_11` | 305 | 0.180 (0.024) | 0.765 (0.0148) |
| `1_150` | 146 | 0.021 (0.012) | 0.634 (**0.0248**) |
| `1_95` | 433 | 0.037 (0.011) | 0.645 (0.0135) |

AUC's SD is only 1.2x smaller on average and is **larger in absolute terms on
the two hardest folds**. The two metrics are on different scales, so raw SD is
not the right comparison — but it does mean the "AUC is obviously more stable"
argument is unsupported as stated, and must not be repeated as if measured.

**What would actually settle it, and it is cheap.** The quantity that matters
for a selector is signal-to-noise *across epochs*: how much the statistic moves
as the model improves, against how much it jiggles from eval sampling. That is
computable for `sens@fpr0.005` today (its curve is persisted) and not for AUC
(no curve). `callbacks.py:SensAtFPR` already computes the buzz activations every
epoch, so logging `val_auc_buzz` and `val_ce_buzz` alongside is a few lines and
near-zero cost. Do that on the next run, then compare the three curves'
signal-to-noise offline before spending a CV on the monitor swap.

Useful context for reading the answer: on `xfold-epoch`'s fixed-budget run the
**pooled** sens curve is flat from epoch ~145 to 400 (253 of 397 epochs within
0.005 of the max), while per-fold argmaxes scatter over 37-337. A selector only
has to land in that plateau, which is a much weaker requirement than finding an
argmax — so a lower-variance statistic may buy less than it appears to.

### Superseded: the one thing owed on the old best config

*Evidence: **E3** — `context-monitor` (2026-09-09), marked `artifact` by
`monitor-leakage` the same day. Kept as the record of what was believed; the
0.307/0.310 below is `max`-over-epochs of its own reported statistic.*

**`yamnet_context` + `--monitor val_sens` = 0.307**, +0.089 over `cv_baseline`
with **5 folds up / 0 down** — the only all-up run in the era. The two
components compose *additively* (+0.040 and +0.031 alone, +0.089 together) and
act on disjoint folds: context carries the rich folds, the monitor carries
`1_150`. Anything built on top of the search should start from this config, not
from `cv_baseline`. Reproduce it with `exp/context-monitor` (the embedder is a
real directory there, not a symlink; the `yamnet_context` cache is already in
the shared tree, so no extraction).

**The gain over `context_embedder` is carried by `1_150`, and that is the
result, not a caveat on it.** Over the other four folds the delta is +0.0095;
the headline is that one near-chance fold going 0.014 → 0.219. Read that as the
thing the project is for — `1_29` sits at ~0.43 whatever we do, and a lever that
only moved the rich folds would be close to worthless.

**Do not discount it with `1_150`'s repeat spread.** That spread (0.007 → 0.062
on two identical runs) is a *within-treatment* number and does not bound a
between-treatment difference. Split all 17 runs of the era by monitor:

| monitor | `1_150` | n |
|---|---|---|
| `val_loss` | 0.007, 0.014 x5, 0.021 x4, 0.027, 0.034, 0.041, 0.062 | 13 |
| `val_sens` | 0.089, 0.158, 0.171, **0.219** | 4 |

**The two groups do not overlap.** Every `val_sens` run beats every `val_loss`
run on this fold, and `context-monitor` is the highest of the four. With the
`best_epoch` 5 → 160 mechanism, that is a treatment effect, not a fold that
swung.

**The repeat is run and it confirms** (`context-monitor-r2`, 2026-09-09,
**E3**). Second draw of the identical config: **0.312** against 0.307, +0.005
on a 0.0095 single-run SD, 5 folds up / 0 down vs `cv_baseline` again. Take
**0.310** as the config's value. Nothing is owed on it any more.

`1_150` landed on **0.219 a second time**, from a different run shipping a
different epoch (160 -> 192); the fold's granularity is 1/146, so both runs
found exactly 32 of 146 buzz frames. The `val_sens` group on that fold is now
0.089 / 0.158 / 0.171 / 0.219 / 0.219 against `val_loss`'s 0.007-0.062. The
n=1 part resolved *in favour of* a real context x monitor lift: the composed
config beats the monitor-alone maximum (0.158) on both draws.

**New, and the live cheap lead from that run: the `val_sens` stopping epoch is
bimodal on RICH folds.** `Fit+Fast/53` (1031 buzz frames, not thin) shipped
epoch **17** in one draw and **226** in the other, 0.465 vs 0.520 — and
`context-monitor` saw the same ambiguity there running the other way against
`context_embedder` (144 -> 17). Most of this config's remaining run-to-run
spread is now coming from the rich folds, a reversal of the usual pattern.
A tie-break on the `val_sens` argmax — prefer the later of two comparable
peaks, or smooth the curve before taking it — is plausible variance reduction
on the era's best config. **It costs no training:**
`val_sens_fpr0.005_curve` is persisted in every fold's `summary.json`, so
replay the candidate rules offline over the runs already on disk and only spend
a CV if one of them clearly wins.

## Seed averaging inside a run — variance reduction over variance measurement

*Evidence: **E3** — `probe-grid` measured baseline run-to-run SD at 0.0095 over
n=3, putting the minimum detectable effect of a single-run comparison at
**~0.027**.*

That MDE is the reason `probe-grid` needed 13 runs to read 8 levers, and it is
why every sub-0.027 result in `log.jsonl` is unreadable as stated. Averaging
2-3 seeds per fold *within* one run would shrink the error on every future
experiment for ~2-3x the compute of one run — cheaper than repeating whole
experiments, and it compounds.

It changes what "a run" means and breaks comparability with everything already
logged, so it is an **era-boundary decision**, not a mid-search change. Raise it
at the next cutover.

**Related standing fact:** `cv_baseline` is a fair draw (0.218 vs a 3-run mean
of 0.2177), so nothing currently in `log.jsonl` is biased by a lucky
denominator. `models/base_r2` and `base_r3` on `exp/probe-grid` are the repeats.

## What survives unfreezing — read before spending a fine-tune

*Evidence: **E3** reasoning over `aves-readout` (2026-09-09) plus **E2**
fine-tuning results. No fine-tune has been run since the CV rework, so this is
a prediction, not a measurement. Written because the frozen-probe grid above is
the intended input to a later unfreeze, and half of it will not transfer.*

The plan this anticipates: settle the probe-convergence levers cheaply on the
frozen probe, *then* unfreeze and retrain. That sequencing is right. What it
must not do is carry the wrong half of the grid across.

**Update (`probe-grid`, 2026-09-09): the grid is run and this section's
prediction held.** L1 was the only survivor on the frozen probe (+0.031) and is
exactly the lever this section says gets *more* valuable once the trunk moves.
L2 did not pay frozen (-0.001), so the L1/L2 pairing below should be read as L1
only. The levers this section predicts will not transfer (L5-L7) are all dead
frozen too, so nothing is owed them.

**Transfers, and gets more valuable:**

- **L1 (restore on `val_sens`) and L2 (stopping slack).** `restore-on-sens`
  measures the sens/`val_loss` divergence it targets at **+0.03-0.05 on
  `unfreeze_more_1e5`** against ~+0.006 on the frozen probe, and names it "a
  backbone-fine-tuning phenomenon." So the stopping rule is worth several times
  more once the trunk moves. Fixing it *before* a long run is the whole point —
  otherwise a 24 h fine-tune gets halted by a rule that stops while the shipped
  metric is still rising.
- **L3 (batch size)** resolves itself: full-batch is a frozen-probe luxury that
  disappears once activations must be stored. But every epoch-denominated
  setting (patience, caps) is then measured in different units — re-derive, do
  not copy.

**Does not transfer:**

- **L5-L7 (dropout, label smoothing, weight decay).** Frozen, these regularise
  ~11.5k trainable parameters (768x15). Unfrozen AVES is ~95M. The archive
  already shows the scale of the shift: trunk fine-tuning needs lr 1e-5 against
  the frozen probe's 2e-3, 200x. Anything capacity-dependent must be re-found
  after unfreezing; treat frozen optima as uninformative priors, not defaults.

**Do not let the frozen ranking choose the embedder.** This is the important
one. YAMNet's frozen advantage is substantially an artifact of *what it is* —
the penultimate layer of a supervised classifier whose 521 AudioSet classes
include `Buzz`, `Bee, wasp, etc.`, `Insect`, `Mosquito`. Its embedding is
pre-shaped so that one linear layer separates exactly this concept: the maximum
possible head start for a frozen linear probe. Unfreezing removes precisely that
advantage, because once the representation can be reshaped, "already linearly
separable" stops being worth anything. AVES is self-supervised and was never
shaped for linear separability of anything, and the PCA result says its buzz
evidence exists but is smeared over ~256 directions — which is the thing
fine-tuning is for. **Expect the 0.194-vs-0.261 gap to narrow under fine-tuning,
and do not rule out an inversion.** The frozen-probe comparison is biased
against SSL backbones by construction.

**Feasibility, before anyone commits.** AVES base is ~95M parameters on a 4 GB
GTX 1650: Adam moments alone are ~1.1 GB at fp32 before activations, so a full
unfreeze is likely infeasible and the realistic version is a top-N transformer
layer unfreeze. YAMNet trunk-ft was ~80 s/epoch against ~1 s frozen; AVES will
be worse. Measure a single epoch before budgeting ([[machine-gpu-constraints]]).

**The cheap decision gate.** A non-linear head on *frozen* AVES costs minutes
and answers the question a fine-tune would answer expensively: if an MLP closes
most of the gap, the information is there and merely needs a better reader, so
unfreezing is very likely to pay; if it doesn't, the representation itself may
need to move. Run that before budgeting a fine-tune, not after.

## near-chance-deployments

*Evidence: **E3**, with **E2** origins. The fold identities and the concept-coverage ruling carried over from E2; every number below was re-measured on the current 5 folds.*

Two of the five rotating folds sit near zero for every model tried:
`Diel Drivers/2026-04-08/1_150` (0.021) and `2026-05-06/1_95` (0.037) on
`cv_baseline`'s predecessor. Since the endpoint averages deployments equally,
**they pin 40% of the headline near zero** — a much bigger share than the 27%
they were at 11 folds, which makes this the single largest lever on the metric.

They are also, as of 2026-09-08, **actively corrupting comparisons**, not just
depressing the mean. `standardize-blocks` produced a null result on every fold
that can resolve one and still moved the headline +0.031, entirely because
1_150 swung +0.151. Two of five folds contributing noise larger than any real
effect this era has produced means the 5-fold mean is not a usable comparator
on its own. Until they are fixed, read `compare_folds.py`'s per-fold table
before the headline — and consider whether the endpoint should weight folds by
buzz-frame count, or exclude a fold whose repeat-run spread exceeds its delta.

One story is ruled out (2026-09-05): it is not leave-one-*concept*-out.
Scoring each fold by how well its buzz sublabels are covered elsewhere gives
r = 0.013 against per-fold sensitivity, and the worst folds are 100% plain
`ins_buzz_medium` with >6,000 frames of exactly that elsewhere. Whatever is
wrong is acoustic or site-level.

The concept-coverage ruling was **re-confirmed 2026-09-08**. A rerun appeared to
overturn it (1_95 looking 89% starved) but had filtered support to `role=='rotate'`
only; against the real training pool, which includes `train`-role idents, every
fold is at 0% thin — 1_95's `ins_buzz_low` has 957 s of support and
`ins_buzz_pollination` 300 s (two "Various Opportunistic Recordings" idents).
Check roles before reviving this.

**The two folds fail in opposite ways** (measured 2026-09-08 from `cv_baseline`'s
surprisal CSVs, no training):

| | 1_150 (0.021) | 1_95 (0.037) |
|---|---|---|
| negatives, p99.5 | 0.139 — normal | **0.391 — ~3x every other fold** |
| buzz frames, mean/median | **0.071 / 0.064 — ≈ its own background** | 0.102 / 0.068 |
| frames setting the threshold | diffuse ambient (11 bg, 6 mech_auto) | **32 of 35 `mech_auto`** |
| fold threshold | -1.819 | **+0.173** (the only positive one) |

So they need different fixes and should stop being treated as one item:

- **1_95 is a false-positive problem** — vehicle noise, not trill, sets its
  threshold, which is why the threshold goes positive while every other fold sits
  near -1.7. Its buzz is 242 s of 328 s `ins_buzz_low`, and low buzz vs engine
  drone is an acoustically plausible confusion. This is the fold **night-negatives**
  would help most. (The other candidate, a buzz-vs-`mech_auto` margin, was run
  and is closed negative — below.) `mech-margin` also measured *why* it is hard:
  its buzz and non-buzz frames are separated by only **0.513** of mean logit in
  `cv_baseline`, against 1.578 at `1_29` and 1.425 at `Fit+Fast`. It is not that
  its threshold is misplaced; the two populations barely separate at all.
- `1_95` was the explicit target of `harmonic-comb` (2026-09-08) and **did not
  move**: +0.002 in each of two runs, and the only fold byte-stable across two
  nondeterministic runs, i.e. its failure is structural rather than stochastic.
  An explicit f0 channel that cleanly separates a 90 Hz engine comb from a
  220 Hz wingbeat comb on synthetic tones does nothing on this fold's real
  audio. **The buzz-vs-`mech_auto` margin that used to be named here as the
  remaining candidate is now closed and negative** — see
  "Closed: absolute-margin confuser penalties" below. `1_95` moved *down* at
  every dose. It has now defeated two targeted interventions (`harmonic-comb`,
  `mech-margin`), so nothing cheap and loss-shaped is left for it. The remaining
  lead for `1_95` specifically is **night-negatives** — in-domain hard negatives
  from its own recorder — which needs a data decision from Luke, and which the
  measurement below now argues for much more strongly than when it was proposed. (The
  listen-to-the-audio item below is `1_150`'s, not this fold's.)
- **`1_95`'s threshold is set by two five-minute windows.** Rebasing
  `cv_baseline`'s false positives onto Even Sample's snips (2026-09-09): of the
  42 negatives above the fold's own fpr0.005 threshold, **22 fall in
  `260506_1924_s95760` (22:00-22:05) and 11 in `_s88560` (20:00-20:05)** — 79% of
  them in 2 of the fold's 24 snips, and 32 of the 42 are `mech_auto`. Of the top
  60 negatives by buzz score, 47 are in those same two snips. Dropping just those
  snips' *negatives* takes the fold from 0.034 to 0.059 (s95760) to **0.097**
  (both) — a diagnostic, not a legitimate metric, but it says the fold's
  near-chance score is a property of ~10 minutes of audio, not of its whole day.

  **This is the same event `tools/night-positives/README.txt` already
  documented** — a ~5% nighttime false-positive spike in a 20-minute bin starting
  at 22:00 on this exact recording, found independently with `model_general_v3`,
  with the audio already extracted to `tools/night-positives/260507_2200.mp3`.
  Two unrelated models, two eras apart, fail in the same ten minutes. That is
  the strongest evidence in the file for **night-negatives**, and it means the
  first dose of it could be built from audio that is already on disk.

- **1_150 is a positives problem** — its negatives are unremarkable and its buzz
  frames are simply indistinguishable from its own background, despite being 88 s
  of plain `ins_buzz_medium` with 6503 s of support. Nothing structural explains it.

**DONE 2026-09-09 — Luke listened, and the annotations are sound.** The buzzes
are real: "fairly quiet and there's background noise, but they're there. Most of
them are very quiet, but still legitimate targets." So the two cheap
explanations are dead — this is **not** a mislabelling and not a phantom. `1_150`
is a genuine low-SNR detection problem, which is the hardest of the possible
answers and the one that says the fold is honest. Do not re-open it as an
annotation-quality question; if it is attacked again, attack it as quiet-buzz
sensitivity (and note `context-embedder` pushed its threshold *down*,
-1.819 -> -2.107, without helping, so a better representation alone did not
touch it).

The original framing, kept because it is what got answered: a narrow target of
**88 seconds of audio.** Is the buzz audible? Is the annotation right? Is
recorder gain or placement different? Note `context-embedder`
pushed its threshold further down (-1.819 → -2.107), so a better representation
does not touch it.

**Caveat on the folds it is measured against:** seven `ins_buzz` annotations
spanning a whole 300 s file supply **63%** of mustard's and **52%** of Fit+Fast's
buzz seconds (none in the other three folds). Those two folds carry the top of the
headline, their `buzz_frames` counts overstate their independent sample size, and
part of what the metric rewards is detecting a continuous drone.

## night-negatives

*Evidence: **Untagged — a proposal, never measured.** Needs a data decision from Luke.*

**Needs a data decision from Luke — an experiment may not add or edit an
annotation effort unilaterally.** Surfaced here because it is the largest
untapped resource in the project.

`01_annotate/2026-05-26 Automatic Annotations/README.md` reports that the
shipped model has a tight false-positive spike near midnight, and that the false
positives "can even dwarf the diel trend in the focal crop (e.g. `Luke - Diel
Drivers/2026-05-06`)" — one of the two near-chance folds above. Every nighttime
detection is false by construction, so labels generate automatically at any
volume.

Why it beats another architecture run: these are **in-domain hard negatives from
the deployment's own recorder**, the thing every `aug-*` experiment tried to
synthesize and failed at. It is **fold-safe for free** (night audio from site X
belongs to site X's fold, so CLAUDE.md's augmentation rule doesn't bind). And it
**scales with deployments, not annotation labour**, so it doesn't get invalidated
as the set grows.

Settle before running: **what label?** A nighttime FP could be trill, plane or
truck, and calling it all `ambient_background` teaches a class the probe meets in
daylight too; `ins_trill` is the likeliest true identity for most, and
mislabelling trill as background would be actively harmful — consider a distinct
`auto_night_negative` with its own translation row. **How much?** A dose-response
(0x / 1x / 4x the fold's existing negatives) is the experiment, not one volume.

## Closed: handcrafted frequency features

*Evidence: **E3.** Supersedes the weaker **E1** closure (`supp-freq-v2`, under the retired metric).*

`harmonic-comb` (2026-09-08) tested the strongest form — 40 f0 candidates over
70-450 Hz with harmonic reinforcement and off-comb subtraction, plus modulation
and band contrast, gain-invariant, no fitted statistics — and it did not move
the metric beyond the repeat-run spread. Crucially the block was **not
ignored**: the saved fold weights put **1.29x** the per-dim `|w|` of a YAMNet
dim on the comb block in all 5 folds, and **1.64x** on the 40-d comb profile.
Used, preferentially weighted, no effect => the f0 information is **redundant**
with what YAMNet's 1024-d already encodes. That is a much stronger closure than
`supp-freq-v2`'s, which rested on four unresolved global scalars under the
retired metric. Don't rerun the fixed version. A *learnable* filterbank is the
only untried variant and the redundancy finding argues against it — the probe
is not starved of this information.

## trill-vs-buzz

*Evidence: **E3** answer to an **E2** hypothesis. The counts below are from `cv_baseline` on the current roster.*

**Hypothesis:** the false positives that set the threshold are mostly
`ins_trill`, and buzz-vs-trill is the real discrimination problem.

`ins_trill` was the largest non-ambient class in the rotating folds by a wide
margin. But it is very unevenly distributed, and the correlation runs the
*opposite* way to the naive story: the folds with essentially no trill were the
worst (0.046, 0.068) and the fold with the most was the best (0.469). Worth
understanding either way — it may just be that trill-rich folds are insect-rich
folds. Recount on the current 5 folds before theorising; the roster changed.

**ANSWERED 2026-09-08, and the hypothesis does not hold.** Raw labels of the
negatives above each fold's own fpr0.005 threshold in `cv_baseline`, pooled by
component label: **`mech_auto` 75, `ambient_background` 46, `ins_trill` 43**,
everything else ≤8. Trill is about a third, not a majority, and it is
fold-dependent — trill leads at willard, vehicles lead at 1_95 (32 of 35) and
mustard. (Read it off the `surprisal/` CSVs, which carry `start`, `label` and
per-class activations and reproduce `folds_sx.csv` exactly;
`cv_baseline/predictions.csv` turned out to hold only
`activation_ins_buzz,correct`, no `labels_raw`.)

**So the pairwise buzz-vs-trill margin this idea recommends would be aimed at the
wrong class.** The buzz-vs-`mech_auto` margin it redirected to was run as
`mech-margin` (2026-09-09) and is **closed negative** — see below. Note the
readout geometry measured there cuts against the redirection too:
`cosine(W_ins_buzz, W_c)` puts `ins_trill` (+0.268) and `ambient_background`
(+0.265) closest to buzz and `mech_auto` near-orthogonal (-0.016), so trill is
the class whose *readout* is entangled with buzz even though vehicles supply
more of the FP frames. Smoke-test any custom loss with `tools/smoke_model.py`
first; `tail-loss` is the cautionary tale.

## eval-sampling-floor → annotation guidance

*Evidence: **E2** bootstrap, **E3** confirmation by direct repetition. The two agree, which is why this one is trustworthy.*

**Measured, not hypothesised** (`tools/eval_sampling_sd.py <model dir>`,
seconds, no training). Bootstrapping frames within each fold gave a per-fold
sampling SD of 0.02–0.03 on normal folds and up to 0.092 on the thinnest,
propagating to ~0.012 of headline SD from the evaluation sample alone.
`noise-floor-cv` put total run-to-run noise at ~0.014 headline — so **roughly
half the noise floor is eval-set sampling, not training stochasticity.** Seed
control would not fix it, and a partial-AUC estimator was tried and tracks
sens@0.005 almost exactly without cutting SD much.

**Independently confirmed 2026-09-08 by direct repetition** (`harmonic-comb`).
Two CVs of one identical config gave 0.223 and 0.239 — **0.016 headline apart,
3 folds up / 1 down / 1 flat**, from TF nondeterminism alone. That is the whole
run-to-run distribution, measured on the current 5-fold roster rather than
inferred. The fold structure matches this section's bootstrap: the two folds
with >1000 buzz frames moved 0.019 and 0.013, while `1_150` (146 buzz frames)
moved **0.055** — 0.007 to 0.062 on the same config. Read any result in this era
under ~0.02 headline, or any result resting on `1_150`, against that.

**RE-MEASURED 2026-09-09, and this section's numbers were stale in a way that
changes the advice.** `tools/eval_sampling_sd.py` on the current roster:

| fold | buzz frames | boot SD (`cv_baseline`) | boot SD (`context_monitor`) |
|---|---|---|---|
| 1_29 | 2144 | 0.014 | 0.029 |
| 53 | 1031 | 0.020 | 0.029 |
| willard | 305 | 0.024 | 0.027 |
| 1_150 | 146 | 0.012 | 0.037 |
| 1_95 | 433 | 0.010 | 0.010 |
| **headline** | | **0.007** | **0.012** |

Per-fold SD is **0.010-0.037**, not the 0.02-0.092 recorded above, and the
"±0.25 or worse" that was in LOOP.md is ~7x too large. Snip equalization is why:
`frames_val` is now within ~1.5x across folds and `neg_frames` is 24-35, so the
FPR side is no longer the ragged thing these numbers were measured on. Buzz
density is not equalized and cannot be — `buzz_frames` still spans 14.7x — but
that turns out to cost much less than assumed.

**The advice inverts for the thinnest fold.** `1_150` has the *smallest*
eval-sampling SD of the five under `cv_baseline` (0.012 — a fold pinned near
chance has little binomial variance) while moving **0.055** between two identical
runs. So its run-to-run noise is **training stochasticity, not eval sampling**,
and more annotated negatives there will not shrink it — a seed control or seed
averaging would. That is an argument for the seed-averaging section below, not
for annotation.

Two limits on the new numbers: the bootstrap resamples frames independently
while frames within a buzz event are correlated, so read 0.037 as a floor; and
it says nothing about training variance, which is the larger term on `1_150`.
More annotated non-buzz frames still buy measurement precision, but the case is
weaker than this section used to make it. Re-run the tool rather than quoting
any of these figures — it is seconds and the roster moves.

## willard-regression

*Evidence: **E2** claim that **did not reproduce in E3.** Kept as a worked example of an inherited caution dissolving, not as a caution.*

`context-stack` gained in 8/11 deployments but lost 0.074 at
`willard/1_11` — still a rotating fold — and the regression scaled monotonically
with context width (0.177 → 0.118 at k=1 → 0.066 at k=2).
`exp/deployment-forensics` attributes it to willard having the highest fraction
of short (<1 s: 68%) and isolated (>5 s gap: 68%) buzz events of any fold:
stacking dilutes a brief isolated buzz with silent neighbours.

**Did not reproduce (2026-09-08).** `context-embedder` on the revised data
**gained** 0.037 at willard, and frame sensitivity bucketed by the span of the
annotation each buzz frame falls under runs the wrong way for dilution: <1 s
**+0.046**, 1-2 s +0.058, 2-5 s +0.039, >5 s **+0.026**, with clustered +0.042 vs
isolated +0.034. Short and isolated buzz gains *more* than long and clustered.

Treat the dilution story as unsupported on live data rather than as a standing
caution — it was measured on the pre-revision annotations, and willard is thin
(305 buzz frames), so the original -0.074/-0.059 may always have been fold
variance. Note the >5 s bucket above is mostly the file-spanning drone
annotations (see **near-chance-deployments**), not long buzz events.

## subframe-head (options 2 and 3 only)

*Evidence: **E2** — 11 folds, pre-revision data. Option 1's negative is an E2 negative: a lead, not a closed door.*

**Option 1 is tested and negative** (2026-09-06): time-max/freq-mean pooling
frozen against `trunk_frozen` gave -0.007, folds split 5/6 — inside the noise
floor. Willard, the named test case, moved only +0.010. Don't rerun it as
stated.

The underlying observation still stands: `yamnet_trunk` caches
`layer12_pointwise_conv_relu` at **(6, 4, 512)** — 6 time steps, 4 frequency
bands per 0.96 s frame — and GAP averages all of it away. A buzz occupying
200 ms of a 960 ms frame has its evidence divided by ~5. Frequency is discarded
too, and buzz is narrowband.

Untried: **(2)** log-sum-exp over time instead of a hard max, if the max's
gradient path was the problem; **(3)** keep the frequency axis — pool over time
only, flatten to 2048-d. Both need the `yamnet_trunk` cache, which no longer
exists for `medium` (see **trunk-fine-tuning** below), and option 1's result
means neither is a priority.

## The `binary` control is confounded — read before running it

*Evidence: **E3** — read off `train_utils.build_weights` and `train.py`, 2026-09-09.
Arithmetic, not a measurement.*

LOOP.md flags the `general`/`binary` pair as "worth rerunning early". It is, but
**a naive rerun does not measure what it looks like it measures**, because
collapsing the taxonomy changes two things that have nothing to do with the
taxonomy:

1. **`ins_buzz`'s own positive weight moves ~5.4x.** `build_weights` uses
   `weight = samples_total / (samples_class * n_classes_present)`, so the class
   *count* is in the denominator of every class's weight. On `medium`:
   **0.76 under `general` (15 classes) vs 4.10 under `binary` (2)**. That is a
   large change to the buzz neuron's positive/negative balance, applied purely
   as an artifact of how many classes exist.
2. **`val_loss` is the mean over neurons, so the stopping signal changes
   composition** — ~1/15 buzz under `general`, ~1/2 under `binary`. The epoch
   that gets restored is therefore chosen on a different curve. `probe-grid`
   measured the stopping rule as the largest lever in the era (+0.031), so this
   is not a second-order concern.

Both channels are real and neither is "does a multi-class taxonomy teach the
model what is not a buzz". **Run the control with `ins_buzz`'s `pos_weight`
pinned to `general`'s value and the monitor held fixed**, or the number is
uninterpretable. If pinned-`binary` then matches `general`, the taxonomy's
apparent benefit is weights-and-stopping and can be had directly, without
carrying 15 classes to get it.

**What the taxonomy provably does *not* do is take gradient away from buzz.**
The head is `Dropout -> Dense(n_classes)` with no activation, into
`weighted_cross_entropy_with_logits` — 15 independent sigmoids, no softmax, so
`W[:, buzz]` sees gradient only from the `ins_buzz` term
([[decoupled-probe-head]]). The "one neuron lights up, the others plummet"
behaviour is real but is *inference geometry*: the readout vectors share one
1024-d space and end up anticorrelated where the labels are near-exclusive
(`cosine(W_ins_buzz, W_mech_background) = -0.390`). Softmax-looking, not softmax.
The live paths from a translation to the buzz neuron are `exclude`/`ignore`
reshaping the frame pool, multi-hot frames losing a buzz positive, the weight
renormalisation above, and the monitor composition above.

## Closed: absolute-margin confuser penalties

*Evidence: **E3** — `mech-margin` (2026-09-09), a 4-CV geometric dose ladder. Closed.*

A class-conditional hinge `lam * 1[mech_auto and not ins_buzz] * relu(z_buzz + m)`
at `m = 2.0`, over `lam` 0.125 / 0.5 / 2.0 / 8.0: **0.202 / 0.169 / 0.090 /
0.021** against `cv_baseline`'s 0.218. Monotone in the dose, 5 folds down and 0
up at every dose at or above 0.5, the two rich folds carrying the largest
losses, and the largest doses 5-7x `probe-grid`'s ~0.027 MDE. `1_95`, the fold
it was designed for, is down at every dose.

**Do not re-run this in another costume.** The mechanism generalises past
`mech_auto`, and it is the reason to read this section:

- **The `mech_auto` share of threshold-setting FPs *rose* with the dose** —
  26.5% -> 32.1% -> 39.3%. The term pushed those frames down in absolute terms
  and lost ground on them in rank.
- **The metric reads rank only.** Every fold is thresholded on its own held-out
  audio, so a term that drives a subpopulation below a fixed logit buys nothing
  by itself; it can only pay by *rotating* the readout. Here rotating away from
  `mech_auto` frames rotates away from buzz: the buzz/non-buzz mean-logit gap
  collapses monotonically (`1_29` 1.578 -> 1.000 -> 0.519), and at `1_95` under
  `lam 2.0` it **inverts** to -0.064.
- **The overlap is in the frame population, not the weights.**
  `cosine(W_ins_buzz, W_mech_auto) = -0.016` — near-orthogonal. `mech_auto`
  frames simply sit high on the buzz direction, which is *why* they lead the FP
  census, and no linear readout of frozen YAMNet separates them.

**What is left of the idea.** A confuser penalty must be **pairwise** — buzz
frame ranked above confuser frame — not absolute, so that it expresses the
ordering the metric actually scores. That shape is `tail-loss`'s, whose E2
failure was a *monitor* artifact (batch-local rank made the compiled loss
invalid as `val_loss`) rather than evidence against ranking losses; the fix is a
pairing that is deterministic per frame, e.g. against a frozen reference model's
scores rather than the live batch. Nobody has run that. Weigh it against the
plainer reading of this result — that a linear readout of frozen YAMNet has no
room to separate these populations at all, which is an argument for
[[shared-trunk-head]] or a better representation rather than for a cleverer
loss.

**Reusable tooling.** The FP census (raw labels of negatives above each fold's
own fpr0.005 threshold, from `models/<m>/surprisal/**/*.csv`) and the
buzz/non-buzz mean-logit gap are a two-minute read on any model with
`surprisal/` on disk, need no training, and recompute for free after a data
revision. The gap in particular is the diagnostic that explains a fold's
sensitivity without reference to its threshold.

## Closed (null): shared-trunk-head — cross-class supervision reaching buzz

*Evidence: **E3** — `shared-trunk-head` (2026-09-09), a 3-CV width ladder. Ran
and closed on the current roster. The **structural fact** below is read off
current code and still holds; only the proposal was tested.*

**Result: null.** h = 64 / 256 / 1024 → 0.219 / 0.221 / 0.226 against baseline
0.218, folds split 3 up / 2 down at every width, all inside `probe-grid`'s
~0.027 MDE and inside the baseline's own n=3 range (0.208-0.227). Not a
negative — no width hurt — but nothing to build on. **1_95, the fold the
mechanism was aimed at, did not move at any width** (-0.005 / +0.002 / -0.007),
which is the direct answer to `mech-margin`'s "a linear readout cannot do this,
a non-linear stage might": given the non-linear stage, it still doesn't.
The trunk reaches the same mean `best_val_loss` (0.907 vs 0.910) in 2.5x fewer
epochs — same destination, shorter path — so YAMNet's code already carries what
the trunk would have learned. `--hidden` is in the code, default 0.

**Step (2) (per-neuron buzz weighting on top of the trunk) is unblocked but
deprioritised, not closed.** Its premise was that a weighted buzz loss needs
something to be weighted *against*; the trunk supplies that and buys nothing,
so the follow-up is loss tuning over a structure with no demonstrated headroom.

**Live residue, and it is a diagnostic rather than an architecture run:** the
two rich folds trade consistently — 1_29 gains ~+0.03 at all three widths while
Fit+Fast loses at all three, both above the ~0.019 repeat movement for
>1000-buzz folds, neither flipping sign across three runs. The mean is null
because they cancel. Nobody has asked what distinguishes those two deployments.

The structural reasoning that motivated it is retained below, because the
`binary`-control prediction at the end of it is still untested and still a
valid check on the framing.

**The structural fact.** `train.py` builds `Input -> Dropout(0.2) ->
Dense(n_classes)` — one layer, no hidden stage. So `ins_buzz`'s logit is a
function of `W[:, buzz]` and `b[buzz]` alone, the loss is a sum of independent
per-neuron terms, and **the 15 classes are 15 decoupled logistic regressions**
sharing only the input dropout mask. No gradient path runs from a `mech_auto`
error to `W[:, buzz]`.

Three consequences, all of which bite:

1. **Non-buzz confusion is already free.** Getting `ambient_noise` wrong costs
   the buzz neuron nothing. There is no "spend less capacity on the other 14
   classes" gain available, because none is being spent.
2. **A per-neuron loss weight is a no-op.** Scaling `ins_buzz`'s neuron loss by
   alpha scales only that neuron's gradients, uniformly — and Adam is invariant
   to uniform per-parameter gradient scaling (up to epsilon). The existing
   `reduce_mean` over 15 neurons is the same constant, equally nullified. Do not
   spend a CV on this lever; it cannot move the number.
3. **`build_weights`' per-class weights only ever act *within* their own
   neuron.** For buzz, `pos_weight` is a positives-vs-negatives tradeoff inside
   one logistic regression. Cross-class weight *comparisons* are meaningless.

**And a fourth, about the metric.** To first order, changing a logistic
regression's positive/negative balance moves the **bias**, and every fold is
thresholded on its own held-out audio — so a pure bias shift is *invisible* to
sens@fpr0.005. This is why raising `ins_buzz`'s weight reads as "overprediction"
with no metric movement: the per-fold threshold absorbs exactly the thing the
weight moved. Only changes to the *direction* `W[:, buzz]` — i.e. to which
negatives pull on it — can move this metric.

**The proposal.** Insert a shared hidden layer:
`Dropout -> Dense(h, relu) -> Dropout -> Dense(n_classes)`. Now all 15 heads
read one learned intermediate representation, so `mech_auto` supervision shapes
features the buzz neuron also uses, and a buzz-weighted loss finally has
something to be weighted *against*. This is the multi-task-transfer argument,
which is a different claim from the capacity argument that MLP heads were
closed on in E1/E2 — those asked "does the probe need more capacity?" (no).
Nobody has asked "should the auxiliary classes inform buzz at all?"

**Sequencing matters, and it is two experiments, not one.** The hidden layer is
the *enabling* change; the buzz weighting is only meaningful afterwards. So:
(1) shared hidden layer alone vs `cv_baseline`, uniform loss — does connecting
the classes help or hurt on its own? (2) only if (1) is not a clear negative,
per-neuron weighting on top, paired against (1). Running them together confounds
an architecture change with a loss change and tells you nothing.

**Costs and honest priors.** Cheap — a frozen-probe CV, ~9 min, no
re-extraction, and head depth is explicitly outside the standing no-unfreezing
injunction ([[injunction-no-long-training-runs]]). But: YAMNet's 1024-d is
already linearly separable for this concept by construction, which is the case
where a hidden layer buys least; the training pool is ~72k frames with ~7.7k
buzz, so a wide hidden layer can overfit the training sites; and the *reason*
the auxiliary classes might help — that vehicles and trill are the confusers
that set the operating point — was attacked directly by a per-confuser margin
and **failed** (`mech-margin`, above). That cuts both ways: it removes the
cheaper alternative, and its mechanism (a *linear* readout of frozen YAMNet
cannot push `mech_auto` frames down without taking buzz with them, despite the
two readout directions being near-orthogonal) is precisely the situation where a
non-linear stage has something to add. Treat `mech-margin`'s failure as mild
support for this section rather than against it. Read this as a real lever with a
plausible mechanism, not as a favourite.

**Free prediction that tests the whole framing.** Because the head is decoupled,
the 15-class head's buzz neuron *is* the binary probe, up to label-mapping
differences. So **L8 (`binary` vs `general` translation) should come back a
near-null.** LOOP.md already flags that control as worth rerunning early; if it
comes back a *large* effect in either direction, the decoupling reasoning above
is missing something and this section should be re-derived before anyone builds
on it.

## Cheap and open

*Evidence: **E2 unless noted.** None of these has been re-established on the current roster.*

- **[E2] Patience 20–25 instead of 50.** Every archived run stopped by early
  stopping, never at the epoch cap, so patience is a flat tax of exactly N
  epochs per fold — 35–94% of a trunk-FT fold's compute. Replaying
  `EarlyStopping` over the saved `val_loss_curve`s puts patience 25 at 58% of
  the compute while restoring a different epoch on 3/4 folds; but those
  differences are the same size as the ±15–50-epoch run-to-run jitter
  `trunk-ft-stop-sweep` measured at patience 50 for an *identical* config. Now
  that `val_sens_fpr0.005_curve` is persisted per fold, this is answerable
  offline from any run — replay it before changing the default.
- **[E2] restore-on-sens' other half, on a trunk fine-tune.** Restoring the true
  val_loss argmin landed in `cv_baseline`. Restoring the *sens@FPR argmax*
  instead was worth another ~+0.006 — inside the noise on a frozen probe, which
  is why it was left out, but the divergence it exploits is a
  backbone-fine-tuning effect (label-smoothing overconfidence). `trunk-ft-restore-sens`
  saw the two curves diverge on all 11 folds, 8 shipping a later epoch than the
  loss argmin. Retest there, not here.
- **[no era — pure throughput, no metric] Split framing from embedding in `--workers`.** `--snip-workers` already
  separates the I/O-bound snip sync, but `--workers` still covers both framing
  (CPU, no GPU) and embedding (VRAM-bound), so protecting 4 GB of VRAM with
  `--workers 1` needlessly serialises the framing too. Pure throughput; no
  effect on any metric.

## Deliberately parked

*Evidence: **Mixed — each bullet is tagged individually.** Read the tag before quoting a number.*

- **[E2] Trunk fine-tuning.** `trunk-ft-1e5` (YAMNet layers 13-14 at lr 1e-5) was
  +0.046 at 9/11 folds, the largest clearly-outside-noise result in the archive,
  and depth is settled: frozen 0.216 → 14-only 0.234 → **13-14 0.262** → 12-14
  0.229, a true interior optimum. It is parked, not dismissed: it needs a
  `yamnet_trunk` re-extraction (the `medium` cache went with the pruned
  worktrees), it is the expensive config (~80 s/epoch vs ~1 s), and the standing
  injunction bans unfreezing backbone layers. Revisit when that lifts — note
  the injunction is about *runtime* and does not restrict head depth, so an
  MLP head over frozen embeddings is available now. LoRA is a
  weak follow-up now that plain FT works — it is a transformer technique and
  inserting it into YAMNet's conv layers is non-standard.
- **[no era — standing policy] `large`-set confirmation.** **Training on `large` is forbidden without Luke
  asking explicitly**, however ready it looks. It is a one-time final
  confirmation after the structural search on `medium` concludes, not another
  set to rotate through. When it happens: `large` is the same annotations and
  folds at 5x frame density, and its `folds_sx.csv` is **not comparable to any
  `medium` number** — changing frame density moves the negative population the
  FPR threshold rests on (`framehop-overlap` is the worked example). The valid
  test is a matched control on `large`, i.e. two CV runs, and a `large` CV is
  much longer. Budget deliberately or don't start.
- **[E3] Frame length, isolated from embedder — the `perch-probe` follow-up.**
  `perch-probe` (2026-09-09) came out +0.026 and **inconclusive**: Perch's 5.0 s
  frame moved the labelling rule at the same time as the embedder, because
  `overlap_event_s = framelength_s * overlap_event_prop`. At 0.2 that is a 1.0 s
  overlap requirement against YAMNet's 0.192 s, and the median buzz annotation
  is 1.00 s — 31% of buzz events can no longer label a grid frame at all and
  enter only through centred rescue (488 rescues vs YAMNet's 187). Perch ends up
  calling 17,970 frame-seconds buzz where YAMNet calls 9,618.
  **Per-fold dilution predicted the per-fold delta exactly**: the two folds where
  buzz-labelled audio inflated ~2.9x gained +0.131 each, the fold where it barely
  moved (1_29, 1.22x, and the only fold rich enough to resolve an effect) lost
  0.195. Read that as a measurement of the relabelling.
  The experiment worth running is **YAMNet re-extracted at a 5 s effective
  frame** — one variable, and it also answers whether a coarser frame is worth
  having on its own. Design note: hold **`overlap_event_s` absolute**, not
  `overlap_event_prop`, or the control reproduces the same confound. Same
  `framehop-overlap` caveat applies — its `folds_sx.csv` is not comparable to
  `cv_baseline`, so this is two CV runs, not one.

- **[no era — infrastructure] Timestamp join for provenance.** `frametimes.csv` isn't on disk for current
  idents — it was added to the extractor after they were last extracted and the
  fingerprint hasn't changed since. `frame_index` joins to it whenever a fold
  re-extracts for some other reason; don't force a re-extraction just for this.
