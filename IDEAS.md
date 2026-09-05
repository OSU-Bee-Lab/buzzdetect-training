# Experiment Ideas

Check this file and `log.jsonl` before proposing an experiment, then take it
through the lifecycle in `LOOP.md`.

`log.jsonl` holds CV-era runs only — measured by leave-one-fold-out on day-long
annotated recordings, against mean per-fold sens@fpr0.005. Those numbers are comparable to
each other and are the ones to beat.

Everything under [Tried before the rework](#tried-before-the-rework) comes from
an era with a different metric, a different eval corpus, and a training set that
was a scattershot of collection methods. Its numbers are gone and its verdicts
are unreliable — see the warning at the head of that section before you treat
any of it as settled.

---

## Where to spend effort while the annotations are still moving

Three of the last four experiments were unfreeze-depth variants, and depth is
now settled on `medium` (13-14, `lora-adapter` below). The rest of the trunk-FT
follow-up list is hyperparameter work, which LOOP deprioritises and which
`trunk-ft-stop-sweep` already showed is inside the noise. That vein is close to
mined out at this frame density.

More importantly, **the rotating folds are still being annotated toward 24 snips
each**, so per-fold buzz counts and the training pool both move under every
result in `log.jsonl`. That changes what an experiment is worth:

- **Hyperparameter results do not survive a data change.** Don't bank them.
- **Large one-directional structural results probably do** — `trunk-ft-1e5`'s
  9/11 will most likely still be 9/11 next month. One confirmation on a
  different frame density (`large-set-confirmation`) is worth more than another
  medium-set variant.
- **Diagnostics and instrumentation survive absolutely.** They cost one cheap
  run, they can be recomputed for free whenever the data moves, and — this is
  the part that matters right now — they tell Luke *what to annotate next*,
  which is the live bottleneck. `prediction-provenance`, `trill-vs-buzz` and
  `eval-sampling-floor` are all of this kind.

A reasonable split for the next several loops: one exploit run (`large`), one
instrumentation run, and the rest on the structural ideas that need no
re-extraction (`subframe-head`, `lab-positives-ablation`).

---

## Open ideas

### subframe-head

**Hypothesis:** `yamnet_trunk` caches `layer12_pointwise_conv_relu` at shape
**(6, 4, 512)** — 6 time steps and 4 frequency bands inside each 0.96 s frame.
Layers 13-14 stride that to (3, 2, 1024) and then
`global_average_pooling2d` **averages all six remaining positions away**. For a
buzz that occupies 200 ms of a 960 ms frame, mean pooling divides its evidence
by ~5 and mixes in the background either side. Replacing the mean over the
*time* axis with a max (or log-sum-exp) should recover it.

**Why this is the best-value structural idea open:**
- **Zero re-extraction.** The `yamnet_trunk` cache already on disk holds the
  full spatial map; only `build_head()` changes.
- **It does not change the frame population**, so unlike `framehop-overlap` the
  result stays comparable to every entry in `log.jsonl`. It buys that
  experiment's intended benefit (better-centred short events) without the
  confound that made it uninterpretable.
- It is the mechanism `willard-regression` and `deployment-forensics` both
  point at from the other direction — willard has the highest fraction of
  short (<1 s, 68%) and isolated events of any fold, is the fold temporal
  context hurt most, and is a big mover for trunk-FT.
- **Frequency is also thrown away.** Buzz is narrowband; the 4-band axis is
  exactly the structure a linear probe on a GAP'd vector cannot reconstruct.

**What to do**, cheapest first, one per CV, on top of `trunk_ft_1e5`
(comparator: `trunk_ft_1e5`, not `cv-baseline`):

1. **Time-max, frequency-mean** — replace GAP with
   `max` over the time axis then `mean` over frequency. 1024-d output, so the
   head is unchanged and nothing else in the pipeline moves.
2. **Log-sum-exp over time** (smooth max, one temperature) if the hard max is
   unstable in training.
3. **Keep the frequency axis**: mean/max over time only, flatten to 2048-d.
   Doubles the head width — watch for the `stopping-rule-scale` confound and
   report `best_epoch` alongside.

**Caveats:** replacing GAP means layers 13-14's downstream statistics are no
longer what AudioSet trained them for; run it *with* the fine-tune (which can
re-adapt) rather than frozen. Max pooling over a ReLU map is also a noisier
gradient path than mean — if it collapses, check `best_epoch` before concluding
the pooling is wrong.

### night-negatives

**Status: not a LOOP experiment yet — it needs a data decision from Luke.**
Adding it means a new/edited annotation effort, which `LOOP.md` prohibits an
experiment from doing unilaterally. Surfacing it here because it is the largest
untapped resource in the project and it is already half-specified in the repo.

`01_annotate/2026-05-26 Automatic Annotations/README.md` states the case: the
shipped model has a consistent, tightly clustered spike of false positives near
midnight, and **"the false positives can even dwarf the diel trend in the focal
crop (e.g. 'Luke - Diel Drivers/2026-05-06')"** — which is one of the two
near-chance folds (`sens` 0.046 under `trunk_ft_1e5`). Every nighttime detection
is false by construction, so labels can be generated automatically at whatever
volume is wanted.

**Why it is worth more than another architecture run:**
- These are **in-domain hard negatives from the deployment's own recorder** —
  the exact thing `aug-*` kept trying to synthesize and failing at.
- **It is fold-safe for free.** Night audio from site X belongs to site X's
  fold, so nothing crosses a boundary; the CLAUDE.md augmentation rule doesn't
  bind.
- **It scales with the deployments, not with annotation labour**, so unlike the
  hand-annotated positives it does not get invalidated as the set grows.
- It targets a *named* failure on a *named* fold, rather than hoping for a
  diffuse gain.

**Open questions to settle before running it:**
- What label? The README flags the specificity problem — a nighttime FP could
  be trill, a plane, or a truck, and calling it all `ambient_background` teaches
  the probe a class it will meet in daylight too. `ins_trill` is the likeliest
  true identity for most of them (see `trill-vs-buzz`), and mislabelling trill
  as background would be actively harmful. Consider a distinct
  `auto_night_negative` label with its own translation row.
- How much? Automatic labels can trivially outnumber the ~5.7k hand-annotated
  buzz frames. A dose-response (0x / 1x / 4x the fold's existing negatives)
  is the experiment, not a single volume.
- Which model generates them? `model_general_v3` is the stale artifact; using
  the current best (`trunk_ft_1e5`) risks a mild self-confirmation loop, though
  a much weaker one than usual since night labels are true by time of day, not
  by the model's judgement.

### lab-positives-ablation

**Hypothesis:** the `train`-role pool is **605 InsectSound1000 idents** plus 70
from the original annotations — lab recordings of insects, a completely
different acoustic condition from a field recorder. They supply roughly
2,700 of the buzz frames a fold trains on, against ~4,700 field frames from the
other ten rotating folds. **Nobody has ever tested whether they help.**

**Why now, and why it is not just an ablation:** on a *frozen* probe, mismatched
positives cost little — the backbone is fixed, and extra positives mostly
regularise a linear head. On a *fine-tuned trunk*, gradient flows into layers
13-14, so the backbone itself is being pulled toward lab acoustics by a third of
its positive signal. **Unfreezing changes what the lab data is worth**, in a
direction nobody has measured. If it's a drag, the current best result is
carrying a handicap; if it's a crutch, that is important to know before the
field annotation grows and dilutes it.

**What to do:** set `role` to `exclude` for the InsectSound1000 source and run
one CV against `trunk_ft_1e5`. **This is free** — roles are training-time policy
only and `02_set` embeds every fold regardless, so no re-extraction. Run it on
the fine-tune, not the frozen probe; if it moves there, a matched frozen run
tells you whether the effect is the domain gap or just the frame count.

**Caveats:** dropping ~33% of positive frames also shrinks the training pool, so
a decline is confounded with plain data volume. Control for it by instead
*downweighting* the lab positives (or subsampling the field positives to match)
in a follow-up before concluding.

### prediction-provenance

*(Supersedes the old `event-level-metric` idea, which is the same file change
for a narrower purpose.)*

**Problem:** `folds/<fold>/predictions.csv` has exactly two columns,
`activation_ins_buzz` and `correct` — and `correct` is a misnomer, it is the
ground-truth buzz label, not whether the prediction was right
(`03_train/train.py:80`). There is no ident, no frame index, no timestamp and no
raw label. Consequences:

- You cannot ask **"what are the ~22 negative frames above threshold?"** — the
  frames that literally define the operating point of every number in the log.
- You cannot ask **"which kind of buzz do we miss?"** (`ins_buzz_low` vs
  `medium` vs `pollination`).
- You cannot ask **"did any frame overlapping this event fire?"**, so every
  comparison is locked to a per-frame rate and any experiment that changes frame
  density (hop, framelength, rescue policy) is uninterpretable — this is
  precisely what sank `framehop-overlap`.

**What to do:** carry `ident`, frame index (join key to the existing per-ident
`frametimes.csv`) and the raw label list into `predictions.csv`, and add an
event-level read alongside the frame-level one. Note this changes what a model
*directory contains*, not the metric, so it does not run afoul of the
`03_train/metrics.py` rule, and `03_train/resummarize.py` means the new columns
can be backfilled for old models without retraining.

**Payoff is retroactive and permanent.** Every future experiment gets a
failure-mode breakdown for free, and the two diagnostics below become one-line
queries instead of their own runs.

### trill-vs-buzz

**Hypothesis:** the false positives that set the threshold are mostly
`ins_trill`, and buzz-vs-trill is the actual discrimination problem.

**Evidence it deserves a look:** `ins_trill` is by a wide margin the largest
non-ambient class in the rotating folds — **13,422 frames against 5,691 buzz
frames** — and it is the nearest acoustic neighbour buzz has (both are sustained
narrowband insect stridulation/wingbeat). It is also very unevenly distributed:
JamesU has 5,126 trill frames against 2,385 buzz, `Diel Drivers/2026-05-06` has
**zero**, `2026-04-08` has 7. The two folds with essentially no trill are
`0.046` and `0.068`; the fold with the most is the best at `0.469`. That
correlation runs the *opposite* way to the naive "trill confuses the model"
story and is worth understanding either way — it may be that trill-rich folds
are simply insect-rich folds.

**What to do:**
1. Diagnostic first, and free once `prediction-provenance` lands: tabulate the
   raw labels of the negatives above each fold's threshold. If trill dominates,
   the whole research program narrows.
2. If it does: the `general` translation already keeps `ins_trill` as its own
   class, so the probe *has* the auxiliary supervision. The next lever is a
   pairwise margin — an explicit ranking term on buzz-vs-trill pairs only —
   rather than another representation change. Smoke-test any custom loss with
   `tools/smoke_model.py` first; `tail-loss` is the cautionary tale.

### eval-sampling-floor

**Measured, not hypothesised (2026-09-05, from existing `predictions.csv` — no
training; `tools/eval_sampling_sd.py <model dir>` reproduces it in seconds):**
bootstrapping frames within each fold and recomputing
sens@fpr0.005 gives a per-fold sampling SD of **0.02-0.03** on the eight normal
folds, and **0.092** on `2025-06-23/1_23` (whose threshold rests on a single
negative frame) and 0.025 on `2025-08-05/31` (two). Propagated to the
eleven-fold mean, that is **0.012 of headline SD from the evaluation sample
alone** (`trunk_ft_1e5`; 0.011 for `cv-baseline`).

`noise-floor-cv` measured total run-to-run noise at ~0.017 median per-fold and
0.014 on the headline. So **roughly half the noise floor is eval-set sampling,
not training stochasticity** — seed control would not fix it, and no cheaper
estimator can either (partial-AUC over an FPR band of 0.001-0.02 was tried in
the same session: it tracks sens@0.005 almost exactly, 0.258 vs 0.262 for
`trunk_ft_1e5` and 0.198 vs 0.197 for baseline, and cuts per-fold SD only
modestly — not worth a metric change).

**What this is actually good for: it is annotation guidance.** The only thing
that shrinks this floor is *more annotated non-buzz frames in the thin folds*.
`1_23` (315 val frames) and `2025-08-05/31` (942) cannot resolve a 0.5% FPR at
all — they contribute 2/11 of the headline on 1 and 2 negative frames
respectively. Annotating negatives in those two deployments buys more
measurement precision per hour than anything the loop can do in software, and
it compounds across every future experiment.

### large-set-confirmation

**The one exploit item.** `trunk-ft-1e5` is +0.046 at 9/11 folds and is the only
result in the log clearly outside the noise floor, but it has only ever been
measured at `framehop_prop 1`. `large` is the same annotations and folds at
`framehop_prop 0.2` (5x frame density), and the working assumption on this
project is that a change can look modest on `medium` and be strong on `large`.

`.local/worktrees/large-trunk-ft/` already exists with extraction started and
`notes.md` written; check whether the extraction finished before relaunching.

**Read it carefully:** `large`'s `folds_sx.csv` is **not comparable to any
`log.jsonl` entry** — changing frame density moves the negative population the
FPR threshold rests on (`framehop-overlap` is the worked example). The valid
comparison is a matched `trunk_frozen` control trained on `large`, i.e. the
delta on `large` against the delta on `medium`. That means **two** CV runs, and
a `large` CV is much longer than a `medium` one. Budget for it deliberately or
don't start.

### near-chance-deployments

**Hypothesis:** two deployments (`Luke - Diel Drivers/2026-05-06/1_95` and
`Luke - Various Opportunistic Recordings/2025-08-27/48`, joined by
`2026-04-08/1_150`) sit near zero for every model tried, including one with 430
buzz frames — so it is not a small-sample artifact. Since the endpoint averages
deployments equally, these three pin ~27% of the headline near zero and dilute
every real gain elsewhere.

**One story has now been ruled out (2026-09-05).** It looked like
leave-one-*concept*-out: `2026-05-06` is 74% `ins_buzz_low` and `2025-08-05/31`
is 33% `ins_buzz_pollination`, sublabels with thin support outside their own
fold. But scoring each fold by how well its buzz sublabels are covered elsewhere
in the training pool gives **r = 0.013 against per-fold sensitivity (n=11)**, and
the two *worst* folds are the cleanest cases of all: `2026-04-08` (0.068) and
`2025-08-27/48` (0.020) are **100% plain `ins_buzz_medium`, with >6,000 frames
of exactly that sublabel available elsewhere**. Whatever is wrong is acoustic or
site-level, not conceptual. `ins_buzz_pollination` remains a real acoustic
outlier (see `2025-08-05`, 0.180) but it is not what is sinking the worst folds.

**What to do:** the diagnostic LOOP has deferred three times now. Listen to the
frames. Are the buzzes audible? Is the annotation right? Is the recorder gain or
placement different? `2026-04-08` has a threshold of -1.401 — its buzz frames
score *below almost every negative in the fold*, which is a stronger statement
than "hard": something about that recording puts buzz on the wrong side of the
score distribution entirely. `03_train/surprisal.py` (on by default) is the
built tool for this and has not been pointed at these folds.

### willard-regression

**Hypothesis:** `exp/context-stack` gained in 8 of 11 deployments but lost 0.074
at `Lily Adam - One Hive/recorders/willard/2024-08-07/1_11`, a large fold, and
the regression scales with context width monotonically — baseline 0.177, k=1
0.118, k=2 0.066. `exp/deployment-forensics` attributes it to willard having the
highest fraction of short (<1 s: 68%) and isolated (>5 s gap: 68%) buzz events
of any fold: stacking dilutes a brief isolated buzz with silent neighbours.

**Note the connection to `subframe-head`:** if the dilution story is right, the
fix is not adaptive context *width* but better time resolution *inside* the
frame — which is what pooling the layer-12 map over time instead of averaging it
gives you, at no cost in frame density. Willard is the fold to check first when
that run lands. Under `trunk_ft_1e5` willard is already one of the big movers
(0.191 → 0.318), which is consistent with the fine-tune partly fixing this.

### normalization-zero-variance

**Status:** the `standardization-convergence` sweep this replaces was run as
`exp/std-convergence` and came back negative-to-inconclusive. What it left
behind is a concrete bug rather than an open question.

52 of `yamnet_combined`'s 1545 input dims have ~zero variance across the
training folds. A `Normalization` layer divides by `sqrt(var + 1e-7) ~ 3e-4` on
those dims, amplifying anything nonzero by ~3000x — a genuine NaN blowup at
lr=5e-4 without gradient clipping, and structural, so no learning rate avoids
it. **If anyone revisits standardization:** mask or floor the near-zero-variance
dims before adapting the layer, or drop them (they carry no signal by
definition). Until then any standardized-input result is numerically fragile and
its failures can't be attributed to the hypothesis.

### aves-intermediate-layer

**Hypothesis:** AVES embeddings from the last transformer layer are too
bird-specific to discriminate insect buzz; an intermediate layer (6-9 of 12)
carries more general acoustic features. `aves_lite` performed at chance, and
AVES embeddings are symmetric around zero for both classes where YAMNet's are
ReLU-sparse and linearly separable. Wav2vec2 transfer literature consistently
favours middle layers.

**What to do:** in `embedders/aves/embedder.py`, change `layer_outputs[-1]` to
`layer_outputs[N]` for N in {5, 7, 9}; re-extract; train. `n_embeddings` stays
768, so no other code changes.

**Caveats:** `LOOP.md` constrains the embedder to YAMNet, so this needs that
constraint lifted first, and it costs a re-extraction per layer. Given
`subframe-head` tests the same "the pooled output discards what we need" idea on
a backbone that already works, and for free, do that one first.

### lora-adapter

**Largely answered — kept for the record.** Differential-rate fine-tuning of
YAMNet layers 13-14 (`exp/trunk-ft`, `trunk-ft-1e5`) is **+0.046 vs a matched
frozen control**, inverting the pre-rework `yamnet-ft` verdict, so LoRA's
motivation (fine-tune with less overfit risk) is much weaker than it was. Depth
is settled on `medium` by `exp/unfreeze-more`: frozen 0.216 → 13-14 **0.262** →
12-14 0.229, with median `best_epoch` falling 35 → 25. 13-14 is the operating
point; deeper only overfits. A cleaner low-rank adapter could still beat plain
FT, but it is a transformer technique and inserting it into YAMNet's conv layers
is non-standard — it is now a low-priority idea, not a promising one.

### stopping-rule-scale

**Mostly closed.** The hypothesis was that `EarlyStopping`'s absolute
`min_delta=0.002` on `val_loss` is mis-scaled for wider inputs and denser epochs,
so every structural result was partly measuring a stopping rule. Two runs since
have largely settled it: `trunk-ft-stop-sweep` found `min_delta` 0.002 → 1e-4
moved a 3-fold mean by +0.015 while a *rerun of the identical config* moved it
+0.018 — a non-lever inside run noise. `restore-on-sens` then found that
restoring at the true `val_loss` argmin (rather than the last min_delta-clearing
epoch) is worth ~+0.008, i.e. real but small, and it is now the default.

**What is left:** the per-fold `best_epoch` column is still the cheapest
tell that a config is stopping for the wrong reason — `trunk-ft-1e5-aug`'s
collapse was diagnosed from `best_epoch 1`. Keep reporting it in every
`notes.md`; don't spend another CV on the stopping rule itself.

---

## Tried before the rework

**Read these as leads, not verdicts.** They were measured on a fixed
train/validate split against a retired hand-curated corpus, with a training set
of mixed provenance, at a time when single-run variance was wide enough
(~0.16-0.25 on the same config) that several entries were later invalidated as
dataset artifacts.

The clearest reason not to trust them: **`temporal-context` — concatenating
[prev, curr, next] frames — was logged as a clear negative (-2pp). The identical
change, rerun as `exp/context-stack` on the current set and metric, is the
largest gain yet (+0.050).** A verdict inverted. Assume any of the below could
do the same, and rerun rather than defer to it.

Full entries: `.local/archive/log_precv.jsonl`. Working trees and notes:
`.local/worktrees-fixed-test/`.

One has already been retested under CV: the fixed-test entry `combined-embedder`
was about validation scope, but the `yamnet_combined` *embedder* it left behind
was finally extracted and trained as `exp/yamnet-combined` (+0.016).

| Area | What was tried | Old verdict |
|---|---|---|
| Regularization | Dropout(0.2) + label smoothing 0.2 | The whole gain over an unregularized probe (3.6pp). Current default. |
| | Label smoothing 0.3 | Collapses. 0.2 was the peak of a monotone trend. |
| | L2(1e-4), alone or added | Indistinguishable from no regularization. |
| | BatchNorm on input embeddings | Clear negative (-5.8pp) — training-set running stats didn't transfer. |
| Head shape | Dense(128, relu) before output | Worse than a linear probe, replicated twice, no overfitting signature. |
| Input surgery | Bandpass 100-3000 Hz before YAMNet | Clear negative (-4.7pp); corrupts YAMNet's expected input. |
| | Zeroing mel bins above 3000 Hz | Catastrophic (-14.3pp). |
| | Handcrafted frequency features | Neutral twice; YAMNet already encodes it. |
| | White-noise samples as 'static' | Neutral; the false positives are structured, not broadband. |
| Backbone | Fine-tune YAMNet layers 13-14 at 1e-5 | Clear negative (-8.3pp); overfit, train 78% vs val 59%. |
| Class weighting | 2x buzz upweight over balanced | Negative-to-neutral; balanced weights already fine. |
| Loss | Focal loss, alpha 0.25 and 0.75 | Shifts the operating point, doesn't lift the curve. |
| Translation | Binary (all non-buzz collapsed) | Hurt; multi-class auxiliary supervision helped. Retested under CV — see `binary-translation-cv`, neutral on the endpoint. |
| Training procedure | min_delta=0.002 early stopping | 6.6x variance reduction, no mean change. Adopted as default. |
| | Forcing buzz out of validation | Negative; early stopping needs buzz in the monitor fold. |
| Temporal | [prev, curr, next] frame concatenation | Negative (-2pp) — **and now known to be wrong**, see above. |
