# Experiment Ideas

**A queue. Candidate experiments, nothing else.** Results go in `log.jsonl`,
protocol in `LOOP.md`, closed eras in `archive/`. When you run an idea, **delete
its section** — the verdict, the mechanism and the "don't rerun this" live in
that run's `log.jsonl` entry and its `notes.md` on `exp/<slug>`. This file grew
to 1047 lines, 46% of it closed material, before that rule was enforced.

**State as of 2026-09-11.** Baseline `cv_baseline` = 0.218 (early-stopped; a
fixed-budget re-anchor is owed). Best config is **`yavf_h1024`** — `yamnet_aves`
concat + a 1024-wide hidden layer at `--fixed-epochs 150` — at **0.307** over
two draws. `yamnet_context` (3072-d) is 0.258 shipped / 0.288 under
`--epoch-rule xfold` / 0.294 replayed at a plain fixed e250. Representation
levers that have paid: `yamnet_context`, `yamnet_aves`, a wide hidden head.
Scoring fixes that have paid: **removing early stopping** — `--fixed-epochs`,
worth +0.031 to +0.040 and measured twice.

**Read the epoch rule before you read any number here.** The audit of
2026-09-11 (`exp/pairwise-rank:notes/new-era-audit.md`) found the era's
distortion is **undertraining, not leakage**: `val_loss` early stopping carries
no measurable selection optimism (-0.002 mean over 17 runs) but stops `1_150`
at epoch 5-32 on every embedder tried, scoring it on a barely-trained probe.
So: **always run a matched control under the same epoch rule as your arm** —
that, not a log boundary, is what makes these numbers comparable, and it is
what the last three entries did. `--fixed-epochs` is preferred over
`--epoch-rule xfold` (no selection at all, and within 0.006 of it everywhere);
250 is a floor for the budget, because every 150-epoch run on disk is still
rising at its cap.

## Which era a number came from

An untagged claim is a proposal, not a measurement.

| tag | era | how to treat it |
|---|---|---|
| **E1** | `archive/2026-06_fixed-test` — 29 runs | Different metric, different data, two revisions ago. **Not a verdict.** Re-establish or don't cite. |
| **E2** | `archive/2026-08_cv-medium-v1` — 30 runs | Right metric, wrong data and roster. Directions survive; numbers don't. |
| **E3** | current `log.jsonl` — 2026-09-08 on | Directly comparable. The only numbers you can beat. |

`temporal-context` was a clear E1 negative and, rerun as `context-stack` in E2,
became the largest gain in the log. **E1/E2 negatives are leads, not answers.**

## Standing facts a proposer needs

- **MDE is ~0.027** at n=1 (`probe-grid`: baseline SD 0.0095 over n=3). Anything
  smaller is unreadable from one run. Prefer dose ladders to single comparisons.
- **Never pass `--monitor val_sens`.** The held-out fold *is* the early-stopping
  monitor, so it reports `max`-over-epochs of the statistic it is scored on.
  `monitor-leakage` marked three entries `artifact` over this. Use
  `--fixed-epochs`, or `--epoch-rule xfold`, or check offline with
  `tools/honest_epoch.py`. The inflation is **budget-dependent** — +0.021 when
  the argmax ranges over 400 epochs, mean +0.006 over the five 150-epoch runs
  on disk — so quote it with its budget rather than as a constant.
- **The stopping rule is worth more than most levers, and it is not leakage.**
  `val_loss` early stopping shows no measurable selection optimism (-0.002 over
  17 runs) but undertrains: `1_150` stops at epoch 5-32 under every embedder.
  Removing it measured **+0.031** (`yamnet_context`) and **+0.040**
  (`yamnet_aves` linear). Run `--fixed-epochs`, and compare only against a
  control under the same rule.
- **Don't quote a noise figure from this file** — run
  `tools/eval_sampling_sd.py <model dir>` (seconds, no training). Current
  per-fold bootstrap SD 0.010-0.037, headline 0.007-0.012. Note `1_150`'s
  run-to-run movement (0.055) is *training* stochasticity, not eval sampling.
- **The hard folds are the target.** `1_150` and `1_95` pin 40% of the headline
  near zero. A gain concentrated there is the result, not a caveat on it.
- **Read `best_epoch` alongside every delta.** If your treatment moved it,
  the delta is confounded with a stopping change — re-score both arms at a
  common epoch (`tools/honest_epoch.py`, no training) before believing it.
  `recorder-center` produced a textbook +0.082 hard-fold gain at `1_150` that
  was entirely its `best_epoch` going 26 → 213; under one rule on both arms it
  is -0.007. Magnitude and repeat spreads do not catch this. **The question is
  whether the treatment could have selected on the reported metric.**
- **Any input normalisation moves the stopping epoch.** Three for three:
  `input-standardization` (E2), `standardize-blocks` (E3, ~6x epochs),
  `recorder-center` (E3, 1.4-8.2x). Pair the next one with a fixed epoch budget
  or a cross-fold epoch rule *from the start*, or its number is confounded.

---

# Queue

## A. Rerun context width k=2 — its negative rests on a dead mechanism

*Evidence: **E2** negative (`context-width`), whose stated cause **failed to
reproduce in E3**.*

`context-width` took k=1 → k=2 (`[t-2..t+2]`, 5120-d) from +0.022 to **-0.021**,
6 of 11 folds down against k=1, and attributed it to a specific mechanism from
`deployment-forensics`: willard has the highest fraction of short (<1 s) and
isolated (>5 s gap) buzz events, so widening dilutes a brief buzz with silent
neighbours. The willard prediction held monotonically (-0.059 at k=1 → -0.111 at
k=2), which is why the verdict was believed.

**That mechanism is dead on the current data.** E3's `context-embedder` *gained*
+0.037 at willard, and frame sensitivity bucketed by event span runs the wrong
way for dilution: <1 s **+0.046**, 1-2 s +0.058, 2-5 s +0.039, >5 s **+0.026** —
short and isolated buzz gains *more* than long and clustered. The k=2 verdict's
load-bearing explanation no longer describes the data it was measured against,
and willard is thin (305 buzz frames), so the original regression may always
have been fold variance.

Context is the only lever that has ever paid, and it has been tested at exactly
one width on this data. Run k=2 against `context_embedder` (not `cv_baseline`),
under `--fixed-epochs` so a wide input's stopping failure doesn't confound
it — `1_150` reached its `val_loss` argmin at epoch 5 on 3072-d, and 5120-d will
be worse. Ladder k=1/2/3 if the first fold's ETA allows; a single k=2 run has
the ~0.027 MDE problem that made the E2 number ambiguous in the first place.

`embedders/yamnet_context_k2/` and `02_set/extract.py`'s `context_frames`
support exist on `refs/archive/context-width`. Extraction is ~11 min / ~1.6 GB
at k=2; write it straight into the shared tree under a new embedder name.

## B. Asymmetric context — feed the contrast, not the concat

*Evidence: **Untagged — a proposal.** Rests on E3's `context-embedder` (+0.061
honest) and on the decoupled-head arithmetic below.*

`yamnet_context` gives the probe `[e_{t-1}, e_t, e_{t+1}]` and one linear layer
over it. A linear readout of a concatenation can only take a fixed weighted sum
of the three frames — it **cannot** compute "this frame stands out from its own
neighbours", because the weights are fixed at training time and the contrast is
a per-frame quantity. Yet that contrast is close to the definition of a buzz: a
transient against a locally stationary background.

Feed the difference explicitly instead:
`[e_t, e_t - mean(e_{t±1})]` (2048-d), or the fuller
`[e_t, mean(e_{t±k}), e_t - mean(e_{t±k})]` (3072-d, same width as the incumbent
so the comparison is clean on parameter count).

Why this is not another head experiment: it adds **no information and no
capacity** — same audio, same dimensionality, same linear probe. It only changes
what is linearly available. That makes it the one input-side move that directly
attacks "a linear readout of frozen YAMNet has no room here", which is
`mech-margin`'s and `shared-trunk-head`'s shared conclusion.

Caveat worth pre-registering: YAMNet's code is 89.6% exact zeros and
non-negative, so a difference channel is signed and roughly zero-centred — a
different input regime for `Dropout(0.2)` with no weight decay. If the result is
negative, check the stopping epoch before calling it a representation verdict
(the `aves-probe` trap).

Cheapest form is a transform at extraction time under a new embedder name; it
can also be done as an input transform in `03_train` over the existing
`yamnet_context` cache, which costs **no extraction at all** — prefer that.

**Pair it with a cross-fold epoch rule from the start.** A difference channel is
genuinely signed, so it is a larger regime change than `recorder-center`'s
median shift was (that moved only 86 of 1024 dims and still shifted `best_epoch`
by up to 8x). Run the stopping fix and the transform together, or the number
will be confounded the way `recorder-center`'s was.

## C. yamnet_aves per-block standardization

*Evidence: **E3** — `yamnet-aves` (+0.025 honest, two draws, 4/5 folds up,
`clean`). Cache is on disk (1792-d, shared tree); needs no extraction.*

The AVES block is ~3.8x the YAMNet block in per-dim |mean| (0.306 vs 0.080) and
0% zeros vs 91.6%, so under one Adam LR the small block is likely
underweighted — the one place `standardize-blocks` has a real motive (the
blocks genuinely differ, unlike YAMNet+sigmoid). `--standardize` exists, off by
default. Pair with a fixed epoch budget or a cross-fold epoch rule from the
start (every input normalisation to date moved `best_epoch`).

## The residue nobody has asked about: 1_29 vs Fit+Fast trade consistently

*Evidence: **E3** — `shared-trunk-head`'s three-width ladder.*

Across h = 64 / 256 / 1024, `1_29` gains ~+0.03 at **all three** widths while
`Fit+Fast/53` loses at all three — both above the ~0.019 repeat movement for
folds with >1000 buzz frames, neither flipping sign across three runs. The mean
was null because they cancel. Both are rich folds, so this is not thin-fold
noise, and a systematic trade between the two richest deployments is a fact
about the data nobody has explained. `DEPLOYMENTS.md` has their crop, site and
date. Diagnostic, no training.

## A pairwise ranking loss with deterministic pairing

*Evidence: **E3** — the untried residue of `mech-margin` (closed negative) and
**E2**'s `tail-loss`.*

`mech-margin`'s absolute hinge failed for a reason that names its own
replacement: **the metric reads rank only.** Every fold is thresholded on its
own held-out audio, so driving a confuser population below a fixed logit buys
nothing — the `mech_auto` *share* of threshold-setting FPs actually **rose** with
the dose (26.5% → 39.3%) while absolute logits fell.

A loss that expresses what the metric scores must be **pairwise**: buzz frame
ranked above confuser frame. That is `tail-loss`'s shape, whose E2 failure was a
*monitor* artifact — batch-local ranking made the compiled loss invalid as
`val_loss` — not evidence against ranking losses. The fix is a pairing that is
deterministic per frame: pair against a **frozen reference model's** scores
rather than the live batch. Nobody has run that.

Weigh it against the plainer reading of `mech-margin`: a linear readout of
frozen YAMNet may have no room to separate these populations at all, in which
case this is a loss fix to a representation problem. Smoke-test any custom loss
with `tools/smoke_model.py` first — `tail-loss` is the cautionary tale.

## The `binary` control is confounded — read before running it

*Evidence: **E3** — read off `train_utils.build_weights` and `train.py`.
Arithmetic, not a measurement.*

`LOOP.md` flags the `general`/`binary` pair as worth rerunning early. It is, but
**a naive rerun measures two things that have nothing to do with the taxonomy**:

1. **`ins_buzz`'s positive weight moves ~5.4x.** `build_weights` puts the class
   *count* in every weight's denominator: **0.76 under `general` (15 classes) vs
   4.10 under `binary` (2)**.
2. **`val_loss` is the mean over neurons**, so the stopping signal's composition
   changes (~1/15 buzz → ~1/2), and the epoch restored is chosen on a different
   curve.

Run it with `ins_buzz`'s `pos_weight` pinned to `general`'s value and the epoch
rule held fixed, or the number is uninterpretable. If pinned-`binary` matches
`general`, the taxonomy's benefit is weights-and-stopping and can be had
directly.

**Free prediction that tests the framing.** The head is `Dropout → Dense(15)`
into `weighted_cross_entropy_with_logits` — 15 independent sigmoids, no softmax,
so `W[:, buzz]` sees gradient only from the `ins_buzz` term
([[decoupled-probe-head]]). The 15-class head's buzz neuron therefore *is* the
binary probe up to label mapping, and this control should come back **near
null**. A large effect either way means the decoupling reasoning is missing
something.

## night-negatives — needs a data decision from Luke

*Evidence: **Untagged proposal**, but the disk survey below is **E3**, measured
2026-09-10.*

**An experiment may not add or edit an annotation effort unilaterally.** The
shipped model has a tight false-positive spike near midnight that "can even
dwarf the diel trend in the focal crop (e.g. `Luke - Diel Drivers/2026-05-06`)"
— one of the two near-chance folds. Every nighttime detection is false by
construction, so labels generate at any volume. These are **in-domain hard
negatives from the deployment's own recorder** — the thing every `aug-*`
experiment tried to synthesize and failed at — they are fold-safe for free, and
they scale with deployments rather than annotation labour.

**What is actually on disk (2026-09-10):**

- `02_set/sets/medium/build.R:11-15` sources exactly two efforts, `Even Sample`
  and `2025-06-04 original annotations`. **Neither night effort is in the set.**
- `01_annotate/2025-06-07 night detections/` — 278 rows, combined, **hand-labeled
  with real identities**. No `folds.csv`, so it violates `MAKE.R`'s contract and
  cannot be sourced (warns, does not fail).
- `01_annotate/2026-05-26 Automatic Annotations/` — `combine.R` complete (1500
  night frames + 4 beehive bouts, `fold='auto'`, `role='train'`), but its
  `data/` inputs are not synced, so it produces nothing.

**The "what label?" question is already answered by the 2025 hand labels:**

| label | n |
|---|---|
| `animal_frog_tree` | 127 |
| `ins_trill` | 60 |
| `mech_hum_traffic` | 26 |
| `mech_auto_truck` | 23 |
| `ambient_rustle` | 13 |
| `mech_auto_car` | 12 |
| `ambient_background` | **2** |

`ambient_background` — which `2026-05-26`'s `combine.R` assigns to all 1500 — is
correct for **0.7%** of them. 46% are tree frog and 22% `ins_trill`, both of
which map to real classes the probe meets in daylight. So the options are a
distinct `auto_night_negative` class with its own translation row, or taking the
shipped model's per-class argmax rather than only `activation_ins_buzz`. Not
"call it all background".

**Corroboration for `1_95` specifically.** Rebasing `cv_baseline`'s FPs onto
Even Sample's snips: 22 of the 42 threshold-setting negatives fall in
`260506_1924_s95760` (22:00-22:05) and 11 in `_s88560` (20:00-20:05) — 79% in 2
of 24 snips, 32 of 42 `mech_auto`. Dropping just those snips' negatives takes
the fold 0.034 → 0.097. `tools/night-positives/README.txt` documents the *same*
event independently with `model_general_v3`, audio already extracted to
`tools/night-positives/260507_2200.mp3`. Two unrelated models, two eras apart,
failing in the same ten minutes.

**Blocking question:** is `data/raw/Luke - Various Opportunistic Recordings/`
syncable to this machine? The automatic path needs it. **How much** is a
dose-response (0x / 1x / 4x the fold's existing negatives), not one volume.

## near-chance-deployments — what is left after four failed interventions

*Evidence: **E3**, re-measured on the current 5 folds.*

`1_150` (0.021) and `1_95` (0.037) pin 40% of the headline. **They fail in
opposite ways and should not be treated as one item:**

| | 1_150 | 1_95 |
|---|---|---|
| negatives, p99.5 | 0.139 — normal | **0.391 — ~3x every other fold** |
| buzz frames, mean/median | **0.071 / 0.064 — ≈ its own background** | 0.102 / 0.068 |
| frames setting the threshold | diffuse ambient | **32 of 35 `mech_auto`** |
| fold threshold | -1.819 | **-0.401** — every other fold is -1.59 to -1.90 |

- **`1_95` is a false-positive problem.** Its buzz/non-buzz mean-logit gap is
  **0.513** against 1.578 at `1_29` — the populations barely separate at all.
  It has now defeated `harmonic-comb` (+0.002 twice, and the only fold
  byte-stable across two nondeterministic runs — structural, not stochastic),
  `mech-margin` (down at every dose), `shared-trunk-head` (flat at every width)
  and `context-embedder`. Nothing cheap and loss-shaped is left. Live leads:
  **night-negatives** and **idea C**.
- **`1_150` is a positives problem, and the annotations are sound.** Luke
  listened 2026-09-09: "fairly quiet and there's background noise, but they're
  there. Most of them are very quiet, but still legitimate targets." Not
  mislabelling, not a phantom — a genuine low-SNR detection problem. Do not
  reopen it as an annotation-quality question. `context-embedder` pushed its
  threshold *down* (-1.819 → -2.107) without helping, so a better representation
  alone does not touch it. **Dropping early stopping does** — `+0.172` under
  `--epoch-rule xfold` on `yamnet_context`, `+0.160` under `--fixed-epochs 150`
  on `yamnet_aves`. Its early stopping was broken, not its model: `val_loss`
  bottoms out at epoch 5-32 there under every embedder tried while its buzz
  curve climbs to e120-185. Any `1_150` result measured under early stopping is
  confounded by this and most of the era's are.

Ruled out (2026-09-05, re-confirmed 2026-09-08): it is **not**
leave-one-concept-out. r = 0.013 between sublabel coverage and per-fold
sensitivity; every fold is at 0% thin against the real training pool. A rerun
that appeared to overturn this had filtered to `role=='rotate'` — check roles
before reviving it.

**Caveat on the rich folds.** Seven `ins_buzz` annotations spanning one 300 s
file supply 63% of mustard's and 52% of Fit+Fast's buzz seconds. Their
`buzz_frames` counts overstate their independent sample size, and part of what
the metric rewards is detecting a continuous drone.

## Seed averaging inside a run

*Evidence: **E3** — `probe-grid`, baseline SD 0.0095 over n=3.*

Averaging 2-3 seeds per fold *within* one run shrinks the error on every future
experiment for ~2-3x the compute of one run — cheaper than repeating whole
experiments, and it compounds. It is why `probe-grid` needed 13 runs to read 8
levers.

**Era-boundary decision** (it changes what "a run" means and breaks
comparability). Raised 2026-09-10; **Luke declined for now** — still hunting for
effects large enough to clear the current floor. Re-raise at the next cutover.

Related standing fact: `cv_baseline` is a fair draw (0.218 vs a 3-run mean of
0.2177), so nothing in `log.jsonl` is biased by a lucky denominator.
`models/base_r2` and `base_r3` on `exp/probe-grid` are the repeats.

## subframe-head — options 2 and 3

*Evidence: **E2** — option 1's negative is an E2 negative: a lead, not a closed
door.*

`yamnet_trunk` caches `layer12_pointwise_conv_relu` at **(6, 4, 512)** — 6 time
steps, 4 frequency bands per 0.96 s frame — and GAP averages all of it away. A
buzz occupying 200 ms of a 960 ms frame has its evidence divided by ~5.
Frequency is discarded too, and buzz is narrowband.

Option 1 (time-max/freq-mean pooling) was -0.007, folds split 5/6 — inside the
noise. Untried: **(2)** log-sum-exp over time instead of a hard max, if the
max's gradient path was the problem; **(3)** keep the frequency axis — pool over
time only, flatten to 2048-d. Both need the `yamnet_trunk` cache, which no
longer exists for `medium`, and option 1's result means neither is a priority.

## Cheap and open

*Evidence: **E2** unless noted. None re-established on the current roster.*

- **[E2] Patience 20-25 instead of 50.** Every archived run stopped by early
  stopping, never at the epoch cap, so patience is a flat tax of exactly N
  epochs per fold. Replaying `EarlyStopping` over saved `val_loss_curve`s puts
  patience 25 at 58% of the compute while restoring a different epoch on 3/4
  folds — but those differences are the same size as run-to-run jitter at
  patience 50 for an identical config. **Answerable offline** from any run now
  that `val_sens_fpr0.005_curve` is persisted. Replay before changing anything.
- **[E2] restore-on-sens' other half, on a trunk fine-tune.** Restoring the
  sens@FPR argmax was worth ~+0.006 frozen — inside the noise — but the
  divergence it exploits is a backbone-fine-tuning effect.
  `trunk-ft-restore-sens` saw the curves diverge on all 11 folds, 8 shipping a
  later epoch. Retest there, not here. **Note this is not `--monitor val_sens`**
  — restoring is not selecting, but verify against the leakage rule first.
- **[no era — pure throughput] Split framing from embedding in `--workers`.**
  `--workers` covers both framing (CPU) and embedding (VRAM-bound), so
  `--workers 1` needlessly serialises the framing. No effect on any metric.

## Deliberately parked

- **[E2] Trunk fine-tuning.** `trunk-ft-1e5` (YAMNet layers 13-14 at lr 1e-5)
  was +0.046 at 9/11 folds, the largest clearly-outside-noise result in the
  archive, and depth is settled: frozen 0.216 → 14-only 0.234 → **13-14 0.262**
  → 12-14 0.229, a true interior optimum. Parked, not dismissed: it needs a
  `yamnet_trunk` re-extraction, it is ~80 s/epoch against ~1 s, and the standing
  injunction bans unfreezing backbone layers ([[injunction-no-long-training-runs]]).
  The injunction is about *runtime* and does not restrict head depth.

  **Do not let the frozen ranking choose the embedder for it.** YAMNet's frozen
  advantage is substantially an artifact of *what it is* — the penultimate layer
  of a supervised classifier whose 521 AudioSet classes include `Buzz`, `Bee,
  wasp, etc.`, `Insect`, `Mosquito`. Its embedding is pre-shaped so one linear
  layer separates exactly this concept. Unfreezing removes that advantage,
  because once the representation can be reshaped, "already linearly separable"
  stops being worth anything. Expect the AVES gap to narrow; do not rule out an
  inversion. Also: dropout / label smoothing / weight decay optima found on
  ~11.5k frozen parameters are **uninformative priors** for ~95M unfrozen ones
  (trunk FT needs lr 1e-5 against the probe's 2e-3, 200x), while the stopping
  rule gets *more* valuable. AVES base on a 4 GB GTX 1650 is likely infeasible
  fully unfrozen — Adam moments alone are ~1.1 GB at fp32
  ([[machine-gpu-constraints]]). Measure one epoch before budgeting.
- **[no era — standing policy] `large`-set confirmation.** **Forbidden without
  Luke asking explicitly.** One-time final confirmation after the structural
  search on `medium` concludes. `large` is the same annotations and folds at 5x
  frame density, and its `folds_sx.csv` is **not comparable to any `medium`
  number** — frame density moves the negative population the FPR threshold rests
  on. The valid test is a matched control on `large`, i.e. two long CV runs.
- **[E3] Frame length, isolated from embedder — the `perch-probe` follow-up.**
  `perch-probe` was +0.026 and inconclusive because Perch's 5.0 s frame moved
  the labelling rule at the same time as the embedder
  (`overlap_event_s = framelength_s * overlap_event_prop`; at 0.2 that is 1.0 s
  against YAMNet's 0.192 s, and the median buzz annotation is 1.00 s, so 31% of
  events can no longer label a grid frame). Per-fold dilution predicted the
  per-fold delta exactly. The experiment worth running is **YAMNet re-extracted
  at a 5 s effective frame** — one variable. Hold **`overlap_event_s`
  absolute**, not `overlap_event_prop`, or the control reproduces the confound.
  Not comparable to `cv_baseline`, so this is two CV runs.
- **[no era — infrastructure] Timestamp join for provenance.** `frametimes.csv`
  isn't on disk for current idents — added to the extractor after they were last
  extracted, and the fingerprint hasn't changed since. `frame_index` joins to it
  whenever a fold re-extracts for another reason; don't force one just for this.
