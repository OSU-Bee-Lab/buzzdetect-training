# Experiment Ideas

Candidate experiments, nothing else. Results go in `log.jsonl`, protocol in
`LOOP.md`, closed eras in `archive/` — each with a README digesting what that
era concluded. Check all three before proposing an experiment.

**State as of 2026-09-08.** `log.jsonl` is empty; the training data was revised
and the era before it is archived. The baseline is `cv_baseline` (frozen YAMNet
probe, `general`, per-class weighting in the loss, restore at the true val_loss
argmin). Every number quoted below was measured on the **previous** data over
**11** rotating folds; there are now **5**. Treat them as directions, not
targets, and re-establish anything you intend to build on.

---

## Closed: standardize the input blocks

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

## near-chance-deployments

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
  would help most; a buzz-vs-`mech_auto` margin is the other candidate.
- `1_95` was the explicit target of `harmonic-comb` (2026-09-08) and **did not
  move**: +0.002 in each of two runs, and the only fold byte-stable across two
  nondeterministic runs, i.e. its failure is structural rather than stochastic.
  An explicit f0 channel that cleanly separates a 90 Hz engine comb from a
  220 Hz wingbeat comb on synthetic tones does nothing on this fold's real
  audio. A buzz-vs-`mech_auto` margin is now the remaining candidate.
- **1_150 is a positives problem** — its negatives are unremarkable and its buzz
  frames are simply indistinguishable from its own background, despite being 88 s
  of plain `ins_buzz_medium` with 6503 s of support. Nothing structural explains it.

**What to do — listen to the frames, and only 1_150 now.** Deferred five times.
It is a narrow target: **88 seconds of audio.** Is the buzz audible? Is the
annotation right? Is recorder gain or placement different? Note `context-embedder`
pushed its threshold further down (-1.819 → -2.107), so a better representation
does not touch it.

**Caveat on the folds it is measured against:** seven `ins_buzz` annotations
spanning a whole 300 s file supply **63%** of mustard's and **52%** of Fit+Fast's
buzz seconds (none in the other three folds). Those two folds carry the top of the
headline, their `buzz_frames` counts overstate their independent sample size, and
part of what the metric rewards is detecting a continuous drone.

## night-negatives

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
wrong class.** If a margin term is worth trying, make it buzz-vs-`mech_auto`.
Smoke-test any custom loss with `tools/smoke_model.py` first; `tail-loss` is the
cautionary tale.

## eval-sampling-floor → annotation guidance

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

**The only thing that shrinks it is more annotated non-buzz frames in the thin
folds.** That buys more measurement precision per hour than anything the loop
can do in software, and it compounds across every future experiment. Re-run the
tool on `cv_baseline` to get the current per-fold ranking — the roster changed,
so the old thin-fold list is stale.

## willard-regression

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

## Cheap and open

- **Patience 20–25 instead of 50.** Every archived run stopped by early
  stopping, never at the epoch cap, so patience is a flat tax of exactly N
  epochs per fold — 35–94% of a trunk-FT fold's compute. Replaying
  `EarlyStopping` over the saved `val_loss_curve`s puts patience 25 at 58% of
  the compute while restoring a different epoch on 3/4 folds; but those
  differences are the same size as the ±15–50-epoch run-to-run jitter
  `trunk-ft-stop-sweep` measured at patience 50 for an *identical* config. Now
  that `val_sens_fpr0.005_curve` is persisted per fold, this is answerable
  offline from any run — replay it before changing the default.
- **restore-on-sens' other half, on a trunk fine-tune.** Restoring the true
  val_loss argmin landed in `cv_baseline`. Restoring the *sens@FPR argmax*
  instead was worth another ~+0.006 — inside the noise on a frozen probe, which
  is why it was left out, but the divergence it exploits is a
  backbone-fine-tuning effect (label-smoothing overconfidence). `trunk-ft-restore-sens`
  saw the two curves diverge on all 11 folds, 8 shipping a later epoch than the
  loss argmin. Retest there, not here.
- **Split framing from embedding in `--workers`.** `--snip-workers` already
  separates the I/O-bound snip sync, but `--workers` still covers both framing
  (CPU, no GPU) and embedding (VRAM-bound), so protecting 4 GB of VRAM with
  `--workers 1` needlessly serialises the framing too. Pure throughput; no
  effect on any metric.
- **Standardize the input blocks.** On the 8 folds with ≥3000 val frames,
  standardization added +0.014 on top of `yamnet-combined` (6/8 up) — the
  scale-mismatch hypothesis was real. Not adopted because 10/11 folds then ran
  the full 400-epoch cap against a median ~120, i.e. the LR and patience were
  tuned for the old input scale. **Known bug if revisited:** 52 of
  `yamnet_combined`'s 1545 dims have ~zero variance, and a `Normalization` layer
  divides by `sqrt(var + 1e-7)` ≈ 3e-4 on those, amplifying noise ~3000x into a
  genuine NaN blowup that no learning rate avoids. Mask, floor, or drop those
  dims before adapting the layer.

## Deliberately parked

- **Trunk fine-tuning.** `trunk-ft-1e5` (YAMNet layers 13-14 at lr 1e-5) was
  +0.046 at 9/11 folds, the largest clearly-outside-noise result in the archive,
  and depth is settled: frozen 0.216 → 14-only 0.234 → **13-14 0.262** → 12-14
  0.229, a true interior optimum. It is parked, not dismissed: it needs a
  `yamnet_trunk` re-extraction (the `medium` cache went with the pruned
  worktrees), it is the expensive config (~80 s/epoch vs ~1 s), and a standing
  injunction limits training to one layer. Revisit when that lifts. LoRA is a
  weak follow-up now that plain FT works — it is a transformer technique and
  inserting it into YAMNet's conv layers is non-standard.
- **`large`-set confirmation.** **Training on `large` is forbidden without Luke
  asking explicitly**, however ready it looks. It is a one-time final
  confirmation after the structural search on `medium` concludes, not another
  set to rotate through. When it happens: `large` is the same annotations and
  folds at 5x frame density, and its `folds_sx.csv` is **not comparable to any
  `medium` number** — changing frame density moves the negative population the
  FPR threshold rests on (`framehop-overlap` is the worked example). The valid
  test is a matched control on `large`, i.e. two CV runs, and a `large` CV is
  much longer. Budget deliberately or don't start.
- **AVES intermediate layers.** `embedders/aves/embedder.py` is 1.0 s / 16 kHz /
  768-d — **frame-matched to YAMNet's 0.96 s**, so unlike Perch its headline
  would join `cv_baseline` directly with no frame-density caveat.
  The idea is to take a middle transformer layer instead of the last
  (`layer_outputs[-1]` → `[N]` for N in {5, 7, 9}; `n_embeddings` stays 768),
  on the wav2vec2 transfer-learning result that middle layers carry more
  general acoustic features than the task-specialized final one. Needs a
  re-extraction per layer, and the YAMNet-only constraint lifted.

  **The "AVES performs at chance" result is not a verdict — check its
  provenance before citing it.** It is from 2026-06-04, two eras back: measured
  on `aves_lite` (a troubleshooting set LOOP.md says is not a place to draw
  conclusions), scored on the **retired** stage-4 endpoint (precision against an
  18% base rate on the fixed test corpus — the metric the CV rework replaced),
  and never logged. No entry, no notes file, no surviving model dir; the only
  trace was a hypothesis section in this file. Two data revisions have landed
  since. `temporal-context` → `context-stack` is the standing warning about
  exactly this kind of inherited negative.

  What *does* survive is the diagnostic under it, because it is a property of
  the embeddings rather than of the metric: AVES embeddings are symmetric
  around zero for both classes with roughly half YAMNet's dimension-wise
  buzz/non-buzz separation (0.13 vs 0.23), where YAMNet's are ReLU-sparse and
  non-negative and linearly separable. **Recompute that on current `medium`
  embeddings first** — it needs no training and no re-extraction, and it is a
  far cheaper filter on whether a layer sweep is worth one.

- **Timestamp join for provenance.** `frametimes.csv` isn't on disk for current
  idents — it was added to the extractor after they were last extracted and the
  fingerprint hasn't changed since. `frame_index` joins to it whenever a fold
  re-extracts for some other reason; don't force a re-extraction just for this.
