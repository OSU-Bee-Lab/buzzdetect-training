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

## Revalidate the two clean embedder wins

**The highest-value next runs, and the reason the baseline doesn't already
include them.** Both are `clean`-trust, above the old noise floor, and ship
properly — but both change the embedder and cost a re-extraction, and neither
was ever tested in combination with the other. Folding them into the baseline
would have bundled three changes into one undecomposable reference point.

| | old delta | folds up | cost |
|---|---|---|---|
| `context-embedder` — [prev, curr, next] concatenated **inside** the embedder, 3072-d | +0.022 | 7/11 | re-extraction |
| `yamnet-combined` — YAMNet embeddings + the 521 AudioSet sigmoid scores, 1545-d | +0.016 | 7/11 | re-extraction |

`context-embedder` is the honest version of `context-stack` (+0.050, logged
`artifact`): stacking at train time let the eval see same-label neighbours that
continuous audio never gives you, and the gap between 0.050 and 0.022 is roughly
that artifact. `model.py` sizes its input from `n_embeddings`, so inference
follows automatically.

`embedders/yamnet_combined/` is in main and needs only a re-extraction.
`yamnet_context` is **not** — that branch was deleted and survives only as
`refs/archive/context-embedder`; recover it with

```bash
git show refs/archive/context-embedder:embedders/yamnet_context/embedder.py
```

Run them one at a time against `cv_baseline`, `context-embedder` first. If both
hold up, *then* ask whether they compose — `yamnet-combined`'s own note warns
that its sigmoid block is ~10x smaller in scale than the embedding block and
nothing normalizes the two (see **standardization** below).

## near-chance-deployments

Two of the five rotating folds sit near zero for every model tried:
`Diel Drivers/2026-04-08/1_150` (0.021) and `2026-05-06/1_95` (0.037) on
`cv_baseline`'s predecessor. Since the endpoint averages deployments equally,
**they pin 40% of the headline near zero** — a much bigger share than the 27%
they were at 11 folds, which makes this the single largest lever on the metric.

One story is ruled out (2026-09-05): it is not leave-one-*concept*-out.
Scoring each fold by how well its buzz sublabels are covered elsewhere gives
r = 0.013 against per-fold sensitivity, and the worst folds are 100% plain
`ins_buzz_medium` with >6,000 frames of exactly that elsewhere. Whatever is
wrong is acoustic or site-level.

**What to do — listen to the frames.** This diagnostic has now deferred four
times. `2026-04-08` had a threshold of -1.401: its buzz frames score *below
almost every negative in the fold*, which is stronger than "hard" and says
something about the recording puts buzz on the wrong side of the distribution
entirely. `03_train/surprisal.py` is on by default and has never been pointed
at these folds. Is the buzz audible? Is the annotation right? Is recorder gain
or placement different?

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

## trill-vs-buzz

**Hypothesis:** the false positives that set the threshold are mostly
`ins_trill`, and buzz-vs-trill is the real discrimination problem.

`ins_trill` was the largest non-ambient class in the rotating folds by a wide
margin. But it is very unevenly distributed, and the correlation runs the
*opposite* way to the naive story: the folds with essentially no trill were the
worst (0.046, 0.068) and the fold with the most was the best (0.469). Worth
understanding either way — it may just be that trill-rich folds are insect-rich
folds. Recount on the current 5 folds before theorising; the roster changed.

**Step 1 is now a query, not a run.** `prediction-provenance` landed 2026-09-07,
so `predictions.csv` carries `labels_raw`: tabulate the raw labels of negatives
above each fold's threshold. `cv_baseline` has the column. If trill dominates,
the research program narrows sharply.

**If it does:** `general` already keeps `ins_trill` as its own class, so the
probe has the auxiliary supervision; the next lever is a pairwise margin — an
explicit ranking term on buzz-vs-trill pairs only — not another representation
change. Smoke-test any custom loss with `tools/smoke_model.py` first;
`tail-loss` is the cautionary tale.

## eval-sampling-floor → annotation guidance

**Measured, not hypothesised** (`tools/eval_sampling_sd.py <model dir>`,
seconds, no training). Bootstrapping frames within each fold gave a per-fold
sampling SD of 0.02–0.03 on normal folds and up to 0.092 on the thinnest,
propagating to ~0.012 of headline SD from the evaluation sample alone.
`noise-floor-cv` put total run-to-run noise at ~0.014 headline — so **roughly
half the noise floor is eval-set sampling, not training stochasticity.** Seed
control would not fix it, and a partial-AUC estimator was tried and tracks
sens@0.005 almost exactly without cutting SD much.

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

Check it when **revalidate-clean-embedder-wins** runs `context-embedder`, which
lost on willard too (-0.059) — so the regression is a property of the
deployment, not the cached-eval bug. If the dilution story is right, the fix is
better time resolution *inside* the frame rather than adaptive context width —
see `subframe-head`.

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
- **AVES intermediate layers.** `aves_lite` performed at chance and AVES
  embeddings are symmetric around zero for both classes where YAMNet's are
  ReLU-sparse and linearly separable; wav2vec2 transfer literature favours middle
  layers. `embedders/aves/embedder.py`, `layer_outputs[-1]` → `[N]` for N in
  {5, 7, 9}; `n_embeddings` stays 768. Needs the YAMNet-only constraint lifted
  and a re-extraction per layer.
- **Timestamp join for provenance.** `frametimes.csv` isn't on disk for current
  idents — it was added to the extractor after they were last extracted and the
  fingerprint hasn't changed since. `frame_index` joins to it whenever a fold
  re-extracts for some other reason; don't force a re-extraction just for this.
