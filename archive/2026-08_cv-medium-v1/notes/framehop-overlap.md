# framehop-overlap

## Hypothesis

`02_set/sets/medium/config_extract.json` has `framehop_prop: 1` — frames from
the non-overlapping grid don't overlap each other. YAMNet's analysis window is
0.96s. `exp/deployment-forensics` found that the fold most resistant to every
model tried (willard, `Lily Adam - One Hive/recorders/willard/2024-08-07/1_11`)
has 68% of its buzz events shorter than 1s, and short events are the ones a
non-overlapping grid aligns worst with: a 0.6s buzz can land split across two
frames, appearing as two weakly-buzzy frames rather than one clearly-buzzy one.

Halving the frame hop (`framehop_prop: 0.5`, 50% overlap) roughly doubles frame
density and gives every short event a better chance at a well-centred frame,
which should improve sensitivity on short-event deployments, willard especially.

**Pre-emption check (done before spending the run):** `02_set/extract.py`
already has a "rescue frame" mechanism (lines ~704-753) — for any labelled
event that no grid-aligned frame captures with sufficient overlap
(`overlap_event_prop`), it cuts one extra frame centred exactly on the event's
midpoint. This means no annotated event is ever *lost* under `framehop_prop=1`,
which partially pre-empts the hypothesis: total loss of a short event isn't
possible today regardless of hop.

What rescue does NOT provide, and what a denser hop could still add:
- Only one rescue frame per missed event — a dense overlapping grid instead
  gives short events *multiple* representations in training, at multiple
  offsets, not just one perfectly-centred synthetic frame.
- Rescue only fires when overlap drops *below* `overlap_event_prop` (0.2) in
  every grid frame. An event that clears that bar in one frame but sits
  off-centre (e.g. right at a frame's edge) is trained on that low-quality
  alignment with no rescue — a denser grid would give it a better-aligned frame
  too, not just a guaranteed-present one.
- Rescue is training-data-only; it doesn't change inference-time frame
  density. A shipped model still scores the deployment audio on whatever hop
  it's run at. (Not relevant to this CV comparison, which trains and evaluates
  under the same fixed hop, but worth flagging for later readers.)

So the hypothesis is not fully pre-empted, but its effect size is likely
smaller than "every short event either aligns well or doesn't reach the
model at all" — rescue already guarantees the second failure mode can't
happen.

## Changes

- Worktree's `02_set/sets/medium/config_extract.json`: `framehop_prop` 1 → 0.5.
- No code changes. Broke the `audio/` and `embeddings/` symlinks in this
  worktree first (real local copies via `cp -rL`) so re-extraction cannot
  touch main's shared cache; verified with `find -type l` (none found) and
  `du -sh` before/after on both the worktree copy and main's set.
- Deleted only the framehop-dependent products in the worktree's local copy
  before re-extracting: `audio/sr16000_fl0.96/raw` (framed-audio cache) and
  `embeddings/yamnet` (embeddings under the old hop). Left `audio/snips`
  alone — snips are annotation-derived, not framehop-dependent — and left
  `embeddings/yamnet_context` / `embeddings/yamnet_combined` alone, since
  this experiment only touches the plain `yamnet` embedder.
- Confirmed main's `02_set/sets/medium/audio` and
  `02_set/sets/medium/embeddings/yamnet` were untouched throughout (checked
  `du -sh` on main's copies after deleting the worktree's).

## Results

`sensitivity_mean @ fpr0.005`: baseline (`models/yamnet_medium_general`) 0.206 →
`framehop_half` **0.054** (Δ −0.152). 10/11 folds down, 1 essentially flat/up.

| fold | baseline sens@fpr0.005 | this exp | delta | frames_val (base→exp) |
|---|---|---|---|---|
| Lily - Fit+Fast/2023_R3_Marysville/53 | 0.386 | 0.012 | −0.374 | 4710 → 9406 |
| Luke - Various Opportunistic/2025-06-23/1_23 | 0.326 | 0.000 | −0.326 | 314 → 627 |
| Lily Adam - One Hive/wooster/2024-07-26/1_143 | 0.260 | 0.019 | −0.241 | 4707 → 9398 |
| **Lily Adam - One Hive/willard/2024-08-07/1_11** | **0.177** | **0.003** | **−0.174** | 4713 → 9412 |
| Luke - Various Opportunistic/2025-08-12/1_114 | 0.163 | 0.000 | −0.163 | 3768 → 7524 |
| Luke - Various Opportunistic/2025-08-05/31 | 0.157 | 0.007 | −0.150 | 942 → 1881 |
| Luke - Various Opportunistic/2025-07-03/1_37 | 0.279 | 0.139 | −0.140 | 4712 → 9408 |
| JamesU - MustardBumbler/1_29 | 0.448 | 0.349 | −0.099 | 6908 → 13794 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.027 | 0.004 | −0.023 | 6606 → 13185 |
| Luke - Various Opportunistic/2025-08-27/48 | 0.013 | 0.010 | −0.003 | 1570 → 3136 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.028 | 0.050 | +0.022 | 4891 → 9727 |

**Willard, the fold this hypothesis specifically targeted, collapsed
(0.177 → 0.003) instead of improving.** But it did not stand out from the
cohort — it fell by roughly the median amount of decline seen across all 11
folds. This is not a clean falsification of the hypothesis, because the whole
run has two comparability problems that would produce a large, broad-based
drop on their own, independent of whether overlapping hop helps or hurts
detection:

**1. The frame population the metric is computed over changed, on both sides
of the ratio, uniformly.** `buzz_frames_total` 5641 → 11225 and
`neg_frames_fold_median` 22 → 43 — every fold's positive and negative pool
grew by almost exactly 2.0x (checked per-fold: ratios cluster at 1.988–1.998,
no outliers). Sensitivity-at-fixed-FPR is a per-frame rate; halving the hop
doesn't just duplicate well-centred frames, it also adds new frames at each
event's edges that only partially overlap it — especially for the short (<1s)
events this hypothesis targets, which don't fill a 0.96s window even at the
best offset. Those edge frames dilute the per-frame hit rate even if the
model's response to the best-aligned frame of an event is unchanged. I could
not construct a hop-invariant, event-level check to separate this from a real
effect: `predictions.csv` stores only `activation_ins_buzz` and `correct` per
frame (`03_train/train.py:148-151`), with no event id, ident, or timestamp, so
there's no way to group a fold's frames back to the annotated events they
came from without re-deriving the frame table from the dataset build — out of
scope here.

**2. Early stopping cut these runs off much sooner, and this looks systemic
rather than specific to hop.** `best_epoch` collapsed: median 13 here vs
baseline's median 56 (full per-fold: 13,27,35,23,10,8,37,13,6,4,12 vs
17,19,22,41,49,56,63,67,73,129,166). The shipped model's `epochs_fixed` did
the same (13 vs 56, same `size_batch` 65568 both sides). Frames/epoch roughly
doubled with the hop (frames_train ~132–145k vs baseline ~67–73k), so 13
epochs here is only ~26 "baseline-equivalent" epochs of exposure, and total
frame-exposure at stop (`epochs_fixed × frames_train`) is ~1.9M here vs ~4.1M
for baseline — about 46%. `EarlyStopping`'s `min_delta=0.002` (`03_train/train.py:257`)
is a fixed absolute threshold on `val_loss`, not scaled to steps-per-epoch, so
doubling frames/epoch plausibly makes it trigger "converged" well before
these models actually are. This matches a pattern flagged across other
experiments run today under different mechanisms (context-width k1/k2,
attention pooling all showed the same best_epoch collapse), which points to
a pipeline-wide stopping-rule miscalibration rather than something particular
to `framehop_prop`.

However, per-fold epoch-ratio (exp best_epoch / baseline best_epoch) only
weakly correlates with per-fold sensitivity delta (Pearson r = 0.155) — the
fold cut shortest relative to baseline (08-12, ratio 0.031) did drop a lot
(−0.163), but so did Lily-Fit (ratio 0.403, −0.374, one of the biggest drops),
and the fold with the least relative cut (08-27, ratio 0.706) had the
smallest drop (−0.003) alongside the one fold that rose (Diel 04-08, ratio
0.526, +0.022). The decline is broad and fairly uniform across folds rather
than concentrated in whichever folds got stopped shortest — more consistent
with something depressing every fold's number somewhat uniformly (the
metric-denominator dilution and/or a training deficit that isn't purely a
per-fold epoch-count story) than with a clean, fold-differentiated capability
signal.

Threshold magnitudes are not wildly different between runs
(`threshold_mean` −0.663 baseline vs −0.829 here), so this isn't a broken
threshold search — the drop is in sensitivity at a comparably-located
threshold, not an artifact of the FPR search failing.

## Conclusion

The measured effect (**−0.152**, 10/11 folds down) is large and consistent in
direction, but I don't trust it as a clean read on whether a denser,
overlapping frame hop helps or hurts short-event detection. Two separate,
independently-plausible artifacts point the same direction as the observed
drop: (1) sensitivity's per-frame denominator changed (~2x more buzz and
negative frames per fold, with the added frames skewed toward
partially-overlapping, harder-to-classify edges of events — exactly the short
events this hypothesis is about), and (2) training pool shape appears to have
tripped a stopping-rule miscalibration (`min_delta=0.002` untuned for
frames/epoch), cutting these models off at roughly half baseline's total
training exposure — a pattern also seen in unrelated experiments today, so
likely a pipeline issue rather than something the hop caused specifically.

Willard did not confirm the hypothesis (it collapsed with the rest of the
cohort rather than improving), but given the above, that's weak evidence
either way — the whole cohort moved in a direction two known artifacts would
predict regardless of the underlying question. I could not build a
hop-invariant, event-level metric from what's on disk (`predictions.csv` has
no event/timestamp identity) to separate a real effect from these two
confounds, so this experiment should be read as **inconclusive on the
hypothesis**, not as evidence that overlapping hop hurts.

If this is worth re-testing: fix `EarlyStopping` to scale with steps rather
than epochs (or fix total frame-exposure instead of epoch count) before
re-running, and ideally instrument predictions with enough identity (ident +
frame start time, or an event id) to let a future run score hop changes by an
event-level hit rate instead of a raw per-frame rate.

**Trust: artifact.** Both known confounds cut in the same direction as the
observed drop and neither is separable from a genuine effect with what's on
disk; the size, and possibly the sign, of −0.152 should not be taken at face
value or used to argue overlapping hop is harmful.
