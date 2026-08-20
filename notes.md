# deployment-forensics

Diagnostic only — no training, no CV. Covers `near-chance-deployments` and
`willard-regression` from `IDEAS.md`.

## Targets

- `diel_1_95` = `Luke - Diel Drivers/2026-05-06/1_95` (near-chance)
- `opportunistic_48` = `Luke - Various Opportunistic Recordings/2025-08-27/48` (near-chance)
- `willard_1_11` = `Lily Adam - One Hive/recorders/willard/2024-08-07/1_11` (context-regression only, fine at baseline)

Baseline fold-held-out numbers (`models/yamnet_medium_general/folds_sx_byfold.csv`,
main, read-only):

| fold | sens@fpr0.005 | own threshold | neg_frames@fpr | val buzz frames |
|---|---|---|---|---|
| diel_1_95 | 0.027 | **+0.306** | 30 | 430 |
| opportunistic_48 | 0.013 | -0.996 | 6 | 299 |
| willard_1_11 | 0.177 | -1.257 | 22 | 300 |
| (typical folds, e.g. mustardbumbler) | 0.27-0.45 | -0.4 to -1.3 | 17-23 | 189-2141 |

`diel_1_95`'s own threshold (+0.306) is wildly out of line with every other
fold (next-highest is -0.445) — its non-buzz frames include a lot that score
high. `willard_1_11` sits mid-pack at baseline; its problem is specific to the
context experiments, not baseline generalization.

## Method

All scripts in `.local/forensics/` of this worktree, read-only against main's
`02_set/sets/medium/` (symlinked) and main's `models/` (absolute path, never
written to).

1. `annot_stats.py` — event counts, durations, sub-label composition per
   rotate fold, from `02_set/sets/medium/annotations.csv`.
2. `score_dist.py` — activation quantiles for buzz vs. non-buzz frames from
   each fold's held-out `predictions.csv` (`correct` column is ground truth,
   not a threshold call — see `03_train/train.py:148`).
3. `willard_gaps.py` — inter-event gap and short-event fraction per fold.
4. `embedding_space.py` — buzz-frame embedding centroid distance. **Gotcha
   found along the way:** `extract.py` writes one `.pickle` file per unique
   label-combination per ident, but each file holds *multiple* sequentially
   `pickle.dump()`-ed frames, not one array — `pickle.load()` once only reads
   the first frame. Fixed by reading to `EOFError`; confirmed correct because
   the recovered buzz-frame counts (430 / 299 / 300) exactly match
   `predictions.csv`'s positive counts for these folds.
5. `export_clips.py` — pulls specific buzz events out of `02_set/sets/medium/audio/snips/`,
   computes RMS/peak/spectral-centroid on the annotated event window, and
   scores each clip with the **shipped** model (`models/yamnet_medium_general/model.keras`).
   Caveat printed in the index: the shipped model trained on *all* folds
   including these three (`config_model.json:folds_train`), so its scores on
   these clips are optimistic (leakage) — read them as "does the model even
   respond to this sound at all under best-case conditions," not as a
   generalization estimate.

## Findings

### diel_1_95 — quiet buzzes + an unusually hard negative population

- Sub-label composition: 61% `ins_buzz_low`, i.e. the annotator already
  flagged most of these as quiet at annotation time. Not a mislabeling
  problem — the label matches what's audible.
- Acoustic: peak amplitude on 8 sampled events was 0.03–0.16 (full scale
  1.0), RMS -37 to -44 dBFS. Compare `mustardbumbler` (a well-scoring fold):
  peak 0.10–0.83, RMS -32 to -39 dBFS — roughly 8-10 dB louder. Diel's
  buzzes sit close to this recording's own noise floor.
- Embedding space: mean cosine similarity to the pooled buzz centroid (built
  from the other 8 rotate folds, 4612 frames) is 0.666 vs. the reference
  folds' own 0.734 — displaced, but not the most extreme of the three targets.
- Score behavior: this is the deployment where hypothesis 4 (hard negatives,
  not weak positives) bites hardest. Non-buzz frames in this fold reach up to
  activation +1.26 at the 99.9th percentile (vs. -0.83 to -1.13 in the other
  two targets), which is why its own fpr-0.005 threshold has to sit at +0.306
  instead of the usual -0.4 to -1.3. `01_annotate/Even Sample/README.md`
  explains why: this is *literally* the file (`260506_1924.mp3`, ~10 PM) that
  motivated Even Sample in the first place, logged there as "peak detection
  rate ~5%" at night with no nighttime coverage in the old training set. Its
  buzz events (checked: they land at roughly dawn/morning hours, not night —
  timing itself isn't the anomaly) are real but quiet, and they compete
  against a deployment whose *other* audio is unusually confusable — both
  effects tax sensitivity at a fixed FPR simultaneously.
- Timing check: buzz event start times fall ~45,000-100,000s into the
  recording, i.e. roughly 7 AM-9 AM the morning after an 19:24 start —
  daytime pollinator activity, not the nighttime false-positive window. So
  the *buzzes* aren't a nighttime artifact; the *negatives* plausibly are
  (night audio elsewhere in the same file inflates the high end of the
  non-buzz score distribution and drags the threshold up).

### opportunistic_48 — a different kind of buzz, thinly represented

- Only 19 annotated buzz events total, **100% one sub-label** (`ins_buzz_medium`),
  vs. every other fold in the roster having 2-4 sub-labels. Structurally
  different from the rest of the training data, not because it's mislabeled,
  but because there's only one example type here.
- Event shape: median duration 7.8s, up to 93.4s — sustained, continuous
  events, not the short discrete pulses (median 0.8-1.7s) everywhere else.
- Acoustic: not quiet — RMS -26 to -30 dBFS on 8 sampled events, louder than
  diel and comparable to willard/wooster. Spectral centroid 1620-1900 Hz,
  clearly lower than mustardbumbler's 1900-7400 Hz on the same measure —
  consistent with a low, sustained drone rather than a buzzy/harmonic
  wingbeat pulse.
- Embedding space: **most out-of-distribution of the three** — mean cosine
  to the pooled buzz centroid is 0.525 (reference 0.734), and 60% of this
  fold's buzz frames fall below the 10th percentile of the reference
  distribution. Score behavior confirms it's a weak-positive problem, not a
  hard-negative one: this fold's own negative population is unremarkable
  (p99 -1.08, p99.9 -0.83, right in line with willard/wooster) and its
  threshold sits at the ordinary -0.996 — the ceiling here is that the
  positives themselves don't look like the pooled "buzz" region.
- Read together: this looks like a genuinely different sound (sustained
  insect drone/chorus vs. discrete wingbeat) that the training set barely
  represents (19 events, one site, one sub-label), so YAMNet's embedding
  space — which the probe is linear over — doesn't place it near where the
  model has learned to expect buzz.

### willard_1_11 — fine at baseline; short, isolated events are what
temporal context smooths away

- Baseline generalizes normally (0.177 sens@fpr0.005, mid-pack — not a
  near-chance deployment).
- Event-shape analysis across all 11 rotate folds (`willard_gaps.py`):
  willard has the **highest fraction of buzz events under 1 second** (68%)
  and the **highest fraction of buzz events separated from the prior event by
  more than 5 seconds** (68%) of any rotate fold — both #1 in the roster,
  with `wooster` (also a "One Hive" bee-recorder deployment) a close #2 on
  both metrics. Short discrete pollinator-visit buzzes surrounded by long
  silence, unlike the multi-minute continuous events in e.g. mustardbumbler.
- This is exactly the mechanism `context-stack`/`context-embedder` would
  hurt: stacking `[t-1, t, t+1]` mixes a genuine short buzz frame with two
  silent/ambient neighbor frames far more often here than in folds with
  longer or denser events. Two independent context experiments regressed
  willard in the same direction (-0.074 cached-eval, -0.059 honest
  re-extraction with real neighboring audio) — the honest version ruling out
  the same-label-neighbor eval bug as the explanation, leaving event
  isolation as the standing hypothesis.
- Not settled by listening (see below) — the annotation-level evidence
  (event duration/gap distributions) is what's driving this conclusion, and
  it's strong enough on its own that I didn't chase per-frame disagreement
  further. I did check: `predictions.csv` between the baseline and
  `context-embedder` models for willard have different row counts (4714 vs.
  4731) — the context embedder's windowing drops/keeps a different set of
  boundary frames, so a naive positional join to find "most disagreed" frames
  is unsafe without adding frame timestamps to `predictions.csv` first. Not
  attempted here; flagged as a tooling gap below.

## Hypotheses: ruled in / ruled out

| # | Hypothesis | diel_1_95 | opportunistic_48 | willard_1_11 |
|---|---|---|---|---|
| 1 | Annotation quality/composition | Ruled out as *mislabeling* (buzzes plausible, timed right); ruled in as *composition* (mostly the "low" sub-label) | **Ruled in** — 100% one sub-label, thin support (19 events) | Ruled out (label mix is the most diverse of any fold: 52/27/21 split across high/medium/low) |
| 2 | Acoustic/recording character | **Ruled in** — 8-10 dB quieter than a well-scoring fold | Ruled out — comparable loudness to mid-pack folds; centroid is lower/duller, not louder/quieter | Not applicable (baseline isn't broken) |
| 3 | Embedding-space position (OOD) | Ruled in, moderate (cos 0.666 vs. 0.734 reference) | **Ruled in, strongest of the three** (cos 0.525, 60% below reference p10) | Ruled in but non-fatal (cos 0.584) — being OOD doesn't break the baseline probe here |
| 4 | Score behavior (hard negatives vs. weak positives) | **Both** — hard negatives (own threshold +0.306, unique in the roster) *and* weak positives | Weak positives only — this fold's negative population is unremarkable | Not applicable at baseline |
| 5 | Willard-specific: short/isolated events smoothed by context | — | — | **Ruled in** — #1 in the roster on both short-duration and long-gap fraction |

## Fixable by a training change, or an upstream data problem?

- **diel_1_95**: partially fixable, partially not. The hard-negative half
  (this file's nighttime audio inflating the non-buzz score ceiling) is a
  training/threshold problem in principle — but per README, threshold is
  set per-deployment at inference time in the field, so a global fix isn't
  really available; what would help is more nighttime-labeled negatives
  training-wide so the model learns to reject whatever this file's confusable
  sounds are, rather than this one fold's threshold absorbing all the cost.
  The weak-positive half (buzzes genuinely near the noise floor) is a
  physical limit of the recording, not something a training change reaches.
- **opportunistic_48**: this reads as an annotation/coverage problem more
  than a modeling one — one deployment, one sub-label, 19 events of a event
  *type* (sustained drone) that's rare elsewhere in the set. More annotated
  examples of this specific acoustic character, ideally from more than one
  site, would let the probe learn to place it near "buzz" instead of relying
  on the pooled distribution's coincidental overlap with what YAMNet does
  with sustained low-frequency sound. Not something a loss/architecture
  change fixes on its own.
- **willard_1_11**: fixable, since it's a training-structure interaction, not
  a data-quality issue. The context experiments should not be abandoned —
  the aggregate gain is real and reproduced across two implementations — but
  before the next context experiment, consider event-aware context (e.g.
  only stack neighbors that share ground truth with the center frame, don't
  stack across an isolated/short event) or report a per-fold breakdown by
  event-duration/gap statistics before committing to the next context run,
  so a similar deployment (wooster, the #2 fold on both metrics) doesn't
  silently eat the same regression.

## Exported clips

`/Users/luke/Documents/bioacoustics/buzzdetect-training/.local/worktrees/deployment-forensics/.local/forensics/clips/`
— 40 clips (8 per fold × 5 folds: `diel_1_95`, `opportunistic_48`,
`willard_1_11`, `wooster_1_143` and `mustardbumbler` as comparisons), index
at `clips/index.csv` (file, fold, ident, event start/end, duration, label,
acoustic stats, shipped-model activation with the leakage caveat repeated in
every row). Clips longer than 25s were truncated to 25s on disk; the
`duration` column still reports the full annotated event length. Total ~53 MB.

Questions the listening pass would settle that the numbers above can't:

1. **diel_1_95**: do the "ins_buzz_low" clips actually sound like insect
   buzzing to a human ear, or could some be wind/mechanical hum that got
   the buzz label by mistake? The RMS/embedding evidence is consistent with
   either "real but quiet buzz" or "mislabeled quiet non-buzz" — only
   listening disambiguates.
2. **opportunistic_48**: is the long sustained event actually one continuous
   insect sound (a chorus, or a single loud stationary buzzer), or is it a
   mechanical drone (generator, HVAC, distant engine) that was annotated as
   `ins_buzz_medium` because it resembled one on the spectrogram? This is the
   single highest-value listen — it would settle whether opportunistic_48 is
   a real detection problem or a labeling error, and the two imply completely
   different fixes.
3. **willard_1_11**: do the short isolated clips sound like a bee arriving/
   leaving (physically brief signal) as opposed to a longer buzz that got
   over-segmented into many short annotated fragments? If the latter, the
   "isolated short events" story is partly an annotation-granularity artifact
   rather than a real acoustic property, which would change the fix from
   "make context event-aware" to "review event segmentation in this effort."
4. **mustardbumbler comparison clips**: included so the ear has a "this is
   what a fold that scores 0.45 sounds like" reference point.

## Recommended next experiments (ranked)

1. **Listening pass on the exported clips** (above) — cheapest, resolves the
   single biggest open fork (mislabeling vs. genuine acoustic difference) for
   both near-chance folds before spending any training compute on them.
2. **Event-aware context stacking** for the next context experiment: don't
   stack a frame with neighbors that don't share its label (or gate the
   stack by event duration). Directly targets the willard/wooster mechanism
   found here, and is a small code change relative to `context-embedder`.
3. **If opportunistic_48's long event is confirmed as real buzz by
   listening**: source or annotate more sustained-buzz examples (ideally
   >1 site) before concluding the model structurally can't learn this
   category — right now it's one deployment's worth of one event shape, which
   is a data-support problem before it's a modeling problem.
4. **Add frame timestamps to `predictions.csv`** (or a parallel side-table
   keyed by fold+frame-index) — would have let this investigation directly
   pull willard's highest-disagreement frames between baseline and
   context-embedder instead of relying on annotation-level proxies. Useful
   generally: the next context/temporal experiment will want the same
   diagnostic.
5. **If diel_1_95's negatives are confirmed nighttime-specific by inspection**
   of its own night-hour predictions vs day-hour predictions (not done here
   — would need per-frame timestamps, same gap as #4): consider whether
   `tools/night_positives.py`-style night-vs-day breakdown should be a
   standard fold diagnostic, since this deployment shows the known
   nighttime-FP failure mode concentrated enough to move its own threshold
   by nearly a full point of activation.

## Should this go in log.jsonl?

No — this produced no metric, no trained model, and LOOP.md's log is for
`"method": "cv"` runs. If a later agent wants a pointer, the one line worth
carrying forward is: diel_1_95's fpr-0.005 threshold (+0.306) is a striking
outlier against every other fold in the roster (next highest -0.445) and is
worth checking again on any future model, since it's a direct read of "how
confusable is this fold's non-buzz audio," independent of sensitivity.
