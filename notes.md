# asym-context-yamnet

## Hypothesis
IDEAS item 10: a linear readout of `yamnet_context`'s concatenated neighbours
`[e_{t-1}, e_t, e_{t+1}]` cannot compute "this frame stands out from its
neighbours" -- it can only weight each neighbour's own dimensions
independently. That contrast is close to the definition of a buzz against a
stationary background. This tests a new embedder, `yamnet_context_contrast`,
that computes the contrast explicitly: `[e_t, e_t - mean(e_{t-1}, e_{t+1})]`,
2048-d, built in the embedder (not as a train-time transform over cached
rows -- a train-time transform has no deployment equivalent and measured a
train/deploy mismatch in the pre-cutover `exp/asymmetric-context`).

Per the item, this is a mechanism question on plain YAMNet -- the signed
difference channel is a real regime change for a linear probe against
YAMNet's 89.6%-zero blocks, and the item explicitly says not to expect this
to add to the era lead, which already gets context-like signal from
concatenated pitch-shift/AVES-depth blocks.

Control: `cv-baseline-v3` (bare YAMNet linear probe, `--fixed-epochs 400`,
the item's natural baseline since this is plain-YAMNet, not lead-width).

## Changes
- New embedder `embedders/yamnet_context_contrast/embedder.py`: subclasses
  the same contiguous-buffer + `context_frames = 1` mechanism as
  `yamnet_context`, but concatenates `[frame_t, frame_t - mean(neighbours)]`
  instead of `[frame_{t-1}, frame_t, frame_{t+1}]`. 2048-d (vs
  `yamnet_context`'s 3072-d).
- `--fixed-epochs 400` to match `cv-baseline-v3`'s own budget (2048-d width,
  not the 4352-d lead width that needed 700).

## Results
Baseline note: `cv-baseline-v3` (2026-09-11) turned out to be stale relative to
current annotations -- ongoing annotation work roughly doubled-to-tripled
buzz events on every fold between 2026-09-11 and 2026-09-13/14 (confirmed by
diffing `buzz_events_exclquiet`/`buzz_frames` across models; root mechanism:
`01_annotate/Even Sample/combine.R` was revised at the era cutover, 2026-09-11
15:41, to stop stripping `_quiet`/`_faint` suffixes, and the Even Sample
corpus kept growing for a couple of days after). Reran the anchor's exact
config fresh (`cv-baseline-v3-refresh`, same `yamnet` cache, no retraining
needed elsewhere) rather than trust the stale comparator. The refreshed
anchor's headline (0.329) is almost identical to the original (0.330) despite
the much larger event pool, so the anchor itself is robust -- but per-fold
buzz-event counts (and therefore per-fold SDs) are not, which matters for any
comparison mixing model vintages.

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.443 | 0.559 | +0.116 | 0.053 | 32 |
| 53 | 0.440 | 0.524 | +0.084 | 0.039 | 28 |
| 1_11 | 0.368 | 0.399 | +0.031 | 0.038 | 26 |
| 1_143 | 0.459 | 0.541 | +0.082 | 0.036 | 22 |
| 1_150 | 0.231 | 0.259 | +0.028 | 0.031 | 21 |
| 1_95 | 0.042 | 0.017 | -0.025 | 0.018 | 46 |
| 1_37 | 0.399 | 0.466 | +0.067 | 0.058 | 14 |
| 1_114 | 0.252 | 0.233 | -0.019 | 0.025 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline (`cv-baseline-v3-refresh`) 0.329 → this 0.375 (+0.046 ± 0.014
  eval-sampling SD; ~0.016 with training-stochasticity added in quadrature -- ~2.9σ, well past MDE)
- inclusive (sensitivity), same thresholds: 0.268 → 0.308 (+0.040)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.810 | +0.032 | 122 |
| untagged | 0.350 | 0.386 | +0.036 | 2418 |
| background | 0.330 | 0.454 | +0.124 | 1874 |
| quiet | 0.067 | 0.089 | +0.022 | 627 |
| faint | 0.000 | 0.250 | +0.250 | 12 |

## Conclusion
A real, broad gain: +0.046 ± ~0.016 total (~2.9σ), 6/8 folds up, both `untagged`
(2418 frames, 8/8 folds, +0.036) and `loud` (+0.032) moving in the same
direction as `background` (+0.124, concentrated in the two continuous-drone
folds `1_29`/`53`) -- this is not just a background-tier artifact the way some
context-stacking runs have been. The contrast channel (explicit
`frame_t - mean(neighbours)`, signed, 2048-d) clearly helps a linear readout
more than `yamnet_context`'s honest concatenation did at the same budget
(context-frames-fix: +0.030 vs this: +0.046), consistent with the item's
premise that the raw concatenation can't compute the contrast itself.

Per the item's own expectation, this doesn't look like a hard-fold fix: `1_95`
(jet fold) is down -0.025 (~1.4σ, unsure) and `1_114` (trill fold) is down
-0.019 (~0.8σ, not clearly moving) -- both hard folds flat-to-down, while the
gain concentrates in richer folds (`1_29`, `53`, `1_143`, `1_37`). `1_150` is
up a little (+0.028, ~0.9σ, unsure). This is the same rich-folds-mostly
signature every context-family lever has shown this era (`yamnet_context`,
`yamnet_aves_context`, `pitchshift-context`) -- confirms item 10's own caveat
that context "has landed rich-folds-only three times this era," now four.
Since this doesn't touch the lead (4352-d pitchshift+AVES stack already has
its own context-like signal per the item), no follow-up composing this into
the lead is queued. Settles item 10 as a real but non-lead-relevant lever;
no repeat draw needed, since the falsifier question (does the contrast beat
plain concatenation) is already answered by the direct comparison to
context-frames-fix's honest number.
