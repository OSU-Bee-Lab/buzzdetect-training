# embed-mixup

## Research context (why revisit augmentation at all)

The loop's prior augmentation runs (`aug-snr-noise` -0.007 caveated, `aug-combine-overlap`
+0.002 caveated) found waveform augmentation ~neutral. A quick literature check says that
result is **consistent with**, not contrary to, best practice:

- The large augmentation gains in the audio-ML literature (SpecAugment, mixup, gain/time-shift)
  are almost all measured when the **feature extractor is being trained** — either encoder
  pretraining or end-to-end fine-tuning. SpecAugment for few-shot bioacoustic event detection
  buys ~1-2 F-score points, and that is on a *trainable* mel->CNN path
  (Reshaping Bioacoustics Event Detection, PMC11274013).
- The bioacoustics transfer-learning papers that match our setup — a linear / shallow probe on
  **frozen** embeddings — deliberately use **no augmentation** downstream. Perch/BirdNET augment
  (MixUp, random gain, time-shift) *when training the bird classifier*; the few-shot probe is
  then "a linear classifier over pre-computed embeddings, trained to convergence" with nothing
  added (Ghani et al. 2023, arXiv:2307.06292; Nature Sci Rep s41598-023-49989-z).
- Mechanism: a frozen AudioSet-scale embedder (YAMNet) is already largely invariant to mild
  gain / additive noise / sub-frame shift, so a weak waveform aug produces an embedding almost
  on top of the clean one (no regularisation pressure on a 1-dense-layer probe), while an aug
  strong enough to move the embedding tends to move it off-manifold and change the effective
  label. `aug-snr-noise` diagnosed exactly this: fixed-amplitude noise sat 13-25 dB above the
  buzz, turning "buzz" frames into "broadband-noise-labeled-buzz" — label-noise injection, not
  augmentation.
- Higher-value structural moves than probe-time waveform aug (all noted for later, not this run):
  a domain-matched embedder (Perch/BirdNET >> YAMNet for insects/birds), temporal context,
  more/harder negatives, class balancing, and test-time augmentation (reporting-time, no leak
  risk).

The one augmentation with a plausible mechanism for a frozen-embedding probe is **mixup in
embedding space** (a.k.a. manifold mixup): linearly interpolate pairs of *embeddings* and their
targets. It stays near the data manifold by construction, needs no re-extraction, and is what
Perch itself used at train time. This run tests it.

## Hypothesis

Convex combinations of training-pool embedding pairs, with matching target interpolation,
regularise the probe's decision boundary (label-smoothing along real embedding directions
rather than uniformly) and lift held-out-deployment sens@fpr0.005. Expect the effect, if any,
to be small and to show as most folds nudging the same way; larger on folds where the baseline
probe looks over-confident (early best_epoch, few buzz frames).

Leakage guard: mixing happens only inside `_to_tf(data_train, ...)`, which never sees the
validation fold. Both parents of every mixed frame are in the current rotation's training pool,
so no fold boundary is crossed and the val fold / early-stopping signal stay clean. Per the new
CLAUDE.md rule, no cross-fold *pairing* logic and no pooled-across-folds statistics.

## Changes

- `03_train/train.py::_to_tf` gains a `mixup_alpha` arg (default 0.0 = off, identical pipeline).
  When >0: after the existing shuffle, draw `lam ~ Beta(alpha, alpha)` per sample, pair each
  frame with another random training frame, emit `lam*x_i + (1-lam)*x_j` and the same convex
  combination of targets. Applied to the train dataset only; `val_tf` keeps `mixup_alpha=0`.
- `alpha = 0.2` (Perch's setting; mild — mass near 0/1).

## Results

`--name yamnet_medium_general_mixup02`, full CV, joined against `models/yamnet_medium_general`.
Frames/epoch unchanged from baseline (~69k train; mixup transforms in place, adds nothing), so
no stopping-rule-scale confound.

| fold | base | exp | delta | val frames |
|---|---|---|---|---|
| Lily Adam / willard 2024-08-07 | 0.191 | 0.151 | -0.040 | 4730 |
| JamesU MustardBumbler 1_29 | 0.447 | 0.423 | -0.024 | 6984 |
| Luke Opportunistic 2025-08-05/31 | 0.148 | 0.128 | -0.020 | 942 |
| Lily Adam / wooster 2024-07-26 | 0.235 | 0.221 | -0.014 | 4708 |
| Lily Fit+Fast Marysville/53 | 0.368 | 0.359 | -0.009 | 4712 |
| Luke Opportunistic 2025-08-27/48 | 0.035 | 0.028 | -0.007 | 1571 |
| Luke Opportunistic 2025-06-23/1_23 | 0.326 | 0.326 | 0.000 | 315 |
| Luke Diel Drivers 2026-05-06/1_95 | 0.028 | 0.030 | +0.002 | 6628 |
| Luke Opportunistic 2025-07-03/1_37 | 0.241 | 0.244 | +0.003 | 4715 |
| Luke Opportunistic 2025-08-12/1_114 | 0.161 | 0.166 | +0.005 | 3768 |
| Luke Diel Drivers 2026-04-08/1_150 | 0.014 | 0.034 | +0.020 | 4947 |

- mean sens@fpr0.005: baseline 0.199 -> this 0.192 (**-0.007**); 6 folds down, 4 up, 1 flat.
- The two down-movers with the most buzz to trust (JamesU -0.024 at 6984 val frames, willard
  -0.040) both regressed. The one real gain (Diel Drivers 04-08, +0.020) is a near-zero-buzz
  fold (baseline 0.014) and unreliable per the README caution.
- Several folds early-stop very early (best epoch 15 / 28 / 35), same pattern flagged in
  `stopping-rule-scale`; not obviously worse than baseline here.

## Conclusion

Mildly negative, leaning negative rather than a clean null: -0.007 mean with the trustworthy
high-buzz folds among the down-movers. Embedding-space mixup (alpha=0.2) does not rescue
augmentation for this frozen-YAMNet-probe setup, matching the research framing in the section
above — the augmentation gains in the literature are at feature-extractor training time, and a
low-capacity probe over a frozen general-purpose embedder has little frame-level nuisance
variance for any augmentation to regularise away. Not worth a follow-up sweep of alpha; the
higher-value levers are a domain-matched embedder, temporal context, and harder negatives.
Trust: caveated (single run, no seed control; direction clearer than size).
