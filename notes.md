# harmonic-comb

## Hypothesis

Give the probe an explicit **f0 channel**. YAMNet's 1024-d GAP embedding is
trained to minimise AudioSet loss over 521 classes; insect buzz is a minor one,
and the pooling averages away the fine harmonic structure that distinguishes a
~200-250 Hz wingbeat comb from a ~50-120 Hz engine comb. Concatenating a 77-d
harmonic-comb block to the embedding should let a *linear* probe read that
difference directly.

The target is named in `IDEAS.md`: the single largest lever on the headline is
`Luke - Diel Drivers/2026-05-06/1_95` (0.037), whose fpr0.005 threshold is set
by **32 `mech_auto` frames out of 35** — vehicle noise, not trill. Engine drone
and bee buzz are both harmonic; they differ mainly in where the comb sits and
how modulated it is. A feature that resolves f0 is aimed exactly there.

**Prior art, and why this is not a rerun of it.**
`archive/2026-06_fixed-test` logged `supp-freq` (+0.012, CIs overlap) and
`supp-freq-v2` (neutral) and concluded "handcrafted frequency features are a
dead end". Both appended **four global scalars** — dominant frequency, spectral
centroid, ZCR, 100-600 Hz band energy — which carry *no f0 resolution at all*:
a 90 Hz engine and a 220 Hz bee are both "low" and both "tonal" under every one
of them. They were also unnormalised against a 1024-d block and measured under
the retired fixed-split / sens@95%-precision metric. The dead-end verdict is
evidence against four unresolved scalars, not against resolving f0. If this
also comes back flat, *that* is what closes the direction.

## Changes

New embedder `embedders/yamnet_harmonic/` (1101-d). YAMNet's 1024-d half is
byte-identical to `embedders/yamnet` (verified). The 77-d block, per 0.96 s
frame, from an STFT at win 2048 / hop 512 (7.81 Hz bins, 27 time steps):

| dims | feature |
|---|---|
| 40 | comb profile: for 40 log-spaced f0 candidates in 70-450 Hz, mean log-power at `h*f0` for h=1..6 minus mean at the off-comb points `h*f0 +/- f0/2` |
| 3 | comb peak, peak-minus-median, and soft-argmax f0 in normalised log-Hz |
| 24 | band contrast: mean log-power in 24 log-spaced bands, 50-8000 Hz |
| 8 | within-frame amplitude modulation: std over the 27 time steps of log band energy, 8 coarse bands |
| 2 | spectral flatness in 100-600 Hz; zero-crossing rate |

Design constraints, both deliberate:

- **Everything is a log10 ratio against a reference computed from the same
  frame**, and frames are RMS-normalised first. Absolute level is *not* a
  feature: it would hand the probe a recorder-gain channel that helps in-fold
  and hurts out-of-fold. Measured gain invariance after the fix: `3.4e-4` max
  abs change across a 20 dB scaling (was `0.30` before RMS normalisation — the
  epsilon floor was leaking level).
- **No fitted statistics.** Every constant is fixed a priori, so nothing is
  estimated from a pool spanning folds. CLAUDE.md bars fold-crossing statistics
  for augmentation; the same reasoning applies to feature scaling, and it also
  sidesteps the `Normalization`-layer NaN trap noted in `IDEAS.md`
  (standardization). Scaling is a fixed `/2` and a clip to +/-3, which lands the
  block at std 0.56 against YAMNet's ReLU activations (mean 0.066, max 3.37).

The whole block is `tf.signal.stft` plus three matmuls against precomputed
matrices, so it runs on GPU wherever TF places it. (Extraction here is pinned to
CPU anyway — the 4 GB card OOMs; see CLAUDE.md.)

Sanity check on synthetic tones, `comb` argmax:
`bee 220 Hz -> 220.0 Hz`, `bee 220 Hz at -20 dB -> 220.0 Hz`,
`engine 90 Hz -> 88.9 Hz`, `white noise -> flat (peak +0.06)`.
The soft-argmax f0 feature reads `+0.115` for the bee and `-0.371` for the
engine — the separation the hypothesis rests on.

## Results

_pending_

## Conclusion

_pending_
