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

Two CV runs of the **identical** config (`harmonic_comb`, `harmonic_comb_r2`).
The second was launched only to get fold weights on disk; it turned into the
more important half of the experiment.

### Run 1 vs `cv_baseline`

| fold | baseline | harmonic_comb | delta | buzz frames |
|---|---|---|---|---|
| willard/1_11 | 0.180 | 0.233 | +0.053 | 305 |
| MustardBumbler/1_29 | 0.426 | 0.437 | +0.011 | 2144 |
| Diel Drivers/1_95 | 0.037 | 0.039 | +0.002 | 433 |
| Diel Drivers/1_150 | 0.021 | 0.007 | -0.014 | 146 |
| Fit+Fast/53 | 0.425 | 0.400 | -0.025 | 1031 |
| **total** | **0.218** | **0.223** | **+0.005** | 4059 |

3 up / 2 down.

### Run 2 vs `cv_baseline` — same config, same data, same folds

| fold | baseline | harmonic_comb_r2 | delta |
|---|---|---|---|
| willard/1_11 | 0.180 | 0.223 | +0.043 |
| MustardBumbler/1_29 | 0.426 | 0.456 | +0.030 |
| Diel Drivers/1_150 | 0.021 | 0.062 | +0.041 |
| Diel Drivers/1_95 | 0.037 | 0.039 | +0.002 |
| Fit+Fast/53 | 0.425 | 0.413 | -0.012 |
| **total** | **0.218** | **0.239** | **+0.021** |

4 up / 1 down.

### Run 1 vs Run 2 — this is the finding

| fold | run 1 | run 2 | delta | buzz frames |
|---|---|---|---|---|
| Diel Drivers/1_150 | 0.007 | 0.062 | **+0.055** | 146 |
| MustardBumbler/1_29 | 0.437 | 0.456 | +0.019 | 2144 |
| Fit+Fast/53 | 0.400 | 0.413 | +0.013 | 1031 |
| Diel Drivers/1_95 | 0.039 | 0.039 | 0.000 | 433 |
| willard/1_11 | 0.233 | 0.223 | -0.010 | 305 |
| **total** | **0.223** | **0.239** | **+0.016** | |

3 up / 1 down / 1 flat, mean +0.0154 — **from nothing but TF's nondeterministic
init and shuffling.** There is no seed control in this pipeline, so this is the
whole run-to-run distribution showing itself.

**The repeat-run spread (0.016) is larger than the experiment's effect
(+0.005), and as large as the run-2 effect (+0.021).** Had these two runs been
two different configs, the 3-up/1-down/1-flat split and the +0.016 mean would
have read as a modest real gain. They are the same config.

`1_150` alone moved 0.007 -> 0.062 between identical runs. It has 146 buzz
frames; `IDEAS.md` already warns per-fold sensitivity is +/-0.25 in the
quietest deployments, and this is that warning made concrete.

### Did the probe use the block? Yes.

From the run-2 fold weights (212 KB each), mean `|w|` per input dim on the
`ins_buzz` output:

| fold | YAMNet dims | comb dims | ratio |
|---|---|---|---|
| 1_29 | 0.0566 | 0.0697 | 1.23x |
| 53 | 0.0601 | 0.0861 | 1.43x |
| 1_11 | 0.0471 | 0.0642 | 1.36x |
| 1_150 | 0.0426 | 0.0452 | 1.06x |
| 1_95 | 0.0636 | 0.0855 | 1.34x |

The block is 7.0% of the dims and carries 8.8% of the weight mass — **1.29x the
per-dim weight of a YAMNet dim, in all five folds.** By sub-block (fold 1_29,
against a YAMNet per-dim baseline of 0.0566): comb profile 0.0928 (**1.64x**),
comb peak/f0 0.0495, modulation 0.0492, band contrast 0.0433, flatness+zcr
0.0360. The 40-d comb profile — the part that resolves f0 — is the most heavily
weighted, exactly as the hypothesis predicted.

So the block is **not inert**. The probe reaches for it, and preferentially for
the f0-resolving part of it. It simply does not change the decisions.

## Conclusion

**Inconclusive on the headline, and the direction is not worth another run.**

The effect (+0.005 run 1, +0.021 run 2) is bounded above by the repeat-run
spread of the identical config (+0.016). Nothing here separates the comb block
from noise, in either direction.

**The mechanism specifically failed at its target.** `1_95` — the fold this was
designed for, whose fpr0.005 threshold is set by 32 `mech_auto` frames of 35 —
moved +0.002 in run 1 and +0.002 in run 2, and its threshold barely shifted
(-0.401 -> -0.336 / -0.424). It is the *only* fold that was byte-stable across
two nondeterministic runs, which says its failure is structural, not stochastic.
An engine/wingbeat f0 separation that works cleanly on synthetic tones
(88.9 Hz vs 220.0 Hz argmax) does not transfer to this fold's real audio.

**Combined with the weight evidence, the interpretation is redundancy, not
inertness.** The probe weights the comb profile 1.64x a YAMNet dim and still
lands in the same place, which means the f0 information is already present in
YAMNet's 1024-d representation — the probe gains a more convenient encoding of
something it could already read. That is a stronger and better-evidenced
version of what `supp-freq-v2` asserted ("YAMNet's embedding already encodes
frequency structure relevant to the task") on much weaker evidence.

**This closes handcrafted frequency features properly.** `supp-freq` /
`supp-freq-v2` closed it on four unresolved global scalars under a retired
metric. This run tested the strongest form of the idea — 40 f0 candidates with
harmonic reinforcement, plus modulation and band contrast, gain-invariant, at
1.29x weight uptake — on the current data and the current metric, and it still
does not move. A learnable filterbank is the only untried variant, and the
redundancy finding argues against it: the probe is not starved of this
information, so letting it reshape the filters addresses a bottleneck the
evidence says is not there.

**The transferable result is the noise measurement.** Two identical CVs, 0.223
and 0.239. `noise-floor-cv` had this at ~0.014 headline from a single
measurement; this is an independent confirmation at 0.016, and it is now
measured on the current 5-fold roster rather than the old 11-fold one. Any
future result in this era under ~0.02 headline, or resting on `1_150`, should
be read against it. Worth noting the fold-level structure: the two folds with
>1000 buzz frames moved +0.019 and +0.013 between identical runs, while
`1_150` (146 frames) moved +0.055 — the noise is concentrated in the thin
folds, which is the same conclusion `eval-sampling-floor` reached by
bootstrapping.

**Kept from this branch regardless of the negative result:** fold models are
now saved (`03_train/train.py`, ~212 KB each, optimizer state still
shipped-only). Without them, asking what a rotation learned requires a full
retrain, and a retrain is a *different model* — which this experiment
demonstrates rather sharply. That change is worth carrying into main
independently of the comb block.
