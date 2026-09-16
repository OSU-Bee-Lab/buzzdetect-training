# avesshift

## Hypothesis
IDEAS item 22: `yamnet-pitchshift` (+0.069, 8/8 folds, twice confirmed) is
attributed to YAMNet's thin mel coverage near ~230 Hz -- a front-end-specific
explanation. AVES has no mel filterbank (raw waveform through a conv front
end), so if the same octave-up shift *also* helps AVES, the win is about
where the pretraining data's spectral energy sat, not about YAMNet's front
end specifically, and shifting becomes a lever worth trying on every encoder
(Perch, item 2; a future bioacoustic SSL model, item 23). If AVES is flat or
down, the win is YAMNet-specific and nobody should spend an extraction
shifting other encoders.

One variable against `aves-mid` / `aves-mid-repeat`: YAMNet block untouched
(plain, unshifted), AVES reads the decimated (seamless) octave-up input
instead of the raw 1.0 s frame. Embedder `yamnet_aves_mid_avesshift` was
already built and committed to main (commit 4c8f6ef) but never run through
extraction or CV -- this experiment does that.

## Changes
- `--embedder yamnet_aves_mid_avesshift` (pre-existing code, unmodified).
- `--fixed-epochs 400` to match `aves-mid-repeat`'s own budget (this is a
  3328-d config, same width as `aves-mid`, not the 4352-d lead width that
  motivated `epoch-budget-700`).

## Results
| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.527 | 0.525 | -0.002 | 0.049 | 32 |
| 53 | 0.479 | 0.532 | +0.053 | 0.031 | 25 |
| 1_11 | 0.500 | 0.429 | -0.071 | 0.033 | 28 |
| 1_143 | 0.543 | 0.514 | -0.029 | 0.049 | 21 |
| 1_150 | 0.346 | 0.298 | -0.048 | 0.035 | 20 |
| 1_95 | 0.123 | 0.130 | +0.007 | 0.026 | 42 |
| 1_37 | 0.348 | 0.399 | +0.051 | 0.041 | 14 |
| 1_114 | 0.168 | 0.330 | +0.162 | 0.049 | 31 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline (`aves-mid-repeat`) 0.379 → this 0.395 (+0.016 ± 0.014 eval-sampling SD; add ~0.007 headline
  training-stochasticity SD in quadrature, ~0.016 total — well inside the ~0.027 MDE)
- inclusive (sensitivity), same thresholds: 0.302 → 0.320 (+0.018)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.685 | 0.822 | +0.137 | 115 |
| untagged | 0.408 | 0.416 | +0.008 | 2317 |
| background | 0.390 | 0.422 | +0.032 | 1804 |
| quiet | 0.068 | 0.067 | -0.001 | 599 |
| faint | 0.000 | 0.250 | +0.250 | 13 |

## Conclusion
Falsifier fires: headline is inside MDE (+0.016 ± ~0.016 total) and the tier that
matters most, `untagged` (2317 frames, 8/8 folds), is flat (+0.008). `quiet`
is also flat. `loud` and `background` move up, but on far smaller n (115 and
1804 frames, the latter concentrated in `1_29`/`53`), and `faint` is a
13-frame artifact, not evidence. This is not the broad detection gain that
would say "the octave-up shift also helps AVES for the reason it helps
YAMNet" -- per the item's own falsifier, treat the win as YAMNet-front-end-
specific and don't spend an extraction shifting Perch (item 2) or a future
encoder (item 23) on this basis.

Per-fold, the picture is genuinely mixed rather than uniformly flat: `1_114`
(trill fold) is up +0.162 ± 0.049 (~3.3σ eval-sampling alone), the third
AVES-side lever this era to move that fold strongly (`aves-mid` layers,
`decimate-lead`'s AVES-shift, and now this) -- worth carrying into any future
`1_114`-focused work, e.g. item 20's layer profile. `1_11` is down -0.071 ±
0.033 (~2.2σ) and `1_150` down -0.048 ± 0.035 (~1.4σ, unsure), a genuine
rich/hard-fold-mixed trade rather than the clean hard-fold gain the hypothesis
hoped for. `1_95` is flat. No repeat draw: the headline isn't large enough to
be worth confirming, and the falsifier already resolved the item's actual
question (is this lever front-end-general) in the negative.
