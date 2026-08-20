# yamnet-native-buzz

## Hypothesis

YAMNet's own classification head, trained on all of AudioSet, might already
score insect buzzing better than our probe trained on ~11 folds of field
recordings — AudioSet is orders of magnitude larger, even if less specific to
our deployments. If so, the "Buzz" neuron's raw logit, used untouched, should
beat or approach `cv-baseline`.

## Changes

No training, no re-extraction. `03_train/run_native.py` (new, this experiment
only) loads YAMNet's full frames model (`embedders/yamnet/yamnet.py`) with its
original AudioSet classification `Dense(521)` layer, pulls out the weight
column and bias for AudioSet class 125 ("Buzz", `embedders/yamnet/
yamnet_class_map.csv`), and wraps them in a frozen one-neuron `Dense` that
takes the exact 1024-d embeddings we already have on disk (the classification
layer's input is the same activation `embedders/yamnet/BUILD.py` cuts the
embedder off at). Scored fold-by-fold through the same `metrics_by_group` /
`metrics_at_fpr` / `summarize_sx` used everywhere else.

## Results

| fold | frames | this exp sens@fpr0.005 |
|---|---|---|
| JamesU - MustardBumbler/1_29 | 6908 | 0.001 |
| Lily - Fit+Fast/.../53 | 4710 | 0.000 |
| willard/2024-08-07/1_11 | 4713 | 0.000 |
| wooster/2024-07-26/1_143 | 4707 | 0.005 |
| Diel Drivers/2026-04-08/1_150 | 4891 | 0.004 |
| Diel Drivers/2026-05-06/1_95 | 6606 | 0.000 |
| Opportunistic/2025-06-23/1_23 | 314 | 0.034 |
| Opportunistic/2025-07-03/1_37 | 4712 | 0.000 |
| Opportunistic/2025-08-05/31 | 942 | 0.007 |
| Opportunistic/2025-08-12/1_114 | 3768 | n/a (too few negatives) |
| Opportunistic/2025-08-27/48 | 1570 | 0.000 |

- sens_persite @ fpr0.005: baseline (cv-baseline) 0.206 → this 0.002
- 9/11 folds scored, all landing at or near zero; nothing moved even
  directionally toward baseline. Not a small-fold artifact — the largest folds
  (6908, 6606, 4891, 4712 frames) all land at 0.000-0.005, so this isn't noise
  from thin val sets.

Looked at the pooled ROC (`folds_pooled_metrics.csv`) to check the neuron
isn't just miscalibrated rather than uninformative: at fpr≈0.40 sensitivity is
only ≈0.30, so there is *some* separation, but nowhere near operating range —
the raw AudioSet "Buzz" score just doesn't rank our insect buzz frames above
our non-buzz frames at any FPR near 0.005.

## Conclusion

Off-the-shelf YAMNet classification is not a substitute for the trained probe
on this task — not close. Plausible reasons: AudioSet's "Buzz" class covers
buzzers/electric buzz/etc. broadly, not specifically insect wingbeat buzz;
AudioSet's clip-level, 521-way training objective doesn't sharpen a
frame-level insect/not-insect boundary the way a probe fit directly to our
labels does; and our field recordings' background (wind, traffic, other
insects) may not resemble AudioSet's distribution closely enough for its
scores to transfer.

Also tried "Bee, wasp, etc." (class 126, the other obviously insect-sounding
neighbour on the class map): sens_persite 0.002, same pattern — all folds near
zero regardless of size. Confirms this isn't a matter of having picked the
wrong AudioSet class; the head doesn't carry a usable insect-buzz signal under
either label.

"Insect" (class 121), "Cricket" (122), "Mosquito" (123), and "Fly, housefly"
(124) remain untested, but given two different classes landed in the same
place, further class picks are unlikely to close a 0.002→0.206 gap. Not
adding to IDEAS.md as a follow-up.
