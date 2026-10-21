# pitch-aug-d8
## Hypothesis
The buzz fundamental scales with the bee (wingbeat frequency differs by species and body size), and each deployment has its own pollinator community: blueberry (1_95), apple (1_150), senna (1_114) are the hard folds, each with flowers that draw a different visitor mix from the pool's. The octave-up view was this era's largest single lever, so the tail is pitch-sensitive. Training the depth8 tail on a copy of half of every training fold's frames, each shifted by its own random pitch uniform in ±2 semitones (librosa phase-vocoder pitch shift: the frame keeps its length, so no tile seam), widens the range of fundamentals it has seen per class. Prediction: headline >= +0.02 over the matched control `ps-depth8-r3` (0.497; three-draw mean 0.507), with `untagged` up and at least one of 1_95 / 1_150 / 1_114 up beyond its SD. Risk: 1_114's confuser is orthopteran trill, and shifted trills could sit nearer buzz; check 1_114 first. Falsifier: within 0.02 of the control and no tier up.

Control: `ps-depth8-r3`. Companions this batch: `eq-aug-d8`, `snr-aug-d8`, `gain-aug-d8`.

## Results
CV on medium, 8 rotating folds, 30 epochs, depth8 recipe with a copy of half of every training fold's snips pitch-shifted by a random ±2 semitones. Buzz logit SD healthy on every fold (0.53-1.26).

| fold | ps-depth8-r3 | pitch-aug-d8 | delta ± SD |
|---|---|---|---|
| 1_29 | 0.486 | 0.579 | +0.093 ± 0.040 |
| 53 | 0.550 | 0.520 | -0.030 ± 0.032 |
| 1_11 (willard) | 0.573 | 0.580 | +0.007 ± 0.024 |
| 1_143 | 0.586 | 0.631 | +0.045 ± 0.041 |
| 1_150 | 0.435 | 0.303 | -0.132 ± 0.044 |
| 1_95 | 0.284 | 0.211 | -0.073 ± 0.032 |
| 1_37 | 0.648 | 0.575 | -0.073 ± 0.056 |
| 1_114 | 0.414 | 0.207 | -0.207 ± 0.047 |

Headline sensitivity_exclquiet 0.451 vs 0.497 (-0.046 ± 0.014; -0.056 vs the depth8 three-draw mean 0.507). Inclusive 0.372 vs 0.411. Tiers: loud -0.005, untagged -0.052, background +0.046, quiet -0.004. No fold missed FPR 0.005.

## Conclusion
Falsified and harmful: -0.046 ± 0.014, ~3x the noise floor. The flagged risk came true: 1_114 fell -0.207 ± 0.047, and the other hard folds fell too (1_150 -0.132 ± 0.044, 1_95 -0.073 ± 0.032), while 1_29 rose +0.093 ± 0.040. The `background` tier rose +0.046, but `untagged` fell -0.052. Widening the fundamentals the tail sees seems to blur the buzz/confuser boundary (trill, hum) on deployments where that boundary is what limits sensitivity.

Batch 40 read together (all against ps-depth8-r3 0.497): eq -0.008, gain -0.013, pitch -0.046, snr -0.063. All four took 1_150 down (-0.055 to -0.132) and none lifted a hard fold beyond its SD. Waveform augmentation of training snips does not help the depth8 tail on held-out deployments. The two harmful ones (snr, pitch) are also the two that put the largest share of altered frames into the pool. Treat further input-augmentation ideas on this recipe as low-prior.
