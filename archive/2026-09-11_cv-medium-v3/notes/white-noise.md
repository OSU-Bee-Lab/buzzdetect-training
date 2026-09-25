# white-noise

## Hypothesis
Adding white noise samples (labeled "static") to the training set gives the linear probe concrete examples of broadband, structurally-null audio. "Static" is a valid class in the general translation but has zero training examples — it's currently an empty placeholder. White noise produces a distinctive YAMNet embedding (broad, uniform spectral activation), unlike buzz (structured 100–600 Hz energy) or any real environmental sound. These examples should help the probe define a clearer "not buzz" boundary for broadband false positives.

Tested on no-reg-baseline config (no dropout, no label smoothing) to isolate the data effect.

## Changes
- `02_set/gen_whitenoise_embeddings.py`: generates 500 frames of white noise (16kHz, unit-RMS normalized, seed=42), extracts YAMNet embeddings, saves to `02_set/sets/medium/embeddings/yamnet/augment_whitenoise/train/static.pickle`
- `03_train/train.py`: no-reg-baseline config (Dropout removed, label_smoothing=0.0)
- Training uses `--augment augment_whitenoise`

## Results
- Baseline (no-reg-baseline / exp_no_reg): 0.183, 0.188, 0.195, 0.197, 0.207  mean=0.193  median=0.195  95% CI=[0.181, 0.205]
- This experiment (white_noise v1–v5): 0.187, 0.199, 0.202, 0.212, 0.214  mean=0.203  median=0.202  95% CI=[0.189, 0.216]

CIs overlap (0.189–0.216 vs 0.181–0.205). +1pp mean improvement is within noise.

## Conclusion
Neutral. Adding 500 white noise frames as "static" provides a marginal upward trend but no significant improvement — CIs overlap substantially. False positives in the test set are likely structured environmental sounds, not broadband noise, so white noise embeddings don't address the actual confusion region. Dead end.
