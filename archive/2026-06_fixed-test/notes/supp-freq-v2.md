# supp-freq-v2

## Hypothesis
`supp-freq` showed a marginal +1.2pp with 4 global frequency features (dominant_freq, spectral_centroid, ZCR, buzz_energy). Two of those features — dominant_freq and spectral_centroid — are computed globally over 0–8 kHz, so they can be dominated by energy outside the buzz band (100–600 Hz). Replacing these with buzz-band-specific variants (spectral centroid within 100–600 Hz, spectral flatness within 100–600 Hz) should give the linear probe a stronger, less diluted signal. Spectral flatness is low for tonal/harmonic signals like buzz and high for broadband noise, making it particularly discriminative. Testing on no-reg-baseline config (no dropout, no LS) to isolate the feature contribution.

## Changes
- New embedder `embedders/yamnet_freq2/embedder.py`: wraps YAMNet 1024-d output, appends 4 buzz-band-specific frequency features. Output: 1028-d.
  - `buzz_energy`: energy in 100–600 Hz / total (same as supp-freq)
  - `spectral_flatness_buzz`: geometric mean / arithmetic mean of PSD in 100–600 Hz (new; low=tonal, high=noisy)
  - `buzz_centroid_norm`: spectral centroid within 100–600 Hz, normalized to [0, 1] (new; replaces global centroid)
  - `zcr`: zero crossing rate (same as supp-freq)
- `03_train/train.py`: no-reg-baseline config (no Dropout, label_smoothing=0.0) to isolate feature contribution

## Results
- Baseline (exp_no_reg, no-reg-baseline): 0.193, CI [0.181, 0.205]
- supp_freq2 v1–v5: mean=0.199  95% CI=[0.178, 0.220]

CIs fully overlap (supp_freq2 upper CI 0.220 vs baseline upper CI 0.205 — barely wider, not better).

## Conclusion
Neutral result. Buzz-band-specific features (flatness, centroid within 100–600 Hz) provide no measurable improvement over a bare YAMNet linear probe. Combined with supp-freq's marginal result (+1.2pp, CIs overlap), the evidence is that handcrafted frequency features appended to YAMNet embeddings don't add meaningful discriminative signal. YAMNet's 1024-d embedding already encodes frequency structure relevant to the task. Frequency feature engineering is a dead end.
