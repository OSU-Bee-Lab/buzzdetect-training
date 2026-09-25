# yamnet-bandpass

## Hypothesis
Pre-filtering audio to 100–3000 Hz before YAMNet embedding removes broadband interference and focuses the embedding on the buzz-relevant frequency range. Non-buzz confusables (mechanical noise, traffic, rain) have significant energy outside this range; restricting input to the buzz band should yield embeddings where buzz is more linearly separable. Tested on no-reg-baseline config (no dropout, no label smoothing) to isolate the embedder effect.

## Changes
- Embedder: `yamnet_bandpass` instead of `yamnet` (FIR bandpass 100–3000 Hz before YAMNet)
- No Dropout, no label smoothing (no-reg-baseline config)
- No Stage 2 (embeddings already extracted)

## Results
- Baseline (exp_no_reg / no-reg-baseline): mean=0.193  95% CI=[0.181, 0.205]
- This experiment (yamnet_bandpass_nrb v1–v5): 0.140, 0.143, 0.144, 0.147, 0.156  mean=0.146  median=0.144  std=0.006  95% CI=[0.139, 0.153]
CIs do not overlap; bandpass is clearly worse (-4.7pp).

## Conclusion
Negative. Bandpass filtering corrupts YAMNet's expected input distribution — the model was trained on full-spectrum audio, and restricting to 100-3000 Hz degrades the learned representations. Even though buzz lives in this range, the embedding quality drops without the full-spectrum context. Dead end.
