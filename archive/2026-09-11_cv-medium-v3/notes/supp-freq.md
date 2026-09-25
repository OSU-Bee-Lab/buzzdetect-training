# supp-freq

## Hypothesis
Concatenating 4 hand-crafted frequency features to YAMNet's 1024-d embedding gives the linear probe a direct channel for buzz-specific acoustic properties that YAMNet's general embeddings don't emphasize. Features: dominant frequency, spectral centroid, ZCR, and buzz-band energy (100–600 Hz relative to total power). All normalized to ~[0,1] range. YAMNet was trained on AudioSet's 521 classes; insect buzz is a minor category and the embedding need not highlight these features to minimize AudioSet loss. These 4 scalars give the classifier a direct handle on the buzz frequency signature.

## Changes
- New embedder `embedders/yamnet_freq/embedder.py`: wraps YAMNet 1024-d output, appends 4 frequency features computed per frame from raw audio. Output: 1028-d.
- Extraction re-run with `--embedder yamnet_freq` (new pickles under `embeddings/yamnet_freq/`).
- Training: 5 runs as `supp_freq_v1–v5`, embedder=yamnet_freq, translation=general, label smoothing=0.2 (from train.py defaults).

## Results
- Baseline (with-dropout, medium): 0.229, CI [0.216, 0.241]
- supp_freq v1–v5: 0.217, 0.229, 0.241, 0.249, 0.272  mean=0.241  median=0.241  95% CI=[0.215, 0.267]

CIs overlap substantially. +1.2pp mean improvement but not statistically distinguishable.

## Conclusion
Marginal positive directional signal (+1.2pp), but CIs overlap too much to call it a clear win. The 4 frequency features add some signal — the upper tail is notably higher (0.272 vs 0.242 max in baseline) — but not enough to confidently claim improvement. Could revisit with more features or better normalization, but the gain is small relative to the noise floor.
