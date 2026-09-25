# yamnet-mask

## Hypothesis

Zeroing mel bins above 3000 Hz (roughly bins 42–63 of 64) inside YAMNet's spectrogram—before
the CNN runs—focuses the representation on buzz-relevant frequencies without the waveform-domain
distribution shift caused by FIR bandpass filtering. The audio-bandpass experiment scored 0.146
(worse than no-reg-baseline 0.193), likely because FIR filtering corrupts the waveform
statistics YAMNet was trained on. Mel-bin masking is a softer intervention: the CNN still sees
the full-spectrum waveform converted to a log mel spectrogram; only the high-frequency bins are
zeroed. This may reduce confusion with high-frequency broadband noise confounders without
disrupting the low/mid-frequency representation YAMNet relies on.

**Additional context**: Out-of-sample nighttime recordings show very strong false positives that
are largely absent in the bandpass model's output. This suggests the test set under-represents
real-world nighttime confounders, and that high-frequency content may be a significant
contributor to those false detections. If mel-bin masking has a similar suppressive effect on
high-frequency confusion as the audio bandpass—but without degrading the test-set metric—it
would be practically valuable even if the numerical gain is modest.

Baseline: no-reg-baseline (linear probe, no dropout, no label smoothing), 0.193 [0.181, 0.205].
Embedder: yamnet_mask (YAMNet with mel bins > ~3000 Hz zeroed).

## Changes

- New embedder `embedders/yamnet_mask/`: loads pretrained YAMNet weights, rebuilds model with
  a mel-bin mask (`mask[i] = 1 if bin_center_hz[i] <= 3000 else 0`) applied to the log mel
  spectrogram patches before the CNN.
- Re-extracts embeddings for the medium set under the new embedder.
- Training config identical to no-reg-baseline (linear probe, LS=0.0, no dropout, LR=0.002,
  patience=50, min_delta=0.002, epochs=400).

## Results

Individual runs (sensitivity at 95% precision):
yamnet_mask_medium v1–v5: see evaluate_set output below.
- Mean: 0.050, Median: 0.043, Std: 0.016, 95% CI: [0.031, 0.070]

Baseline (no-reg-baseline, exp_no_reg v1–v5):
- Mean: 0.193, Median: ?, 95% CI: [0.181, 0.205]

CIs do not overlap at all — 14.3pp gap, masked model is dramatically worse.

## Conclusion

Strong negative: mel-bin masking is even more destructive than audio bandpass filtering
(0.050 vs 0.193 baseline; audio bandpass scored 0.146). Zeroing 22 of 64 mel bins
(those > ~3000 Hz) cripples YAMNet's representation regardless of whether the removal
happens in the waveform domain or the spectrogram domain. The high-frequency mel bins
encode discriminative structure that the linear probe depends on heavily.

The hypothesis that mel masking would be a "softer" intervention than audio bandpass was
wrong — it is harsher in practice. The YAMNet CNN appears to use high-frequency spectral
context globally (possibly texture/harmonic relationships), not just as a local frequency
indicator, so zeroing those bins destroys the entire representation.

The nighttime false-positive improvement observed with the audio bandpass model must
therefore be an artifact of a degraded, low-sensitivity model that simply detects
less of everything — not a genuine reduction of out-of-band confusion.
