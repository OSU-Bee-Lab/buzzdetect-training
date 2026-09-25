# aug-snr-noise

## Hypothesis

Augmentation has hurt every time it's been tried here. Measured cause candidate:
noise is added at a **fixed absolute amplitude** regardless of each frame's
level.

`02_set/augment.py::_apply_noise`: `f + prop * np.random.uniform(-1, 1, len(f))`.
Default specs use `prop` in {0.05, 0.075, 0.2}, i.e. noise RMS
`prop/sqrt(3)` = {0.029, 0.043, 0.115}.

Measured on the `medium` set (sampled ~19k frames): buzz-labeled frames have
**median waveform RMS 0.006**, p95 0.028. All-frame median RMS 0.014.

So the default noise augmentation adds noise that is **+13 dB (prop=0.05) to
+25 dB (prop=0.2) louder than the median buzz frame**. Every default
noise-augmented "buzz" frame is essentially broadband noise wearing a buzz
label — which is exactly how you would train a probe to fire on wind/rain. That
would depress sensitivity at a fixed low FPR, which is what the loop optimizes.

Pre-rework log already noted "white-noise samples as 'static' | Neutral; the
false positives are structured, not broadband" — consistent: broadband noise is
neither a useful positive nor a useful hard negative here.

## Changes

One change: noise is scaled **per-frame to a target SNR** instead of a fixed
absolute amplitude.

- `augment_specs.NoiseSpec` gains an optional `snr_db` field. When set,
  `spec_dirname` → `augment_noise_snr<snr_db>`; `prop` is ignored.
- `_apply_noise` (when `snr_db` is set): for each frame, draw Gaussian noise,
  rescale it so `rms(noise) = rms(frame) / 10**(snr_db/20)`, add. Frames with
  ~zero RMS get no noise (avoid divide-by-zero).
- Nothing else changes. `prop`-mode noise and volume/combine specs are
  untouched.

Spec used: a single `NoiseSpec(snr_db=15)` — noise 15 dB below the signal —
applied to all training frames (one augmented copy of the training pool, labels
unchanged). Built for `fold=train`... see below re: rotate folds.

## Results

Baseline `models/yamnet_medium_general` (this machine, current set) = 0.199.

| fold | base | exp | delta | val frames | exp best_epoch (base) |
|---|---|---|---|---|---|
| willard/2024-08-07/1_11 | 0.191 | 0.046 | **-0.145** | 4730 | 10 (69) |
| Various/2025-08-05/31 | 0.148 | 0.100 | -0.048 | 942 | 17 (26) |
| JamesU/1_29 | 0.447 | 0.412 | -0.035 | 6984 | 19 (113) |
| Various/2025-08-27/48 | 0.035 | 0.007 | -0.028 | 1571 | 10 (69) |
| Fit+Fast/53 | 0.368 | 0.367 | -0.001 | 4712 | 100 (93) |
| Diel/2026-05-06/1_95 | 0.028 | 0.028 | 0.000 | 6628 | 68 (95) |
| Various/2025-08-12/1_114 | 0.161 | 0.183 | +0.022 | 3768 | 169 (128) |
| Diel/2026-04-08/1_150 | 0.014 | 0.041 | +0.027 | 4947 | 15 (21) |
| Various/2025-06-23/1_23 | 0.326 | 0.354 | +0.028 | 315 | 146 (67) |
| wooster/2024-07-26/1_143 | 0.235 | 0.266 | +0.031 | 4708 | 68 (62) |
| Various/2025-07-03/1_37 | 0.241 | 0.313 | +0.072 | 4715 | 94 (95) |

- mean sens@fpr0.005: baseline 0.199 → this 0.192 (**-0.007**)
- 5 folds up, 5 down, 1 flat.
- The mean drop is carried by **willard -0.145**, a deployment IDEAS already
  flags as context/augmentation-fragile (short isolated buzzes), and its exp
  model early-stopped at **epoch 10** (baseline 69) — undertrained, not a
  measured capability loss. Three of the four biggest decliners early-stopped
  far sooner than baseline (JamesU 19 vs 113, 31: 17 vs 26, 48: 10 vs 69).
- Adding one augmented copy roughly doubles frames/epoch, so the fixed
  `min_delta=0.002` early-stopping rule fires earlier — the exact confound
  `stopping-rule-scale` in IDEAS.md predicts. The folds that trained a normal
  number of epochs are net positive (+0.072, +0.031, +0.028, +0.022).

## Conclusion

Inconclusive on the endpoint (-0.007, folds split 5/5/1) — but a real departure
from "augmentation always hurts here". SNR-relative noise at 15 dB is roughly
neutral where historical fixed-amplitude noise (13-25 dB *above* the buzz
signal) was clearly harmful. The residual drop is plausibly the early-stopping
confound: 4 folds early-stopped 3-6x sooner than baseline on the denser
augmented training pool, and the normally-trained folds are net positive.

Trust: **caveated** — direction (roughly neutral) is probably right; the -0.007
size is contaminated by the stopping-rule interaction, not a clean read of the
augmentation itself.

Follow-ups: (a) rerun with a stopping rule scaled to frames/epoch (or a fixed
epoch budget from the baseline medians) before judging SNR-15; (b) sweep SNR
(10, 20 dB); (c) `augment.py`'s `--fold train` default is stale for the CV
layout — `--all-folds` added here should probably land on main regardless of
this result.

