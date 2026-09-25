# aves-mid

## Hypothesis

IDEAS.md item 4: `embedders/yamnet_aves/embedder.py` reads only wav2vec2's
final layer (`layer_outputs[-1]`). For SSL encoders the transfer literature
puts the most transferable information in *middle* layers, not the layer
shaped by the pretraining task; AVES is wav2vec2 (SSL, 12 layers) reading
layer 12 only for a non-bird target. Mean-pool layers 6, 9 and 12 (indices 5,
8, 11) instead, letting the probe's own weights decide how much of each to
use. Falsifier: if the learned per-block weight norm puts >=80% on the
layer-12 block and the headline is inside MDE, the final layer is the right
read and this closes for real.

## Changes

New embedder `embedders/yamnet_aves_mid/` (shared tree), a standalone copy of
`yamnet_aves/embedder.py` reading `layer_outputs[5]`, `layer_outputs[8]`,
`layer_outputs[11]` (mean-pooled each) instead of `layer_outputs[-1]` alone.
1024 YAMNet + 3*768 AVES = 3328-d. Unit-verified: YAMNet half and the
layer-12 block byte-identical to `yamnet_aves`'s own blocks, layers 6/9/12
genuinely differ from each other. `to_onnx()` verified against `embed()`
directly (max diff 5.5e-5, matching `yamnet_aves`'s own documented
tolerance). One `medium` extraction (reused the shared raw-audio cache keyed
by `framelength_s`/`samplerate`, same as `yamnet_aves` and `aves-p3`; no
crashes). Same pre-existing `1_99` workaround as prior AVES-family
extractions this session. Config matched to `yamnet-aves-verify`:
`--translation general --fixed-epochs 400`, no dropout.

**Same annotation-drift caveat as `aves-p3`:** `yamnet-aves-verify` was
extracted 2026-09-11; this run extracted fresh today (2026-09-14). Loudness
tagging is ongoing, so the two runs' buzz-event pools differ.

## Results

vs `yamnet-aves-verify`:

| fold | control sens@fpr0.005 | this run | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| **1_150** | 0.304 | **0.452** | **+0.148** | 0.049 | 14 |
| willard/1_11 | 0.444 | 0.527 | +0.083 | 0.026 | 14 |
| 1_29 | 0.465 | 0.537 | +0.072 | 0.027 | 14 |
| 1_95 | 0.098 | 0.114 | +0.016 | 0.015 | 23 |
| 1_143 | 0.487 | 0.486 | -0.001 | 0.034 | 9 |
| 1_114 | 0.156 | 0.154 | -0.002 | 0.017 | 14 |
| 53 | 0.493 | 0.479 | -0.014 | 0.023 | 9 |
| 1_37 | 0.382 | 0.339 | -0.043 | 0.031 | 5 |

mean sens@fpr0.005 (excl. quiet): 0.354 -> 0.386 (**+0.032 +/- 0.010**, ~3.2
sigma -- clears the ~0.027 headline MDE). Inclusive figure 0.283 -> 0.308
(+0.025). Tiers: `untagged` 0.380 -> 0.413 (+0.033), `background` 0.364 ->
0.395 (+0.031), `loud` flat (0.704 -> 0.704, thin tier), `quiet` flat.

## Conclusion

**Both named hard folds move up, and `1_150` moves by far the largest single
representation-change gain logged on it this era: +0.148 +/- 0.049 (~3
sigma).** `willard`/`1_11` and `1_29` also gain strongly (+0.083, +0.072,
both ~2.7-3.2 sigma); `1_37` is the one clear loser (-0.043, ~1.4 sigma). The
`untagged`/`background` tiers move together, not `loud`/`quiet`, so this
reads as a genuine detection gain on the tiers a middle-layer read would be
expected to help, not a quiet-buzz-only artifact.

This is the single largest headline-clearing gain from an AVES-side change
this era (headline +0.032 +/- 0.010, ~3.2 sigma), and it lands squarely on
`1_150` -- the target this project exists to move. **Not yet trusted as
`clean`**: no repeat draw yet, given the size and that `1_150` is exactly the
fold LOOP.md's hard-fold-gain protocol asks to confirm before trusting.
Logging this run `caveated` and launching `aves-mid-repeat` immediately.
