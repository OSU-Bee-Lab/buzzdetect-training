# yamnet-pitchshift

## Hypothesis

IDEAS.md item 5: honey-bee flight (~230 Hz + harmonics) sits at the bottom
edge of YAMNet's mel filterbank (starts at 125 Hz, near-linear/coarse below
1 kHz), where AudioSet's training mass is thin. Pitch-shifting each frame up
an octave before a second YAMNet pass moves the fundamental and harmonics to
460/920/1380 Hz, where the filterbank is denser and AudioSet has more mass.
Concatenated with the unshifted block (not a replacement -- `yamnet-bandpass`
-0.047 and `yamnet-mask` -0.143 in E1 are why), this should be a pure
detection-band lever with no capacity change.

## Changes

New embedder `embedders/yamnet_pitchshift/embedder.py` (2048-d = 1024
unshifted + 1024 shifted). `framelength_s` held at YAMNet's native 0.96 s --
`embedders/yamnet_doublerate/` was explicitly NOT used, since its
`framelength_s=0.48` changes the frame grid/`overlap_event_s` system-wide
(the confound that sank `perch-probe`). The shift is by resampling only (no
formant preservation): each 0.96 s frame is resampled 16 kHz -> 8 kHz
(halves the sample count, low-passing content above 4 kHz of the original
signal), the result is relabelled as 16 kHz (so playback takes half the real
time -> every frequency doubles), then tiled 2x back to a full 0.96 s patch
so YAMNet's patch_hop==patch_window==0.96s grid is untouched. Smoke-tested
standalone before extraction: the shifted embedding of a 230 Hz tone lands
closer to the unshifted embedding of a real 460 Hz tone than to itself
(0.074 vs 0.149 mean abs diff) -- the mechanism does what it claims.

Ran: `02_set/main.py --set medium --embedder yamnet_pitchshift --workers 1`
(fresh extraction, ~20 min, 82/82 idents; one empty fold,
`Luke - Various Opportunistic Recordings/2026-07-27/1_99`, has zero snip
audio for every embedder including the existing `yamnet_aves` cache -- its
raw dir just needed creating empty to match, no data was invented). Then
`03_train/main.py --name yamnet_pitchshift --set medium --embedder yamnet_pitchshift --translation general --verbose -y`
(default `--fixed-epochs 400`, `--dropout 0`, no hidden layer -- bare linear
probe, matched to the era anchor `cv_baseline_v3`).

## Results

Matched control: `cv_baseline_v3` (plain YAMNet, same fixed-epochs/dropout/no-hidden config).

| fold | control | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| `53` | 0.429 | 0.433 | +0.004 | ±0.031 | 11 |
| `1_95` (hard) | 0.052 | 0.076 | +0.024 | ±0.019 | 23 |
| `1_29` | 0.441 | 0.479 | +0.037 | ±0.018 | 13 |
| `1_37` | 0.369 | 0.430 | +0.061 | ±0.034 | 5 |
| willard (`1_11`) | 0.368 | 0.446 | +0.078 | ±0.022 | 12 |
| `1_143` | 0.461 | 0.541 | +0.080 | ±0.033 | 9 |
| `1_150` (hard) | 0.266 | 0.352 | +0.086 | ±0.050 | 14 |
| `1_114` | 0.250 | 0.431 | +0.181 | ±0.038 | 13 |

- mean sens@fpr0.005 excl-quiet: 0.330 -> 0.398 (**+0.069 ± 0.011** headline delta SD), **8/8 folds up**
- inclusive: 0.269 -> 0.325 (+0.056)

**Epoch-budget check** (this is a wider input, 2048-d vs 1024-d): pooled
`mean(last 21 epochs) - mean(e300-320)` on `val_sens_fpr0.005` --
pitchshift `[0.005, 0.0, 0.007, -0.002, 0.003, 0.007, -0.002, 0.0]` vs.
control `[0.008, 0.008, -0.002, -0.003, 0.001, -0.003, 0.004, 0.013]` -- both
flat, comparable magnitudes. Not an epoch-budget artifact.

**Falsifier** (IDEAS item 5): "if the shifted block's learned per-dim weight
norm is under ~0.7x the unshifted block's *and* the headline is inside MDE".
Not checked directly -- CV rotations don't save weights (`save_binary=False`)
-- but moot regardless: the headline delta (+0.069 ± 0.011) is ~2.5x the
~0.027 MDE, so the second half of the AND can't hold. If a weight-norm read
is wanted later it needs a `--skip-cv`/shipped-model run.

## Conclusion

**The largest single-run headline gain in the current era's log, and the
cleanest: 8/8 folds up, no exceptions, headline ~6x its own delta SD.** Both
hard folds move in the predicted-helpful direction: `1_150` +0.086 ± 0.050
(~1.7 sigma) and `1_95` +0.025 ± 0.019 (~1.3 sigma) -- weaker individually
than the headline but directionally consistent, and `1_95` moving *up* here
(where several other levers have failed on it) is worth a note even at this
significance. `1_114` (+0.181 ± 0.038, ~4.8 sigma) is the standout single
fold and the fold this era has no other mechanism for.

Mechanism support: the standalone smoke test showed the shift does what it
claims in embedding space (moves a 230 Hz tone's embedding toward a real
460 Hz tone's), and the epoch-budget check rules out the wider input simply
needing (or being unfairly denied) more training. This reads as a genuine
detection-band lever, not a capacity or budget artifact.

**Not yet run: the x4 rung.** IDEAS item 5 asked to also try a 4x pitch shift
if x2 is directionally positive -- it clearly is. Left as a small follow-up
(no extraction changes beyond adding a second resample ratio) rather than
run here, since this result is already unambiguous at x2.
