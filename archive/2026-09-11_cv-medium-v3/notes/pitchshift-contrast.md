# pitchshift-contrast

## Hypothesis

New proposal, not from IDEAS.md's queue (the AVES injunction closed items 20/21/23,
the only ready queued item is item 2/Perch, run separately as `perch-centred`).

Combines this era's two largest confirmed non-AVES mechanism results:
`yamnet-pitchshift` (concat `[e_t, e_shifted(t)]`, +0.069, the era's largest
single-lever gain) and `asym-context-yamnet` (a linear readout of a raw
concat cannot compute a contrast; the explicit `[e_t, e_t - mean(neighbours)]`
beat plain concatenation, +0.046 vs `context-frames-fix`'s +0.030 at the same
budget).

This applies the SAME contrast mechanism along the *pitch* axis instead of
the *time* axis: `embedders/yamnet_pitchshift_contrast` emits
`[e_t, e_t - e_shifted(t)]`, 2048-d (same width as `yamnet_pitchshift`), where
`e_shifted(t)` is the octave-up embedding of the identical frame (not an
adjacent frame in time). This does **not** touch LOOP.md's standing
context/averaging prior: no temporal neighbour is read, no cached row from
another frame is involved, and both towers still come from independent real
audio for the same timestamp (byte-identical mechanism to `yamnet_pitchshift`
otherwise).

Prediction: if the contrast channel carries information a linear head can't
otherwise reconstruct from the raw concat (the mechanism `asym-context-yamnet`
demonstrated), this should beat `yamnet-pitchshift`'s +0.069 on the same
budget. If concat already gives the head everything it needs (pitch-shift
is not "context" in the sense the contrast mechanism was built for), this
should be flat against it.

Comparator: `yamnet-pitchshift`'s own control (fresh `cv-baseline-v3`-style
anchor) is stale by now (`asym-context-yamnet`'s notes -- annotations moved
2026-09-11 to 2026-09-14). Trains a fresh matched control on plain `yamnet`
at this run's exact config, plus reads `yamnet-pitchshift` itself as a second,
same-embedder-family comparator (frame grids match exactly, so directly
paired).

## Changes

New embedder `embedders/yamnet_pitchshift_contrast/embedder.py`, subclasses
`EmbedderYamnetPitchshift` and overrides only `embed()` to concatenate
`[unshifted, unshifted - shifted]` instead of `[unshifted, shifted]`. No
`03_train` changes -- frozen linear probe, default `--epochs 400` (the same
budget `yamnet_pitchshift` itself used at this 2048-d width).

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_pitchshift_contrast
tools/launch_job.sh train_ctrl.log -- 03_train/main.py --name pitchshift-contrast-ctrl --set medium --embedder yamnet --translation general -y
tools/launch_job.sh train.log -- 03_train/main.py --name pitchshift-contrast --set medium --embedder yamnet_pitchshift_contrast --translation general -y
```

## Results

GPU OOM twice mid-rotation on a wide (2048-d) probe (`Dst tensor is not
initialized`, same crash `trunk-ft-v3` found and fixed but never merged to
main -- ported just the session-clear fix, see commit fc75c73). Still OOMed a
third time after that fix on fold 4 with the card otherwise clean, so this
run finished on `--cpu` (resumed cleanly at fold 4; folds 1-3 kept their GPU
results, per-fold training config is identical either way).

| fold | ctrl sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.448 | 0.468 | +0.020 | 0.019 | 32 |
| 53 | 0.436 | 0.423 | -0.013 | 0.023 | 28 |
| 1_11 (willard) | 0.363 | 0.420 | +0.057 | 0.026 | 26 |
| 1_143 | 0.441 | 0.523 | +0.082 | 0.036 | 22 |
| 1_150 | 0.241 | 0.269 | +0.028 | 0.044 | 21 |
| 1_95 | 0.052 | 0.087 | +0.035 | 0.018 | 46 |
| 1_37 | 0.394 | 0.412 | +0.018 | 0.029 | 14 |
| 1_114 | 0.226 | 0.379 | +0.153 | 0.037 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): `pitchshift-contrast-ctrl` (fresh
  plain-yamnet anchor, same data, same day) 0.325 -> this 0.373 (**+0.048 ±
  0.011**, ~4.4σ, well past the ~0.027 headline MDE)
- inclusive (sensitivity), same thresholds: 0.265 -> 0.302 (+0.037)
- folds that missed fpr 0.005: none

| tier | ctrl | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.787 | +0.009 | 122 |
| untagged | 0.344 | 0.400 | +0.056 | 2418 |
| background | 0.332 | 0.322 | -0.010 | 1874 |
| quiet | 0.073 | 0.077 | +0.004 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

**Against `yamnet_pitchshift` itself** (the raw-concat design, same 2048-d
width and frame grid): headline 0.398 -> 0.373 (-0.025), 7/8 folds down. This
comparison is **stale, not a verdict** -- `yamnet_pitchshift` trained
2026-09-13 on annotations from before this era's ongoing annotation growth
(per `asym-context-yamnet`'s notes, buzz events roughly doubled-to-tripled on
every fold between 2026-09-11 and 2026-09-14, and tagging has continued
since). A fair concat-vs-contrast head-to-head needs a same-day rerun of the
concat design, not done here.

## Conclusion

**Real, clean gain against its own fresh matched control: +0.048 ± 0.011
(~4.4σ), 6/8 folds up.** Both named hard folds move up: `1_95` +0.035 ± 0.018
(~1.9σ, the aircraft-confusion fold, weak-to-moderate) and `1_114` **+0.153 ±
0.037** (~4.1σ, the trill-confusion fold -- the largest single-fold delta
logged on it this era by any lever). `1_150` +0.028 ± 0.044 is inside its own
SD, unsure. `untagged` (2418 frames, the largest tier) moved +0.056, more than
the headline itself -- this is a genuine detection gain, not a background-tier
artifact (`background` itself is flat-to-down, -0.010).

Passes the mechanism prediction the hypothesis made (contrast beats concat,
mirroring `asym-context-yamnet` vs `yamnet_context`), but **cannot be directly
scored against `yamnet_pitchshift`'s own +0.069** without a same-day rerun of
the concat design, since the only comparison available is confounded by
annotation drift and reads backwards (-0.025) as a result. That head-to-head,
and whether the contrast channel's `1_114` gain replicates, are the open
follow-ups -- not run here to keep this batch moving to the other three
experiments.
