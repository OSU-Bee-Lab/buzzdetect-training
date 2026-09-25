# shift-ratio-ladder

## Hypothesis

IDEAS.md item 24. `yamnet_pitchshift`'s x2 rung (+0.069, 8/8 folds, this
era's largest clean gain) and its seamless-decimated form
(`yamnet_pitchshift_decimate`, run as `pitchshift-decimate-up` /
`-repeat`: headline flat vs the tiled x2, but `1_95`/`1_114` up both
draws) bracket ratio 2. Nothing between 1 and 2 has been tried. Bee flight
fundamentals span roughly 130-250 Hz across bumble and honey bees, so a fixed
octave may overshoot for some taxa and undershoot for others.

Two rungs, each concatenated with the unshifted YAMNet block, 2048-d:

- **x1.26** (`yamnet_pitchshift_decimate_x126`, a minor third, 2**(4/12))
- **x1.5** (`yamnet_pitchshift_decimate_x150`, a perfect fifth)

Both built in the same seamless decimated form as `yamnet_pitchshift_decimate`
(no tiling, no seam): `context_frames = 1` supplies one real look-ahead frame
(1.92 s total per row), of which only the leading `round(ratio *
frame_samples)` samples are used and resampled down to one frame's width.

*Control:* `yamnet_pitchshift_decimate` (`pitchshift-decimate-up-repeat`,
0.393 / baseline `yamnet-pitchshift` 0.398) — the seamless x2 rung, not the
tiled `yamnet_pitchshift`.

*Falsifier:* if neither rung beats x2 on the headline and the fold signatures
match x2's, the octave is the right shift — stop there, do not run a
multi-ratio concat. If a rung wins or shows a different fold signature (e.g.
`1_114` up), the follow-up is `[x_best, x2]` concatenated, and only then the
swap into the lead.

## Changes

Two new embedders committed straight to main's shared tree (per LOOP.md: "a
new embedder directory... goes straight into the shared tree"), commit
`3ad01e9`:

- `embedders/yamnet_pitchshift_decimate_x126/embedder.py`
- `embedders/yamnet_pitchshift_decimate_x150/embedder.py`

Both are copies of `yamnet_pitchshift_decimate`'s embedder generalized to an
arbitrary ratio (`RATIO` constant), reading only as much of the real
look-ahead frame as the ratio needs rather than always the full second frame.
`x126` was smoke tested end-to-end on the `lite` set (clean run, 2048-d
output, no errors). `x150` was unit tested (`embed()` on synthetic audio,
correct output shape) since the code is otherwise identical.

No changes to `03_train` or any training code — this is YAMNet-only, same
architecture as every other `yamnet_pitchshift*` run.

## Results

Both rungs extracted and CV'd (8 rotating folds, `--fixed-epochs 400`,
`--translation general`). x126's CV training OOM'd on GPU 3/8 folds in
(`Dst tensor is not initialized`, the 4 GB card's TF memory limit under
`from_tensor_slices`) and was relaunched with `--cpu`, resuming cleanly from
fold 4; x150 ran on `--cpu` throughout. Neither is a measurement defect —
noted for the record since a GPU retry mid-run is unusual.

Comparator: `pitchshift-decimate-up-repeat` (headline 0.393), the seamless x2
rung, per the pre-registered control.

**x1.26** (`tools/results.py` vs control):

| fold | control | x126 | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.462 | 0.538 | +0.076 | 0.044 | 32 |
| 53 | 0.431 | 0.396 | -0.035 | 0.022 | 28 |
| 1_11 | 0.430 | 0.423 | -0.007 | 0.029 | 26 |
| 1_143 | 0.470 | 0.486 | +0.016 | 0.045 | 22 |
| 1_150 | 0.329 | 0.229 | **-0.100** | 0.054 | 21 |
| 1_95 | 0.100 | 0.101 | +0.001 | 0.033 | 46 |
| 1_37 | 0.448 | 0.462 | +0.014 | 0.031 | 14 |
| 1_114 | 0.474 | 0.273 | **-0.201** | 0.083 | 28 |

- headline (`sensitivity_exclquiet`): 0.393 → **0.364 (-0.029 ± 0.016)**
- tiers: untagged -0.033, background +0.030, loud -0.008 — no tier moved up
  cleanly; this is not a detection gain by the standing-facts test.

**x1.5:**

| fold | control | x150 | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.462 | 0.484 | +0.022 | 0.029 | 32 |
| 53 | 0.431 | 0.411 | -0.020 | 0.021 | 28 |
| 1_11 | 0.430 | 0.416 | -0.014 | 0.036 | 26 |
| 1_143 | 0.470 | 0.495 | +0.025 | 0.026 | 22 |
| 1_150 | 0.329 | 0.238 | **-0.091** | 0.051 | 21 |
| 1_95 | 0.100 | 0.159 | +0.059 | 0.032 | 46 |
| 1_37 | 0.448 | 0.460 | +0.012 | 0.030 | 14 |
| 1_114 | 0.474 | 0.510 | +0.036 | 0.035 | 28 |

- headline: 0.393 → **0.397 (+0.004 ± 0.012)** — flat, well inside the ~0.014
  combined (eval + training-stochasticity) SD for a single-run comparison.
- tiers: untagged +0.005, background +0.000, loud -0.017, faint +0.104 (12
  frames — noise, not read as signal).

## Conclusion

**Falsifier fired: neither rung beats x2, and neither shows a fold signature
clearly better than x2's own known pattern.** x1.26 is an outright worse
headline (-0.029 ± 0.016, ~1.8σ) with both hard folds (`1_150` -0.100,
`1_114` -0.201) down 1.8-2.4σ — the clearer of the two negatives. x1.5 is flat
on the headline (+0.004 ± 0.012, unreadable from one run) with a mixed
per-fold picture: `1_150` down about as much as x1.26's (-0.091), `1_95` up
+0.059 (~1.8σ, the one fold worth a second look but not on its own) and
`1_114` up a small, unresolved +0.036. Nothing here clears the ~0.027 MDE in
the direction that would justify a concat follow-up.

**Per the pre-registered falsifier, this closes item 24: the octave (x2) is
the right shift, and no multi-ratio concat is warranted.** `1_150` losing
ground at both intermediate ratios (and at a similar size to each other) is
the one consistent thread across both rungs, worth flagging for whoever next
touches the pitch-shift family, but it's a single hard fold on 21 events and
not a reason to reopen this ladder.

`x150` is reported as the primary model (closer to the control, less negative)
with `x126`'s result folded into this conclusion — see HANDOFF.md's framing,
this is one experiment covering both rungs.
