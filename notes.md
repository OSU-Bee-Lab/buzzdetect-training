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

(pending — see HANDOFF.md)

## Conclusion

(pending — see HANDOFF.md)
