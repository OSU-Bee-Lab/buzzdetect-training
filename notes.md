# decimate-lead

## Hypothesis

IDEAS item 19. The era lead (`pitchshift-aves-mid`) uses the **tiled** octave
shift (0.48 s of real audio played twice, hard seam). `pitchshift-decimate-up`
(+repeat) showed the seamless **decimated** shift moves `1_95` (+0.052/+0.024)
and `1_114` (+0.034/+0.043) -- two of the lead's three FP-limited folds --
while trading away rich-fold sensitivity that `aves-mid` supplies back. This
embedder (`yamnet_pitchshift_decimate_aves_mid`) swaps the lead's shifted
block for the decimated form, one variable against `yamnet_pitchshift_aves_mid`.

Falsifier (from IDEAS.md item 19): `1_114` and `1_95` must both rise above the
control on the run, with the headline no worse than -0.027 (one MDE).

## Changes

New embedder `embedders/yamnet_pitchshift_decimate_aves_mid/embedder.py`
(committed to main, shared tree): `[yamnet(crop_t),
yamnet(decimate(crop_t..crop_t+30720)), aves_layer6/9/12(frame_t)]`, 4352-d,
`framelength_s=1.0`, `context_frames=1`. Unit-checked against
`yamnet_pitchshift_aves_mid` on synthetic audio: unshifted and AVES blocks are
byte-identical (max diff 0.0), only the decimated-shifted block differs, as
expected.

## Results

(pending)

## Conclusion

(pending)
