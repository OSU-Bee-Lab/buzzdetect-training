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

vs `pitchshift-aves-mid` (the tiled-shift lead, main comparator):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.551 | 0.563 | +0.012 | 0.019 | 32 |
| 53 | 0.484 | 0.473 | -0.011 | 0.020 | 25 |
| 1_11 | 0.515 | 0.495 | -0.020 | 0.022 | 28 |
| 1_143 | 0.571 | 0.514 | -0.057 | 0.040 | 21 |
| 1_150 | 0.468 | 0.404 | -0.064 | 0.038 | 20 |
| 1_95 | 0.170 | 0.154 | -0.016 | 0.019 | 42 |
| 1_37 | 0.410 | 0.414 | +0.004 | 0.025 | 14 |
| 1_114 | 0.210 | 0.266 | +0.056 | 0.022 | 31 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.422 → this 0.410 (-0.012 ± 0.009)
- inclusive (sensitivity), same thresholds: 0.336 → 0.330 (-0.006)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.712 | 0.780 | +0.068 | 115 |
| untagged | 0.453 | 0.437 | -0.016 | 2317 |
| background | 0.401 | 0.404 | +0.003 | 1804 |
| quiet | 0.073 | 0.075 | +0.002 | 599 |
| faint | 0.000 | 0.000 | +0.000 | 13 |

The optional secondary comparator (`yamnet_pitchshift_decimate` as a trained
model) doesn't exist — grepped `log.jsonl`, nothing was ever trained under
that name; the closest thing, `pitchshift-decimate-up`, decimates a different
signal (the context frame, not the embedder's own shift block) and isn't a
valid stand-in. Skipped; only the main comparator is reported.

**Falsifier check** (IDEAS item 19: `1_114` and `1_95` must both rise above
the control, headline no worse than -0.027): **fires**. `1_114` rises clearly
(+0.056 ± 0.022, ~2.5σ) — a genuine surprise against
`diagnostics/2026-09-14_trill_1_114/README.md`'s expectation that the
decimated shift's marginal raw-separation edge (0.044 vs 0.049) would be too
small to survive concatenation with AVES; it evidently did survive here. But
`1_95` does not rise (-0.016 ± 0.019, <1σ, indistinguishable from flat) — the
joint condition the hypothesis was built on fails on this fold alone, so no
repeat draw is needed to settle it (IDEAS.md's own stricter "both draws"
wording only tightens a failure that's already unconditional here). Headline
stays inside the -0.027 floor (-0.012 ± 0.009, ~1.3σ, not distinguishable from
zero) — not a broad win, but not the trade-off's worst case either.

The two rich folds this hypothesis expected to give back (per
`pitchshift-decimate-up`'s own trade) do drop, `1_150` (-0.064 ± 0.038, ~1.7σ)
and `1_143` (-0.057 ± 0.040, ~1.4σ) — both near but not clearing 2σ on their
own; "unsure, leaning down" rather than confirmed, matching the anticipated
direction. The other four folds (1_29, 53, 1_11, 1_37) are flat, all within
1σ.

Tiers: `loud` moves up substantially (+0.068, but on a thin 115-frame tier);
`untagged` — the tier that would show a `1_114`-shaped detection gain, since
that fold has no loud/quiet tags — moves slightly *down* (-0.016) despite
`1_114`'s own large gain, meaning the drop is coming from elsewhere in that
tier (consistent with `1_150`/`1_143` both being untagged-heavy folds and
both dropping). `background`/`quiet`/`faint` are flat. Not a quiet-buzz-only
artifact, but not a clean broad gain either — a lateral trade, same shape as
`pitchshift-decimate-up`'s own result.

## Conclusion

Falsifier fires: the decimated shift does not rescue `1_95` inside the
`aves-mid` stack, so this is **not** the joint 1_95+1_114 win IDEAS item 19
was betting on. `1_114` alone moves solidly (+0.056, ~2.5σ) — a real,
representation-level surprise against the trill-confusion diagnostic's
prediction, worth a note for item 20b (block weighting) rather than further
shift variants — but it comes packaged with the same rich-fold cost
(`1_150`, `1_143`, both down, near-2σ) that `pitchshift-decimate-up` already
showed, and the headline is flat-to-slightly-down (-0.012 ± 0.009), not a
detection gain. Trust: clean — one draw is sufficient here since the
falsifier's fold-level condition already fails outright on `1_95`, independent
of any repeat variance. Settles IDEAS item 19: the tiled-vs-decimated shift
axis alone does not fix `1_95`; whatever moved `1_95` in `pitchshift-decimate-up`
doesn't survive concatenation with AVES the way `1_114` does. `1_150`/`1_143`'s
near-2σ drops here (not clearly confirmed, not clearly noise) are worth a
repeat only if a future item specifically needs a trusted magnitude on them.
