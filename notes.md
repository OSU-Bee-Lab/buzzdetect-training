# perch-pitchshift-concat

## Hypothesis

Follow-up to `perch-centred` (this batch, 2026-09-23): centred Perch on a
real 5.0 s window per 0.96 s frame clears its pre-registered falsifier
decisively -- `1_95` (jet) +0.180 +/- 0.048 (~3.75σ), `1_114` (trill) +0.294
+/- 0.101 (~2.9σ), the largest single-fold gains logged on either fold this
era by any lever -- but three rich folds (`1_29` -0.209, `1_143` -0.252,
`1_11` -0.118) collapse, netting a flat headline (0.329 -> 0.313). This
inverts the era's standing context/averaging prior exactly: context levers
have landed rich-folds-only, four for four; centred Perch lands
hard-folds-only, on the two specific confusions it was built for.

If that inversion means the two mechanisms carry genuinely separable
signal -- Perch resolving the aircraft/trill confusions, YAMNet's pitch-shift
concat carrying the era's confirmed general sensitivity lever (+0.069,
`yamnet-pitchshift`) -- concatenating them should keep both folds' gains
rather than trading one for the other. If instead they compete for the same
linear capacity (interference, not addition), the concat will land between
the two parents rather than above both.

New embedder `perch_centred_pitchshift` (3584-d): `[yamnet(t) 1024,
yamnet(octave-up t) 1024, perch_centred(t) 1536]`, frozen, no fine-tuning --
matching perch-centred's own regime and yamnet_pitchshift's confirmed frozen
lever, not the (separate, more expensive) trunk fine-tuning line.

**Falsifier:** both `1_95` and `1_114` must sit at or above `perch-centred`'s
own deltas (within noise) for this to be a real stack, not a wash. If either
hard fold reverts toward the `yamnet_pitchshift`-alone level, the two
mechanisms interfere rather than add. Rich-fold recovery is a bonus, not the
test -- `perch-centred`'s design already showed Perch's window costs them;
recovering `1_29`/`1_143`/`1_11` back toward baseline would be the more
surprising, stronger result but isn't required to call this a win.

**Comparators:** `perch-centred` (this batch) and `yamnet-pitchshift`
(archived era-confirmed +0.069 lever, but from 9+ days earlier and possibly
stale against the current annotation snapshot -- read with that caveat) and
`cv-baseline-v3-refresh`.

## Changes

New embedder `embedders/perch_centred_pitchshift/embedder.py`: composes (not
subclasses) a held `EmbedderPerchCentred` instance for the Perch half
(avoids `self.n_embeddings`/`self.embed` polymorphic-binding bugs a
subclass would hit calling `super().embed()`), plus YAMNet's own
`yamnet.keras` loaded directly for the plain + octave-up pair,
downsampling each 0.96 s frame from Perch's native 32 kHz to YAMNet's native
16 kHz before the existing `_pitch_up_octave` mechanism (identical to
`yamnet_pitchshift`/`yamnet_trunk_pitchshift`, just at a different
intermediate sample rate). `context_frames=3`, inherited from
`perch_centred`'s own requirement. Smoke-tested: Perch half is byte-identical
to standalone `perch_centred` on the same input (max abs diff 0.0); YAMNet
half produces sane nonzero ReLU-like activations.
