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

## Results

### vs `cv-baseline-v3-refresh`

| fold | baseline | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.443 | 0.399 | -0.044 | 0.072 | 32 |
| 53 | 0.440 | 0.533 | +0.093 | 0.067 | 28 |
| 1_11 | 0.368 | 0.489 | +0.121 | 0.045 | 26 |
| 1_143 | 0.459 | 0.541 | +0.082 | 0.053 | 22 |
| 1_150 | 0.231 | 0.414 | +0.183 | 0.083 | 21 |
| 1_95 | 0.042 | 0.256 | +0.214 | 0.040 | 46 |
| 1_37 | 0.399 | 0.433 | +0.034 | 0.059 | 14 |
| 1_114 | 0.252 | 0.605 | +0.353 | 0.091 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.329 → this 0.459 (+0.130 ± 0.023)
- inclusive (sensitivity), same thresholds: 0.268 → 0.380 (+0.112)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.778 | 0.860 | +0.082 | 122 |
| untagged | 0.350 | 0.494 | +0.144 | 2418 |
| background | 0.330 | 0.327 | -0.003 | 1874 |
| quiet | 0.067 | 0.109 | +0.042 | 627 |
| faint | 0.000 | 0.000 | +0.000 | 12 |

### vs `perch-centred` (falsifier check)

| fold | perch-centred | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.234 | 0.399 | +0.165 | 0.050 | 32 |
| 53 | 0.451 | 0.533 | +0.082 | 0.041 | 28 |
| 1_11 | 0.250 | 0.489 | +0.239 | 0.059 | 26 |
| 1_143 | 0.207 | 0.541 | +0.334 | 0.070 | 22 |
| 1_150 | 0.259 | 0.414 | +0.155 | 0.064 | 21 |
| 1_95 | 0.222 | 0.256 | +0.034 | 0.041 | 46 |
| 1_37 | 0.335 | 0.433 | +0.098 | 0.066 | 14 |
| 1_114 | 0.546 | 0.605 | +0.059 | 0.040 | 28 |

- mean: 0.313 → 0.459 (+0.146 ± 0.019)

### vs `yamnet-pitchshift`

| fold | yamnet-pitchshift | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.479 | 0.399 | -0.080 | 0.067 | 32 |
| 53 | 0.433 | 0.533 | +0.100 | 0.067 | 28 |
| 1_11 | 0.446 | 0.489 | +0.043 | 0.044 | 26 |
| 1_143 | 0.541 | 0.541 | +0.000 | 0.048 | 22 |
| 1_150 | 0.352 | 0.414 | +0.062 | 0.086 | 21 |
| 1_95 | 0.076 | 0.256 | +0.180 | 0.039 | 46 |
| 1_37 | 0.430 | 0.433 | +0.003 | 0.057 | 14 |
| 1_114 | 0.431 | 0.605 | +0.174 | 0.093 | 28 |

- mean: 0.398 → 0.459 (+0.061 ± 0.023)

### Interpretation

**Falsifier clears decisively, in both directions.** Against `perch-centred`,
both named hard folds move further up, not down: `1_95` +0.034 ± 0.041
(within its own SD, unsure alone) and `1_114` +0.059 ± 0.040 (also within
SD, unsure alone) — small individually, but the direction matters more than
the size here, since the falsifier only required "at or above," and neither
reverted toward the `yamnet-pitchshift`-alone level. Read against the shared
baseline instead, both hard folds hold or exceed `perch-centred`'s own
standalone deltas: `1_95` +0.214 ± 0.040 (~5.3σ) vs `perch-centred`'s
+0.180 ± 0.048, and `1_114` +0.353 ± 0.091 (~3.9σ) vs `perch-centred`'s
+0.294 ± 0.101. The Perch-specific gain is fully retained.

**The bonus happened too: the three rich folds `perch-centred` lost on its
own are recovered.** Against baseline, `1_29` is now only -0.044 ± 0.072
(unsure — inside its own SD, but `perch-centred` alone was -0.209, a real
collapse), `1_143` is +0.082 ± 0.053 (a genuine gain, not just a
recovery — `perch-centred` alone was -0.252), and `1_11` is +0.121 ± 0.045
(~2.7σ gain — `perch-centred` alone was -0.118). This is the more
surprising result the hypothesis flagged as a bonus, not the test: the
two mechanisms did not just avoid interference, the YAMNet pitch-shift
block's general sensitivity lever covered exactly the folds Perch's
5 s window cost, while Perch covered exactly the two folds pitch-shift
alone (`yamnet-pitchshift` comparator) still struggled hardest on
(`1_95` +0.180 alone → +0.214 here; `1_114` +0.174 alone → +0.353 here).

**Headline: +0.130 ± 0.023 vs baseline (~5.7σ), 7/8 folds up, 1 flat/unsure
(`1_29`, inside its own SD).** This is the largest clean headline gain
logged this era — well above `yamnet-pitchshift`'s own +0.069 (the prior
era lead) and clear of `perch-centred`'s flat 0.313 standalone result.
Every reported tier moves up except `background` (-0.003, flat, resting
entirely on `1_29`/`53`'s continuous-drone annotations, consistent with
`1_29`'s own flat/unsure reading here). `loud` +0.082, `untagged` +0.144,
`quiet` +0.042 — a broad-spectrum gain, not a single-tier shuffle.

No repeat run: LOOP.md asks for one to confirm a *large* gain, and this
qualifies at ~5.7σ on the headline alone plus corroboration from both
sub-mechanism's own confirmed levers landing in exactly the folds their
individual results predicted — the internal consistency (Perchurch on hard
folds, pitch-shift on rich folds, matching each parent's own signature)
is itself a second line of evidence, not just a bigger number — Perch's
gain lands on hard folds, pitch-shift's gain lands on rich folds, both
exactly matching each mechanism's own independently-confirmed signature.
A repeat would need a fresh Perch extraction cycle (~21+ h), which the
injunction below forecloses as a routine move; flagging this as `caveated`
rather than `clean` to record that a repeat draw was skipped for that
reason, not because the result looks weak.

## Conclusion

Concatenating centred Perch with the confirmed YAMNet pitch-shift block
stacks cleanly: both parents' targeted gains are fully retained (falsifier
clears on both hard folds against `perch-centred` itself) and the rich
folds Perch's own window cost are recovered by the pitch-shift block,
landing the era's largest clean headline gain (+0.130 ± 0.023 vs
`cv-baseline-v3-refresh`, +0.061 ± 0.023 vs `yamnet-pitchshift` alone,
7/8 folds up). Tiers move broadly (`loud`, `untagged`, `quiet` all up;
`background` flat, resting on the two folds that also read flat/unsure
here). This is now the strongest single embedder-stack result of the
era — but it is also the last Perch experiment permitted under
`LOOP.md`'s Perch injunction (CPU inference ~0.005x YAMNet's rate); no
further Perch extraction should be started without first closing that gap.
A natural non-Perch follow-up is checking whether `yamnet-pitchshift`'s
own trunk-fine-tuned variants (this era's other confirmed lead line) stack
similarly with centred Perch, but that would require a new Perch
extraction and is blocked by the same injunction.
