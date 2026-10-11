# ps-d8-aves
## Hypothesis
Training arm. One-off test under LOOP.md's AVES injunction, allowed by Luke on 2026-10-11 ("I'll allow
your cheap test this once"). A frozen AVES block next to YAMNet features was a small, repeated gain on
frozen probes in closed eras (`yamnet-aves-verify` +0.024, `aves-mid` +0.032 / +0.025,
`pitchshift-aves-mid` +0.043 / +0.032 over their controls), but it has never been tried on a fine-tuned
trunk, and fine-tuning has since passed it without AVES (`ps-depth8` ~0.50 against a 0.324 baseline).
The question is whether AVES still adds information once the YAMNet tail is trained on the task.

Embedder `yamnet_trunk_pitchshift_depth8_aves`: ps-depth8's two trunk views (bit-identical to
`yamnet_trunk_pitchshift_depth8` on a test clip) plus AVES layers 6, 9, 12 mean-pooled over the same
0.96 s frame (2304 columns, frozen); the class Dense reads [2048 tail code, 2304 AVES]. Everything else
is the ps-depth8 recipe.

Prediction: headline above the same-recipe control `ps-depth8-r3` (0.497; three draws 0.497-0.515) by
more than the headline delta's SD (~0.014), with `untagged`/`loud` up. Falsified if it lands within
~0.014 of the control, or if the gain is only on `1_29`/`53` background drones. `1_114` is the fold to
watch: adding AVES layers to pitch-shift took it down 0.204 in era v3.

Caveat written before the run: the head is ps-depth8's (Dense at lr 2e-4, 30 epochs), tuned for a sparse
non-negative YAMNet code; AVES columns are dense and signed. A null here gets one offline readout sweep
on the cached AVES columns before it is logged as an embedder verdict (LOOP.md, "A new embedder doesn't
inherit the probe config").
