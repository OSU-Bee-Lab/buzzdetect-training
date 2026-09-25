# aves-p3

## Hypothesis

IDEAS.md item 3, re-aimed 2026-09-11 at `untagged`/`loud` sensitivity (quiet-
buzz recall is out of the headline). `embedders/yamnet_aves/embedder.py`
mean-pools wav2vec2's ~49 tokens per 1.0 s frame before the probe ever sees
it -- a matched filter for a stationary signal and the worst pooling for a
transient one. Replace the mean with `[mean, max, std]` over the token axis:
1024 YAMNet + 3*768 AVES = 3328-d. Falsifier: if `1_150` does not move above
its own bootstrap SD in the right direction, the dilution story is wrong and
the extra dims are capacity, not signal.

## Changes

New embedder `embedders/yamnet_aves_p3/` (shared tree), a standalone copy of
`yamnet_aves/embedder.py` with the AVES half's `.mean(dim=1)` replaced by
`torch.cat([mean, amax, std], dim=1)`, plus a matching `to_onnx()` (the same
hand-built YAMNet+AVES bridge, only the AVES-side pooling wrapper differs).
Unit-verified: YAMNet half and AVES mean-block byte-identical to
`yamnet_aves`'s own blocks, max/std blocks differ from the mean block, and
`to_onnx()` matches `embed()` directly (max diff 5.5e-5 -- the same CPU/GPU
reduction-order noise `yamnet_aves` itself documents). One `medium`
extraction, `--workers 0 BUZZDETECT_NO_GPU=1` (AVES-GPU convention): 1528 s --
faster than the ~1.0 h estimate because the raw-audio cache (keyed by
`framelength_s`/`samplerate`, shared with `yamnet_aves`) was already built, so
this run only computed new embeddings, no re-decode. No crashes. Config
matched to `yamnet-aves-verify`: `--translation general --fixed-epochs 400`,
no dropout.

**Caveat that applies to the comparison, not the treatment:** `yamnet-aves-
verify` was extracted 2026-09-11; this run extracted fresh today (2026-09-14)
against the current `annotations.csv`. Loudness tagging is ongoing, so the two
runs' buzz-event pools differ (`buzz_events_exclquiet` per fold: 5-23 here vs
`yamnet-aves-verify`'s own count at the time), not just the treatment. Read
the delta as approximate for that reason.

## Results

vs `yamnet-aves-verify`:

| fold | control sens@fpr0.005 | this run | delta | delta SD (eval sampling) | buzz events |
|---|---|---|---|---|---|
| willard/1_11 | 0.444 | 0.478 | +0.034 | 0.020 | 14 |
| wooster/1_143 | 0.487 | 0.516 | +0.029 | 0.042 | 9 |
| 1_37 | 0.382 | 0.406 | +0.024 | 0.032 | 5 |
| 1_150 | 0.304 | 0.327 | +0.023 | 0.029 | 14 |
| 1_29 | 0.465 | 0.479 | +0.014 | 0.010 | 14 |
| 1_114 | 0.156 | 0.160 | +0.004 | 0.013 | 14 |
| 53 | 0.493 | 0.487 | -0.006 | 0.015 | 9 |
| 1_95 | 0.098 | 0.087 | -0.011 | 0.012 | 23 |

mean sens@fpr0.005 (excl. quiet): 0.354 -> 0.368 (**+0.014 +/- 0.008**, ~1.75
sigma). Inclusive figure 0.283 -> 0.291 (+0.008). 6/8 folds up, both down-
movers small (`53` -0.006, `1_95` -0.011).

Tiers: `loud` 0.704 -> 0.709 (+0.005), `untagged` 0.380 -> 0.396 (+0.016),
`background` 0.364 -> 0.364 (flat), `quiet` 0.063 -> 0.062 (flat), `faint`
0.25 -> 0.0 (thin tier, a handful of frames, not meaningful).

## Conclusion

**A small, broad, right-direction signal, not a clean pass of its own
falsifier.** `untagged`/`loud` both move up modestly, so this is not a quiet-
buzz-only shuffle -- the re-aimed hypothesis' target tiers do move. But
`1_150`'s movement (+0.023 +/- 0.029) sits *below* its own delta SD, not above
it, so the falsifier's specific bar ("moves above its bootstrap SD") is not
cleanly cleared -- the dilution story is weakly supported, not confirmed.
`1_95` (the other named hard fold) moves the wrong way, though inside its own
SD (~0.9 sigma). Given the annotation-drift confound on the comparator noted
above, this result should be read as **caveated**: probably a small real gain
from richer token-axis pooling, broadly distributed rather than concentrated
on one hard fold, but neither the size nor (for `1_150` specifically) the
mechanism is confirmed. A repeat draw against a same-day `yamnet_aves` control
would be the right next step before trusting the magnitude; a same-arm offline
`[mean, max]`-only ablation (no `std`) would separate whether `std` is
earning its third of the added dimensions.

Settles the falsifier check for item 3 as "inconclusive, weakly positive" --
worth a repeat, not yet a clean structural win.
