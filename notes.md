# yamnet-aves-verify

## Hypothesis

Re-verify IDEAS.md item 1b — the E3 `yamnet-aves` result (YAMNet 1024-d concat
AVES 768-d, +0.025 honest over two draws, 4/5 folds up, the only intervention
ever to move `1_95`) — against the `cv-medium-v3` era anchor: fresh
`yamnet_aves` extraction under the current (2026-09-11) annotations, 8 rotating
folds, `--dropout 0`, `--fixed-epochs 400`, `sensitivity_exclquiet` headline.
No code changes — `yamnet_aves` is a standing embedder on `main`, re-extracted
fresh and trained under the anchor's exact config with embedder swapped.

## Changes

None to code. `02_set/main.py --embedder yamnet_aves` re-extracted all 79
idents whose annotations moved since the last `yamnet_aves` extraction (3 were
already current). `03_train/main.py --embedder yamnet_aves` under otherwise
default (anchor) flags.

## Results

| fold | baseline (`cv_baseline_v3`) | this exp | delta | val frames |
|---|---|---|---|---|
| 1_114 (new, no E3 precedent) | 0.250 | 0.156 | -0.094 | 5025 |
| 1_37 | 0.369 | 0.382 | +0.013 | 5033 |
| 1_29 | 0.441 | 0.465 | +0.024 | 7617 |
| 1_143 | 0.461 | 0.487 | +0.026 | 5024 |
| 1_150 (hard, low-SNR positives) | 0.266 | 0.304 | +0.038 | 5091 |
| **1_95 (hard, jet-flyover)** | **0.052** | **0.098** | **+0.046** | 7572 |
| 53 | 0.429 | 0.493 | +0.064 | 7540 |
| willard (1_11) | 0.368 | 0.444 | +0.076 | 7562 |

- mean sens@fpr0.005 excl-quiet: 0.330 -> 0.354 (+0.024), 7/8 folds up
- inclusive (incl. quiet): 0.269 -> 0.283 (+0.014)

## Conclusion

**Confirms the E3 lead at almost the same magnitude** (E3: +0.025 honest;
here: +0.024), on fresh annotations, 8 folds instead of 5, no dropout, and a
fixed 400-epoch budget instead of early stopping — a genuinely different
measurement setup landing on the same number is a strong result.

**`1_95` moved again** (+0.046, 0.052 -> 0.098), on its `untagged` column
(0.052 -> 0.098, since that fold carries no other tier). This is the fourth
draw (across two eras) in which `yamnet_aves` is the only intervention to
move that fold, and it is still near chance — but a repeatable +0.046 on a
fold nothing else touches is the strongest structural lead in the log.

**Tier reading is mostly a detection gain, with one caveat.** `untagged` (the
largest tier, audible-but-unclassified buzz) moved up in 7/8 folds carrying it
(only `1_114` down) and the `background` tier moved up in both folds that
carry it (`1_29` +0.011, `53` +0.071) — that is real detection gain on audible
buzz, not a quiet-buzz shuffle. `loud` moved **down** pooled (0.778 -> 0.704),
but that tier is thin (122 -> 115 frames total, concentrated in `53` and
`1_143`/`1_37`) and the per-fold moves are small (-0.032 to -0.081 on n=4-58
frames) — read as noise on a small tier, not a contradiction of the untagged
gain, but flagging it rather than hiding it.

**`1_114` (new this era, no E3 precedent) went down substantially** (-0.094,
entirely on `untagged`). No mechanism proposed — it is the newest fold in the
roster (added 2026-09-11) and has the fewest training passes of any fold's
history to compare against. Worth watching on the next `yamnet_aves`-family
run rather than investigating now.

**`yamnet_aves` is now a confirmed structural gain for this era**, on top of
`yamnet_context` (+0.085). IDEAS.md item 1b called for testing it against
`yamnet_context` too, and item 1c (wide hidden head) should now be run against
both `yamnet` and this representation before either is combined with the other.

trust: clean — matched budget and stopping rule against the anchor, no
selection-on-held-out-fold hazard (no `--monitor`, fixed epochs only).
