# decimate-lead-e700

## Hypothesis

Follow-up to `epoch-budget-700` (IDEAS item 1e, settled 2026-09-15):
`--fixed-epochs 400` undercounted the 4352-d era lead by +0.026 (~3.25σ,
pooled argmax ~e640-680). `decimate-lead` is the *same width* (4352-d,
`yamnet_pitchshift_decimate_aves_mid`) trained at the same default 400, and
its own falsifier fired on a headline that was flat (-0.012 ± 0.009) with two
rich folds (1_150, 1_143) dropping near 2σ — settling IDEAS item 19 as a
negative ("the shift-seam axis alone does not fix 1_95").

That verdict may be an epoch-budget artifact rather than a true negative: if
`decimate-lead`'s curve was also still climbing at e400 (plausible, same
width as the lead), re-scoring at e700 could change the sign or size of the
falsifying deltas, particularly on 1_150/1_143 (the folds that drove the
negative) and 1_95/1_114 (the folds the item was betting on).

This is a re-run of `decimate-lead`'s exact config at `--fixed-epochs 700`,
nothing else changed. No extraction (cache already on disk, shared tree).

*Falsifier:* if `tools/honest_epoch.py`'s pooled argmax is ≤ e400 and the
e400→e700 headline move is < 0.01 (same threshold `epoch-budget-700` used),
decimate-lead's e400 verdict stands and this is a true negative, not an
epoch-budget artifact.

## Changes

None. Same command as `decimate-lead`, `--fixed-epochs 700` instead of 400:

```bash
03_train/main.py --name decimate-lead-e700 --set medium --embedder yamnet_pitchshift_decimate_aves_mid --translation general --fixed-epochs 700 -y
```

## Results

**vs `decimate-lead` (same config, e400→e700):**

| fold | e400 | e700 | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.563 | 0.594 | +0.031 | 0.018 | 32 |
| 53 | 0.473 | 0.474 | +0.001 | 0.016 | 25 |
| 1_11 | 0.495 | 0.511 | +0.016 | 0.014 | 28 |
| 1_143 | 0.514 | 0.533 | +0.019 | 0.022 | 21 |
| 1_150 | 0.404 | 0.452 | +0.048 | 0.032 | 20 |
| 1_95 | 0.154 | 0.139 | -0.015 | 0.013 | 42 |
| 1_37 | 0.414 | 0.391 | -0.023 | 0.020 | 14 |
| 1_114 | 0.266 | 0.251 | -0.015 | 0.012 | 31 |

Headline (exclquiet): 0.410 → 0.418 (+0.008 ± 0.007, ~1.1σ). Inclusive:
0.330 → 0.337. Small, under the 0.01 threshold on its own.

`tools/honest_epoch.py`'s pooled argmax is e658-696 of 700 for every fold
(not capped), i.e. far above e400 — the falsifier's first clause fails.
`shipped` (e700) and `xfold-pooled` agree to within 0.005 per fold, so e700
is close to this config's own true cross-fold optimum, same pattern as
`epoch-budget-700`. **The falsifier does not cleanly hold**: the budget was
structurally short (pooled optimum ≫ e400), even though the *realized*
headline gain from training to e700 happened to be small for this
particular config.

**The comparison that actually matters — vs `epoch-budget-700` (the tiled
lead, also at e700, fair matched budgets on both arms):**

| fold | lead@e700 | decimate-lead@e700 | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.586 | 0.594 | +0.008 | 0.016 | 32 |
| 53 | 0.495 | 0.474 | -0.021 | 0.022 | 25 |
| 1_11 | 0.533 | 0.511 | -0.022 | 0.023 | 28 |
| 1_143 | 0.571 | 0.533 | -0.038 | 0.037 | 21 |
| 1_150 | 0.471 | 0.452 | -0.019 | 0.036 | 20 |
| 1_95 | 0.170 | 0.139 | -0.031 | 0.020 | 42 |
| 1_37 | 0.417 | 0.391 | -0.026 | 0.041 | 14 |
| 1_114 | 0.251 | 0.251 | +0.000 | 0.016 | 31 |

Headline: 0.437 → 0.418 (-0.019 ± 0.010, ~1.9σ), 7/8 folds down. Tiers:
untagged -0.020, background -0.009, loud +0.010 (thin, 115 frames) — a
broad-ish cost, not one fold.

## Conclusion

At matched e400 budgets, `decimate-lead`'s original falsifier read `1_114`
as the one clear win (+0.056) against a flat-to-slightly-down `1_95`
(-0.016). **At matched e700 budgets, `1_114`'s gain is gone entirely**
(+0.000 ± 0.016 against the e700 lead) and `1_95` is more clearly down
(-0.031 vs -0.016). The item-19 falsifier still fires, and does so more
cleanly under fair epoch budgets than it did originally — this is not an
epoch-budget artifact rescuing a false negative, it's the opposite: the
original e400 comparison was *flattering* the decimated shift's one
apparent win, which was itself an undertraining artifact on the lead's own
side (the tiled lead was more undertrained at e400 than the decimated
variant happened to be, by coincidence of curve shape, not a real gap).

**This amends `decimate-lead`'s (IDEAS item 19) conclusion.** The
mechanism note — "1_114 is a real representation-level surprise worth a
note for item 20b" — does not survive matched-epoch re-measurement and
should not be carried into item 20b or any future AVES-layer work. Setting
`decimate-lead`'s `trust` to `caveated` with a dated `amended` field
pointing here; not overwriting its original text per LOOP.md's amend
convention.

Separately: the pooled-optimal epoch (~e660-700) sits well above e400 for
this config too, same as `epoch-budget-700` — `--fixed-epochs 700` should
be the default for any 4352-d config being compared to the lead, not just
the lead itself, so this and any future run at this width stay
epoch-matched.
