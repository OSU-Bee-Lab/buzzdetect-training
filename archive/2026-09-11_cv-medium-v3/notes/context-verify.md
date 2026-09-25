# context-verify

## Hypothesis

Re-verify E3's `yamnet_context` result (IDEAS.md item 1a) against the
cv-medium-v3 era anchor. `context-embedder` (E3) measured +0.022 `clean` on
5-fold, dropout, val_loss-stopped data — an honest rebuild of `context-stack`'s
inflated +0.050. Under the new era (8 folds, no dropout, --fixed-epochs 400,
sensitivity_exclquiet headline) that number has never been re-measured.

`yamnet_context` (`embedders/yamnet_context/embedder.py`) is a standalone
embedder already on `main` — concat(frame t-1, t, t+1) built from real
neighbouring audio at extraction time, 3072-d. No train-time flags to port; no
code changes needed. Just extract it fresh for the current annotations/fold
roster and train under the anchor's exact config, embedder swapped.

**Read the tier row, not just the headline**: cv_baseline_v3 is already at
0.778 on `loud`. A context gain has to show up in `untagged` or `loud` to be a
detection gain rather than a shuffle among frames nobody was promised.

*Falsifier:* no movement in `untagged` or `loud`, whatever the headline does.

## Changes

None — no code changes. `--embedder yamnet_context` in place of `yamnet` on an
otherwise unchanged anchor invocation.

## Results

Ran `context_verify_v1`: `--embedder yamnet_context --set medium --translation
general --fixed-epochs 400 --dropout 0.0` — identical to the `cv_baseline_v3`
anchor invocation except the embedder. All 8 folds trained the full 400 epochs
(no early stopping, `best_epoch` == 400 on every fold in both arms, so no
epoch-selection confound).

| fold | baseline sens@fpr0.005 (exclquiet) | this exp | delta | val frames |
|---|---|---|---|---|
| `1_114` | 0.250 | 0.167 | -0.083 | 5025 |
| `1_95` | 0.052 | 0.017 | -0.035 | 7572 |
| `1_150` | 0.266 | 0.343 | +0.077 | 5091 |
| `53` | 0.429 | 0.525 | +0.096 | 7540 |
| `willard/1_11` | 0.368 | 0.466 | +0.098 | 7562 |
| `1_29` | 0.441 | 0.601 | +0.160 | 7617 |
| `1_143` | 0.461 | 0.640 | +0.179 | 5024 |
| `1_37` | 0.369 | 0.562 | +0.193 | 5033 |

mean sens@fpr0.005 (exclquiet): baseline **0.330** -> this **0.415** (+0.085)
mean sens@fpr0.005 (inclusive): baseline 0.269 -> this 0.341 (+0.072)

Tier breakdown, total row, same thresholds:

| tier | baseline | this | delta |
|---|---|---|---|
| `loud` | 0.778 | 0.870 | +0.092 |
| `untagged` | 0.350 | 0.440 | +0.090 |
| `background` | 0.322 | 0.455 | +0.133 |
| `quiet` | 0.070 | 0.119 | +0.049 |
| `faint` | 0.000 | 0.000 | 0 |

6/8 folds up, 2 down (`1_95`, `1_114`). +0.085 mean is well outside the ~0.027
MDE and the 0.010-0.037 per-fold noise floor.

## Conclusion

**Clean detection gain, not a quiet-buzz shuffle.** `untagged` (+0.090) and
`loud` (+0.092) both moved up as much as the headline, so this passes the
falsifier stated in the hypothesis — the anchor was already at 0.778 on `loud`
and context still added another +0.092 there. `background` moved the most
(+0.133), driven by `1_29`/`53`, but their `untagged` columns (the discrete-buzz
read per LOOP.md item 11) also went up substantially (0.564->0.718,
0.651->0.755), so this isn't only the two folds' continuous-drone component.

**`1_150` (the low-SNR positives hard fold) moved on `untagged`, not just
`quiet`**: 0.266 -> 0.343 is entirely the `untagged` column (that fold has no
`background`/`loud`/`normal` frames), so context is helping on audible buzz
there too, not just picking up faint positives. This is the result LOOP.md says
to read as a result, not a caveat.

**`1_95` (the jet-flyover hard fold) got worse**, and on `untagged` specifically
(0.052 -> 0.017), not confined to `quiet`. Plausible mechanism, not verified:
the jet flyover is itself a sustained ~90 s event, so concatenating a frame with
its real temporal neighbours gives the jet's false-positive frames *more*
context supporting "this is continuous," the same property that helps genuine
sustained buzz. Four other interventions have failed on `1_95`
(`harmonic-comb`, `mech-margin`, `shared-trunk-head`, `context-embedder`'s own
E3 predecessor); this is a fifth, and the direction (negative) matches
`context-embedder`'s E3 result at `1_95` too, so it's consistent rather than a
new finding.

**`1_114` (added this era, 5025 val frames, no `_background`/`loud` tags yet)**
dropped -0.083 on `untagged`, its only scored tier besides `quiet` (also flat at
0.0 in both arms). No mechanism proposed here; flagging for whoever looks at
`1_114` next.

**Verdict: re-verification confirmed.** E3's `context-embedder` (+0.022,
`clean`, 5-fold) direction holds on E4 data at 8 folds under `--fixed-epochs`,
and the size is larger (+0.085 exclquiet), consistent with LOOP.md's framing
that a large one-directional structural result is the likeliest kind to survive
a data change. `yamnet_context` should be a serious candidate representation
for the next structural comparisons (1c's hidden head, item 2's Perch concat),
in place of or alongside plain `yamnet`.

trust: clean — both arms ran the identical rule and budget (`--fixed-epochs
400`, `best_epoch`==400 on every fold both sides), no epoch selection, no
extraction confound (yamnet_context re-extracted fresh from the current
annotations before this run).
