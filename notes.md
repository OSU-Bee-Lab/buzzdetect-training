# noise-floor-cv

## Hypothesis

Run-to-run variance on this pipeline (no seed control anywhere: TF nondeterministic
init + shuffle, plus val_loss-trajectory noise in early stopping) is large enough
that single-run per-fold deltas of ~0.02-0.05 carry little signal. Every recent
entry in `log.jsonl` (`trunk-ft` +0.046, `context-embedder` +0.022, the whole
`aug-*` cluster) is read against an unquantified denominator. `std-convergence`
and the `trunk-ft` stopping sweep both bumped into this wall: a same-config
3-fold rerun there moved the mean +0.018 and swung `best_epoch` by 15-50.

This experiment does the control LOOP.md names as expensive and never run:
repeat `cv-baseline`'s exact config as a full 11-fold CV under a fresh name,
join per fold against `models/yamnet_medium_general`, and report the per-fold
spread. That spread is the denominator for every delta in the log.

Prediction: per-fold |Δ| against `yamnet_medium_general` is ~0.02 median with
some quiet folds (Diel, Opp/08-27) swinging 0.05+. Headline mean sens@fpr0.005
lands within ~0.015 of 0.206.

## Changes

None. Stock `main` config, fresh `--name`. Linear probe on frozen YAMNet GAP
embeddings, Dropout(0.2), BinaryCrossentropy(label_smoothing=0.2), Adam 2e-3,
`general` translation, `medium` set, 11 rotating folds. No stage 2 (reuses the
existing GAP embedding cache).

## Results

`noise_floor_r1` (this run) vs `models/yamnet_medium_general` (the logged
baseline model), paired per fold via `tools/compare_folds.py`:

| fold | baseline | r1 | delta | frames_val |
|---|---|---|---|---|
| Luke - Various Opp/2025-08-05/31 | 0.148 | 0.017 | **-0.131** | 942 (~2 neg frames) |
| Lily Adam/willard/2024-08-07/1_11 | 0.191 | 0.102 | **-0.089** | 4730 |
| Luke - Various Opp/2025-08-27/48 | 0.035 | 0.008 | -0.027 | 1571 (~6 neg) |
| JamesU - MustardBumbler/1_29 | 0.447 | 0.437 | -0.010 | 6984 |
| Luke - Diel Drivers/2026-04-08/1_150 | 0.014 | 0.014 | 0.000 | 4947 |
| Lily - Fit+Fast/…/53 | 0.368 | 0.373 | +0.005 | 4712 |
| Luke - Diel Drivers/2026-05-06/1_95 | 0.028 | 0.034 | +0.006 | 6628 |
| Luke - Various Opp/2025-08-12/1_114 | 0.161 | 0.172 | +0.011 | 3768 |
| Luke - Various Opp/2025-06-23/1_23 | 0.326 | 0.343 | +0.017 | 315 (~1 neg) |
| Lily Adam/wooster/2024-07-26/1_143 | 0.235 | 0.254 | +0.019 | 4708 |
| Luke - Various Opp/2025-07-03/1_37 | 0.241 | 0.277 | +0.036 | 4715 |

- mean sens@fpr0.005: baseline 0.199 → r1 0.185 (**-0.014**)
- 6 folds up, 4 down, 1 flat. Mean |Δ| = 0.035; **median |Δ| ≈ 0.017**.
- The three biggest movers are all the quiet/small folds: Opp/08-05 (only ~2
  negative frames set its threshold), willard, Opp/08-27. Their swings
  (0.03–0.13) match README's "±0.25 or worse where there's little buzz" warning.
- Among the folds with real buzz, |Δ| is 0.005–0.036 **except willard at
  −0.089** — willard is a 4730-frame fold, not thin, and it still swung ~0.09
  from nothing but reinit/shuffle. It was also the biggest mover in `trunk-ft`
  (+0.114) and the biggest loser in `context-stack` (−0.074): willard is
  intrinsically high-variance, not a reliable per-fold signal in any experiment.

## Conclusion

**The run-to-run noise floor on this pipeline is ~±0.017 median per fold, ~0.014
on the headline mean, and 0.03–0.13 on the four quiet folds.** This is now
measured, once, against `yamnet_medium_general`.

Implications for the log:

- Any headline delta under ~0.015 with folds split either way (`binary-translation`
  −0.003, `embed-mixup` −0.007, `class-weight-fix` +0.013, the whole `aug-*`
  cluster, the `trunk-ft` min_delta sweep) is **inside this floor** — the
  `caveated`/inconclusive calls on those stand and none should be built on.
- `trunk-ft-1e5`'s **+0.046 vs frozen (9/11 folds one direction, several
  >0.05)** clears it: a pure-noise CV here gave 6up/4down/1flat, not 9/11
  one-way, so the *direction* is real. But the *size* is inflated by whatever
  willard did that run (+0.114, vs −0.089 from noise alone here) — strip
  willard and it's ~+0.039 over 10 folds. Still the largest real gain in the
  log; the `caveated` trust flag was the right call.
- `context-embedder` +0.022 `clean` is ~1.3× the median floor — real but small.
- Per-fold deltas on Opp/08-05, Opp/08-27, Diel/05-06, Diel/04-08 carry
  essentially no signal and should be reported but never used to drive a
  decision.

This is one repeat, not a distribution — it bounds the noise as "at least this
big", not exactly this big. But it's enough to stop reading sub-0.015 mean
shifts as results.

Trust: **caveated** — a single repeat, so it establishes a floor, not a
variance estimate; and the headline −0.014 is itself within the very noise it
measures (direction of the mean shift is meaningless, only the magnitude
matters).
