# hidden-aves-verify

## Hypothesis

IDEAS item 1c, third and final leg: does a shared 1024-wide ReLU hidden layer
(`--hidden 1024`) help on top of `yamnet_aves`, the era's other confirmed
structural gain? The first two legs (plain YAMNet, `yamnet_context`) were both
null on the headline and both cost `1_150` specifically on its `untagged`
tier. Given that pattern, the working prediction going in was another null
with `1_150` down.

## Changes

Ported the `--hidden` flag from `exp/yamnet-aves-head-fixed` /
`exp/hidden-context-verify` onto this era's anchor (`03_train/main.py`,
`03_train/train.py`): an optional shared `Dense(hidden, relu)` between the
input dropout and the class logits, off by default (`hidden=0`, decoupled
head, matches every other run this era). No other change. Smoke-tested with
`tools/smoke_model.py` before launching.

```
03_train/main.py --name hidden_aves_verify --set medium --embedder yamnet_aves \
                 --translation general --hidden 1024 --dropout 0 --fixed-epochs 400 -y --verbose
```

Matched control: `yamnet-aves-verify`'s `yamnet_aves_verify` model (same
embedder, same budget, hidden=0).

## Results

Against the matched control (`yamnet-aves-verify`, hidden=0). `± SD` is the
paired per-fold delta SD from `tools/eval_sampling_sd.py --other`, eval
sampling only:

| fold | control (h=0) | this (h=1024) | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_114 | 0.156 | 0.075 | -0.081 | 0.030 | 14 |
| 1_37 | 0.382 | 0.347 | -0.035 | 0.042 | 5 |
| 53 | 0.493 | 0.521 | +0.028 | 0.039 | 9 |
| 1_29 | 0.465 | 0.495 | +0.030 | 0.017 | 14 |
| wooster/1_143 | 0.487 | 0.533 | +0.046 | 0.052 | 9 |
| 1_95 (jet, hard) | 0.098 | 0.149 | +0.051 | 0.020 | 23 |
| 1_150 (positives, hard) | 0.304 | 0.365 | +0.061 | 0.041 | 14 |
| willard/1_11 | 0.444 | 0.526 | +0.082 | 0.026 | 14 |

- mean sens@fpr0.005 (excl. quiet): 0.354 -> 0.376 (**+0.022 ± 0.012**)
- inclusive: 0.283 -> 0.301 (+0.018)
- 6/8 folds up, 2 down (1_114, 1_37 — same two folds that were also weak in
  `yamnet-aves-verify` itself, no new mechanism proposed here)

Tier movement (pooled, same thresholds): `untagged` 0.380 -> 0.401 (+0.021),
`background` 0.364 -> 0.403 (+0.039), `loud` 0.704 -> 0.686 (-0.018, thin,
115 frames), `quiet` 0.063 -> 0.073. `untagged`/`background` both up — a real,
if modest, detection gain, not a quiet-buzz shuffle.

## Conclusion

`--hidden 1024` on `yamnet_aves` is worth **+0.022 ± 0.012** on the headline,
confirmed by the repeat draw in `notes-r2.md` (+0.031) — pooled ~+0.027, a real
and modest positive. That is the result.

**The per-fold reading in the original version of this file was wrong, and so
was r2's rebuttal of it.** This section first argued that both named hard folds
moving up together (1_150 +0.061, 1_95 +0.051) was "the result, not a caveat on
it", and proposed a representation-dependent mechanism for it: `--hidden` hurts
1_150 on YAMNet and yamnet_context but helps it on yamnet_aves. Then r2 drew
1_150 at -0.044, and was written up as showing the r1 pattern was a one-draw
artifact.

Neither claim survives contact with the fold's n. `1_150` holds **14 buzz
events**, not 137 independent frames, and its paired delta SD is **0.041** from
evaluation sampling alone, with training stochasticity larger again on top. So
+0.061 was weak evidence for a positive effect — real evidence, not noise to
discard, but nowhere near enough to found a mechanism on — and the later -0.044
is a second draw from the same wide distribution, not a refutation. A
threshold-free read (pAUC over FPR<=0.005: control 0.217, r1 0.270, r2 0.208)
agrees the two models genuinely differ on that fold, but not resolvably.

`1_95` is the sturdier of the two, at +0.051 ± 0.020 and positive again in r2
(+0.022 ± 0.015) — still not a mechanism, but the right place to look next.

Item 1c (all three legs) is closed. The instrumentation gap this exposed is
closed too: `folds_sx.csv` now carries `buzz_events_exclquiet`,
`tools/eval_sampling_sd.py` bootstraps events instead of frames (it understated
per-fold SD by up to ~8x), and LOOP.md requires a per-fold delta to be quoted
with its SD.
