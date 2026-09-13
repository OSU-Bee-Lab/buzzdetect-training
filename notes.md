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

| fold | control (h=0) | this (h=1024) | delta |
|---|---|---|---|
| 1_114 | 0.156 | 0.075 | -0.081 |
| 1_37 | 0.382 | 0.347 | -0.035 |
| 53 | 0.493 | 0.521 | +0.028 |
| 1_29 | 0.465 | 0.495 | +0.030 |
| wooster/1_143 | 0.487 | 0.533 | +0.046 |
| **1_95 (jet, hard)** | 0.098 | 0.149 | **+0.051** |
| **1_150 (positives, hard)** | 0.304 | 0.365 | **+0.061** |
| willard/1_11 | 0.444 | 0.526 | +0.082 |

- mean sens@fpr0.005 (excl. quiet): 0.354 -> 0.376 (**+0.022**)
- inclusive: 0.283 -> 0.301 (+0.018)
- 6/8 folds up, 2 down (1_114, 1_37 — same two folds that were also weak in
  `yamnet-aves-verify` itself, no new mechanism proposed here)

Tier movement (pooled, same thresholds): `untagged` 0.380 -> 0.401 (+0.021),
`background` 0.364 -> 0.403 (+0.039), `loud` 0.704 -> 0.686 (-0.018, thin,
115 frames), `quiet` 0.063 -> 0.073. `untagged`/`background` both up — a real,
if modest, detection gain, not a quiet-buzz shuffle.

## Conclusion

Headline delta (+0.022) sits inside the ~0.027 MDE, but the fold pattern is
coherent rather than scattered — 6/8 up, and **both named hard folds moved up
together for the first time this era**: 1_150 +0.061, 1_95 +0.051. Per
LOOP.md, that concentration is the result, not a caveat on it; it is not
grounds for `caveated` on its own.

This **reverses the sign** of the other two legs of item 1c
(`hidden-head-verify`: plain YAMNet, null, 1_150 down; `hidden-context-verify`:
yamnet_context, null, 1_150 down). All three ran the identical `--hidden 1024`
mechanism at the identical budget — the only variable is the representation
underneath. So `--hidden`'s effect is representation-dependent: it hurts
1_150 on YAMNet and yamnet_context, but helps both hard folds on yamnet_aves.
No mechanism proposed for why AVES concat specifically benefits from a shared
representation where the other two don't; worth a look if this gets revisited
(AVES's block is dense/signed vs YAMNet's sparse/non-negative code, which is
also why `--dropout` behaves differently across representations per
`03_train/CLAUDE.md`).

Item 1c (all three legs) is now closed. `--hidden 1024` on `yamnet_aves` is a
positive, inside-MDE lead worth combining with future `yamnet_aves`-based
work, not something to build a headline claim on yet — one more run (a repeat
draw) would settle whether the hard-fold coherence replicates.
