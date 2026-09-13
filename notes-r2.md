# hidden-aves-verify-r2

## Hypothesis

Repeat draw of `hidden-aves-verify`: is the +0.022 headline gain from
`--hidden 1024` on `yamnet_aves`, and specifically the coherent both-hard-folds-up
pattern (1_150 +0.061, 1_95 +0.051), a real effect or a lucky draw? No seed
control exists in this pipeline, so an independent rerun of the identical
config is the only way to check. Per `LOOP.md`: confirm a hard-fold gain with a
repeat run rather than discounting it on a within-treatment spread argument.

## Changes

None — identical command to `hidden-aves-verify`, different `--name` only:

```
03_train/main.py --name hidden_aves_verify_r2 --set medium --embedder yamnet_aves \
                 --translation general --hidden 1024 --dropout 0 --fixed-epochs 400 -y --verbose
```

## Results

Against the matched control (`yamnet-aves-verify`, hidden=0):

| fold | control | r1 | r2 | r1 delta | r2 delta |
|---|---|---|---|---|---|
| 1_114 | 0.156 | 0.075 | 0.112 | -0.081 | -0.044 |
| 1_150 | 0.304 | 0.365 | 0.260 | **+0.061** | **-0.044** |
| 1_37 | 0.382 | 0.347 | 0.367 | -0.035 | -0.015 |
| 1_95 | 0.098 | 0.149 | 0.120 | **+0.051** | **+0.022** |
| 53 | 0.493 | 0.521 | 0.556 | +0.028 | +0.063 |
| 1_29 | 0.465 | 0.495 | 0.550 | +0.030 | +0.085 |
| wooster/1_143 | 0.487 | 0.533 | 0.581 | +0.046 | +0.094 |
| willard/1_11 | 0.444 | 0.526 | 0.533 | +0.082 | +0.089 |

- headline vs control: r1 +0.022 (0.354->0.376), r2 **+0.031** (0.354->0.385) —
  direction replicated, both inside/at the ~0.027 MDE
- r1 vs r2 directly: +0.009, 6/8 folds up, draw-to-draw noise mostly small
  **except 1_150, which swung 0.105** (0.365 -> 0.260) between two runs of the
  identical config

## Conclusion

**The headline direction replicated; the hard-fold-coherence claim did not.**
`1_150` was the standout of r1 (+0.061) and is the standout *failure* of r2
(-0.044) — a 0.105 draw-to-draw swing on that fold alone, larger than the
0.055 run-to-run figure already on record for `1_150` under a *different*
config (`LOOP.md`'s noise-floor section), so this fold's training-stochasticity
noise is at least that large under `--hidden` too. `1_95` stayed positive both
draws (+0.051, +0.022) — smaller the second time, but consistently on the
correct side of the control, which is the more interesting result now: it's
the fourth-through-what-is-now-fifth-ish observation across two eras that
something about `yamnet_aves`-family configs is the only lever that ever
moves that fold in the right direction, however noisily.

**Amended `hidden-aves-verify`'s log entry** (trust `clean` -> `caveated`) —
its "both hard folds moved up together" framing claimed more than the number
supports; the mechanism note added there is wrong as stated. Read the pooled
picture as: `--hidden 1024` on `yamnet_aves` is a real, modest, headline-level
positive (+0.022 then +0.031, i.e. consistently a touch above the ~0.027 MDE
rather than clearly above it), but do not build any hard-fold narrative on
top of it without a third draw — `1_150` specifically needs to be read as
noisy under this config, not as evidence either way.
