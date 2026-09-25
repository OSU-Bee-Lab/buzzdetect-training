# yamnet-aves-standardize

## Hypothesis

IDEAS.md item 9: the AVES block is ~3.8x YAMNet's per-dim |mean| and 0% zeros
vs. 91.6%, so under one Adam LR the small block is likely underweighted.
`--standardize` (per-dim Normalization fit on the training pool only, fold-safe)
should let the optimizer treat both blocks fairly and lift the headline.
Ported from `exp/standardize-blocks` (pre-cutover) onto current `main`'s
`train.py`/`main.py`, since that branch predates `--fixed-epochs`, the loudness
tiers and the `sample` column and can't be merged directly. No hidden layer —
bare linear probe, matched to the era anchor and to `yamnet-aves-verify`.

## Changes

`03_train/train.py`, `03_train/main.py`: added `--standardize` (off by
default), `_input_stats` (per-dim mean/variance of the training pool only,
`VAR_FLOOR` pass-through for near-zero-variance dims), and a
`keras.layers.Normalization` first layer built from those fixed stats (no
`adapt()`, so it can never see validation data). Config recorded in
`config_model.json`. No change to any embedder or to `metrics.py`.

Ran: `03_train/main.py --name yamnet_aves_standardize --set medium --embedder yamnet_aves --translation general --standardize --verbose -y` (default `--fixed-epochs 400`, `--dropout 0`). No extraction — `yamnet_aves` cache already on disk.

## Results

Matched control: `yamnet-aves-verify` (`.local/worktrees/yamnet-aves-verify/models/yamnet_aves_verify`), same embedder, same fixed-epochs/dropout config, no `--standardize`.

| fold | control | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| `53` | 0.493 | 0.427 | -0.066 | ±0.027 | 9 |
| `1_37` | 0.382 | 0.321 | -0.061 | ±0.050 | 5 |
| `1_150` (hard) | 0.304 | 0.260 | -0.045 | ±0.037 | 14 |
| `1_143` | 0.487 | 0.449 | -0.038 | ±0.066 | 9 |
| `1_95` (hard) | 0.098 | 0.062 | -0.036 | ±0.016 | 23 |
| `1_29` | 0.465 | 0.484 | +0.019 | ±0.026 | 14 |
| `1_114` | 0.156 | 0.190 | +0.033 | ±0.032 | 14 |
| willard (`1_11`) | 0.444 | 0.505 | +0.061 | ±0.022 | 14 |

- mean sens@fpr0.005 excl-quiet: 0.354 -> 0.337 (-0.016 ± 0.013 headline delta SD), 3/8 up, 5/8 down
- inclusive: 0.283 -> 0.267 (-0.016)

**Epoch-budget check (item 9's own falsifier: does standardization need more
epochs than the fixed budget gives it?).** Pooled `mean(last 21 epochs) -
mean(e300-320)` on `val_sens_fpr0.005`: standardize per-fold values
`[0.016, 0.004, 0.018, 0.006, 0.001, 0.011, 0.004, -0.008]` vs. control
`[0.006, 0.026, 0.004, 0.01, 0.003, 0.01, 0.005, -0.007]` — both arms show the
same small late-training drift (~0.006-0.007 mean), so `--standardize` is not
quietly needing a longer budget under `--fixed-epochs 400`. The negative is
not an epoch-budget artifact.

## Conclusion

**Negative, both hard folds down, inside-to-just-outside their own SD.**
`1_95` moves down by more than 2x its delta SD (-0.036 ± 0.016) — the more
confident of the two — and `1_150` moves down at about 1.2x its SD (-0.045 ±
0.037); both in the same direction weakens rather than strengthens the case for
standardization specifically on this pair, though neither alone would clear
the MDE. The headline (-0.016 ± 0.013, ~1.2 sigma) is a real but modest cost,
not a wash: 5/8 folds down against 3/8 up, and the two folds this era cares
about most are both on the down side. `yamnet_aves`'s per-block scale mismatch
does not appear to be the bottleneck `--hidden 1024` (hidden-aves-verify,
+0.022 to +0.031) already addresses on the same embedder — a learned linear
layer can already rescale each block itself, so an explicit fixed
normalization may just be removing information (the AVES block's absolute
scale, not just its direction) that the linear readout was using. Settles
IDEAS item 9 as a negative; do not re-run at this budget without a new
mechanism.
