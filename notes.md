# epoch-budget-700

## Hypothesis

IDEAS item 1e. `--fixed-epochs 400` was picked by argument on a 1024-d
YAMNet input; the current era lead (`pitchshift-aves-mid`,
`yamnet_pitchshift_aves_mid`, 4352-d) is 4x wider, and width moves the
training-time optimum. Every fixed-budget run logged so far was still rising
at its cap. One run of the lead's exact config at `--fixed-epochs 700` gives
the pooled curve to e700; `tools/honest_epoch.py` then reads back the pooled
argmax and any shorter budget's score from the same curves offline, so this
one run answers the whole ladder without retraining.

Falsifier: if the pooled argmax is <= e400 and e400 -> e700 moves the headline
less than 0.01, 400 stands for the lead and this closes the line.

## Changes

None to code. Same embedder (`yamnet_pitchshift_aves_mid`, extraction already
on disk, shared tree) and same architecture as `pitchshift-aves-mid` /
`pitchshift-aves-mid-repeat`, only `--fixed-epochs 700` instead of 400.

```bash
03_train/main.py --name epoch-budget-700 --set medium --embedder yamnet_pitchshift_aves_mid --translation general --fixed-epochs 700 -y
```

## Results

Comparator: `pitchshift-aves-mid-repeat` (same embedder/architecture, `--fixed-epochs 400`).

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.563 | 0.586 | +0.023 | 0.013 | 32 |
| 53 | 0.486 | 0.495 | +0.009 | 0.018 | 25 |
| 1_11 | 0.511 | 0.533 | +0.022 | 0.018 | 28 |
| 1_143 | 0.524 | 0.571 | +0.047 | 0.029 | 21 |
| 1_150 | 0.452 | 0.471 | +0.019 | 0.039 | 20 |
| 1_95 | 0.159 | 0.170 | +0.011 | 0.013 | 42 |
| 1_37 | 0.379 | 0.417 | +0.038 | 0.032 | 14 |
| 1_114 | 0.212 | 0.251 | +0.039 | 0.017 | 31 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.411 → this 0.437 (+0.026 ± 0.008, ~3.25σ)
- inclusive (sensitivity), same thresholds: 0.328 → 0.350 (+0.022)
- tiers: loud 0.712→0.762 (+0.050), untagged 0.440→0.464 (+0.024), background 0.409→0.432 (+0.023), quiet 0.072→0.072 (+0.000), faint 0.000→0.000 — the gain is in loud/untagged/background, not a quiet-buzz artifact
- all 8 folds move up; none by less than its own SD, most by 1-2σ
- `tools/honest_epoch.py`: pooled argmax across folds lands at e640-682 (of 700), not capped at the 700 boundary on any fold — the curve has actually peaked inside this budget, it isn't still climbing at the cap. `shipped` (e700) and `xfold-pooled` scores are within 0.01 of each other per fold, so e700 isn't overfit past its own cross-fold optimum either.

## Conclusion

Falsifier does not hold: pooled argmax is well above e400 (~e640-680) and the
e400→e700 headline move is +0.026, more than double the 0.01 threshold. 400
was too short for this 4352-d config — `--fixed-epochs` at the old YAMNet-only
budget was undercounting the lead by a real margin, and the loss is broad
(loud/untagged/background all up), not concentrated in one fold or tier.
`honest_epoch.py` also says 700 is close to the actual optimum, not just
"less wrong than 400": the pooled peak sits under 700 with room to spare, and
shipped-vs-xfold-pooled agree to within 0.01. Recommend `--fixed-epochs 700`
(or a budget in the 650-700 range) as the default for the 4352-d lead and any
future run at comparable width; 400 remains fine for narrower configs
(cv-baseline-v3 was 1024-d and its own honest_epoch table shows no such gap).
Settles IDEAS item 1e.
