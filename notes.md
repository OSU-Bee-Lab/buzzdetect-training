# dropout-verify

## Hypothesis

IDEAS item 1d: `Dropout(0.2)` was hardcoded for three eras and never tested
against the current head, current data, or a fixed budget (the anchor runs
`--dropout 0`). One run at `--dropout 0.2` settles whether those eras were
carrying a regulariser that helps, hurts, or does nothing.

## Changes

None — `--dropout` already exists on `main`. Ran plain YAMNet at `--dropout 0.2`
against `cv_baseline_v3` (`--dropout 0`), everything else identical
(`--fixed-epochs 400`, `general`, `medium`).

```
03_train/main.py --name dropout_verify --set medium --embedder yamnet \
                 --translation general --dropout 0.2 --fixed-epochs 400 -y --verbose
```

## Results

| fold | dropout=0 (anchor) | dropout=0.2 | delta |
|---|---|---|---|
| 1_150 | 0.266 | 0.241 | -0.025 |
| wooster/1_143 | 0.461 | 0.450 | -0.011 |
| 1_114 | 0.250 | 0.240 | -0.010 |
| 1_95 | 0.052 | 0.045 | -0.007 |
| 1_29 | 0.441 | 0.445 | +0.004 |
| 1_37 | 0.369 | 0.376 | +0.007 |
| willard/1_11 | 0.368 | 0.380 | +0.012 |
| 53 | 0.429 | 0.454 | +0.025 |

- mean sens@fpr0.005 (excl. quiet): 0.330 -> 0.329 (-0.001)
- inclusive: 0.269 -> 0.270 (+0.001)
- 4/8 up, 4/8 down — perfectly split, no directional signal at all

## Conclusion

Dead flat and evenly split — the cleanest possible null. `Dropout(0.2)` did
nothing measurable on plain YAMNet at this budget, in either direction. Three
eras of results carrying it were not carrying a regulariser that mattered one
way or the other, on this representation. Consistent with `03_train/CLAUDE.md`'s
argument that dropout on YAMNet's 89.6%-sparse non-negative code is mild
regularisation, not the heavy multiplicative noise it would be on a dense
signed code (AVES, Perch) — this run doesn't test that half of the claim, only
confirms the YAMNet half is inert. Settles item 1d; no reason to default
`--dropout` on for YAMNet-based runs.
