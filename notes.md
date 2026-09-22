# pitchshift-headtohead

## Hypothesis

`pitchshift-contrast` (this era, 2026-09-21/22) tested whether the explicit
signed contrast `[e_t, e_t - e_shifted(t)]` beats the plain concat
`[e_t, e_shifted(t)]` that `yamnet-pitchshift` used (+0.069, 2026-09-13). Its
own notes.md flagged the comparison as **not fair**: `yamnet_pitchshift`'s
cache and model are from 2026-09-13, and annotations have grown
substantially since (roughly doubled-to-tripled buzz events per fold by
2026-09-14 alone, per `asym-context-yamnet`, and tagging has continued). The
only available comparison read backwards (concat looked better, -0.025)
purely from the stale snapshot.

This reruns `yamnet_pitchshift` (concat) fresh, same annotation snapshot as
`pitchshift-contrast`'s run, same exact training config, so the two designs
are paired: same frame grid, same day, same control
(`pitchshift-contrast-ctrl`, already trained 2026-09-21 on plain `yamnet` at
this exact config). This directly answers whether the contrast mechanism's
own +0.048 beats concat's true current-data gain, not just whether it beats
a three-annotation-revisions-old number.

Prediction, per `pitchshift-contrast`'s own hypothesis: contrast should beat
concat on the same current data, mirroring `asym-context-yamnet` beating
`context-frames-fix` at the same budget. Concat's `1_95` gain and contrast's
`1_114` gain in their (unpaired) historical numbers hinted they may extract
different information; this run finds out on matched data.

## Changes

None -- no new code. Extracts `embedders/yamnet_pitchshift` (existing,
unchanged) fresh on `medium`'s current annotation snapshot, then trains it at
the exact same config `pitchshift-contrast`/`pitchshift-contrast-ctrl` used
(default `--epochs 400`, `--translation general`, frozen linear probe).

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium --embedder yamnet_pitchshift
tools/launch_job.sh train.log -- 03_train/main.py --name pitchshift-headtohead --set medium --embedder yamnet_pitchshift --translation general -y
```

Comparators (both already trained, read directly, not retrained):
- `pitchshift-contrast-ctrl` (`.local/worktrees/pitchshift-contrast/models/pitchshift-contrast-ctrl`) -- fresh plain-yamnet anchor, same day, same config.
- `pitchshift-contrast` (`.local/worktrees/pitchshift-contrast/models/pitchshift-contrast`) -- the contrast design, same day, same config.

*Falsifier:* if this run's concat headline (vs `pitchshift-contrast-ctrl`) is
inside noise of contrast's own +0.048 ± 0.011, the two designs are
equivalent and the contrast mechanism bought nothing extra on pitch-shift
specifically (unlike on temporal context). A clean concat > contrast result
would reverse `pitchshift-contrast`'s conclusion.

## Results

**vs a fresh matched control** (`pitchshift-contrast-ctrl`, plain `yamnet`, same day/config):

| fold | ctrl sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.448 | 0.483 | +0.035 | 0.018 | 32 |
| 53 | 0.436 | 0.436 | +0.000 | 0.029 | 28 |
| 1_11 (willard) | 0.363 | 0.435 | +0.072 | 0.029 | 26 |
| 1_143 | 0.441 | 0.532 | +0.091 | 0.040 | 22 |
| 1_150 | 0.241 | 0.324 | +0.083 | 0.047 | 21 |
| 1_95 | 0.052 | 0.069 | +0.017 | 0.020 | 46 |
| 1_37 | 0.394 | 0.462 | +0.068 | 0.036 | 14 |
| 1_114 | 0.226 | 0.402 | +0.176 | 0.034 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): 0.325 -> 0.393 (**+0.068 ± 0.012**, ~5.7σ)
- inclusive (sensitivity), same thresholds: 0.265 -> 0.321 (+0.056)

This lands very close to `yamnet-pitchshift`'s original 2026-09-13 number
(+0.069, 0.330 -> 0.398) -- on today's larger/different annotation pool, the
concat design's own gain over plain YAMNet reproduces almost exactly.

**Direct head-to-head against `pitchshift-contrast`** (same day, same config,
same frame grid -- the fair comparison this experiment exists to make):

| fold | contrast (baseline) | concat (this exp) | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.468 | 0.483 | +0.015 | 0.012 | 32 |
| 53 | 0.423 | 0.436 | +0.013 | 0.014 | 28 |
| 1_11 (willard) | 0.420 | 0.435 | +0.015 | 0.022 | 26 |
| 1_143 | 0.523 | 0.532 | +0.009 | 0.018 | 22 |
| 1_150 | 0.269 | 0.324 | +0.055 | 0.033 | 21 |
| 1_95 | 0.087 | 0.069 | -0.018 | 0.014 | 46 |
| 1_37 | 0.412 | 0.462 | +0.050 | 0.028 | 14 |
| 1_114 | 0.379 | 0.402 | +0.023 | 0.025 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): `pitchshift-contrast` 0.373 ->
  `pitchshift-headtohead` 0.393 (**+0.020 ± 0.008**, ~2.5σ), 6/8 folds up
- inclusive (sensitivity), same thresholds: 0.302 -> 0.321 (+0.019)
- tiers: loud +0.050, untagged +0.024, background +0.012, quiet +0.011 --
  broad, not a single-tier artifact

## Conclusion

**Falsifier fires: concat beats contrast, reversing `pitchshift-contrast`'s
conclusion.** The comparison that experiment itself flagged as unfair (its
own concat comparator, `yamnet_pitchshift`, was three annotation revisions
stale) turns out to have been hiding the true result. Rerun on the exact
same annotation snapshot, same day, same training config: plain concatenation
`[e_t, e_shifted(t)]` beats the explicit signed contrast `[e_t, e_t -
e_shifted(t)]` by +0.020 ± 0.008 (~2.5σ, 6/8 folds), not the other way
around. `asym-context-yamnet`'s mechanism (a linear head cannot recover a
contrast from a raw concat) does not transfer from the time axis to the
pitch axis -- on pitch-shift specifically, giving the head both raw views
already lets it separate hard folds, e.g. `1_114` (+0.176 over the fresh
plain-YAMNet control) is this era's single largest gain on that fold, edging
out contrast's own +0.153.

The named hard folds split, not sweep, between the two designs: `1_95`
(aircraft) favors contrast (concat is -0.018 worse, weak, ~1.3σ); `1_150`,
`1_37`, `1_114` all favor concat, `1_150` and `1_37` at ~1.7-2.0σ. Neither
design is a clean win across all three -- the two mechanisms extract
partially different information, as `trunk-ft-pitchshift`'s own mixed
1_95-up/1_114-down signature (concat, but through a fine-tuned tail)
already hinted.

Practical read: the fresh `yamnet_pitchshift` run here (0.393) is now this
era's best still-frozen, non-trunk pitch-shift result, and a plain concat
remains the right default for this axis -- do not swap it for the contrast
mechanism based on `pitchshift-contrast`'s own (stale) comparator.
