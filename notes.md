# aves-layer-profile

## Hypothesis

IDEAS item 20a. `aves-mid`'s layers 6/9/12 were picked by argument, not by
looking at which AVES depth actually carries buzz. The fold pattern says
depth matters: `aves-mid` gained on `1_150` but cost `1_37` and (inside the
lead) `1_114`. This extracts all 12 AVES transformer layers, mean-pooled,
alongside the lead's YAMNet blocks (`yamnet_pitchshift_aves_all12`,
11264-d), then sweeps each layer individually offline with a fast sklearn
linear readout (real leave-one-fold-out rotation, `03_train/metrics.py`'s
own `sens_at_fpr`) rather than a full Keras CV per layer. This is a map, not
a verdict: sklearn won't match the Keras probe's numbers, only its ranking.

Falsifier (from IDEAS.md): if the best layers turn out to be 6/9/12 and a
follow-up learned mix (item 20b) puts >=70% weight on them with a headline
inside MDE, `aves-mid` already found the depth answer and item 20 closes. If
a different peak shows up (e.g. layers 3-5), the follow-up is a concat of
those layers instead.

## Changes

New embedder `yamnet_pitchshift_aves_all12` (11264-d = 1024 unshifted YAMNet +
1024 pitch-shifted YAMNet + 12*768 AVES layers, all mean-pooled per frame).
Extraction only; no Keras CV yet. Offline analysis:
`diagnostics/2026-09-16_aves-layer-profile/layer_profile.py` -- for each
rotation and each of the 12 layers, fits a `StandardScaler` +
`LogisticRegression(class_weight='balanced')` on `[yamnet blocks (2048-d),
that layer (768-d)]`, scores the held-out fold with `sens_at_fpr`, and
reports the headline (mean over the 8 rotating folds) plus the four hard
folds (`1_150`, `1_114`, `1_37`, `1_95`) per layer.

```bash
tools/launch_job.sh --no-notify extract_lite_smoke.log -- 02_set/main.py --set lite --embedder yamnet_pitchshift_aves_all12 --workers 0   # BUZZDETECT_NO_GPU=1
tools/launch_job.sh extract_medium.log -- 02_set/main.py --set medium --embedder yamnet_pitchshift_aves_all12 --workers 0   # BUZZDETECT_NO_GPU=1
conda run -n buzzdetect-train python diagnostics/2026-09-16_aves-layer-profile/layer_profile.py
```

## Results

<pending — extraction in flight>

## Conclusion

<pending>
