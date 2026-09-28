# ps-untiled

## Hypothesis
v4-ft-ps's resampled up view, untiled (yamnet_trunk_pitchshift_untiled_depth12).
If untiling is what costs 1_114, this lands near v4-ft-ps-fast on 1_114; if the
shared STFT is, it lands near v4-ft-ps.
Isolates which of ps-fast's two changes cost 1_114 (-0.174 vs v4-ft-ps). The 2x2:

                     tiled                         untiled
  resampled audio    v4-ft-ps (0.452)              ps-untiled
  shared STFT        ps-sharedtiled                v4-ft-ps-fast (0.432)

Comparators: v4-ft-ps and v4-ft-ps-fast. Config identical to both (30 epochs,
TRUNK_LR_BACKBONE=1e-5, TRUNK_LR_HEAD=2e-4, TRUNK_BATCH=1024, TRUNK_FP16=1,
CHUNK_FRAMES=48). Caveat: medium's translation changed between the comparators
and these runs (b4747d4: mech_quadcopter -> mech_machinery), ~46 s of non-buzz
audio; buzz labels unchanged.

## Changes
New embedder only (main b61da7a).

## Results

Headline sens@fpr0.005 (excl. quiet): **0.439**, down against v4-ft-ps (0.452,
-0.013 ± 0.013) and up against v4-ft-ps-fast (0.432, +0.007 ± 0.012). Seven
folds sit within their SD of both comparators.

1_114: **0.393**, down -0.078 ± 0.037 from v4-ft-ps (0.471) and up +0.096 ±
0.039 from v4-ft-ps-fast (0.297). Untiling alone accounts for about a third
of ps-fast's full -0.174 drop on this fold — a smaller share than the shared
STFT's -0.114 (see ps-sharedtiled/notes.md).

## Conclusion

Untiling costs 1_114 on its own, but less than the shared STFT does. Combined
with ps-sharedtiled's result, the two changes look roughly additive on this
fold (-0.078 + -0.114 ≈ -0.192, close to fast's actual -0.174), not a single
change or an interaction. Unlike the shared STFT, untiling alone does cost
something on the headline (-0.013), so it isn't a free win even ignoring
1_114. No cell in the 2x2 keeps 1_114 at baseline while dropping either
change.
