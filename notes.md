# ps-sharedtiled

## Hypothesis
Shared-STFT up view, tiled (yamnet_trunk_pitchshift_sharedtiled_depth12). If
the shared STFT's coarser frequency resolution is what costs 1_114, this lands
near v4-ft-ps-fast on 1_114; if untiling is, it lands near v4-ft-ps.
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

Headline sens@fpr0.005 (excl. quiet): **0.451**, flat against v4-ft-ps (0.452,
-0.001 ± 0.013) and up against v4-ft-ps-fast (0.432, +0.019 ± 0.012). Seven
folds sit within their SD of both comparators.

1_114: **0.357**, down -0.114 ± 0.037 from v4-ft-ps (0.471) and up +0.060 ±
0.032 from v4-ft-ps-fast (0.297). The shared STFT alone accounts for about
two thirds of ps-fast's full -0.174 drop on this fold.

## Conclusion

Shared STFT is not free on 1_114: tiled-but-shared-STFT already loses more
than untiled-but-resampled does (-0.114 vs ps-untiled's -0.078), while costing
nothing on the headline. So the 1_114 loss isn't one change reproducing the
whole effect — it's dominated by the shared STFT's coarser frequency
resolution, with untiling adding a smaller, roughly independent hit on top
(-0.114 + -0.078 ≈ -0.192, close to fast's actual -0.174 combined). No cell in
the 2x2 keeps 1_114 at baseline while dropping either change; v4-ft-ps itself
is still the only one that doesn't lose it.
