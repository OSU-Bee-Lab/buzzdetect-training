# ps-depth4
## Hypothesis
If depth6 still gains, extend to layer 3 cut (tail = layers 4-14), the earliest cut before the spatial map grows to 24x16x128 (4x the cache of depth8). Falsifier: gain turns over, or hard folds fall.

Control: `v4-ft-ps` (0.452) and ps-depth10 / ps-depth8 draws; config identical to ps-depth8 (30 epochs, backbone LR 1e-5, head LR 2e-4, batch 1024, fp16).

## Changes
New embedders yamnet_trunk_depth4 / yamnet_trunk_pitchshift_depth4, cut at layer3_pointwise_conv_relu.

Run config (batch 24): ps-depth6 OOMed at batch 1024 (train step) and at scoring chunk 1024, so this branch carries ps-depth6's `TRUNK_ACCUM` and `SCORE_CHUNK` commits. Depth 4's activations are larger still: `TRUNK_BATCH=256 TRUNK_ACCUM=4` (batch 1024's update, BN frozen), `SCORE_CHUNK=128`, and `TRUNK_STREAM=1` (the 16 GB two-view pool exceeds what the in-RAM lowmem path holds on the 23 GB host). `run_depth4.sh` holds the exact command; it waits on ps-depth6's pid first.

## Results
## Conclusion
