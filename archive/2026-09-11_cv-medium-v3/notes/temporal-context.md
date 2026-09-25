# temporal-context

## Hypothesis
Every experiment so far treats each 0.96s YAMNet frame independently. Insect buzz has a temporal signature: it persists for multiple frames and has characteristic dynamics. Mechanical hum (the main confusable) may look buzz-like in a single frame but differ in how that frame relates to its neighbors.

New embedder `yamnet3`: for each frame, concatenates [prev_frame || curr_frame || next_frame] into a 3072-d vector (zero-padded at clip boundaries). Frame count and hop are identical to standard yamnet (0.96s), so all training and inference code is unchanged. Extraction re-embeds from cached framed audio — fast.

Expected: the linear probe can now distinguish genuine buzz (where all 3 slots show buzz-like features) from transient confusable events (where only the center slot looks buzz-like), lifting sensitivity at 95% precision by 3–6pp.

## Changes
- `embedders/yamnet3/embedder.py`: new embedder class `EmbedderYamnet3`

## Results
- Baseline (with_dropout, yamnet, medium): 0.229, CI=[0.216, 0.241]
- This experiment (temporal_context v1–v5): 0.194, 0.209, 0.213, 0.215, 0.213  mean=0.209  median=0.213  95% CI=[0.194, 0.224]
- CIs barely overlap (upper 0.224 vs lower 0.216); temporal_context is likely worse.

## Conclusion
Negative result. Mean drops 2pp; variance is low (std=0.012) so this is consistent. Root cause unclear — likely the 3x wider input space (3072-d) with the same amount of buzz training data makes the linear probe harder to fit, and/or the 2.88s window dilutes the buzz signal per frame.
