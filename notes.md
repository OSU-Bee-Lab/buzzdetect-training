# pauc-l01
## Hypothesis
IDEAS #33's remaining rung. `pauc-l03` (lambda 0.3: squared hinge, margin 1, between each `ins_buzz` positive's logit and the top 5% of the batch's negative logits) lost -0.091 +- 0.015 vs ps-depth8-r3, most on 1_114/1_150/1_95, with no logit collapse. At lambda 0.1 the BCE dominates more; if the loss at 0.3 was a dose overshoot, 0.1 lands at or above the three-draw depth8 mean (0.507) with hard folds up. Falsifier: monotone decline with dose (0.1 below 0.507, between it and pauc-l03's 0.406), as `pairwise-rank` showed; then the low-FPR-loss family closes. Watch each fold's `buzz logit SD` for collapse.
## Changes
None to code beyond exp/pauc-l03's `TRAIN_PAUC_LAMBDA` (patched in). Run: `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 TRAIN_PAUC_LAMBDA=0.1`, embedder `yamnet_trunk_pitchshift_depth8`, 30 epochs, medium.
