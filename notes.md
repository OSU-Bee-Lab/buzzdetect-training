# focal-d8
## Hypothesis
The hard folds fail on hard frames: 1_95's and 1_114's thresholds are set by a few dozen confuser negatives (aircraft, trill) scoring like buzz, and 1_150's misses are low-SNR positives. The loss is a frame mean dominated by easy frames (silence, obvious buzz). A focal modulation, (1 - p_t)^2 on each class neuron's weighted BCE (gamma 2, p_t from the unsmoothed target; label smoothing and inverse-frequency pos_weight kept), moves the fine-tuned depth8 tail's gradient onto exactly those frames. Focal loss was only ever tried on a frozen probe (E1, `focal-loss`: shifted the operating point, which a per-fold threshold makes inert); with a trainable tail it can reshape features. Unlike `pauc-l03`/`pauc-l01` (monotone negatives) it adds no ranking term and no named confuser.

Prediction: headline >= +0.02 over `ps-depth8-r3` (0.497; three-draw mean 0.507), with 1_95 or 1_114 up and untagged not down. Falsifier: within 0.02, or hard folds down as under pAUC.

Control: `ps-depth8-r3` (same cache, 30 epochs, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`).
