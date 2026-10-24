# quietup6-d8
## Hypothesis
`noquiet-d8` removed quiet/faint buzz samples from training and lost -0.037 ± 0.013, with the hard folds 1_150, 1_95 and 1_114 each down about 0.10-0.13 and untagged -0.048: quiet buzz is the training signal those folds' low-SNR positives depend on (1_150's misses are "very quiet, but still legitimate targets", Luke 2026-09-09). If removing it hurts that much, emphasising it should help. Repeating every quiet/faint-tier training sample 6x (its frames presented 6 times per epoch; class weights computed on the un-repeated pool so only the emphasis changes) on the ps-depth8 recipe lifts the headline.

Dose ladder with `quietup3-d8` (same batch). Prediction: headline >= +0.02 over `ps-depth8-r3` (0.497; depth8 three-draw mean 0.507), with 1_150 up and untagged not down. Falsifier: within 0.02 of the control, or a monotone decline with dose (the emphasis then trades audible-buzz precision for quiet-buzz recall and the threshold eats it).

Control: `ps-depth8-r3` (same cache `yamnet_trunk_pitchshift_depth8`, 30 epochs, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024`).
