# eq-aug-d8
## Hypothesis
IDEAS #32. Each deployment is one recorder, and recorders differ in frequency response (mic, housing, enclosure). Device-response augmentation is the strongest single lever for unseen recording devices in acoustic scene classification (Morocutti et al. 2023, arXiv 2305.07499; DCASE 2023-24 Task 1). No recorder-response augmentation has run in any era. Training the depth8 tail on a random-EQ copy of half of every training fold's snips should make it rely less on a recorder's spectral colouring and lift held-out deployments.

Prediction: headline >= +0.02 over the matched control `ps-depth8-r3` (0.497; depth8 three-draw mean 0.507), with `untagged` or `loud` up, and the movers being folds whose recorders differ most from the pool. The hard folds' confusers (aircraft at 1_95, trill at 1_114) are not a device effect, so they are not the expected movers. Falsifier: headline within 0.02 of the control and no tier up.

Control: `ps-depth8-r3` (same recipe, same embedder cache, no augmentation).
