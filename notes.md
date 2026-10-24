# bn-site-d8
## Hypothesis
IDEAS #37 (b) site-adaptive and (c) shrunk BN re-estimation. `bn-pooled` showed post-hoc statistics collapse a model whose weights co-adapted to frozen AudioSet BN (0.481 -> 0.019), so (b)/(c) are run on a model trained with BN tracking: `bntrain-d8` = ps-depth8-r3's recipe with the tail's BatchNorm trainable (`TRUNK_BN_TRAIN=1`, as `ps-bntrain` on depth12, +0.027 there). Its moving statistics then describe the training pool, and the weights were trained under batch statistics, which is what AdaBN (Li et al., arXiv 1603.04779) assumes.

(b) `bn-site-d8`: each held-out fold's BN statistics re-estimated, layer by layer in graph order, on that fold's own frames (labels unused, every view). A deployment's own normalisation should cancel site-level shifts (recorder gain, background spectrum) that set hard folds' thresholds. Prediction: >= +0.02 over `bntrain-d8` (same weights, unadapted), with `untagged` up. Falsifier: within 0.02, or a loss; failure mode to check: 1_95 (jet flyover) and the buzz-rich folds 1_29/53, where the statistics may absorb whatever dominates the fold. Counter-evidence: Boudiaf et al. (arXiv 2302.06658), source-free adaptation mostly hurts on bioacoustic shift.

(c) `bn-shrunk-d8` (logged separately): the same site statistics shrunk toward the pooled ones, lam = n_site / (n_site + 6000) (folds hold 5.0-7.6k frames, so lam ~0.46-0.56; one value, not tuned), mixture moments. Prediction: between (b) and the unadapted model if (b) overshoots, above both if site statistics help but are noisy.

Side result: `bntrain-d8` against `ps-depth8-r3` (0.497) re-measures trainable BN at depth 8.

**Policy caveat (IDEAS #37): (b) and (c) use the eval deployment's audio at test time. In deployment that means two-pass or streaming per-recording inference; Luke decides on that. Results surface to him; no design work toward deploying them.** Also: the site statistics here come from the fold's annotated snips (the frames that are scored), whose class mix over-represents buzz relative to a whole deployment's audio.

## Changes
- `embedders/yamnet_trunk_depth8/embedder.py`: `TRUNK_BN_TRAIN=1` makes the tail's BatchNorm trainable (gamma/beta at the backbone LR 1e-5, moving stats tracking the pool) instead of frozen at AudioSet stats.
- `03_train/train.py`: `TRUNK_SAVE_WEIGHTS=1` saves each fold's final weights (`fold.weights.h5`).
- `tools_exp/soup.py --bn-site N [--bn-shrink N0]`: per held-out fold, re-estimates the 14 tail BN layers' moving mean/variance layer by layer in graph order on that fold's own frames (labels unused, all views, up to N), optionally blended with the pooled stats at lam = n_site/(n_site+N0) (mixture moments), then rescores the fold.
- Run (`chain_b42b.sh`): `bntrain-d8` = medium, `yamnet_trunk_pitchshift_depth8`, 30 epochs, `TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BN_TRAIN=1`. **Batch 512 x 2 accumulated** (`TRUNK_BATCH=512 TRUNK_ACCUM=2`): batch 1024 with trainable BN OOMed the 4 GB card in the smoke (f32 activations kept for BN backward). Update size matches ps-depth8-r3's 1024, but BN batch statistics are over 512, and ps-depth8-r3 had frozen BN; the side comparison differs in both.
- `bn-site-d8` = soup.py `--bn-site 20000` on bntrain-d8's fold weights; `bn-shrunk-d8` = the same plus `--bn-shrink 6000` (logged separately, its own worktree).

## Results
Buzz logit SD: bntrain-d8 folds 0.55-1.23 (healthy). Under site BN 0.80-1.60 except **1_150 at 7.72**: the site statistics blew that fold's logit scale up (per-fold thresholds make the metric rank-only, so it still scores, but deployment thresholds would not transfer). Shrunk: 0.72-1.95.

(b) `tools/results.py bntrain-d8 bn-site-d8` (same weights, only BN stats differ):

| fold | baseline sens@fpr0.005 | this exp | delta | ± SD | buzz events |
|---|---|---|---|---|---|
| 1_29 | 0.487 | 0.435 | -0.052 | 0.024 | 32 |
| 53 | 0.530 | 0.356 | -0.174 | 0.060 | 28 |
| 1_11 | 0.554 | 0.487 | -0.067 | 0.061 | 26 |
| 1_143 | 0.631 | 0.568 | -0.063 | 0.044 | 22 |
| 1_150 | 0.312 | 0.250 | -0.062 | 0.057 | 21 |
| 1_95 | 0.188 | 0.208 | +0.020 | 0.029 | 46 |
| 1_37 | 0.600 | 0.618 | +0.018 | 0.040 | 14 |
| 1_114 | 0.271 | 0.651 | +0.380 | 0.065 | 28 |

- mean sens@fpr0.005 (sensitivity_exclquiet): baseline 0.447 → this 0.447 (+0.000 ± 0.018)
- inclusive (sensitivity), same thresholds: 0.364 → 0.375 (+0.011)
- folds that missed fpr 0.005: none

| tier | baseline | this exp | delta | frames |
|---|---|---|---|---|
| loud | 0.845 | 0.764 | -0.081 | 122 |
| untagged | 0.466 | 0.477 | +0.011 | 2418 |
| background | 0.409 | 0.278 | -0.131 | 1874 |
| quiet | 0.112 | 0.139 | +0.027 | 627 |
| faint | 0.000 | 0.062 | +0.062 | 12 |

Headline exactly flat (+0.000 ± 0.018), but it is a redistribution, not a no-op. 1_114 +0.380 ± 0.065 is far outside its SD: site statistics rescued a fold that bntrain-d8 handled badly (0.271, vs 0.414 under ps-depth8-r3 and 0.670 under ps2-seamfree-d8; its val_loss was still falling at epoch 30). 53 (buzz-rich) -0.174 ± 0.060 is the predicted failure mode: the statistics absorb the fold's dominant content. 1_29 -0.052 ± 0.024 leans down; 1_11, 1_143, 1_150 each about -0.06 with SDs 0.044-0.061 (unsure, but together a lean down); 1_95 (jet flyover) +0.020 ± 0.029 and 1_37 +0.018 ± 0.040 unsure. Tiers: loud -0.081 and background -0.131 lost contrast, quiet +0.027 and untagged +0.011 slightly up.

Side: `tools/results.py ps-depth8-r3 bntrain-d8` (trainable BN at depth 8, batch 512x2): 0.497 → 0.447, **-0.050 ± 0.017**; untagged -0.058; hard folds fell: 1_150 -0.123 ± 0.057, 1_95 -0.096 ± 0.032, 1_114 -0.143 ± 0.040. Unlike depth 12 (`ps-bntrain`, +0.022 two-draw mean), trainable BN hurts at depth 8 under this recipe; the 512-sample BN batch is a confound, not separable here.

## Conclusion
(b) site-adaptive BN: null on the headline (+0.000 ± 0.018), falsifying the >= +0.02 prediction, with large fold-level redistribution: 1_114 +0.380 ± 0.065 and buzz-rich 53 -0.174 ± 0.060, loud/background tiers down. Not a candidate to deploy as is; surfaced to Luke (test-time adaptation is his policy call). The baseline it rests on, bntrain-d8, is itself -0.050 ± 0.017 below ps-depth8-r3, so (b)/(c) were tested on a weaker model than the frozen-BN incumbent.
