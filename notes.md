# bn-site-d8
## Hypothesis
IDEAS #37 (b) site-adaptive and (c) shrunk BN re-estimation. `bn-pooled` showed post-hoc statistics collapse a model whose weights co-adapted to frozen AudioSet BN (0.481 -> 0.019), so (b)/(c) are run on a model trained with BN tracking: `bntrain-d8` = ps-depth8-r3's recipe with the tail's BatchNorm trainable (`TRUNK_BN_TRAIN=1`, as `ps-bntrain` on depth12, +0.027 there). Its moving statistics then describe the training pool, and the weights were trained under batch statistics, which is what AdaBN (Li et al., arXiv 1603.04779) assumes.

(b) `bn-site-d8`: each held-out fold's BN statistics re-estimated, layer by layer in graph order, on that fold's own frames (labels unused, every view). A deployment's own normalisation should cancel site-level shifts (recorder gain, background spectrum) that set hard folds' thresholds. Prediction: >= +0.02 over `bntrain-d8` (same weights, unadapted), with `untagged` up. Falsifier: within 0.02, or a loss; failure mode to check: 1_95 (jet flyover) and the buzz-rich folds 1_29/53, where the statistics may absorb whatever dominates the fold. Counter-evidence: Boudiaf et al. (arXiv 2302.06658), source-free adaptation mostly hurts on bioacoustic shift.

(c) `bn-shrunk-d8` (logged separately): the same site statistics shrunk toward the pooled ones, lam = n_site / (n_site + 6000) (folds hold 5.0-7.6k frames, so lam ~0.46-0.56; one value, not tuned), mixture moments. Prediction: between (b) and the unadapted model if (b) overshoots, above both if site statistics help but are noisy.

Side result: `bntrain-d8` against `ps-depth8-r3` (0.497) re-measures trainable BN at depth 8.

**Policy caveat (IDEAS #37): (b) and (c) use the eval deployment's audio at test time. In deployment that means two-pass or streaming per-recording inference; Luke decides on that. Results surface to him; no design work toward deploying them.** Also: the site statistics here come from the fold's annotated snips (the frames that are scored), whose class mix over-represents buzz relative to a whole deployment's audio.
