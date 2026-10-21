# sam-d8
## Hypothesis
Every held-out fold is an unseen deployment, so the headline is a domain-generalization score. Sharpness-aware minimization (Foret et al. 2021, arXiv 2010.01412) seeks flat minima, and flatness is the best-supported single predictor of domain generalization in the DG literature (Cha et al. 2021, SWAD, arXiv 2102.08604; DomainBed SAM rows). This loop's flatness proxies were weight averaging, which acts after training (`ps-ema` -0.008, `ps-d8-soup` far below its members): SAM changes the trajectory itself, taking each step from the worst point in an L2 ball of radius rho around the weights.

Prediction: SAM (rho 0.05, on the trainable tail and head) on the ps-depth8 recipe lifts the headline >= +0.02 over ps-depth8-r3 (0.497) with `untagged` up and no hard fold (1_95/1_114/1_150) down beyond its SD. Falsifier: within 0.02 or a loss.

Control: `ps-depth8-r3` (same recipe, cache, epochs and batch; SAM doubles the gradient cost per step but not the epoch count).
