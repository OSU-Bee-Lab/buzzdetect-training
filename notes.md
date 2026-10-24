# sf-ens-d8
## Hypothesis
Same-recipe ps-depth8 ensembles saturate at three (`ps-d8-ens3` 0.558 = `ps-d8-ens-same5`), and a cross-recipe member only helped when weak members were not swapped in: `plain-d8` (0.418) cost -0.024 to -0.032 against ens3, its 1_114 collapse dragging the mean. `plain-d8`'s conclusion: "ensemble diversity has to come from members that are individually as strong." `ps2-seamfree-d8` is such a member (0.509, at the depth8 three-draw mean) with a different fold profile (1_114 0.670 vs ps draws' 0.41-0.51; 1_150 weaker).

Prediction: swapping one ps-depth8 draw for a seam-free member raises the triple above ens3 by >= +0.02, with 1_114 up and 1_150 not down beyond its SD. Falsifier: the cross-recipe triples at or below ens3, as plain-d8's were.

Disclosure: one triple (ps2-seamfree-d8 + ps-depth8 + ps-depth8-repeat) was scored as a row-compatibility probe before this section was written (0.570, i.e. +0.012 over ens3). The prediction stands as written and every triple below is reported, not only that one.

Design (eval only, `tools_exp/ensemble_predictions.py` from `exp/ps-d8-ensemble`, logit mean): the three swap-one triples {sf + two of a/b/c} against ens3 and against the ps pair alone; once this batch's runs finish, the same with `ps2sf-d8-r2`, both seam-free draws together with one ps draw, and the 4-member {a, b, sf, sf-r2} against ens3. a = ps-depth8, b = ps-depth8-repeat, c = ps-depth8-r3.
