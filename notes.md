# pitch-aux-d8
## Hypothesis
`binary-d8` showed the 14 non-buzz classes' supervision shapes trunk features the buzz boundary uses (collapsing them cost -0.116 at e30). Buzz is the one class still collapsed. Adding pitch subclasses (`buzz_low`/`buzz_medium`/`buzz_high`) as extra multi-label outputs beside the scored `ins_buzz` (unchanged targets, unchanged loss weight) gives the fine-tuned tail more structure within buzz and should sharpen the buzz/non-buzz boundary.

Prediction (IDEAS #38): headline >= +0.02 over the matched control `ps-depth8-r3` (0.497) with `untagged` or a hard fold (1_95/1_114/1_150) up. Falsifier: within ~0.02.

Control: `ps-depth8-r3` (same ps-depth8 cache, recipe, 30 epochs, batch 1024; ran 2026-10-21 era-cohort with plain-d8/sam-d8; medium annotations unchanged since).
