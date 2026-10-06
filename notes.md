# d1-yamnet-a050-clean
## Hypothesis
IDEAS D1. The top of the student frontier (the candidate *standard* tier) is the
YAMNet-front-end a0.50 student, whose only rung-C readings (0.723/0.722 at 7k/14k,
1.38x) trained on the contaminated `C` pack. Retrained on the clean rebuilt pack
(rung C, WSD to 56k, `--wsd-stop 1.3`, classes buzz+rain+human, seed 1), it should land
within ~0.02 of 0.722. Matched control: `fe_C_yamnet_a0.25_s1_select_c-buzz-rain-human_wsd*`
(same everything but width; 0.693/0.691/0.704/0.707). Falsifier: clean headline more than
~0.02 below 0.722 says the contamination was worth something.
## Changes
None to code. Run: `05_distill/main.py --rung C --runs "yamnet:a0.50:select:classes=ins_buzz+ambient_rain+human" --wsd-max 56000 --wsd-stop 1.3`.
## Results
Clean rung C, seed 1, WSD branches (headline = sensitivity_exclquiet @ fpr 0.005; x YAMNet GPU 20 s / 200 s):

| steps | headline | inclusive | hit@K % | x20 | x200 |
|---|---|---|---|---|---|
| 7k  | 0.723 | 0.608 | 68.58 | 1.51 | 1.39 |
| 14k | 0.720 | 0.604 | 70.38 | 1.50 | 1.39 |
| 28k | 0.725 | 0.609 | 71.03 | 1.47 | 1.40 |

`--wsd-stop 1.3` stopped it at 28k (hit@K gains +1.81 then +0.65: two small doublings in a row), so no 56k branch.

Against the matched control at 28k (`tools/results.py`, a0.25 wsd28000 → a0.50 wsd28000):
mean 0.704 → 0.725 (+0.021 ± 0.008 eval sampling). Per fold (± eval SD; training noise per fold is larger and
not in it): 1_29 +0.030 ± 0.013, 53 +0.014 ± 0.012, 1_11 -0.015 ± 0.011, 1_143 +0.056 ± 0.027, 1_37 +0.018 ± 0.017.
Loudness tiers: loud +0.007, untagged +0.018, background +0.027, quiet +0.012 (faint has 35 frames).
At 7k/14k the gap to a0.25 is +0.030/+0.029. Seed noise on a WSD branch is ~0.01 (FRONTENDS.md 2026-10-05).

## Conclusion
Hypothesis holds: the clean a0.50 reads 0.723/0.720 at 7k/14k against the contaminated 0.723/0.722, so the
contamination was worth nothing measurable here (falsifier was > 0.02 below). The standard-tier candidate now has
a clean, comparable row: 0.725 at 28k, ~1.4x YAMNet at 200 s, the top of the rung-C frontier, ~0.02-0.03 above
a0.25 (1.62x) at every budget. Like the other YAMNet-front-end students it is flat in steps (7k is already within
noise of 28k), so more budget will not move it; width is what separates it from a0.25. One seed; the a0.50-a0.25
gap is 2-3x seed noise, a moderate, not strong, reading.
