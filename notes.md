# d2-fast-56k
## Hypothesis
IDEAS D2. The rung-C WSD trunks of fast32 a0.50 (2.17x) and twofast32 a0.50 (1.76x) were
stopped at 28k by `--wsd-stop 1.3` (hit@K gains +1.3/+1.4) while the headline still rose
+0.012 per doubling (0.654/0.668/0.680 and 0.659/0.678/0.690). A 56k branch of each should
add ~+0.01 (inside seed noise ~0.01), so this reads curve shape: does it flatten like
fast32h16 a0.50 (0.650 -> 0.649 at 112k) or keep rising like fast32h16 a0.25 (+0.009 at 56k)?
Matched control: each trunk's own 28k branch (same trunk, seed, data).
## Changes
None to code. Resume both trunks to 56k, no stop rule:
`05_distill/main.py --rung C --runs "fast32:a0.50:classes=ins_buzz+ambient_rain+human twofast32:a0.50:classes=ins_buzz+ambient_rain+human" --wsd-max 56000`.
