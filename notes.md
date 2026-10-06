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
## Results
Rows in main's `05_distill/log.jsonl` (rung C, V-pool hit@K; headline = sensitivity_exclquiet @ FPR 0.005 on the 5 eval folds).

| trunk | x YAMNet (gpu200) | 7k | 14k | 28k | **56k** | hit@K 28k -> 56k |
|---|---|---|---|---|---|---|
| fast32 a0.50 | 2.15x | 0.654 | 0.668 | 0.680 | **0.687** | 66.45 -> 67.07 (+0.62) |
| twofast32 a0.50 | 1.76x | 0.659 | 0.678 | 0.690 | **0.691** | 67.05 -> 67.44 (+0.39) |

fast32 56k vs its 28k branch (`tools/results.py`): +0.007 ± 0.005 (eval sampling only);
per fold 1_29 +0.014, 53 -0.008, 1_11 +0.020, 1_143 +0.018, 1_37 -0.010 (each ± ~0.01).
Loud +0.013, untagged +0.007, background +0.004, quiet +0.001. Inclusive 0.565 -> 0.571.
twofast32 56k vs 28k: +0.001 ± 0.005; per fold 1_29 +0.004, 53 +0.000, 1_11 -0.001, 1_143 -0.006,
1_37 +0.011 (each ± ~0.01); every loudness tier within ±0.003. Inclusive 0.578 -> 0.578.
Speed unchanged (same graph). hit% at logit 0 fell for fast32 (56.99 -> 55.39); the
offset at a fixed cutoff doesn't matter in deployment (users set the threshold).

## Conclusion
Both curves flatten after 28k: the per-doubling gain drops from +0.012 to +0.007
(fast32) and +0.001 (twofast32), both inside seed noise (~0.01). That's the fast32h16 a0.50
shape, not fast32h16 a0.25's. The hit@K stop rule's 28k call cost at most ~0.007 here,
which backs keeping `--wsd-stop 1.3` as it is. The 56k branches are the new best
points for these two trunks (fast32 0.687 @ 2.15x, twofast32 0.691 @ 1.76x), but they don't
move the frontier beyond noise. Doubling past 28k is not a lever for a0.50 fast32-family
students. Width/alpha and front end are where the remaining gains would come from.
