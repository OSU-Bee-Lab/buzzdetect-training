# HANDOFF: step-budget curve for distillation (warmup-stable-decay)

The code is in: `main.py --wsd`, `distill_train.py --schedule wsd`, `ladder_record.py wsd`
(05_distill/README.md, "Step budget", has the design and the choices made). What is left is the experiment.
Delete this file when the curve is reported.

## Running (launched 2026-10-02 from the main checkout, replacing the cosine rung-C job)

    main.py --teacher v4-ft-ps-e60-moderate --rung C --wsd-max 56000 --wsd-stop 1.3 \
      --runs "yamnet:a0.50:select:classes=ins_buzz+ambient_rain+human fast32h16:a0.50:classes=ins_buzz+ambient_rain+human"

The 28k cosine job it replaced had not started training yet (it was caching; the cache keeps every finished
slice). Trunks `fe_C_..._wsd`, branches `..._wsd7000/14000/28000/56000`, each judged. The trunk stops
early once a doubling raises teacher-hit % (V buzz detections shared with the teacher, 100 - lost%) by
< 1.3 points (checked 2026-10-02: across the 52 ladder runs it predicts the fold headline, r = 0.86 overall,
0.96 for buzz+rain+human students; 1.3 points is about 0.009 headline). Log `05_distill/data/main_rungC_wsd.log`. Same command resumes. If 56k still improves,
rerun with `--wsd-max 112000 --wsd-halvings 4` (the trunk extends from its last checkpoint).

## Still to do

1. **Equivalence check** (README's last paragraph on WSD): a `wsd7000` branch should land near the 7k cosine run of
   the same student on `mae_live` and lost%. Cheapest: rung B, `--wsd 7000` on
   `yamnet:a0.50:select:classes=ins_buzz+ambient_rain+human` vs the existing
   `fe_B_yamnet_a0.50_s1_select_c-buzz-rain-human`. If clearly worse, the stable lr (1e-3) is wrong:
   try `lr=2e-3` / `lr=5e-4` on the run spec.
2. The rung-B curve on a frontier student: `yamnet:a0.25:select:classes=ins_buzz+ambient_rain+human`,
   `--wsd 7000,14000,28000,70000` (17 min per 7k steps).
3. Report each curve with `ladder_record.py wsd --name <trunk>` and the noise floor (lost% spread 1.3 points,
   headline seed noise ~0.01-0.02); say plainly if it is flat past some budget or still rising at the last.
   Then revisit LADDER.md's "B to C did not advance": the ladder was confounded by passes (A/B/C got ~19/5/1.2).
