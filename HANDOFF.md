# HANDOFF: rung-C frontier fill (`chain_fill.sh`)

Delete this file once the results are written up in 05_distill/FRONTENDS.md and sent to Luke.

Context in one paragraph: distillation trains small fast "students" to mimic the deployed teacher; the
frontier is sensitivity (headline, `sensitivity_exclquiet` @ fpr 0.005 on the rotating folds) against speed
(x YAMNet, GPU). FRONTENDS.md "Update 2026-10-04" has the WSD step-budget results and the test-set
contamination story: 64 older students saw ~0.6% test-set audio; Luke dropped the retrain, so those numbers
stand as informative (rows in `05_distill/ladder/<teacher>.contaminated_2026-10-02.jsonl`). Everything this
chain trains is clean and lands in the tracked log `05_distill/ladder/<teacher>.jsonl` (README, "The
experiment log"). Luke's chart: `tools/human/frontier.svg` (buzz+rain+human students only, outline = rung).

## Running

    tools/launch_job.sh 05_distill/data/chain_fill.log -- bash 05_distill/chain_fill.sh    # 2026-10-04 23:47, pid 2098505

Expected ~17 h (finish ~2026-10-05 17:00), up to ~26 h if no stop rule fires early. Steps (header of the script):

1. fast32h16 a0.50 **rung B** WSD 7k-56k, no stop (~4.5 h). The question: rung C's gain for small students
   came with more steps (C 7k 0.605 = B 7k 0.607; C 28k 0.639, 56k 0.650). Does B with the same steps match it?
   Same steps & B ≈ C means steps, not data; B clearly below means C's data matters.
2. yamnet a0.25 rung C, stop rule (~2 h): frontier near the 1.5x speed floor.
3. rung-C mels for fast32 + twofast32 (one CPU decode, ~50 min), then fast32 a0.50 rung C, stop rule.
4. twofast32 a0.50 rung C, stop rule.
5. Repeat rule: fast32h16 a0.50 rung C seed 2, 7k/14k/28k: seed noise on the best clean frontier point.
6. proxy table, `ladder_record.py wsd` for each trunk, regenerated frontier.svg / frontier.html / wsd.svg.

`--wsd-stop` is 1.85 (last measured rung-A hit@K spread; only one clean rung-A run exists now).

**Checking it** (once, not a loop):

    ps -p 2098505 >/dev/null && echo running || echo ended
    grep -E "\[fillchain\]|Traceback|launch_job\] exit" 05_distill/data/chain_fill.log | tail -30

`attempt 2 FAILED` = that step was skipped: read the Traceback above it, fix, relaunch the same command (every
stage resumes). `skipped (WSD plateau, --wsd-stop 1.85)` lines are the stop rule working.
**Killing:** kill launch_job's pid and the python under it (`pgrep -af "[d]istill_train|[0]5_distill/main"`),
then check `nvidia-smi --query-compute-apps=pid --format=csv`.

## When it ends

1. Write up in FRONTENDS.md (new dated update): the B-vs-C curve answer (say plainly if inside noise; the
   seed-2 repeat in step 5 gives the noise on a WSD branch), where each new rung-C student stops and lands,
   and how the frontier moved. Compare at equal steps, never C-at-28k vs B-at-7k.
2. Send Luke `tools/human/frontier.svg` and `tools/human/wsd.svg` (SendUserFile), two lines of headline.
3. Delete this file; commit.
