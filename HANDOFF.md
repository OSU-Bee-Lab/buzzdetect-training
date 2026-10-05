# HANDOFF: clean retrain of the distillation students (`chain_clean.sh`)

Delete this file when the retrained results are written up. Background, in order: 05_distill/FRONTENDS.md,
"Update 2026-10-04" (the WSD curves, then the test-set contamination and its fix). Short version: packs built
before the 2026-10-02 SeeNote blacklist had no fingerprint and were trusted, so 64 students trained on test-set
audio (`Luke - Pollinator Habitat/2025-07-11/gru`) and V held another test deployment. `pack()` now rebuilds
fingerprint-less packs; the 64 runs are quarantined in `05_distill/data/v4-ft-ps-e60-moderate/_contaminated_2026-10-02/`
(runs, models, eval, ladder rows, manifest.json of their args); `chain_clean.sh` retrains them all.

## Running

    tools/launch_job.sh 05_distill/data/chain_clean.log -- bash 05_distill/chain_clean.sh    # launched 2026-10-04 21:25, pid 2085964

ETA: about 2026-10-06 19:30 (~45 h total). Order: caches (done) -> re-score V on the 9 clean fast32h16 rung-C
branches (done) -> lad_A_s1, lad_A_s2 -> 46 rung-B students (one main.py each) -> 3 WSD steps -> lad_B_s1,
lad_B_a0.375_s1, lad_B_a0.50_d12_s1, lad_C_s1 -> proxy/frontier tables and plots.

Luke asked for no monitoring past the first stable runs: check once when the cron reminder fires (or when asked).

**Checking it** (one look, not a loop):

    ps -p 2085964 >/dev/null && echo running || echo ended
    grep -E "\[cleanchain\]|Traceback|launch_job\] exit" 05_distill/data/chain_clean.log | tail -30
    wc -l 05_distill/data/v4-ft-ps-e60-moderate/ladder.jsonl      # 9 clean rows at launch; +1 per finished run

## Known: the `lad_*` runs fail their last step in the running copy (expected, not a fault)

The running chain's `lad_run` reads `wall.txt` after a `cd`, so its ladder `record` step fails for all six
`lad_*` runs (`attempt 2 FAILED` lines for them). Training, export, eval, probe and speed all complete. The file
on disk is fixed (replaced by rename, so the running bash still reads its old copy). Already handled:
lad_A_s1 recorded by hand; `05_distill/data/record_lad_A_s2.sh` (pid 2090616, log `record_lad_A_s2.log`) records
lad_A_s2 once it is timed, so the WSD steps get the real rung-A hit@K spread.

**When the chain ends** it will report `failed steps: lad_A_s1 lad_A_s2 lad_B_s1 lad_B_a0.375_s1
lad_B_a0.50_d12_s1 lad_C_s1` and `exit 1`. Rerun the same launch command: every finished stage is skipped, the
fixed `lad_run` records the four remaining rows in seconds, and the plots regenerate. Any *other* failed step
needs its Traceback read (above its `attempt 2 FAILED`); fix and rerun the same command.

Also check: every ladder row has `"val_frames": 375019, "buzz_teacher_pos": 7415` (the clean V pool). The
quarantined rows say 7685.

**Killing it:** kill launch_job's pid and also the python under it (`pgrep -af "[d]istill_train|[0]5_distill/main"`),
then confirm the card is free with `nvidia-smi --query-compute-apps=pid --format=csv`.

## Then

1. Rewrite FRONTENDS.md's 2026-10-04 numbers (WSD table, equivalence, frontier, B-vs-C) from the clean
   ladder; compare each clean run with its quarantined twin (`_contaminated_2026-10-02/ladder.jsonl`) and say
   plainly whether the contamination moved anything beyond seed noise (~0.01-0.02). First pair: lad_A_s1
   0.519 contaminated -> 0.512 clean (lad_A_s2 contaminated 0.510).
2. `ladder_record.py proxy`, `ladder_record.py wsd --name <trunk>`, and send Luke `tools/human/wsd.svg` and
   `tools/human/frontier.svg`.
3. The fast32h16 rung-C branches (7k/28k/56k/112k a0.50) are clean and ready for Luke's held-out SeeNote test.
4. Delete this file; commit.
