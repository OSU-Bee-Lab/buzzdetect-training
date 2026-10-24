# HANDOFF sf-ens-d8
## Job
One chain runs all of batch 43's training: pid **2749159** (launch_job.sh), log `/home/luke/projects/buzzdetect-training/.local/worktrees/pssf-d8/chain_b43.log`, script `exp/pssf-d8:chain_b43.sh`. Order: lite extractions of pssf/ps12sf embedders → medium extractions → smokes → CVs ps2sf-d8-r2 → pssf-d8 → ps12sf-d8. Launched 2026-10-23 22:30; ETA ~10 h.

Resume: `tools/watch_job.sh --adopt 2749159`, then one `tools/watch_job.sh` Monitor. **If it's still running, report progress and stop** (park again if > 1.5 h remain).

If it died: from `/home/luke/projects/buzzdetect-training/.local/worktrees/pssf-d8`, `/home/luke/projects/buzzdetect-training/tools/launch_job.sh chain_b43.log -- bash chain_b43.sh` (every step resumes; finished folds and extracted idents are skipped). Read the log's last lines first; an OOM after a fold's training is an unbatched call (CLAUDE.md Gotchas).

Batch 43 state: 0 of 4 experiments logged (ps2sf-d8-r2, pssf-d8, ps12sf-d8, sf-ens-d8 remain).

## This experiment
Eval only; needs ps2sf-d8-r2 (and optionally pssf-d8/ps12sf-d8) finished. Run the design in notes.md with `tools_exp/ensemble_predictions.py <name> <member dirs>` (members: .local/worktrees/ps-depth8/models/ps-depth8, ps-depth8-repeat/models/ps-depth8-repeat, ps-depth8-redraw/models/ps-depth8-r3, ps2-seamfree-d8/models/ps2-seamfree-d8, ps2sf-d8-r2/models/ps2sf-d8-r2); results.py each vs .local/worktrees/ps-d8-ensemble/models/ps-d8-ens3; log the best-justified triple (pre-registered: report all) with --baseline-model ps-d8-ens3.
