# HANDOFF ps12sf-d8
## Job
One chain runs all of batch 43's training: pid **2749159** (launch_job.sh), log `/home/luke/projects/buzzdetect-training/.local/worktrees/pssf-d8/chain_b43.log`, script `exp/pssf-d8:chain_b43.sh`. Order: lite extractions of pssf/ps12sf embedders → medium extractions → smokes → CVs ps2sf-d8-r2 → pssf-d8 → ps12sf-d8. Launched 2026-10-23 22:30; ETA ~10 h.

Resume: `tools/watch_job.sh --adopt 2749159`, then one `tools/watch_job.sh` Monitor. **If it's still running, report progress and stop** (park again if > 1.5 h remain).

If it died: from `/home/luke/projects/buzzdetect-training/.local/worktrees/pssf-d8`, `/home/luke/projects/buzzdetect-training/tools/launch_job.sh chain_b43.log -- bash chain_b43.sh` (every step resumes; finished folds and extracted idents are skipped). Read the log's last lines first; an OOM after a fold's training is an unbatched call (CLAUDE.md Gotchas).

Batch 43 state: 0 of 4 experiments logged (ps2sf-d8-r2, pssf-d8, ps12sf-d8, sf-ens-d8 remain).

## When this experiment's CV finishes
`tools/results.py ps-depth8-r3 ps12sf-d8` (check buzz logit SD per fold in train.log first), fill notes.md Results/Conclusion (also compare to ps2-seamfree-d8 0.509), then step 5: delete HANDOFF.md, `tools/finish_experiment.sh ps12sf-d8 --model .local/worktrees/ps12sf-d8/models/ps12sf-d8 --baseline-model ps-depth8-r3 ...`.
